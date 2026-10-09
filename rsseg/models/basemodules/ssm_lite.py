"""Light state-space (Mamba-2 style) blocks written in plain PyTorch (no CUDA extension to compile).

SSDLite          selective SSM over a 1-D token sequence: scalar decay per head, B/C shared by all heads,
                 chunked "structured state space duality" scan (the dense matmul form inside a chunk, a short
                 recurrence over chunks), so it is parallel over the sequence and needs no custom kernel.
SparseScanBlock  what is scanned and in which order, for land-cover maps:
                   * only the `ratio` most uncertain pixels (classifier entropy) are scanned, the rest are
                     bypassed (cost scales with the number of uncertain pixels, not H*W);
                   * order = 'raster'  row-major (optionally also column-major) order of the chosen pixels,
                     'conf'    grouped by predicted class, most confident first, so the SSM state has already
                               summarised each class before it reaches the ambiguous pixels (semantically
                               related pixels become neighbours however far apart they are in the image),
                     'hybrid'  both a class-ordered and a row-major scan;
                   * every order is scanned forward and backward with shared weights;
                   * a depthwise 3x3 conv before the scan and a 2-D coordinate embedding keep local detail
                     and position, which a sorted 1-D sequence would otherwise lose.
                   * order = 'landcover' builds one independent sequence per predicted class, prepends a
                     prototype pooled from confident pixels, and scans towards frequency-balanced uncertain
                     queries.  Packing classes as separate batch rows resets the SSM state at every class
                     boundary and prevents arbitrary label-order contamination.
                   * selector = 'landcover' spends the same sparse budget on entropy, local-boundary
                     disagreement and known LoveDA confusion pairs, while reserving a small conditional
                     quota for classes that are actually present.  This improves error coverage without
                     increasing the number of unique scanned pixels.
"""
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _segsum(a):
    """a: (..., T) -> (..., T, T) with out[t, s] = sum_{r=s+1..t} a[r] for s <= t, -inf above the diagonal."""
    T = a.size(-1)
    cs = torch.cumsum(a, dim=-1)
    seg = cs[..., :, None] - cs[..., None, :]
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=a.device))
    return seg.masked_fill(~mask, float('-inf'))


def ssd_scan(x, dt, A, Bm, Cm, D, chunk=64):
    """Selective scan  h_t = exp(dt_t A) h_{t-1} + dt_t x_t B_t^T ,  y_t = h_t C_t + D x_t.

    x: (B, L, H, P)   dt: (B, L, H) > 0   A: (H,) < 0   Bm, Cm: (B, L, N)   D: (H,)  ->  y: (B, L, H, P)
    """
    Bsz, L, H, P = x.shape
    N = Bm.shape[-1]
    pad = (-L) % chunk
    if pad:                                   # padded steps: no input, no decay -> state untouched
        x = F.pad(x, (0, 0, 0, 0, 0, pad))
        dt = F.pad(dt, (0, 0, 0, pad))
        Bm = F.pad(Bm, (0, 0, 0, pad))
        Cm = F.pad(Cm, (0, 0, 0, pad))
    Lp = L + pad
    C_ = Lp // chunk

    X = (x * dt.unsqueeze(-1)).view(Bsz, C_, chunk, H, P)
    a = (dt * A).view(Bsz, C_, chunk, H).permute(0, 3, 1, 2)          # (B, H, C, T)
    Bc = Bm.view(Bsz, C_, chunk, N)
    Cc = Cm.view(Bsz, C_, chunk, N)

    a_cum = torch.cumsum(a, dim=-1)                                    # (B, H, C, T)
    # 1) inside a chunk: dense (T x T) causal decay matrix
    Lmat = torch.exp(_segsum(a))                                       # (B, H, C, T, T)
    CB = torch.einsum('bcln,bcsn->bcls', Cc, Bc)                       # (B, C, T, T)
    Y_diag = torch.einsum('bhcls,bcshp->bclhp', CB.unsqueeze(1) * Lmat, X)
    # 2) state at the end of every chunk
    decay_to_end = torch.exp(a_cum[..., -1:] - a_cum)                  # (B, H, C, T)
    states = torch.einsum('bcln,bhcl,bclhp->bchpn', Bc, decay_to_end, X)
    # 3) carry states across chunks
    chunk_decay = torch.exp(_segsum(F.pad(a_cum[..., -1], (1, 0))))    # (B, H, C+1, C+1)
    states = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)
    states = torch.einsum('bhzc,bchpn->bzhpn', chunk_decay, states)[:, :-1]   # state entering each chunk
    # 4) state -> output
    Y_off = torch.einsum('bcln,bchpn,bhcl->bclhp', Cc, states, torch.exp(a_cum))

    y = (Y_diag + Y_off).reshape(Bsz, Lp, H, P)[:, :L]
    return y + x[:, :L] * D.view(1, 1, H, 1)


class SSDLite(nn.Module):
    """Mamba-2-style block on (B, L, C) tokens: in_proj -> selective scan -> gated norm -> out_proj."""

    def __init__(self, dim, expand=1, d_state=8, n_heads=4, dt_min=1e-3, dt_max=1e-1, chunk=64,
                 backend='torch', use_checkpoint=False):
        """backend: 'torch'    chunked scan above (no extra install),
                    'compile'  the same function through torch.compile (fuses the small ops, needs Triton),
                    'triton'   the mamba-ssm Triton SSD kernel (pip install mamba-ssm; checked against 'torch'
                               by tools/profile_scan.py before you rely on it).
        use_checkpoint: recompute the scan in the backward pass (less VRAM, ~+30% time of the scan)."""
        super().__init__()
        assert backend in ('torch', 'compile', 'triton'), backend
        E = int(expand * dim)
        assert E % n_heads == 0, (E, n_heads)
        self.E, self.H, self.P, self.N, self.chunk = E, n_heads, E // n_heads, d_state, chunk
        self.backend, self.use_checkpoint, self._scan_fn = backend, use_checkpoint, None
        self.in_proj = nn.Linear(dim, 2 * E, bias=False)               # scan input x and gate z
        self.bcdt_proj = nn.Linear(E, 2 * d_state + n_heads, bias=False)   # B_t, C_t and one dt per head
        # multi-timescale: heads start with log-spaced step sizes (fast heads = local, slow heads = long memory)
        dt = torch.exp(torch.linspace(math.log(dt_min), math.log(dt_max), n_heads))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))        # inverse softplus
        self.A_log = nn.Parameter(torch.log(torch.linspace(1., 16., n_heads)))
        self.D = nn.Parameter(torch.ones(n_heads))
        self.norm = nn.LayerNorm(E)
        self.out_proj = nn.Linear(E, dim, bias=False)

    def _scan(self, x, dt, A, Bm, Cm, D):
        if self.backend == 'triton':
            from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
            return mamba_chunk_scan_combined(x, dt, A, Bm.unsqueeze(2), Cm.unsqueeze(2), chunk_size=self.chunk, D=D)
        if self._scan_fn is None:
            fn = partial(ssd_scan, chunk=self.chunk)
            self._scan_fn = torch.compile(fn, dynamic=False) if self.backend == 'compile' else fn
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(self._scan_fn, x, dt, A, Bm, Cm, D, use_reentrant=False)
        return self._scan_fn(x, dt, A, Bm, Cm, D)

    def forward(self, tokens):
        Bsz, L, _ = tokens.shape
        x, z = self.in_proj(tokens).chunk(2, dim=-1)
        x = F.silu(x)
        Bm, Cm, dt = torch.split(self.bcdt_proj(x), [self.N, self.N, self.H], dim=-1)
        dt = F.softplus(dt + self.dt_bias)
        y = self._scan(x.view(Bsz, L, self.H, self.P).float(), dt.float(), -torch.exp(self.A_log.float()),
                       Bm.float(), Cm.float(), self.D.float())
        y = y.reshape(Bsz, L, self.E).to(tokens.dtype)
        return self.out_proj(self.norm(y) * F.silu(z))


class SparseScanBlock(nn.Module):
    ORDERS = ('raster', 'conf', 'hybrid', 'landcover')
    SELECTORS = ('uncertainty', 'landcover', 'landcover_v2', 'landcover_fast')

    def __init__(self, dim, order='conf', dirs=2, ratio=0.25, expand=1, d_state=8, n_heads=4,
                 dt_min=1e-3, dt_max=1e-1, chunk=64, backend='torch', use_checkpoint=False,
                 balance=0.5, anchor_topk=16, pos_bands=4, scene_condition=True,
                 selector='uncertainty', boundary_weight=0.5, confusion_weight=0.5,
                 class_quota=0.1, presence_threshold=0.01, presence_peak=0.35,
                 confusion_pairs=((5, 6), (4, 6), (2, 0), (2, 4), (1, 0)),
                 selector_weights=(0.35, 0.25, 0.25, 0.15),
                 selector_quotas=(0.35, 0.25, 0.25, 0.15),
                 num_prototypes=1, confidence_gate=False,
                 warmup_epochs=0, ramp_epochs=0):
        super().__init__()
        assert order in self.ORDERS and dirs in (2, 4), (order, dirs)
        assert selector in self.SELECTORS, selector
        assert 0. <= ratio <= 1., ratio
        assert 0. <= balance <= 1., balance
        assert boundary_weight >= 0. and confusion_weight >= 0.
        assert 0. <= class_quota <= 1.
        assert 0. <= presence_threshold <= 1. and 0. <= presence_peak <= 1.
        assert anchor_topk > 0 and pos_bands > 0
        assert len(selector_weights) == 4 and sum(selector_weights) > 0.
        assert len(selector_quotas) == 4 and sum(selector_quotas) > 0.
        assert num_prototypes in (1, 2), num_prototypes
        self.order, self.dirs, self.ratio = order, dirs, ratio
        self.balance, self.anchor_topk = balance, anchor_topk
        self.selector = selector
        self.boundary_weight, self.confusion_weight = boundary_weight, confusion_weight
        self.class_quota = class_quota
        self.presence_threshold, self.presence_peak = presence_threshold, presence_peak
        self.confusion_pairs = tuple(tuple(int(v) for v in pair) for pair in confusion_pairs)
        weight_sum = float(sum(selector_weights))
        self.selector_weights = tuple(float(v) / weight_sum for v in selector_weights)
        quota_sum = float(sum(selector_quotas))
        self.selector_quotas = tuple(float(v) / quota_sum for v in selector_quotas)
        self.num_prototypes = int(num_prototypes)
        self.confidence_gate = bool(confidence_gate)
        self.warmup_epochs, self.ramp_epochs, self.current_epoch = int(warmup_epochs), int(ramp_epochs), 0
        self.pos_bands = pos_bands if order == 'landcover' else 1
        self.local = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.pos = nn.Linear(4 * self.pos_bands, dim)
        self.scene_proj = (nn.Linear(dim, dim, bias=False)
                           if order == 'landcover' and scene_condition else None)
        self.gate_proj = nn.Linear(3, 1) if order == 'landcover' and confidence_gate else None
        if self.gate_proj is not None:
            with torch.no_grad():
                self.gate_proj.weight.copy_(torch.tensor([[1.0, 0.5, 0.5]]))
                self.gate_proj.bias.fill_(-0.5)
        self.ssm = SSDLite(dim, expand, d_state, n_heads, dt_min, dt_max, chunk, backend, use_checkpoint)
        self.last_conf = None
        self.last_query_conf = None
        self.last_anchor_conf = None
        self.last_selected_per_class = None
        self.last_boundary_coverage = None
        self.last_confusion_coverage = None
        self.last_quota_fraction = None
        self.last_gate = None

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def _routing_scale(self):
        if not self.training:
            return 1.0
        if self.current_epoch < self.warmup_epochs:
            return 0.0
        if self.ramp_epochs <= 0:
            return 1.0
        return min(1.0, (self.current_epoch - self.warmup_epochs + 1) / self.ramp_epochs)

    @staticmethod
    def _perm(key):
        return key.argsort(dim=1)

    def _position(self, idx, H, W, dtype):
        """Multi-frequency 2-D Fourier coordinates; one band reproduces the old encoding."""
        ys, xs = idx // W, idx % W
        yn, xn = ys.float() / max(H - 1, 1), xs.float() / max(W - 1, 1)
        bands = torch.arange(self.pos_bands, device=idx.device, dtype=xn.dtype)
        freq = (2. ** bands) * math.pi
        x, y = xn.unsqueeze(-1) * freq, yn.unsqueeze(-1) * freq
        return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=-1).to(dtype)

    @staticmethod
    def _reverse_queries(x, lengths, prefix_len=1):
        """Keep scene/class prototypes first and reverse only the valid query suffix."""
        L = x.shape[1]
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        is_query = (pos >= prefix_len) & (pos < lengths.unsqueeze(1))
        rev = torch.where(is_query, lengths.unsqueeze(1) + prefix_len - 1 - pos, pos)
        return x.gather(1, rev.unsqueeze(-1).expand(-1, -1, x.shape[-1]))

    @staticmethod
    def _rank01(value, valid=None):
        """Per-image percentile rank, with invalid/zero-support entries pinned to zero."""
        Bsz, HW = value.shape
        order = value.argsort(dim=1)
        rank = torch.empty_like(value)
        scale = max(HW - 1, 1)
        values = torch.arange(HW, device=value.device, dtype=value.dtype) / scale
        rank.scatter_(1, order, values.unsqueeze(0).expand(Bsz, -1))
        if valid is not None:
            rank = rank * valid.to(rank.dtype)
        return rank

    @staticmethod
    def _minmax01(value, valid=None):
        """Cheap per-image normalization used by the fast quota selector."""
        if valid is None:
            lo = value.amin(1, keepdim=True)
            hi = value.amax(1, keepdim=True)
            return (value - lo) / (hi - lo).clamp_min(1e-6)
        masked_lo = value.masked_fill(~valid, float('inf')).amin(1, keepdim=True)
        masked_hi = value.masked_fill(~valid, float('-inf')).amax(1, keepdim=True)
        has_valid = valid.any(1, keepdim=True)
        lo = torch.where(has_valid, masked_lo, torch.zeros_like(masked_lo))
        hi = torch.where(has_valid, masked_hi, torch.ones_like(masked_hi))
        return ((value - lo) / (hi - lo).clamp_min(1e-6)).clamp_(0., 1.) * valid.to(value.dtype)

    def _landcover_signals(self, probs, pred):
        """Return boundary, known-pair confusion, rarity and soft class mass."""
        _, K, H, W = probs.shape
        HW = H * W
        flat_probs = probs.flatten(2)
        local_probs = F.avg_pool2d(F.pad(probs, (1, 1, 1, 1), mode='replicate'),
                                   kernel_size=3, stride=1)
        boundary = (0.5 * (probs - local_probs).abs().sum(1)).flatten(1).clamp_(0., 1.)
        confusion = torch.zeros_like(boundary)
        for a, b in self.confusion_pairs:
            if a < K and b < K:
                confusion = torch.maximum(confusion, 2. * torch.minimum(flat_probs[:, a], flat_probs[:, b]))
        class_mass = flat_probs.sum(2).clamp_min_(1.)
        rarity = (HW / class_mass.gather(1, pred)).pow(self.balance)
        rarity = (rarity / rarity.mean(1, keepdim=True).clamp_min(1e-6)).clamp_(0.5, 4.)
        return boundary, confusion, rarity, class_mass

    def _landcover_queries(self, probs, entropy, pred, budget):
        """Choose exactly ``budget`` unique pixels using land-cover failure signals.

        Entropy alone misses confident boundary errors.  The boundary term measures disagreement with a
        local 3x3 probability average, while the confusion term targets the LoveDA pairs that dominate the
        weak classes (forest/agriculture, barren/agriculture, road/background, road/barren and
        building/background).  A small quota is only activated for a class when both its predicted count
        and its soft probability evidence indicate that it is present; absent classes are never forced.
        """
        Bsz, K, H, W = probs.shape
        HW = H * W
        flat_probs = probs.flatten(2)

        # Replicate padding prevents crop/tile borders from looking like artificial semantic boundaries.
        local_probs = F.avg_pool2d(F.pad(probs, (1, 1, 1, 1), mode='replicate'),
                                   kernel_size=3, stride=1)
        boundary = 0.5 * (probs - local_probs).abs().sum(1).flatten(1)
        boundary = boundary.clamp_(0., 1.)

        confusion = torch.zeros_like(boundary)
        for a, b in self.confusion_pairs:
            if a < K and b < K:
                # Peaks when both classes are plausible; zero when either class has no support.
                pair_score = 2. * torch.minimum(flat_probs[:, a], flat_probs[:, b])
                confusion = torch.maximum(confusion, pair_score)

        # Use soft class mass rather than hard counts to avoid extreme boosts from isolated false labels.
        class_mass = flat_probs.sum(2).clamp_min_(1.)
        pixel_mass = class_mass.gather(1, pred)
        rarity = (HW / pixel_mass).pow(self.balance)
        rarity = (rarity / rarity.mean(1, keepdim=True).clamp_min(1e-6)).clamp_(0.5, 4.)
        score = (entropy + self.boundary_weight * boundary +
                 self.confusion_weight * confusion) * rarity

        # Reserve at most class_quota of the budget.  Each active predicted class receives the same small
        # minimum; unused/duplicate slots automatically return to the global score through the final top-k.
        forced = torch.zeros(Bsz, HW, dtype=torch.bool, device=probs.device)
        quota_total = int(round(self.class_quota * budget))
        if quota_total > 0:
            pred_counts = torch.zeros(Bsz, K, device=probs.device, dtype=probs.dtype)
            pred_counts.scatter_add_(1, pred, torch.ones_like(pred, dtype=probs.dtype))
            peak = flat_probs.amax(2)
            present = ((class_mass / HW) >= self.presence_threshold) & (peak >= self.presence_peak)
            present &= pred_counts > 0
            # Divide exactly quota_total slots among active classes (remainder goes to the first active
            # classes).  This keeps the quota bounded even at the coarse stage where 10% can be < K.
            active_count = present.sum(1).clamp_min(1)
            base = torch.div(quota_total, active_count, rounding_mode='floor')
            remainder = quota_total - base * active_count
            active_rank = present.cumsum(1) - 1
            quota_per_class = (base.unsqueeze(1) +
                               (active_rank < remainder.unsqueeze(1)).to(base.dtype)) * present
            classes = torch.arange(K, device=probs.device).view(1, K, 1)
            candidate_score = score.unsqueeze(1).masked_fill(pred.unsqueeze(1) != classes, float('-inf'))
            quota_value, quota_idx = candidate_score.topk(min(quota_total, HW), dim=2)
            slot = torch.arange(quota_idx.shape[2], device=probs.device).view(1, 1, -1)
            quota_valid = (slot < quota_per_class.unsqueeze(-1)) & torch.isfinite(quota_value)
            batch_idx = torch.arange(Bsz, device=probs.device).view(Bsz, 1, 1).expand_as(quota_idx)
            forced[batch_idx[quota_valid], quota_idx[quota_valid]] = True

        # Forced entries rank before ordinary entries but retain their relative land-cover score.  topk
        # still returns exactly budget unique indices even if fewer classes are present.
        boost = score.amax(1, keepdim=True).clamp_min(1.) + 1.
        q_idx = (score + forced.to(score.dtype) * boost).topk(budget, dim=1).indices
        selected = torch.zeros_like(forced).scatter(1, q_idx, True)
        denom_boundary = (boundary > boundary.mean(1, keepdim=True)).sum(1).clamp_min(1)
        denom_confusion = (confusion > 0.25).sum(1).clamp_min(1)
        self.last_boundary_coverage = ((selected & (boundary > boundary.mean(1, keepdim=True))).sum(1) /
                                       denom_boundary).mean().detach()
        self.last_confusion_coverage = ((selected & (confusion > 0.25)).sum(1) /
                                        denom_confusion).mean().detach()
        self.last_quota_fraction = ((selected & forced).sum(1).float() / budget).mean().detach()
        return q_idx

    def _quota_landcover_queries(self, probs, pred, budget, signals, normalized_signals):
        """Apply explicit signal/class quotas, then fill to an exact unique-token budget."""
        Bsz, K, H, W = probs.shape
        HW = H * W
        flat_probs = probs.flatten(2)
        boundary, confusion, _, class_mass = signals
        score = sum(w * signal for w, signal in zip(self.selector_weights, normalized_signals))

        raw_quota = [budget * q for q in self.selector_quotas]
        quota = [int(v) for v in raw_quota]
        remainder = budget - sum(quota)
        order = sorted(range(4), key=lambda i: raw_quota[i] - quota[i], reverse=True)
        for i in order[:remainder]:
            quota[i] += 1

        forced = torch.zeros(Bsz, HW, dtype=torch.bool, device=probs.device)
        signal_valid = (None, boundary > 0, confusion > 0)
        for signal, valid, slots in zip(normalized_signals[:3], signal_valid, quota[:3]):
            if slots <= 0:
                continue
            _, idx = signal.topk(min(slots, HW), dim=1)
            keep = torch.ones_like(idx, dtype=torch.bool) if valid is None else valid.gather(1, idx)
            batch_idx = torch.arange(Bsz, device=probs.device).unsqueeze(1).expand_as(idx)
            forced[batch_idx[keep], idx[keep]] = True

        class_slots = quota[3]
        if class_slots > 0:
            pred_counts = torch.zeros(Bsz, K, device=probs.device, dtype=probs.dtype)
            pred_counts.scatter_add_(1, pred, torch.ones_like(pred, dtype=probs.dtype))
            peak = flat_probs.amax(2)
            present = ((class_mass / HW) >= self.presence_threshold) & (peak >= self.presence_peak)
            present &= pred_counts > 0
            active_count = present.sum(1).clamp_min(1)
            base = torch.div(class_slots, active_count, rounding_mode='floor')
            remainder_per_image = class_slots - base * active_count
            active_rank = present.cumsum(1) - 1
            quota_per_class = (base.unsqueeze(1) +
                               (active_rank < remainder_per_image.unsqueeze(1)).to(base.dtype)) * present
            classes = torch.arange(K, device=probs.device).view(1, K, 1)
            candidate = score.unsqueeze(1).masked_fill(pred.unsqueeze(1) != classes, float('-inf'))
            value, idx = candidate.topk(min(class_slots, HW), dim=2)
            slot = torch.arange(idx.shape[2], device=probs.device).view(1, 1, -1)
            keep = (slot < quota_per_class.unsqueeze(-1)) & torch.isfinite(value)
            batch_idx = torch.arange(Bsz, device=probs.device).view(Bsz, 1, 1).expand_as(idx)
            forced[batch_idx[keep], idx[keep]] = True

        boost = score.amax(1, keepdim=True).clamp_min(1.) + 1.
        q_idx = (score + forced.to(score.dtype) * boost).topk(budget, dim=1).indices
        selected = torch.zeros_like(forced).scatter(1, q_idx, True)
        boundary_mask = boundary > boundary.mean(1, keepdim=True)
        confusion_mask = confusion > 0.25
        self.last_boundary_coverage = ((selected & boundary_mask).sum(1) /
                                       boundary_mask.sum(1).clamp_min(1)).mean().detach()
        self.last_confusion_coverage = ((selected & confusion_mask).sum(1) /
                                        confusion_mask.sum(1).clamp_min(1)).mean().detach()
        self.last_quota_fraction = ((selected & forced).sum(1).float() / budget).mean().detach()
        return q_idx

    def _landcover_queries_v2(self, probs, entropy, pred, budget, signals):
        """Percentile-rank selector retained as the V2 negative ablation."""
        _, _, H, W = probs.shape
        HW = H * W
        boundary, confusion, _, class_mass = signals
        normalized = (
            self._rank01(entropy),
            self._rank01(boundary, boundary > 0),
            self._rank01(confusion, confusion > 0),
            self._rank01((HW / class_mass).pow(self.balance)).gather(1, pred),
        )
        return self._quota_landcover_queries(probs, pred, budget, signals, normalized)

    def _landcover_queries_fast(self, probs, entropy, pred, budget, signals):
        """Linear-time normalized quota selector; no full token sorting beyond required top-k calls."""
        _, _, H, W = probs.shape
        HW = H * W
        boundary, confusion, _, class_mass = signals
        class_priority = self._minmax01((HW / class_mass).pow(self.balance)).gather(1, pred)
        normalized = (
            self._minmax01(entropy),
            self._minmax01(boundary, boundary > 0),
            self._minmax01(confusion, confusion > 0),
            class_priority,
        )
        return self._quota_landcover_queries(probs, pred, budget, signals, normalized)

    def _forward_landcover(self, feat, logits, tokens, probs, entropy, conf, pred):
        """Anchor-to-uncertain scan with independent state for every image/class pair.

        A class prototype is pooled from its most confident pixels and placed before that class's
        uncertain queries.  Each image/class sequence is a separate batch row, which is equivalent to
        resetting the recurrent state at class boundaries.  Only query outputs are scattered back.
        """
        Bsz, C, H, W = feat.shape
        HW, K = H * W, logits.shape[1]
        budget = HW if self.ratio >= 1 else max(1, int(self.ratio * HW))
        scene = feat.mean(dim=(2, 3))
        if self.scene_proj is not None:
            scene = torch.tanh(self.scene_proj(scene))
        else:
            scene = torch.zeros_like(scene)

        # V1 remains available for exact historical ablations. V2 changes selection, prototypes and
        # residual gating while preserving the exact same number of unique scanned image tokens.
        signals = self._landcover_signals(probs, pred)
        if self.selector == 'landcover_fast':
            q_idx = self._landcover_queries_fast(probs, entropy, pred, budget, signals)
        elif self.selector == 'landcover_v2':
            q_idx = self._landcover_queries_v2(probs, entropy, pred, budget, signals)
        elif self.selector == 'landcover':
            q_idx = self._landcover_queries(probs, entropy, pred, budget)
        else:
            class_counts = torch.zeros(Bsz, K, device=feat.device, dtype=conf.dtype)
            class_counts.scatter_add_(1, pred, torch.ones_like(conf))
            pixel_count = class_counts.gather(1, pred).clamp_min_(1.)
            rarity = (HW / pixel_count).pow(self.balance)
            q_idx = ((1. - conf) * rarity).topk(budget, dim=1).indices
        q_pred = pred.gather(1, q_idx)
        q_conf = conf.gather(1, q_idx)

        # Within every class: prototype -> easier selected query -> hardest selected query.
        perm = (q_pred.float() * 2. + (1. - q_conf)).argsort(dim=1)
        q_idx = q_idx.gather(1, perm)
        q_pred = q_pred.gather(1, perm)
        q_conf = q_conf.gather(1, perm)
        q_tokens = tokens.gather(1, q_idx.unsqueeze(-1).expand(-1, -1, C))
        q_tokens = self.norm(q_tokens) + self.pos(self._position(q_idx, H, W, q_tokens.dtype))

        counts = torch.zeros(Bsz, K, dtype=torch.long, device=feat.device)
        counts.scatter_add_(1, q_pred, torch.ones_like(q_pred))
        one_hot = F.one_hot(q_pred, K)
        pos_in_class = (one_hot.cumsum(1) - 1).gather(2, q_pred.unsqueeze(-1)).squeeze(-1)

        # Prototype 0 uses reliable pixels. Prototype 1, when enabled, uses a disjoint set weighted
        # towards boundaries/confusion regions to represent intra-class land-cover variation.
        classes = torch.arange(K, device=feat.device).view(1, K, 1)
        class_mask = pred.unsqueeze(1) == classes
        anchor_n = min(self.anchor_topk, HW)
        expanded = tokens.unsqueeze(1).expand(-1, K, -1, -1)

        def pool_prototype(anchor_score):
            value, idx = anchor_score.topk(anchor_n, dim=2)
            valid = torch.isfinite(value)
            anchor = expanded.gather(2, idx.unsqueeze(-1).expand(-1, -1, -1, C))
            anchor = self.norm(anchor) + self.pos(self._position(idx, H, W, anchor.dtype))
            weight = valid.unsqueeze(-1).to(anchor.dtype)
            proto = (anchor * weight).sum(2) / weight.sum(2).clamp_min_(1.)
            return proto, value, idx, valid

        global_score = conf.unsqueeze(1).expand(-1, K, -1).masked_fill(~class_mask, float('-inf'))
        global_proto, global_value, global_idx, global_valid = pool_prototype(global_score)
        prototypes = [global_proto]
        anchor_values = [global_value[global_valid]]
        if self.num_prototypes == 2:
            used = torch.zeros_like(class_mask)
            used.scatter_(2, global_idx, global_valid)
            boundary, confusion, _, _ = signals
            hard_evidence = conf * (0.5 + boundary + confusion)
            hard_score = hard_evidence.unsqueeze(1).expand(-1, K, -1)
            hard_score = hard_score.masked_fill(~class_mask | used, float('-inf'))
            hard_proto, hard_value, _, hard_valid = pool_prototype(hard_score)
            hard_proto = torch.where(hard_valid.any(2, keepdim=True), hard_proto, global_proto)
            prototypes.append(hard_proto)
            anchor_values.append(hard_value[hard_valid])
        prototypes = torch.stack(prototypes, dim=2) + scene.unsqueeze(1).unsqueeze(2)

        # Pack every active class into its own batch row.  This is the class-boundary state reset: no state
        # can flow from an arbitrary annotation ID to the next one.  Query packing is fully vectorized to
        # avoid the GPU synchronizations caused by per-image/per-class Python loops.
        active = counts.flatten() > 0
        active_ids = torch.nonzero(active, as_tuple=False).flatten()
        row_lookup = torch.full((Bsz * K,), -1, dtype=torch.long, device=feat.device)
        row_lookup[active_ids] = torch.arange(active_ids.numel(), device=feat.device)
        row_global = (torch.arange(Bsz, device=feat.device).unsqueeze(1) * K + q_pred).flatten()
        query_rows = row_lookup[row_global].view(Bsz, budget)
        query_cols = pos_in_class + self.num_prototypes
        lengths_t = counts.flatten()[active] + self.num_prototypes
        max_len = int(lengths_t.max())
        packed = tokens.new_zeros((active_ids.numel(), max_len, C))
        packed[:, :self.num_prototypes] = prototypes.reshape(
            Bsz * K, self.num_prototypes, C)[active]
        packed = packed.index_put(
            (query_rows.flatten(), query_cols.flatten()), q_tokens.reshape(Bsz * budget, C))

        rev_in = self._reverse_queries(packed, lengths_t, self.num_prototypes)
        # One SSM launch for both directions; weights are shared exactly as in the original block.
        fwd, bwd = self.ssm(torch.cat([packed, rev_in], dim=0)).chunk(2, dim=0)
        bwd = self._reverse_queries(bwd, lengths_t, self.num_prototypes)
        out = 0.5 * (fwd + bwd)

        query_out = out[query_rows.flatten(), query_cols.flatten()].view(Bsz, budget, C)
        if self.gate_proj is not None:
            boundary, confusion, _, _ = signals
            gate_input = torch.stack((1. - q_conf, boundary.gather(1, q_idx),
                                      confusion.gather(1, q_idx)), dim=-1)
            gate = torch.sigmoid(self.gate_proj(gate_input.to(query_out.dtype)))
            query_out = query_out * gate
            self.last_gate = gate.mean().detach()
        else:
            self.last_gate = query_out.new_tensor(1.)
        full = tokens.new_zeros(Bsz, HW, C)
        full = full.scatter(1, q_idx.unsqueeze(-1).expand(-1, -1, C), query_out)

        self.last_conf = conf.mean().detach()
        self.last_query_conf = q_conf.mean().detach()
        valid_anchor_conf = torch.cat(anchor_values)
        self.last_anchor_conf = valid_anchor_conf.mean().detach()
        self.last_selected_per_class = counts.detach()
        return full.transpose(1, 2).reshape(Bsz, C, H, W)

    def forward(self, feat, logits):
        routing_scale = self._routing_scale()
        if routing_scale == 0.0:
            self.last_conf = logits.new_tensor(0.0)
            return torch.zeros_like(feat)
        Bsz, C, H, W = feat.shape
        HW = H * W
        K = logits.shape[1]
        tokens = (feat + self.local(feat)).flatten(2).transpose(1, 2)                  # (B, HW, C)

        with torch.no_grad():
            p = F.softmax(logits.detach().float(), dim=1)
            ent = -(p * torch.log(p.clamp_min(1e-8))).sum(1) / math.log(K)               # (B, H, W) in [0, 1]
            conf = (1. - ent).flatten(1)                                                  # (B, HW)
            pred = p.argmax(1).flatten(1)                                                 # (B, HW)

        if self.order == 'landcover':
            # Indices are discrete routing decisions; feature/prototype computation remains differentiable.
            return self._forward_landcover(feat, logits, tokens, p, ent.flatten(1), conf, pred) * routing_scale

        with torch.no_grad():
            k = HW if self.ratio >= 1 else max(1, int(self.ratio * HW))
            if k == HW:
                idx = torch.arange(HW, device=feat.device).unsqueeze(0).expand(Bsz, -1)
            else:                                                                         # most uncertain, raster order
                idx = (1. - conf).topk(k, dim=1).indices.sort(dim=1).values
            conf_s, pred_s = conf.gather(1, idx), pred.gather(1, idx)
            ys, xs = idx // W, idx % W
            pos = self._position(idx, H, W, tokens.dtype)
            perms = []                                                                    # permutations of the chosen tokens
            if self.order in ('raster', 'hybrid'):
                perms.append(torch.arange(k, device=feat.device).unsqueeze(0).expand(Bsz, -1))   # already row-major
                if self.order == 'raster' and self.dirs == 4:
                    perms.append(self._perm(xs * H + ys))                                 # column-major
            if self.order in ('conf', 'hybrid'):
                perms.append(self._perm(pred_s.float() + (1. - conf_s) * 0.999))          # class, then confident first
                if self.order == 'hybrid':
                    perms = perms[::-1]

        sel = tokens.gather(1, idx.unsqueeze(-1).expand(-1, -1, C))
        sel = self.norm(sel) + self.pos(pos.to(sel.dtype))

        seqs = []
        for pm in perms:
            s = sel.gather(1, pm.unsqueeze(-1).expand(-1, -1, C))
            seqs += [s, s.flip(1)]                                                        # forward and backward
        out = self.ssm(torch.cat(seqs, dim=0)).chunk(len(seqs), dim=0)

        delta = torch.zeros_like(sel)
        for j, pm in enumerate(perms):
            fwd, bwd = out[2 * j], out[2 * j + 1].flip(1)
            delta = delta.scatter_add(1, pm.unsqueeze(-1).expand(-1, -1, C), fwd + bwd)   # back to the chosen-token order
        delta = delta / (2 * len(perms))

        full = torch.zeros(Bsz, HW, C, dtype=delta.dtype, device=delta.device)
        full = full.scatter(1, idx.unsqueeze(-1).expand(-1, -1, C), delta)                # zero at bypassed pixels
        self.last_conf = conf.mean().detach()
        return full.transpose(1, 2).reshape(Bsz, C, H, W) * routing_scale
