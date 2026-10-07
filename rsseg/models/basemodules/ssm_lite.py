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

    def __init__(self, dim, order='conf', dirs=2, ratio=0.25, expand=1, d_state=8, n_heads=4,
                 dt_min=1e-3, dt_max=1e-1, chunk=64, backend='torch', use_checkpoint=False,
                 balance=0.5, anchor_topk=16, pos_bands=4, scene_condition=True,
                 warmup_epochs=0, ramp_epochs=0):
        super().__init__()
        assert order in self.ORDERS and dirs in (2, 4), (order, dirs)
        assert 0. <= ratio <= 1., ratio
        assert 0. <= balance <= 1., balance
        assert anchor_topk > 0 and pos_bands > 0
        self.order, self.dirs, self.ratio = order, dirs, ratio
        self.balance, self.anchor_topk = balance, anchor_topk
        self.warmup_epochs, self.ramp_epochs, self.current_epoch = int(warmup_epochs), int(ramp_epochs), 0
        self.pos_bands = pos_bands if order == 'landcover' else 1
        self.local = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.pos = nn.Linear(4 * self.pos_bands, dim)
        self.scene_proj = (nn.Linear(dim, dim, bias=False)
                           if order == 'landcover' and scene_condition else None)
        self.ssm = SSDLite(dim, expand, d_state, n_heads, dt_min, dt_max, chunk, backend, use_checkpoint)
        self.last_conf = None
        self.last_query_conf = None
        self.last_anchor_conf = None
        self.last_selected_per_class = None

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
    def _reverse_queries(x, lengths):
        """Keep the class prototype first and reverse only the valid query suffix."""
        L = x.shape[1]
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        is_query = (pos > 0) & (pos < lengths.unsqueeze(1))
        rev = torch.where(is_query, lengths.unsqueeze(1) - pos, pos)
        return x.gather(1, rev.unsqueeze(-1).expand(-1, -1, x.shape[-1]))

    def _forward_landcover(self, feat, logits, tokens, conf, pred):
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

        # Frequency-balanced global routing.  balance=0 is the original global uncertainty top-k;
        # balance=0.5 (default) boosts rare predicted classes by inverse-sqrt frequency without Python loops
        # or a hard per-class quota.  Exactly `budget` image tokens are still selected.
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

        # One differentiable prototype per image/class, pooled from the class's most confident pixels.
        # Invalid top-k entries of absent/very small classes are masked before the mean.
        classes = torch.arange(K, device=feat.device).view(1, K, 1)
        class_mask = pred.unsqueeze(1) == classes
        anchor_n = min(self.anchor_topk, HW)
        anchor_score = conf.unsqueeze(1).expand(-1, K, -1).masked_fill(~class_mask, float('-inf'))
        anchor_value, anchor_idx = anchor_score.topk(anchor_n, dim=2)
        anchor_valid = torch.isfinite(anchor_value)
        expanded = tokens.unsqueeze(1).expand(-1, K, -1, -1)
        anchor_tokens = expanded.gather(2, anchor_idx.unsqueeze(-1).expand(-1, -1, -1, C))
        anchor_tokens = self.norm(anchor_tokens) + self.pos(
            self._position(anchor_idx, H, W, anchor_tokens.dtype))
        anchor_weight = anchor_valid.unsqueeze(-1).to(anchor_tokens.dtype)
        prototypes = (anchor_tokens * anchor_weight).sum(2) / anchor_weight.sum(2).clamp_min_(1.)
        prototypes = prototypes + scene.unsqueeze(1)

        # Pack every active class into its own batch row.  This is the class-boundary state reset: no state
        # can flow from an arbitrary annotation ID to the next one.  Query packing is fully vectorized to
        # avoid the GPU synchronizations caused by per-image/per-class Python loops.
        active = counts.flatten() > 0
        active_ids = torch.nonzero(active, as_tuple=False).flatten()
        row_lookup = torch.full((Bsz * K,), -1, dtype=torch.long, device=feat.device)
        row_lookup[active_ids] = torch.arange(active_ids.numel(), device=feat.device)
        row_global = (torch.arange(Bsz, device=feat.device).unsqueeze(1) * K + q_pred).flatten()
        query_rows = row_lookup[row_global].view(Bsz, budget)
        query_cols = pos_in_class + 1
        lengths_t = counts.flatten()[active] + 1
        max_len = int(lengths_t.max())
        packed = tokens.new_zeros((active_ids.numel(), max_len, C))
        packed[:, 0] = prototypes.reshape(Bsz * K, C)[active]
        packed = packed.index_put(
            (query_rows.flatten(), query_cols.flatten()), q_tokens.reshape(Bsz * budget, C))

        rev_in = self._reverse_queries(packed, lengths_t)
        # One SSM launch for both directions; weights are shared exactly as in the original block.
        fwd, bwd = self.ssm(torch.cat([packed, rev_in], dim=0)).chunk(2, dim=0)
        bwd = self._reverse_queries(bwd, lengths_t)
        out = 0.5 * (fwd + bwd)

        query_out = out[query_rows.flatten(), query_cols.flatten()].view(Bsz, budget, C)
        full = tokens.new_zeros(Bsz, HW, C)
        full = full.scatter(1, q_idx.unsqueeze(-1).expand(-1, -1, C), query_out)

        self.last_conf = conf.mean().detach()
        self.last_query_conf = q_conf.mean().detach()
        valid_anchor_conf = anchor_value[anchor_valid]
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
            return self._forward_landcover(feat, logits, tokens, conf, pred) * routing_scale

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
