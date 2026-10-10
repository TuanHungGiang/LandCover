import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from rsseg.models.segheads.logcanplus_head import RVSA_MRAM, SpatialGatherModule, conv_3x3, upsample_add
from rsseg.models.basemodules.ssm_lite import CoverageScanBlock, SparseScanBlock


class SceneExplore(nn.Module):
    """Explore path: every pixel attends to a small set of pooled scene tokens.

    The feature map is average-pooled to a fixed grid x grid, so the cost is O(HW * grid^2)
    and does not depend on the input resolution.
    """

    def __init__(self, dim, num_heads=8, grid=8):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.pool = nn.AdaptiveAvgPool2d(grid)
        self.pos = nn.Parameter(torch.zeros(1, grid * grid, dim))
        trunc_normal_(self.pos, std=.02)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def tokens(self, x):
        return self.pool(x).flatten(2).transpose(1, 2) + self.pos            # B, M, C

    def attend(self, q_in, tokens):
        """q_in: (B, N, C) query pixels, tokens: (B, M, C) pooled scene tokens -> (B, N, C)."""
        B, N, C = q_in.shape
        h = self.num_heads
        q, k, v = self.q(q_in), self.k(tokens), self.v(tokens)
        q, k, v = [t.reshape(B, -1, h, C // h).transpose(1, 2) for t in (q, k, v)]
        out = F.scaled_dot_product_attention(q, k, v)                         # B, h, N, C/h
        return self.proj(out.transpose(1, 2).reshape(B, N, C))

    def forward(self, x):
        B, C, H, W = x.shape
        out = self.attend(x.flatten(2).transpose(1, 2), self.tokens(x))      # every pixel is a query
        return out.transpose(1, 2).reshape(B, C, H, W)


def entropy_gate(logits):
    """1 where the classifier is confident (exploit the class prior), 0 where it is uncertain (explore)."""
    p = F.softmax(logits.detach().float(), dim=1)
    ent = -(p * torch.log(p.clamp_min(1e-8))).sum(dim=1, keepdim=True)
    return 1. - ent / math.log(logits.shape[1])


class GEE_Head(nn.Module):
    """Gated exploit/explore decoder on top of the LoGCAN++ class-center attention.

    mode:
        'gated'        g * exploit + (1 - g) * explore, g from the entropy of the stage classifier
        'sum'          0.5 * exploit + 0.5 * explore (ablation: no gate)
        'exploit_only' class-center attention only (LoGCAN++-style decoder, ablation baseline)
        'explore_only' scene-token attention only (ablation)
        'sparse'       exploit everywhere; the scene-token attention is only run for the `sparse_ratio`
                       fraction of pixels with the highest classifier entropy (residual add), so the
                       explore cost scales with the number of uncertain pixels, not with H*W
        'mamba'        exploit + a light state-space scan (basemodules/ssm_lite.py) over the uncertain pixels,
                       configured by `scan_cfg` (order raster|conf|hybrid|landcover, dirs, ratio, expand, d_state,
                       n_heads, dt_min, dt_max, chunk, stages); residual add, bypass elsewhere
        'mamba_only'   the same scan without the class-center attention (feature + scan residual)

        The landcover order prepends confident class prototypes to frequency-balanced uncertain queries,
        resets SSM state between classes, adds multi-frequency 2-D position, and optionally conditions the
        prototypes on a pooled scene vector. Extra keys are passed through `scan_cfg`.
        `scan_cfg.block_type='coverage'` instead condenses every 2x2 cell to one spatial token, applies
        soft prototypes and a bidirectional serpentine scan, then returns dense context to every pixel.
    """

    def __init__(self,
                 transform_channel,
                 in_channel,
                 num_class,
                 num_heads,
                 patch_size,
                 explore_grid=8,
                 mode='gated',
                 sparse_ratio=0.25,
                 scan_cfg=None):
        super().__init__()
        assert mode in ('gated', 'sum', 'exploit_only', 'explore_only', 'sparse', 'mamba', 'mamba_only')
        self.mode = mode
        self.sparse_ratio = sparse_ratio
        C = transform_channel

        self.bottleneck = nn.ModuleList([conv_3x3(c, C) for c in in_channel])
        # aux[i] classifies the stage-i feature (i=0 is the finest, stride 4); it feeds the gate and deep supervision
        self.aux = nn.ModuleList([nn.Conv2d(C, num_class, kernel_size=1) for _ in in_channel])
        self.global_gather = SpatialGatherModule()

        if mode not in ('explore_only', 'mamba_only'):
            self.exploit = nn.ModuleList([
                RVSA_MRAM(dim=C, out_dim=C, num_heads=num_heads, patch_size=patch_size, num_classes=num_class)
                for _ in in_channel])
        if mode in ('mamba', 'mamba_only'):
            requested = dict(scan_cfg or {})
            block_type = requested.pop('block_type', 'sparse')
            stages = [int(i) for i in requested.pop('stages', (1, 2, 3))]
            if block_type == 'coverage':
                cfg = dict(ratio=0.25, expand=1, d_state=8, n_heads=4,
                           dt_min=1e-3, dt_max=1e-1, chunk=64)
                cfg.update(requested)
                block_cls = CoverageScanBlock
            elif block_type == 'sparse':
                cfg = dict(order='conf', dirs=2, ratio=0.25, expand=1, d_state=8, n_heads=4,
                           dt_min=1e-3, dt_max=1e-1, chunk=64)
                cfg.update(requested)
                block_cls = SparseScanBlock
            else:
                raise ValueError(f'Unknown scan block_type: {block_type}')
            self.scan = nn.ModuleDict({str(i): block_cls(C, **cfg) for i in stages})
        if mode in ('gated', 'sum', 'explore_only', 'sparse'):
            self.explore = nn.ModuleList([SceneExplore(C, num_heads, explore_grid) for _ in in_channel])
            if mode != 'sparse':
                self.explore_fuse = nn.ModuleList([conv_3x3(C * 2, C) for _ in in_channel])

        self.catconv = nn.ModuleList([conv_3x3(C * 2, C) for _ in range(len(in_channel) - 1)])
        self.final = conv_3x3(C, C)

    def set_epoch(self, epoch):
        if hasattr(self, 'scan'):
            for block in self.scan.values():
                block.set_epoch(epoch)

    def _sparse_stage(self, i, feat, logits, global_center):
        out = self.exploit[i](feat, global_center)
        B, C, H, W = feat.shape
        uncertainty = 1. - entropy_gate(logits)                                   # B, 1, H, W
        k = max(1, int(self.sparse_ratio * H * W))
        idx = uncertainty.flatten(1).topk(k, dim=1).indices                       # B, k (most uncertain pixels)
        idx = idx.unsqueeze(-1).expand(-1, -1, C)
        x_flat = feat.flatten(2).transpose(1, 2)                                  # B, HW, C
        explore = self.explore[i]
        ctx = explore.attend(x_flat.gather(1, idx), explore.tokens(feat))         # B, k, C
        delta = torch.zeros_like(x_flat).scatter(1, idx, ctx.to(x_flat.dtype))    # zero at confident pixels
        self._gate_means[i] = (1. - uncertainty.mean()).detach()                  # mean confidence, for the log
        return out + delta.transpose(1, 2).reshape(B, C, H, W)

    def _stage(self, i, feat, logits, global_center):
        if self.mode == 'exploit_only':
            return self.exploit[i](feat, global_center)
        if self.mode in ('mamba', 'mamba_only'):
            out = self.exploit[i](feat, global_center) if self.mode == 'mamba' else feat
            if str(i) in self.scan:
                block = self.scan[str(i)]
                out = out + block(feat, logits)
                self._gate_means[i] = block.last_conf          # mean classifier confidence, for the log
            return out
        if self.mode == 'sparse':
            return self._sparse_stage(i, feat, logits, global_center)

        ctx = self.explore_fuse[i](torch.cat([feat, self.explore[i](feat)], dim=1))
        if self.mode == 'explore_only':
            return ctx

        exploit = self.exploit[i](feat, global_center)
        if self.mode == 'sum':
            g = 0.5
        else:
            g = entropy_gate(logits).to(feat.dtype)
            # 1 = fully on the class-prior (exploit) path, 0 = fully on the scene (explore) path.
            # Exposed so train.py can log it: a gate stuck near 0 or 1 for every stage means it
            # is not actually routing anything, which the loss curve alone would not show.
            self._gate_means[i] = g.mean().detach()   # stays a GPU tensor: float() here forced a sync every stage
        return g * exploit + (1. - g) * ctx

    def forward(self, x_list):
        self._gate_means = [None] * 4
        f = [b(x) for b, x in zip(self.bottleneck, x_list)]      # f[0] stride 4 ... f[3] stride 32

        logits = [None] * 4
        logits[3] = self.aux[3](f[3])
        global_center = self.global_gather(f[3], logits[3])

        feat = f[3]
        outs = [None] * 4
        for i in (3, 2, 1, 0):
            if i < 3:
                feat = self.catconv[i](upsample_add(outs[i + 1], f[i]))
                logits[i] = self.aux[i](feat)
            outs[i] = self._stage(i, feat, logits[i], global_center)

        fused = outs[0]
        for i in (1, 2, 3):
            fused = fused + F.interpolate(outs[i], scale_factor=2 ** i, mode="bilinear", align_corners=False)
        out = self.final(fused)

        # main feature (classified by the classifier), then aux logits coarse -> fine
        return [out, logits[3], logits[2], logits[1], logits[0]]
