import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from rsseg.models.segheads.logcanplus_head import RVSA_MRAM, SpatialGatherModule, conv_3x3, upsample_add


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

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.num_heads
        tokens = self.pool(x).flatten(2).transpose(1, 2) + self.pos          # B, M, C
        q = self.q(x.flatten(2).transpose(1, 2))                              # B, HW, C
        k, v = self.k(tokens), self.v(tokens)
        q, k, v = [t.reshape(B, -1, h, C // h).transpose(1, 2) for t in (q, k, v)]
        out = F.scaled_dot_product_attention(q, k, v)                         # B, h, HW, C/h
        out = out.transpose(1, 2).reshape(B, H * W, C)
        return self.proj(out).transpose(1, 2).reshape(B, C, H, W)


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
    """

    def __init__(self,
                 transform_channel,
                 in_channel,
                 num_class,
                 num_heads,
                 patch_size,
                 explore_grid=8,
                 mode='gated'):
        super().__init__()
        assert mode in ('gated', 'sum', 'exploit_only', 'explore_only')
        self.mode = mode
        C = transform_channel

        self.bottleneck = nn.ModuleList([conv_3x3(c, C) for c in in_channel])
        # aux[i] classifies the stage-i feature (i=0 is the finest, stride 4); it feeds the gate and deep supervision
        self.aux = nn.ModuleList([nn.Conv2d(C, num_class, kernel_size=1) for _ in in_channel])
        self.global_gather = SpatialGatherModule()

        if mode != 'explore_only':
            self.exploit = nn.ModuleList([
                RVSA_MRAM(dim=C, out_dim=C, num_heads=num_heads, patch_size=patch_size, num_classes=num_class)
                for _ in in_channel])
        if mode != 'exploit_only':
            self.explore = nn.ModuleList([SceneExplore(C, num_heads, explore_grid) for _ in in_channel])
            self.explore_fuse = nn.ModuleList([conv_3x3(C * 2, C) for _ in in_channel])

        self.catconv = nn.ModuleList([conv_3x3(C * 2, C) for _ in range(len(in_channel) - 1)])
        self.final = conv_3x3(C, C)

    def _stage(self, i, feat, logits, global_center):
        if self.mode == 'exploit_only':
            return self.exploit[i](feat, global_center)

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
            self._gate_means[i] = float(g.mean())
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
