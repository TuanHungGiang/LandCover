"""Check that RVSA_MRAM treats every sample of a batch independently.

In eval mode the output for sample 0 must be identical whether it is run alone (B=1) or together with
another sample (B=2). If the max difference is not ~0, tensors are being mixed across the batch
(e.g. a repeat() vs reshape() ordering mismatch).

    python tools/check_batch_consistency.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from rsseg.models.segheads.logcanplus_head import RVSA_MRAM

torch.manual_seed(0)
m = RVSA_MRAM(dim=96, out_dim=96, num_heads=8, num_classes=7, patch_size=(4, 4)).eval()
# offsets/scales are zero-initialised, which would hide ordering bugs: randomise them
for branch in (m.sampling_offsets, m.sampling_scales, m.sampling_angles):
    torch.nn.init.normal_(branch[-1].weight, std=0.05)

x = torch.randn(2, 96, 32, 32)
gc = torch.randn(2, 7, 96)
with torch.no_grad():
    both = m(x, gc)
    alone = m(x[:1], gc[:1])
print("max |sample0(batch=2) - sample0(batch=1)| =", (both[:1] - alone).abs().max().item())
