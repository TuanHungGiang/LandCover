"""Fast CPU/GPU correctness checks for the land-cover-specific sparse Mamba block.

Run from the repository root:
    python tools/check_landcover_scan.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rsseg.models.basemodules.ssm_lite import SparseScanBlock


def main():
    torch.manual_seed(7)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    block = SparseScanBlock(
        dim=32,
        order='landcover',
        ratio=0.25,
        selector='landcover_v2',
        selector_weights=(0.35, 0.25, 0.25, 0.15),
        selector_quotas=(0.35, 0.25, 0.25, 0.15),
        balance=0.5,
        anchor_topk=8,
        num_prototypes=2,
        confidence_gate=True,
        pos_bands=3,
        d_state=4,
        n_heads=4,
        chunk=16,
    ).to(device)

    feat = torch.randn(2, 32, 16, 16, device=device, requires_grad=True)
    logits = torch.randn(2, 7, 16, 16, device=device)
    # Force the second sample to contain only one predicted class: empty class rows must remain safe.
    logits[1].fill_(-10.)
    logits[1, 3].fill_(10.)

    out = block(feat, logits)
    assert out.shape == feat.shape
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    assert feat.grad is not None and torch.isfinite(feat.grad).all()
    assert block.ssm.in_proj.weight.grad is not None
    assert torch.isfinite(block.ssm.in_proj.weight.grad).all()
    assert block.scene_proj.weight.grad is not None
    assert block.gate_proj.weight.grad is not None
    assert torch.isfinite(block.scene_proj.weight.grad).all()
    assert int(block.last_selected_per_class.sum(1)[0]) == 64
    assert int(block.last_selected_per_class.sum(1)[1]) == 64
    assert torch.isfinite(block.last_anchor_conf) and torch.isfinite(block.last_query_conf)
    assert torch.isfinite(block.last_boundary_coverage)
    assert torch.isfinite(block.last_confusion_coverage)
    assert 0. <= float(block.last_quota_fraction) <= 1.0
    assert 0. < float(block.last_gate) < 1.

    params = sum(p.numel() for p in block.parameters())
    print(f'OK device={device} shape={tuple(out.shape)} params={params:,}')
    print('selected queries per image/class:')
    print(block.last_selected_per_class.cpu())
    print(f'anchor confidence={block.last_anchor_conf.item():.4f} '
          f'query confidence={block.last_query_conf.item():.4f}')
    print(f'boundary coverage={block.last_boundary_coverage.item():.4f} '
          f'confusion coverage={block.last_confusion_coverage.item():.4f} '
          f'quota fraction={block.last_quota_fraction.item():.4f} '
          f'gate={block.last_gate.item():.4f}')


if __name__ == '__main__':
    main()
