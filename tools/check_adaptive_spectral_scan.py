"""Forward/backward and conservation checks for AdaptiveCoverageScanBlock."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rsseg.models.basemodules.ssm_lite import AdaptiveCoverageScanBlock


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    block = AdaptiveCoverageScanBlock(
        dim=32,
        ratio=0.25,
        adapt_fraction=0.25,
        d_state=4,
        n_heads=4,
        chunk=16,
        pos_bands=3,
        spectral_iters=4,
    ).to(device).train()
    block.set_epoch(20)
    feat = torch.randn(2, 32, 16, 16, device=device, requires_grad=True)
    logits = torch.randn(2, 7, 16, 16, device=device, requires_grad=True)
    out = block(feat, logits)
    out.square().mean().backward()

    assert out.shape == feat.shape
    assert block.last_context_tokens == 64
    assert float(block.last_token_ratio) == 0.25
    assert block.last_adaptive_pairs == 4  # 16 candidate 4x4 regions * 0.25
    assert block.last_partition_complete
    assert feat.grad is not None and torch.isfinite(feat.grad).all()
    assert (feat.grad.abs().sum(1) > 0).all()
    assert block.ssm.in_proj.weight.grad is not None
    assert block.condense.weight.grad is not None
    assert logits.grad is None

    params = sum(p.numel() for p in block.parameters())
    print(f'OK device={device} shape={tuple(out.shape)} params={params:,}')
    print(f'context_tokens={block.last_context_tokens}/256 ratio={float(block.last_token_ratio):.2f}')
    print(f'merge_split_pairs={block.last_adaptive_pairs} partition_complete=True')
    print(f'all_source_pixels_receive_gradient=True mean_region_mass={float(block.last_region_mass):.2f}')


if __name__ == '__main__':
    main()
