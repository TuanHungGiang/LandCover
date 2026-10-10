"""Fast forward/backward and invariant checks for CoverageScanBlock."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rsseg.models.basemodules.ssm_lite import CoverageScanBlock


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    block = CoverageScanBlock(
        dim=32,
        ratio=0.25,
        d_state=4,
        n_heads=4,
        chunk=16,
        pos_bands=3,
        soft_prototypes=True,
    ).to(device).train()
    feat = torch.randn(2, 32, 16, 16, device=device, requires_grad=True)
    logits = torch.randn(2, 7, 16, 16, device=device, requires_grad=True)
    out = block(feat, logits)
    out.square().mean().backward()

    assert out.shape == feat.shape
    assert block.last_context_tokens == 64
    assert float(block.last_token_ratio) == 0.25
    assert feat.grad is not None and torch.isfinite(feat.grad).all()
    # All source positions influence the condensed context path; none is discarded by hard top-k.
    assert (feat.grad.abs().sum(1) > 0).all()
    assert block.ssm.in_proj.weight.grad is not None
    assert block.condense.weight.grad is not None
    assert block.expand_context.weight.grad is not None
    assert block.semantic_proj.weight.grad is not None
    # Probabilities are deliberately detached: routing/conditioning must not destabilize aux logits.
    assert logits.grad is None

    params = sum(p.numel() for p in block.parameters())
    print(f'OK device={device} shape={tuple(out.shape)} params={params:,}')
    print(f'context_tokens={block.last_context_tokens}/256 ratio={float(block.last_token_ratio):.2f}')
    print(f'all_source_pixels_receive_gradient=True confidence={float(block.last_conf):.4f}')


if __name__ == '__main__':
    main()
