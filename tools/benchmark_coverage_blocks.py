"""Compare C1/C2 scan latency at one decoder stage without dataset dependencies."""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rsseg.models.basemodules.ssm_lite import AdaptiveCoverageScanBlock, CoverageScanBlock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--dim', type=int, default=96)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--amp', action='store_true')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if args.amp and device == 'cuda' else torch.float32
    feat = torch.randn(args.batch, args.dim, args.size, args.size, device=device, dtype=dtype)
    logits = torch.randn(args.batch, 7, args.size, args.size, device=device, dtype=dtype)
    common = dict(dim=args.dim, ratio=.25, n_heads=4, chunk=32)
    blocks = {
        'C1-uniform': CoverageScanBlock(**common),
        'C2-adaptive-spectral': AdaptiveCoverageScanBlock(**common, adapt_fraction=.25),
    }
    print(f'device={device} shape={tuple(feat.shape)} dtype={dtype}')
    for name, block in blocks.items():
        block = block.to(device=device, dtype=dtype).eval()
        with torch.no_grad():
            for _ in range(args.warmup):
                block(feat, logits)
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(args.iters):
                block(feat, logits)
            if device == 'cuda':
                torch.cuda.synchronize()
        elapsed = 1000. * (time.perf_counter() - start) / args.iters
        params = sum(p.numel() for p in block.parameters())
        print(f'{name:<22} {elapsed:8.2f} ms  params={params:,}  tokens={block.last_context_tokens}')


if __name__ == '__main__':
    main()
