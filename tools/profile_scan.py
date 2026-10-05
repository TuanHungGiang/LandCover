"""Time and memory of the light scan block on THIS machine, before committing hours of training to it.

    python tools/profile_scan.py                    # crop 512, batch 8 per GPU (the training setting)
    python tools/profile_scan.py --no_compile --no_triton

For the three stages that carry a scan (stride 8/16/32 = 64x64, 32x32, 16x16 maps at crop 512) it measures
forward+backward of one SparseScanBlock for several configurations, chunk sizes and backends, prints the
peak GPU memory, and compares every non-default backend with the plain-PyTorch result (max abs difference).
Pick the fastest row whose difference is ~1e-5 or smaller; then set it with
    --set model_config.seghead.scan_cfg.chunk=32 model_config.seghead.scan_cfg.backend=compile
"""
import argparse
import copy
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from rsseg.models.basemodules.ssm_lite import SparseScanBlock


def bench(block, feat, logits, iters, warmup, device):
    def step():
        feat.grad = None
        block(feat, logits).float().sum().backward()
    for _ in range(warmup):
        step()
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            step()
        b.record()
        torch.cuda.synchronize()
        return a.elapsed_time(b) / iters, torch.cuda.max_memory_allocated() / 1e6
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    return (time.perf_counter() - t0) / iters * 1000, float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--crop', type=int, default=512)
    ap.add_argument('--dim', type=int, default=96)
    ap.add_argument('--iters', type=int, default=10)
    ap.add_argument('--no_compile', action='store_true')
    ap.add_argument('--no_triton', action='store_true')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device={device} ({torch.cuda.get_device_name(0) if device == "cuda" else "cpu"}) | crop={args.crop} | '
          f'batch={args.batch} | dim={args.dim}')
    torch.manual_seed(0)
    stages = {f'stride{s}': args.crop // s for s in (8, 16, 32)}
    feats = {k: torch.randn(args.batch, args.dim, n, n, device=device, requires_grad=True) for k, n in stages.items()}
    logits = {k: torch.randn(args.batch, 7, n, n, device=device) for k, n in stages.items()}

    configs = [('raster', 4, 1.0), ('raster', 2, 0.25), ('conf', 2, 0.25), ('hybrid', 2, 0.25),
               ('landcover', 2, 0.25), ('conf', 2, 0.10)]
    print(f'\n{"config":<22}{"chunk":>6}{"backend":>10}{"ckpt":>6}' + ''.join(f'{k + " ms":>14}' for k in stages)
          + f'{"sum ms":>10}{"peak MB":>10}{"max diff":>10}')

    def run(order, dirs, ratio, chunk, backend, ckpt, ref_blocks=None):
        block = SparseScanBlock(args.dim, order=order, dirs=dirs, ratio=ratio, chunk=chunk, backend=backend,
                                use_checkpoint=ckpt).to(device).train()
        if ref_blocks is not None:
            block.load_state_dict(ref_blocks.state_dict())
        times, peak = [], 0.
        for k in stages:
            t, m = bench(block, feats[k], logits[k], args.iters, 3, device)
            times.append(t)
            peak = max(peak, m)
        return block, times, peak

    best = {}
    for order, dirs, ratio in configs:
        name = f'{order}/d{dirs}/r{ratio}'
        for chunk in (32, 64, 128):
            try:
                _, times, peak = run(order, dirs, ratio, chunk, 'torch', False)
            except Exception as e:                                           # e.g. out of memory at a big chunk
                print(f'{name:<22}{chunk:>6}{"torch":>10}{"-":>6}  failed: {str(e)[:60]}')
                continue
            print(f'{name:<22}{chunk:>6}{"torch":>10}{"no":>6}' + ''.join(f'{t:14.1f}' for t in times)
                  + f'{sum(times):10.1f}{peak:10.0f}{"-":>10}')
            if name not in best or sum(times) < best[name][1]:
                best[name] = (chunk, sum(times))

    # other backends / checkpointing on the fastest chunk of the sparse config you are most likely to use
    order, dirs, ratio = 'conf', 2, 0.25
    name = f'{order}/d{dirs}/r{ratio}'
    chunk = best.get(name, (64, 0))[0]
    ref, _, _ = run(order, dirs, ratio, chunk, 'torch', False)
    variants = [('torch', True)]
    if device == 'cuda' and not args.no_compile:
        variants.append(('compile', False))
    if device == 'cuda' and not args.no_triton:
        variants.append(('triton', False))
    print(f'\nother backends on {name}, chunk {chunk} (same weights as the torch row):')
    for backend, ckpt in variants:
        try:
            block, times, peak = run(order, dirs, ratio, chunk, backend, ckpt, ref_blocks=ref)
            with torch.no_grad():
                k = next(iter(stages))
                diff = (block(feats[k], logits[k]) - ref.eval()(feats[k], logits[k])).abs().max().item()
                ref.train()
            print(f'{name:<22}{chunk:>6}{backend:>10}{"yes" if ckpt else "no":>6}' + ''.join(f'{t:14.1f}' for t in times)
                  + f'{sum(times):10.1f}{peak:10.0f}{diff:10.1e}')
        except Exception as e:
            print(f'{name:<22}{chunk:>6}{backend:>10}{"yes" if ckpt else "no":>6}  failed: {type(e).__name__}: {str(e)[:90]}')


if __name__ == '__main__':
    main()
