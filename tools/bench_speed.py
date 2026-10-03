"""Size and speed of a model config (no training, random weights): params, GFLOPs, latency, throughput, memory.

    python tools/bench_speed.py -c configs/loveda/gee_ablation.py
    python tools/bench_speed.py -c configs/loveda/gee_ablation.py --set model_config.seghead.mode=mamba_only \
        model_config.backbone.type=repvit_m1_1 model_config.backbone.out_indices=[3,7,21,24] \
        model_config.seghead.in_channel=[64,128,256,512] --sizes 512 1024

Precisions: fp32; amp = torch.autocast fp16 (what training with precision=16 would use); half = the whole network in
fp16 (a deployment-style run; the class-center attention (RVSA) cannot run in pure fp16 and is reported as such).
The full pipeline is timed, including the bilinear upsampling of the logits to the input size.
GFLOPs = multiply-accumulates counted by torch's FlopCounterMode (conv / matmul / attention; grid_sample is not counted).
"""
import argparse
import ast
import os
import sys
import time
import warnings

import torch

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from torch.utils.flop_counter import FlopCounterMode

from rsseg.models.build_model import build_model
from utils.config import Config


def parse_sets(pairs):
    out = {}
    for p in pairs or []:
        k, _, v = p.partition('=')
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            pass
        out[k] = v
    return out


def timed(fn, iters, device):
    for _ in range(5):
        fn()
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            fn()
        b.record()
        torch.cuda.synchronize()
        return a.elapsed_time(b) / iters, torch.cuda.max_memory_allocated() / 1e6
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000, float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('--set', nargs='+', default=None)
    ap.add_argument('--sizes', nargs='+', type=int, default=[512, 1024])
    ap.add_argument('--batches', nargs='+', type=int, default=[1, 8])
    ap.add_argument('--iters', type=int, default=30)
    ap.add_argument('--channels_last', action='store_true')
    ap.add_argument('--compile', action='store_true', help='also time torch.compile + amp (needs Triton)')
    ap.add_argument('--compile_mode', default='default', choices=['default', 'reduce-overhead', 'max-autotune-no-cudagraphs'],
                    help='reduce-overhead = CUDA graphs: removes launch overhead, the limit of this head at small sizes')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = Config.fromfile(args.config)
    if args.set:
        cfg.merge_from_dict(parse_sets(args.set))
    cfg.model_config.backbone.init_cfg = None            # random weights are enough for speed
    bb_type = cfg.model_config.backbone.type              # build_model pops 'type' from the config, so read it first
    seg = dict(cfg.model_config.seghead)
    net = build_model(cfg.model_config).eval().to(device)
    if args.channels_last:
        net = net.to(memory_format=torch.channels_last)
    params = sum(p.numel() for p in net.parameters()) / 1e6
    bb = sum(p.numel() for p in net.backbone.parameters()) / 1e6
    print(f'device={device} ({torch.cuda.get_device_name(0) if device == "cuda" else "cpu"}) | backbone={bb_type} '
          f'| head mode={seg.get("mode", "?")} scan={dict(seg.get("scan_cfg", {}))}')
    print(f'params {params:.2f} M (backbone {bb:.2f} M, rest {params - bb:.2f} M)')

    def run(x, mode):
        if mode == 'amp':
            with torch.autocast('cuda', dtype=torch.float16):
                return net(x)[0]
        return net(x)[0]

    print(f'\n{"size":>5}{"batch":>6}{"mode":>6}{"GFLOPs(b1)":>12}{"latency ms":>12}{"FPS":>9}{"img/s":>9}{"peak MB":>9}')
    for size in args.sizes:
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            net(torch.randn(1, 3, size, size, device=device))
        gf = fc.get_total_flops() / 2e9
        for batch in args.batches:
            for mode in (('fp32', 'amp', 'half') if device == 'cuda' else ('fp32',)):
                x = torch.randn(batch, 3, size, size, device=device)
                model = net
                try:
                    if mode == 'half':
                        model = net.half()
                        x = x.half()
                    if args.channels_last:
                        x = x.contiguous(memory_format=torch.channels_last)
                    fn = (lambda: model(x)[0]) if mode != 'amp' else (lambda: run(x, 'amp'))
                    with torch.no_grad():
                        ms, mem = timed(fn, args.iters, device)
                    print(f'{size:>5}{batch:>6}{mode:>6}{gf:>12.1f}{ms:>12.1f}{1000 / ms * (1 if batch == 1 else 1):>9.1f}'
                          f'{batch * 1000 / ms:>9.1f}{mem:>9.0f}')
                except Exception as e:
                    print(f'{size:>5}{batch:>6}{mode:>6}{gf:>12.1f}  not possible: {type(e).__name__}: {str(e)[:70]}')
                finally:
                    net.float()
                    torch.cuda.empty_cache() if device == 'cuda' else None
    if args.compile and device == 'cuda':
        mode = None if args.compile_mode == 'default' else args.compile_mode
        print(f'\ntorch.compile (mode={args.compile_mode}) + amp, every size/batch above (first call of each shape compiles, ~1 min):')
        print(f'{"size":>5}{"batch":>6}{"latency ms":>12}{"FPS":>9}{"img/s":>9}{"peak MB":>9}')
        try:
            comp = torch.compile(net, dynamic=False, mode=mode)
            for size in args.sizes:
                for batch in args.batches:
                    x = torch.randn(batch, 3, size, size, device=device)
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                        ms, mem = timed(lambda: comp(x)[0], args.iters, device)
                    print(f'{size:>5}{batch:>6}{ms:>12.1f}{1000 / ms:>9.1f}{batch * 1000 / ms:>9.1f}{mem:>9.0f}')
        except Exception as e:
            print(f'torch.compile failed: {type(e).__name__}: {str(e)[:100]}')


if __name__ == '__main__':
    main()
