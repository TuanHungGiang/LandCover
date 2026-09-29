"""Break down where the GEE_Head's wall-clock time goes: backbone vs head, and inside the head,
class-prior attention (exploit / RVSA_MRAM) vs scene-token attention (explore / SceneExplore) vs
gated (both, current default).

Run on the actual training GPU/resolution/batch size, e.g.:
    python tools/profile_head.py -c configs/loveda/gee_repvit.py --iters 20

Uses CUDA events when available (accurate for async GPU kernels); falls back to wall-clock on CPU.
"""
import argparse
import time

import torch

from utils.config import Config
from rsseg.models.backbones import repvit_m2_3, get_resnet34_OS32
from rsseg.models.segheads.gee_head import GEE_Head


def timed(fn, iters, warmup, device):
    for _ in range(warmup):
        fn()
    if device == 'cuda':
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters / 1000.0  # ms -> s
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def build_backbone(cfg):
    bcfg = dict(cfg.model_config.backbone)
    btype = bcfg.pop('type')
    bcfg.pop('init_cfg', None)
    bcfg['pretrained'] = False
    return {'repvit_m2_3': repvit_m2_3, 'get_resnet34_OS32': get_resnet34_OS32}[btype](**bcfg) \
        if btype == 'get_resnet34_OS32' else repvit_m2_3(init_cfg=None, out_indices=bcfg['out_indices'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--backward', action='store_true', help='include the backward pass (matches training cost more closely)')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    size = cfg.dataset_config.train_mode.transform.RandomSizeAndCrop.size
    bs = cfg.dataset_config.train_mode.loader.batch_size
    hcfg = {k: v for k, v in dict(cfg.model_config.seghead).items() if k != 'type'}
    print(f"device={device} | crop={size} | batch={bs} | backward={args.backward}")

    backbone = build_backbone(cfg).to(device).train()
    x = torch.randn(bs, 3, size, size, device=device)
    with torch.no_grad():
        feats_ref = [f.detach() for f in backbone(x)]
    print("feature map sizes:", [tuple(f.shape) for f in feats_ref])

    def bench(label, module):
        module = module.to(device).train()
        params = sum(p.numel() for p in module.parameters()) / 1e6

        def step():
            feats = [f.clone().requires_grad_(args.backward) for f in feats_ref]
            out = module(feats)
            if args.backward:
                sum(o.float().sum() for o in out).backward()
        t = timed(step, args.iters, args.warmup, device)
        print(f"{label:<16} params={params:6.2f}M  time/iter={t*1000:8.2f} ms")

    def bench_backbone():
        def step():
            xi = x.clone().requires_grad_(args.backward)
            out = backbone(xi)
            if args.backward:
                sum(o.float().sum() for o in out).backward()
        t = timed(step, args.iters, args.warmup, device)
        params = sum(p.numel() for p in backbone.parameters()) / 1e6
        print(f"{'backbone':<16} params={params:6.2f}M  time/iter={t*1000:8.2f} ms")

    bench_backbone()
    for mode in ('exploit_only', 'explore_only', 'sum', 'gated'):
        bench(mode, GEE_Head(**{**hcfg, 'mode': mode}))


if __name__ == '__main__':
    main()
