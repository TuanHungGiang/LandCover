"""Validation mIoU of one checkpoint under several inference settings (no training).

    python tools/eval_ckpt.py -c configs/loveda/gee_ablation.py --set <same --set as the run> \
        --ckpt work_dirs/long/<run>/epoch=47.ckpt --settings full tile512 tile512s256 --batch 2

Why: validation during training cuts every 1024x1024 image into 512x512 tiles (no overlap) and stitches the logits. The large
homogeneous classes (water, forest, farmland) need context, and are exactly the classes where our IoU is far below the
published numbers (published methods evaluate whole 1024 images). This shows how much of the gap is the protocol.

settings:  full          the whole image in one forward pass
           tile<C>       C x C tiles without overlap            (tile512 = what training-time validation does)
           tile<C>s<S>   C x C tiles every S pixels, logits of overlapping pixels averaged (tile512s256)
           full-lr       whole image + the flip TTA of online_test.py (4 passes)
           full-ms       whole image + flip and scales 0.75/1/1.25 (6 passes)
           full-d4       the d4 TTA of online_test.py (flips + rot90 + 5 scales, 40 passes: ~1 h on the whole val set)
The published LoveDA numbers of this code base were produced with the d4 TTA (online_test.py default), so the TTA lines
are the ones to compare with them.
--flip     also average with the horizontally flipped image (extra flip for the non-TTA settings, 2x the cost)
"""
import argparse
import ast
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root


def make_tta(kind):
    import ttach as tta
    if kind == 'lr':
        t = tta.Compose([tta.HorizontalFlip(), tta.VerticalFlip()])
    elif kind == 'ms':
        t = tta.Compose([tta.HorizontalFlip(), tta.Scale(scales=[0.75, 1.0, 1.25], interpolation='bicubic', align_corners=False)])
    elif kind == 'd4':
        t = tta.Compose([tta.HorizontalFlip(), tta.VerticalFlip(), tta.Rotate90(angles=[90]),
                         tta.Scale(scales=[0.5, 0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False)])
    else:
        raise SystemExit(f'unknown TTA {kind!r}')
    return t


def parse_setting(name):
    if name == 'full':
        return None
    m = re.fullmatch(r'tile(\d+)(?:s(\d+))?', name)
    if not m:
        raise SystemExit(f'unknown setting {name!r}: use full, tile<C> or tile<C>s<S>')
    crop = int(m.group(1))
    return dict(crop=crop, stride=int(m.group(2) or crop))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('--set', nargs='+', default=None, help='the same --set overrides the run was trained with')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--settings', nargs='+', default=['full', 'tile512', 'tile512s256', 'full-lr', 'full-ms'])
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--flip', action='store_true')
    args = ap.parse_args()

    from train import myTrain                       # lazy: needs pytorch_lightning
    from rsseg.datasets import build_dataloader
    from utils.config import Config

    cfg = Config.fromfile(args.config)
    if args.set:
        opts = {}
        for p in args.set:
            k, _, v = p.partition('=')
            try:
                v = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                pass
            opts[k] = v
        cfg.merge_from_dict(opts)
    cfg.model_config.backbone.init_cfg = None
    cfg.dataset_config.val_mode.loader.batch_size = args.batch
    model = myTrain.load_from_checkpoint(args.ckpt, cfg=cfg, strict=False).cuda().eval()
    K = cfg.metric_cfg2['num_classes'] - 1
    ignore = cfg.metric_cfg2['ignore_index']
    names = list(cfg.class_name)[:K]

    print(f'{args.ckpt}  (flip TTA: {args.flip})')
    print(f'\n{"setting":<14}{"mIoU":>7}{"OA":>7}  ' + ' '.join(f'{n[:7]:>8}' for n in names) + f'{"min":>6}')
    for setting in args.settings:
        base, _, tta_kind = setting.partition('-')
        if tta_kind and base != 'full':
            raise SystemExit('TTA settings are whole-image only: full-lr, full-ms, full-d4')
        model.cfg.val_sliding = parse_setting(base)
        runner = model
        if tta_kind:
            import ttach as tta
            runner = tta.SegmentationTTAWrapper(model, make_tta(tta_kind))
        loader = build_dataloader(cfg.dataset_config, mode='val')
        cm = torch.zeros(K, K, dtype=torch.long, device='cuda')
        t0 = time.time()
        with torch.no_grad():
            for batch in loader:
                image, mask = batch[0].cuda(), batch[1].cuda()
                if tta_kind:
                    logits = runner(image, True)                  # same call as online_test.py
                else:
                    logits, _ = model._val_forward(image, mask)
                if args.flip and not tta_kind:
                    flipped, _ = model._val_forward(image.flip(-1), mask.flip(-1))
                    logits = (logits + flipped.flip(-1)) / 2
                pred = logits.argmax(1)
                valid = (mask != ignore) & (mask >= 0) & (mask < K)
                cm += torch.bincount(mask[valid] * K + pred[valid], minlength=K * K).view(K, K)
        cm = cm.double()
        tp = cm.diag()
        iou = tp / (cm.sum(0) + cm.sum(1) - tp).clamp_min(1)
        oa = (tp.sum() / cm.sum()).item()
        print(f'{setting:<14}{iou.mean().item() * 100:7.2f}{oa * 100:7.2f}  ' + ' '.join(f'{v * 100:8.1f}' for v in iou.tolist())
              + f'{(time.time() - t0) / 60:6.1f}', flush=True)


if __name__ == '__main__':
    main()
