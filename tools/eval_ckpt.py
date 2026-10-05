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
           tile512s256-lr / tile512s256-ms
                         apply the same TTA around an overlapping sliding-window predictor
The published LoveDA numbers of this code base were produced with the d4 TTA (online_test.py default), so the TTA lines
are the ones to compare with them.
--flip     also average with the horizontally flipped image (extra flip for the non-TTA settings, 2x the cost)
"""
import argparse
import ast
import os
import sys
import time
import json

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('--set', nargs='+', default=None, help='the same --set overrides the run was trained with')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--settings', nargs='+', default=['full', 'tile512', 'tile512s256', 'full-lr', 'full-ms'])
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--flip', action='store_true')
    ap.add_argument('--json-out', default=None, help='write all metrics here so a notebook can select the best setting')
    args = ap.parse_args()

    from train import myTrain                       # lazy: needs pytorch_lightning
    from rsseg.datasets import build_dataloader
    from tools.inference import build_predictor
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
    results = []
    for setting in args.settings:
        try:
            runner = build_predictor(model, setting)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        loader = build_dataloader(cfg.dataset_config, mode='val')
        cm = torch.zeros(K, K, dtype=torch.long, device='cuda')
        t0 = time.time()
        with torch.no_grad():
            for batch in loader:
                image, mask = batch[0].cuda(), batch[1].cuda()
                logits = runner(image)
                if args.flip:
                    flipped = runner(image.flip(-1))
                    logits = (logits + flipped.flip(-1)) / 2
                pred = logits.argmax(1)
                valid = (mask != ignore) & (mask >= 0) & (mask < K)
                cm += torch.bincount(mask[valid] * K + pred[valid], minlength=K * K).view(K, K)
        cm = cm.double()
        tp = cm.diag()
        iou = tp / (cm.sum(0) + cm.sum(1) - tp).clamp_min(1)
        oa = (tp.sum() / cm.sum()).item()
        minutes = (time.time() - t0) / 60
        print(f'{setting:<14}{iou.mean().item() * 100:7.2f}{oa * 100:7.2f}  ' + ' '.join(f'{v * 100:8.1f}' for v in iou.tolist())
              + f'{minutes:6.1f}', flush=True)
        results.append(dict(setting=setting, miou=iou.mean().item(), oa=oa,
                            iou=iou.tolist(), class_names=names, minutes=minutes))

    if args.json_out:
        parent = os.path.dirname(os.path.abspath(args.json_out))
        os.makedirs(parent, exist_ok=True)
        with open(args.json_out, 'w') as f:
            json.dump(dict(checkpoint=args.ckpt, results=results), f, indent=2)
        best = max(results, key=lambda row: row['miou'])
        print(f'JSON: {args.json_out}\nBEST_SETTING={best["setting"]} mIoU={best["miou"] * 100:.2f}', flush=True)


if __name__ == '__main__':
    main()
