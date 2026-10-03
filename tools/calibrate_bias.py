"""Per-class logit bias that fixes systematic over/under-prediction, fitted on validation pixels (no retraining).

The confusion analysis of the fast models shows one pattern in every class: 25-55% of the pixels of barren, road, forest,
agriculture and building are predicted as *background* (the catch-all class), so recall is low where precision is high.
Adding a constant to each class logit (the same as changing the bias of the last 1x1 conv, so inference cost is zero)
moves the decision boundaries; this tool searches the 7 constants that maximise mIoU.

    python tools/calibrate_bias.py -c configs/loveda/gee_ablation.py --set <same --set as the run> \
        --ckpt work_dirs/ablation/<run>/epoch=11.ckpt --apply work_dirs/ablation/<run>/calibrated.ckpt

Honesty about the estimate: fitting and scoring on the same pixels is optimistic, so the images are split in two halves
(odd / even index); the bias fitted on one half is scored on the other and the two held-out halves are pooled. Quote that
number ("held-out"), not the in-sample one. The final bias is fitted on all pixels. mIoU is computed from `--pixels`
randomly sampled pixels per image (about +-0.1 of the full-image value), never from the classifier output itself.
"""
import argparse
import ast
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root


def miou_from(logits, labels, bias, K):
    """mIoU (and per-class IoU) of argmax(logits + bias) against labels (all on the same device)."""
    pred = (logits + bias).argmax(1)
    cm = torch.bincount(labels * K + pred, minlength=K * K).view(K, K).double()
    tp = cm.diag()
    iou = tp / (cm.sum(0) + cm.sum(1) - tp).clamp_min(1)
    return iou.mean().item(), iou


def fit_bias(logits, labels, K, passes=3, grid=None):
    """coordinate ascent over the class biases, maximising mIoU; returns a (K,) tensor with zero mean"""
    grid = torch.arange(-3., 3.01, 0.1) if grid is None else grid
    bias = torch.zeros(K, device=logits.device)
    best, _ = miou_from(logits, labels, bias, K)
    for _ in range(passes):
        improved = False
        for c in range(K):
            cur = bias[c].item()
            for delta in grid.tolist():
                cand = bias.clone()
                cand[c] = cur + delta
                m, _ = miou_from(logits, labels, cand, K)
                if m > best + 1e-6:
                    best, bias, improved = m, cand, True
        if not improved:
            break
    return bias - bias.mean()


def cross_fit(logits, labels, folds, K):
    """fit on one fold, score on the other, pool: returns (miou_before, miou_heldout, per-class before, per-class heldout)"""
    pred = torch.empty_like(labels)
    for f in (0, 1):
        tr, te = folds != f, folds == f
        b = fit_bias(logits[tr], labels[tr], K)
        pred[te] = (logits[te] + b).argmax(1)
    cm = torch.bincount(labels * K + pred, minlength=K * K).view(K, K).double()
    tp = cm.diag()
    iou_after = tp / (cm.sum(0) + cm.sum(1) - tp).clamp_min(1)
    before, iou_before = miou_from(logits, labels, torch.zeros(K, device=logits.device), K)
    oa_after = (pred == labels).float().mean().item()
    oa_before = ((logits.argmax(1)) == labels).float().mean().item()
    return before, iou_after.mean().item(), iou_before, iou_after, oa_before, oa_after


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('--set', nargs='+', default=None, help='the same --set overrides the run was trained with')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--pixels', type=int, default=4000, help='random pixels sampled per image')
    ap.add_argument('--apply', default=None, help='write a checkpoint with the bias added to the classifier')
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
    model = myTrain.load_from_checkpoint(args.ckpt, cfg=cfg, strict=False).cuda().eval()
    K = cfg.metric_cfg2['num_classes'] - 1          # classes without the ignore slot
    ignore = cfg.metric_cfg2['ignore_index']

    loader = build_dataloader(cfg.dataset_config, mode='val')
    Ls, Ys, Fs, n_img = [], [], [], 0
    with torch.no_grad():
        for batch in loader:
            image, mask = batch[0].cuda(), batch[1].cuda()
            logits, _ = model._val_forward(image, mask)             # same tiled inference as validation
            B, _, H, W = logits.shape
            idx = torch.randperm(H * W, device=logits.device)[:args.pixels]
            lg = logits.flatten(2)[:, :, idx].permute(0, 2, 1)       # (B, n, K)
            lb = mask.flatten(1)[:, idx]
            valid = lb != ignore
            fold = (torch.arange(B, device=lg.device) + n_img) % 2
            Ls.append(lg[valid].float()); Ys.append(lb[valid]); Fs.append(fold.unsqueeze(1).expand_as(lb)[valid])
            n_img += B
    logits, labels, folds = torch.cat(Ls), torch.cat(Ys).long(), torch.cat(Fs)
    print(f'{n_img} validation images, {len(labels):,} sampled pixels, {K} classes')

    before, after, iou_b, iou_a, oa_b, oa_a = cross_fit(logits, labels, folds, K)
    bias = fit_bias(logits, labels, K)
    in_sample, iou_in = miou_from(logits, labels, bias, K)
    names = list(cfg.class_name)[:K]
    print(f'\nmIoU before            : {before * 100:6.2f}')
    print(f'mIoU after (held-out)  : {after * 100:6.2f}   <- the honest estimate  ({(after - before) * 100:+.2f})')
    print(f'mIoU after (in-sample) : {in_sample * 100:6.2f}   (optimistic)')
    print(f'OA   before -> held-out: {oa_b * 100:6.2f} -> {oa_a * 100:6.2f}   (mIoU-optimal biases usually trade a little OA for the rare classes)')
    print('\nclass         bias    IoU before -> held-out')
    for i, n in enumerate(names):
        print(f'{n:<13}{bias[i].item():+6.2f}    {iou_b[i].item() * 100:6.1f} -> {iou_a[i].item() * 100:6.1f}')
    if args.apply:
        ck = torch.load(args.ckpt, map_location='cpu')
        key = 'net.classifier.classifier.bias'
        assert key in ck['state_dict'], f'{key} not found: this classifier type needs its own handling'
        ck['state_dict'][key] = ck['state_dict'][key] + bias.cpu()
        torch.save(ck, args.apply)
        print(f'\nwrote {args.apply} (bias folded into {key}; inference cost unchanged)')
    print('\nbias vector:', [round(x, 2) for x in bias.tolist()])


if __name__ == '__main__':
    main()
