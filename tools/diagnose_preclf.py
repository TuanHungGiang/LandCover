"""H1 diagnostic: does the class prior lock in the pre-classifier's mistakes?

For a trained model whose head returns [main_logits, pre_classifier_logits, ...] (GEE_Head, LoGCAN++),
this measures, per ground-truth class on the validation images:

  * IoU / recall of the final prediction and of the pre-classifier (preds[1], the coarse stride-32 classifier)
  * rescue rate     P(final right | pre-classifier wrong): low = the decoder cannot recover from a wrong prior
  * corruption rate P(final wrong | pre-classifier right)
  * AUROC of the pre-classifier's normalised entropy as an error detector (does uncertainty flag its mistakes?)

Entropy is a stand-in for the evidential vacuity that is not implemented yet.
Run once per domain (Urban / Rural) to see whether uncertainty rises under a domain shift.

    python tools/diagnose_preclf.py -c configs/loveda/gee_repvit_exploit.py --ckpt work_dirs/<exp>/<epoch>.ckpt
"""
import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BINS = 1000


def new_stats(num_classes, device):
    z = lambda *shape: torch.zeros(*shape, dtype=torch.long, device=device)
    return dict(conf_final=z(num_classes, num_classes), conf_pre=z(num_classes, num_classes),
                pre_right_final_right=z(num_classes), pre_right_final_wrong=z(num_classes),
                pre_wrong_final_right=z(num_classes), pre_wrong_final_wrong=z(num_classes),
                hist_right=z(num_classes, BINS), hist_wrong=z(num_classes, BINS))


@torch.no_grad()
def update(stats, final_logits, pre_logits, target, num_classes, ignore_index):
    """final_logits, pre_logits: (B, K, H, W); target: (B, H, W)."""
    K = num_classes
    valid = (target != ignore_index) & (target >= 0) & (target < K)
    y = target[valid]
    final = final_logits.argmax(1)[valid]
    prob = torch.softmax(pre_logits.float(), dim=1)
    pre = prob.argmax(1)[valid]
    ent = -(prob * torch.log(prob.clamp_min(1e-8))).sum(1)[valid] / math.log(K)
    b = (ent * BINS).long().clamp_(0, BINS - 1)

    stats['conf_final'] += torch.bincount(y * K + final, minlength=K * K).view(K, K)
    stats['conf_pre'] += torch.bincount(y * K + pre, minlength=K * K).view(K, K)
    pre_ok, fin_ok = pre == y, final == y
    for name, m in (('pre_right_final_right', pre_ok & fin_ok), ('pre_right_final_wrong', pre_ok & ~fin_ok),
                    ('pre_wrong_final_right', ~pre_ok & fin_ok), ('pre_wrong_final_wrong', ~pre_ok & ~fin_ok)):
        stats[name] += torch.bincount(y[m], minlength=K)
    stats['hist_right'] += torch.bincount(y[pre_ok] * BINS + b[pre_ok], minlength=K * BINS).view(K, BINS)
    stats['hist_wrong'] += torch.bincount(y[~pre_ok] * BINS + b[~pre_ok], minlength=K * BINS).view(K, BINS)


def _iou(conf):
    conf = conf.double()
    tp = conf.diag()
    return tp / (conf.sum(0) + conf.sum(1) - tp).clamp_min(1)


def _auroc(hist_right, hist_wrong):
    """P(entropy of a wrong pixel > entropy of a right pixel), ties count half."""
    right, wrong = hist_right.double(), hist_wrong.double()
    if right.sum() == 0 or wrong.sum() == 0:
        return float('nan')
    below_right = torch.cumsum(right, 0) - right
    return float((wrong * (below_right + 0.5 * right)).sum() / (right.sum() * wrong.sum()))


def summarize(stats, class_names):
    K = len(class_names)
    prw, prf = stats['pre_right_final_wrong'].double(), stats['pre_right_final_right'].double()
    pwf, pww = stats['pre_wrong_final_right'].double(), stats['pre_wrong_final_wrong'].double()
    n = stats['conf_final'].sum(1).double()
    iou_f, iou_p = _iou(stats['conf_final']), _iou(stats['conf_pre'])
    rows = []
    for c in range(K):
        rows.append(dict(
            cls=class_names[c], pixel_share=float(n[c] / n.sum().clamp_min(1)),
            iou_final=float(iou_f[c]), iou_pre=float(iou_p[c]),
            recall_final=float((prf[c] + pwf[c]) / n[c].clamp_min(1)), recall_pre=float((prf[c] + prw[c]) / n[c].clamp_min(1)),
            rescue_rate=float(pwf[c] / (pwf[c] + pww[c]).clamp_min(1)),
            corruption_rate=float(prw[c] / (prw[c] + prf[c]).clamp_min(1)),
            entropy_auroc=_auroc(stats['hist_right'][c], stats['hist_wrong'][c])))
    total = dict(cls='ALL', pixel_share=1.0, miou_final=float(iou_f.mean()), miou_pre=float(iou_p.mean()),
                 rescue_rate=float(pwf.sum() / (pwf.sum() + pww.sum()).clamp_min(1)),
                 corruption_rate=float(prw.sum() / (prw.sum() + prf.sum()).clamp_min(1)),
                 entropy_auroc=_auroc(stats['hist_right'].sum(0), stats['hist_wrong'].sum(0)))
    return rows, total


def print_table(title, rows, total):
    print(f"\n=== {title} ===")
    print(f"{'class':<13}{'share':>7}{'IoU fin':>9}{'IoU pre':>9}{'rec fin':>9}{'rec pre':>9}{'rescue':>8}{'corrupt':>9}{'AUROC(H)':>10}")
    for r in rows:
        print(f"{r['cls']:<13}{r['pixel_share']:>7.3f}{r['iou_final']:>9.3f}{r['iou_pre']:>9.3f}{r['recall_final']:>9.3f}"
              f"{r['recall_pre']:>9.3f}{r['rescue_rate']:>8.3f}{r['corruption_rate']:>9.3f}{r['entropy_auroc']:>10.3f}")
    print(f"{'ALL':<13}{'':>7}{total['miou_final']:>9.3f}{total['miou_pre']:>9.3f}{'':>9}{'':>9}"
          f"{total['rescue_rate']:>8.3f}{total['corruption_rate']:>9.3f}{total['entropy_auroc']:>10.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('--ckpt', required=True, help='Lightning checkpoint saved by train.py')
    parser.add_argument('--domains', nargs='+', default=['Urban', 'Rural'], help='LoveDA scenes to evaluate separately')
    parser.add_argument('--max-batches', type=int, default=0, help='0 = the whole val split')
    parser.add_argument('--num-workers', type=int, default=None, help='override the val loader workers of the config')
    parser.add_argument('--out', default=None, help='optional JSON file for the results')
    args = parser.parse_args()

    from utils.config import Config
    from rsseg.models.build_model import build_model
    from rsseg.datasets import build_dataloader

    cfg = Config.fromfile(args.config)
    backbone = cfg.model_config.backbone
    backbone.pop('init_cfg', None)          # the weights come from the checkpoint, not from the ImageNet file
    if 'pretrained' in backbone:
        backbone['pretrained'] = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = build_model(cfg.model_config)
    state = torch.load(args.ckpt, map_location='cpu')['state_dict']
    state = {k[len('net.'):]: v for k, v in state.items() if k.startswith('net.')}
    net.load_state_dict(state, strict=True)
    net.to(device).eval()

    K = cfg.num_class
    class_names = list(cfg.class_name)[:K]
    ignore_index = cfg.ignore_index

    results = {}
    if args.num_workers is not None:
        cfg.dataset_config.val_mode.loader.num_workers = args.num_workers

    for domain in args.domains:
        cfg.dataset_config.val_mode.domains = [domain]
        loader = build_dataloader(cfg.dataset_config, mode='val')
        stats = new_stats(K, device)
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if args.max_batches and i >= args.max_batches:
                    break
                preds = net(batch[0].to(device))
                update(stats, preds[0], preds[1], batch[1].to(device), K, ignore_index)
        rows, total = summarize(stats, class_names)
        print_table(f"{domain} val ({len(loader.dataset)} images)", rows, total)
        results[domain] = dict(per_class=rows, total=total)

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=1)
        print('\nsaved', args.out)


if __name__ == '__main__':
    main()
