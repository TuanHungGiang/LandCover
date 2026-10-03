"""Read the finished runs of tools/run_ablation.py and answer: which variant is best, which is most accurate,
and is the difference real?  Writes <out>/analysis.md.

    python tools/analyze_ablation.py                      # work_dirs/ablation
    python tools/analyze_ablation.py --baseline exploit_only --out work_dirs/ablation

best          = highest mean best-epoch mIoU over seeds
most accurate = highest mean overall accuracy (OA) at the best epoch (OA favours the big classes; mIoU the rare ones)
real?         = paired difference to the baseline over seeds (same seed = same data order) with a t statistic;
                with 1-2 seeds nothing can be called significant, and the table says so.
"""
import argparse
import json
import math
import os
import statistics as st
import sys

# two-sided 95% critical values of Student's t, df = 1..10
T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23}

# what each run is made of: (class-center attention = exploit, scene attention, light Mamba scan, scan order)
ANATOMY = {
    'exploit_only': ('yes', '-', '-', '-'), 'explore_only': ('-', 'dense', '-', '-'),
    'gated': ('yes', 'dense, soft gate', '-', '-'), 'sum': ('yes', 'dense, 0.5/0.5', '-', '-'),
    'sparse25': ('yes', 'top-25% uncertain', '-', '-'), 'sparse10': ('yes', 'top-10% uncertain', '-', '-'),
    'mamba_raster_dense': ('yes', '-', 'all pixels', 'raster x4'),
    'mamba_raster25': ('yes', '-', 'top-25% uncertain', 'raster'),
    'mamba_conf25': ('yes', '-', 'top-25% uncertain', 'class/confidence'),
    'mamba_hybrid25': ('yes', '-', 'top-25% uncertain', 'class + raster'),
    'mamba_conf_dense': ('yes', '-', 'all pixels', 'class/confidence'),
    'mamba_conf25_only': ('-', '-', 'top-25% uncertain', 'class/confidence'),
    'm11_mamba_conf_only': ('-', '-', 'top-25% uncertain', 'class/confidence (M1.1 backbone)'),
    'm11_mamba_raster_only': ('-', '-', 'top-25% uncertain', 'raster (M1.1 backbone)'),
    'm11_explore_only': ('-', 'dense', '-', '- (M1.1 backbone)'),
    'm11_exploit_only': ('yes', '-', '-', '- (M1.1 backbone)'),
    'm15_mamba_conf_only': ('-', '-', 'top-25% uncertain', 'class/confidence (M1.5 backbone)'),
    'm15_explore_only': ('-', 'dense', '-', '- (M1.5 backbone)'),
    'm11_conf_bg07': ('-', '-', 'top-25% uncertain', 'class/confidence, background weight 0.7'),
    'm11_conf_bg05': ('-', '-', 'top-25% uncertain', 'class/confidence, background weight 0.5'),
}


def load(out_dir, metric='last'):
    runs = {}
    for d in sorted(os.listdir(out_dir)):
        rp = os.path.join(out_dir, d, 'result.json')
        if not os.path.isfile(rp):
            continue
        r = json.load(open(rp))
        jl = os.path.join(out_dir, d, 'val_metrics.jsonl')
        rows = [json.loads(l) for l in open(jl) if l.strip()] if os.path.isfile(jl) else []
        best = max(rows, key=lambda x: x['val_miou']) if rows else None
        r['oa'] = r.get('oa', best['val_oa'] if best else float('nan'))
        r['curve'] = [(x['epoch'], x['val_miou']) for x in rows]
        r['peak_miou'] = r['best_miou']            # mIoU at the epoch picked on the same val set (optimistic)
        r['cm'] = rows[-1].get('cm') if rows else None
        if metric == 'last' and rows:              # no epoch selection: what the finished model scores
            last = rows[-1]
            r['best_miou'], r['iou'], r['oa'] = last['val_miou'], last['iou'], last['val_oa']
        runs.setdefault(r['name'], []).append(r)
    return runs


def mean_std(xs):
    return (sum(xs) / len(xs), st.stdev(xs) if len(xs) > 1 else float('nan'))


def paired(runs, name, base):
    """paired difference (in mIoU points) per seed; falls back to unpaired means if seeds do not match"""
    a = {r.get('seed'): r['best_miou'] * 100 for r in runs[name]}
    b = {r.get('seed'): r['best_miou'] * 100 for r in runs[base]}
    seeds = sorted(set(a) & set(b), key=lambda x: (x is None, x))
    d = [a[s] - b[s] for s in seeds]
    if len(d) < 2:
        return (d[0] if d else float('nan')), float('nan'), float('nan'), len(d)
    m, sd = mean_std(d)
    t = m / (sd / math.sqrt(len(d))) if sd > 0 else float('inf')
    return m, sd, t, len(d)


def verdict(m, t, n):
    if n < 2:
        return 'one seed: cannot tell'
    if n < 3:
        return 'only 2 seeds: cannot tell'
    crit = T95.get(n - 1, 2.0)
    if abs(t) > crit:
        return 'better (p<0.05)' if m > 0 else 'worse (p<0.05)'
    return 'no real difference'


def pareto(points):
    """points: name -> (miou, minutes, params); keep the runs nobody beats on all three"""
    front = []
    for n, (a, b, c) in points.items():
        dominated = any((a2 >= a and b2 <= b and c2 <= c) and (a2 > a or b2 < b or c2 < c)
                        for n2, (a2, b2, c2) in points.items() if n2 != n)
        if not dominated:
            front.append(n)
    return front


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')     # the tables contain '±' (Windows consoles default to cp1252)
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='work_dirs/ablation')
    ap.add_argument('--baseline', default='exploit_only')
    ap.add_argument('--metric', choices=['last', 'best'], default='last',
                    help='last = mIoU/OA/IoU of the final epoch (default, no selection bias); best = at the best val epoch')
    ap.add_argument('--classes', nargs='+', default=['background', 'building', 'road', 'water', 'barren', 'forest', 'agricultural'])
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = args.out if os.path.isabs(args.out) else os.path.join(root, args.out)
    runs = load(out, args.metric)
    if not runs:
        raise SystemExit(f'no finished runs (result.json) in {out}')
    base = args.baseline if args.baseline in runs else None
    L = []
    P = L.append

    stats = {}
    for n, rs in runs.items():
        m, s = mean_std([r['best_miou'] * 100 for r in rs])
        oa = sum(r['oa'] for r in rs) / len(rs) * 100
        stats[n] = dict(n=len(rs), miou=m, std=s, oa=oa, minutes=sum(r['minutes'] for r in rs) / len(rs),
                        params=rs[0]['params_M'], iou=[sum(r['iou'][i] for r in rs) / len(rs) * 100 for i in range(len(args.classes))])

    P('# Ablation analysis\n')
    P(f'{sum(v["n"] for v in stats.values())} finished runs, {len(stats)} methods; baseline = {base or "none found"}; '
      f'metric = {"final epoch (no epoch selection)" if args.metric == "last" else "best validation epoch (optimistic)"}.\n')

    P('## 1. Ranking by mIoU (best)\n')
    P('| rank | method | seeds | mIoU mean | std | vs baseline (paired, points) | verdict | min/run | params (M) | peak-epoch mIoU |')
    P('|---|---|---|---|---|---|---|---|---|---|')
    for i, (n, v) in enumerate(sorted(stats.items(), key=lambda kv: -kv[1]['miou']), 1):
        if base and n != base:
            m, sd, t, k = paired(runs, n, base)
            diff = f'{m:+.2f}' + (f' ± {sd:.2f}' if not math.isnan(sd) else '')
            vd = verdict(m, t, k)
        else:
            diff, vd = '-', '(baseline)' if n == base else '-'
        std = f'{v["std"]:.2f}' if not math.isnan(v['std']) else '-'
        peak = sum(r['peak_miou'] for r in runs[n]) / len(runs[n]) * 100
        P(f'| {i} | {n} | {v["n"]} | {v["miou"]:.2f} | {std} | {diff} | {vd} | {v["minutes"]:.0f} | {v["params"]:.2f} | {peak:.2f} |')

    P('\n## 2. Ranking by accuracy (OA at the best epoch)\n')
    P('| rank | method | OA mean | mIoU mean |')
    P('|---|---|---|---|')
    for i, (n, v) in enumerate(sorted(stats.items(), key=lambda kv: -kv[1]['oa']), 1):
        P(f'| {i} | {n} | {v["oa"]:.2f} | {v["miou"]:.2f} |')
    top_miou = max(stats, key=lambda n: stats[n]['miou'])
    top_oa = max(stats, key=lambda n: stats[n]['oa'])
    P(f'\nBest by mIoU: **{top_miou}**. Most accurate by OA: **{top_oa}**' + (' (same run).' if top_miou == top_oa else ' (different runs: they trade rare classes against large ones).'))

    P('\n## 3. Per-class IoU (mean over seeds) and the winner per class\n')
    P('| method | ' + ' | '.join(args.classes) + ' |')
    P('|---|' + '---|' * len(args.classes))
    for n, v in stats.items():
        P(f'| {n} | ' + ' | '.join(f'{x:.1f}' for x in v['iou']) + ' |')
    win = [max(stats, key=lambda n: stats[n]['iou'][i]) for i in range(len(args.classes))]
    P('| **winner** | ' + ' | '.join(f'**{w}**' for w in win) + ' |')
    if base:
        P(f'\nGain over {base} per class (points):\n')
        P('| method | ' + ' | '.join(args.classes) + ' |')
        P('|---|' + '---|' * len(args.classes))
        for n, v in stats.items():
            if n != base:
                P(f'| {n} | ' + ' | '.join(f'{v["iou"][i] - stats[base]["iou"][i]:+.1f}' for i in range(len(args.classes))) + ' |')

    P('\n## 4. Explore / exploit anatomy: what each run is made of\n')
    P('| method | class-center attention (exploit) | scene attention (explore) | light Mamba scan | scan order | mIoU | vs baseline |')
    P('|---|---|---|---|---|---|---|')
    for n, v in sorted(stats.items(), key=lambda kv: -kv[1]['miou']):
        a = ANATOMY.get(n, ('?', '?', '?', '?'))
        d = f'{v["miou"] - stats[base]["miou"]:+.2f}' if base and n != base else '-'
        P(f'| {n} | {a[0]} | {a[1]} | {a[2]} | {a[3]} | {v["miou"]:.2f} | {d} |')

    P('\n## 5. Accuracy / cost trade-off (Pareto front: no other run is better on mIoU, time and parameters at once)\n')
    front = pareto({n: (v['miou'], v['minutes'], v['params']) for n, v in stats.items()})
    P('Pareto-optimal: ' + ', '.join(f'**{n}**' for n in front))
    P('\n| method | mIoU | min/run | params (M) | on front |')
    P('|---|---|---|---|---|')
    for n, v in sorted(stats.items(), key=lambda kv: -kv[1]['miou']):
        P(f'| {n} | {v["miou"]:.2f} | {v["minutes"]:.0f} | {v["params"]:.2f} | {"yes" if n in front else ""} |')

    P('\n## 6. Noise floor\n')
    P('Seed-to-seed spread inside each method (max - min of the final mIoU) against the spread between methods:\n')
    P('| method | seeds | min | max | spread |')
    P('|---|---|---|---|---|')
    spreads = []
    for n, rs in runs.items():
        vals = [r['best_miou'] * 100 for r in rs]
        spreads.append(max(vals) - min(vals))
        P(f'| {n} | {len(vals)} | {min(vals):.2f} | {max(vals):.2f} | {max(vals) - min(vals):.2f} |')
    between = max(v['miou'] for v in stats.values()) - min(v['miou'] for v in stats.values())
    P(f'\nBetween-method spread of the means: **{between:.2f}**; typical within-method seed spread: **{sum(spreads) / len(spreads):.2f}**. '
      + ('Differences between methods are of the same size as seed noise: do not read them as improvements.'
         if between <= 2.0 * (sum(spreads) / len(spreads)) else 'Between-method differences exceed seed noise.'))

    with_cm = {n: [r['cm'] for r in rs if r.get('cm')] for n, rs in runs.items()}
    with_cm = {n: v for n, v in with_cm.items() if v}
    if with_cm:
        P('\n## 7. What each class is confused with (final epoch, share of the true pixels, mean over seeds)\n')
        P('| method | class | recall | most confused with | share | second | share |')
        P('|---|---|---|---|---|---|---|')
        for n, cms in with_cm.items():
            K = len(args.classes)
            tot = [[sum(cm[i][j] for cm in cms) for j in range(K)] for i in range(K)]
            for i, cname in enumerate(args.classes):
                row_sum = sum(tot[i]) or 1
                wrong = sorted(((tot[i][j] / row_sum, args.classes[j]) for j in range(K) if j != i), reverse=True)[:2]
                P(f'| {n} | {cname} | {tot[i][i] / row_sum * 100:.1f}% | {wrong[0][1]} | {wrong[0][0] * 100:.1f}% | {wrong[1][1]} | {wrong[1][0] * 100:.1f}% |')

    n_seeds = min(v['n'] for v in stats.values())
    P(f'\n## Reading guide\n\n- Fewest seeds in any method: {n_seeds}. ' +
      ('With fewer than 3 seeds no difference below about 1 point can be trusted; run `--seeds 0 1 2`.' if n_seeds < 3 else
       'Differences are tested with a paired t-test over seeds (95%).'))
    P('- mIoU is the 7-class mean on LoveDA val (512 tiles, stitched); it is not comparable with the paper\'s test-set numbers.')
    text = '\n'.join(L) + '\n'
    open(os.path.join(out, 'analysis.md'), 'w', encoding='utf-8').write(text)
    print(text)


if __name__ == '__main__':
    main()
