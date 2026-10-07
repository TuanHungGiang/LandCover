"""Run several GEE_Head variants back to back inside one Kaggle session and tabulate them.

    python tools/run_ablation.py                       # all experiments, default budget
    python tools/run_ablation.py --epochs 10 --budget_hours 9
    python tools/run_ablation.py --only exploit_only gated sparse25
    python tools/run_ablation.py --group ee --seeds 0 1          # 6 methods x 2 seeds, mean +- std in the table
    python tools/analyze_ablation.py                              # ranking, significance, per-class winners, Pareto

Each experiment is a normal `train.py` run (DDP when the config lists two GPUs) in work_dirs/ablation/<name>/.
It is restartable: an experiment with a result.json is skipped, so after a session timeout you rerun the same
command and it continues. Before each experiment it estimates the wall time from the finished ones and stops
if that would overrun --budget_hours. The comparison table is printed and saved to <out>/summary.md.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# keys starting with 'backbone.' go to model_config.backbone, 'loss.' to loss_config, all others to model_config.seghead
FAST = {'backbone.type': 'repvit_m1_1', 'backbone.out_indices': [3, 7, 21, 24],
        'backbone.init_cfg.checkpoint': 'pretrain/repvit_m1_1_distill_450e.pth', 'in_channel': [64, 128, 256, 512]}

# RepViT-M1.5 (13.6M backbone): the middle point between M1.1 (7.8M) and M2.3 (22.4M); weights verified to convert cleanly
MID = {'backbone.type': 'repvit_m1_5', 'backbone.out_indices': [5, 11, 37, 42],
       'backbone.init_cfg.checkpoint': 'pretrain/repvit_m1_5_distill_450e.pth', 'in_channel': [64, 128, 256, 512]}


def landcover_scan(ratio=0.25, selector='landcover'):
    """Shared settings keep selector/budget ablations identical in every other respect."""
    return {'scan_cfg.order': 'landcover', 'scan_cfg.ratio': ratio, 'scan_cfg.selector': selector,
            'scan_cfg.balance': 0.5, 'scan_cfg.boundary_weight': 0.5, 'scan_cfg.confusion_weight': 0.5,
            'scan_cfg.class_quota': 0.1, 'scan_cfg.presence_threshold': 0.01,
            'scan_cfg.presence_peak': 0.35, 'scan_cfg.anchor_topk': 16, 'scan_cfg.pos_bands': 4,
            'scan_cfg.scene_condition': True, 'scan_cfg.chunk': 32}

# ordered by importance: if the budget runs out, the last ones are the ones that get skipped
EXPERIMENTS = [
    ('exploit_only', dict(mode='exploit_only')),                 # LOGCAN++-style decoder = baseline
    ('gated',        dict(mode='gated')),                        # current design (soft entropy gate)
    ('sparse25',     dict(mode='sparse', sparse_ratio=0.25)),    # explore only on the 25% most uncertain pixels
    ('explore_only', dict(mode='explore_only')),
    ('sum',          dict(mode='sum')),
    ('sparse10',     dict(mode='sparse', sparse_ratio=0.10)),
    # light state-space scan (rsseg/models/basemodules/ssm_lite.py); scan_cfg.* are passed through to it
    ('mamba_raster_dense', dict(mode='mamba', **{'scan_cfg.order': 'raster', 'scan_cfg.dirs': 4, 'scan_cfg.ratio': 1.0})),   # VMamba-like
    ('mamba_raster25',     dict(mode='mamba', **{'scan_cfg.order': 'raster', 'scan_cfg.dirs': 2, 'scan_cfg.ratio': 0.25})),
    ('mamba_conf25',       dict(mode='mamba', **{'scan_cfg.order': 'conf', 'scan_cfg.ratio': 0.25})),          # class / confidence order
    ('mamba_hybrid25',     dict(mode='mamba', **{'scan_cfg.order': 'hybrid', 'scan_cfg.ratio': 0.25})),
    ('mamba_conf_dense',   dict(mode='mamba', **{'scan_cfg.order': 'conf', 'scan_cfg.ratio': 1.0})),           # order effect without sparsity
    ('mamba_conf25_only',  dict(mode='mamba_only', **{'scan_cfg.order': 'conf', 'scan_cfg.ratio': 0.25})),     # no class-center attention
    # fast model: RepViT-M1.1 backbone (7.8M params instead of 22.4M); needs pretrain/repvit_m1_1_distill_450e.pth
    ('m11_mamba_conf_only',   dict(mode='mamba_only', **FAST, **{'scan_cfg.order': 'conf', 'scan_cfg.ratio': 0.25})),
    ('m11_mamba_landcover_legacy', dict(mode='mamba_only', **FAST, **landcover_scan(selector='uncertainty'))),
    ('m11_mamba_landcover_only', dict(mode='mamba_only', **FAST, **landcover_scan())),
    ('m11_mamba_landcover50', dict(mode='mamba_only', **FAST, **landcover_scan(ratio=0.50))),
    ('m11_mamba_landcover_dense', dict(mode='mamba_only', **FAST, **landcover_scan(ratio=1.0))),
    ('m11_mamba_raster_only', dict(mode='mamba_only', **FAST, **{'scan_cfg.order': 'raster', 'scan_cfg.dirs': 2, 'scan_cfg.ratio': 0.25})),
    ('m11_explore_only',      dict(mode='explore_only', **FAST)),
    ('m11_exploit_only',      dict(mode='exploit_only', **FAST)),                       # accuracy reference, class-center attention
    # catch-all class down-weighted in the loss (the confusion table shows every class leaking into background)
    ('m11_conf_bg07', dict(mode='mamba_only', **FAST, **{'scan_cfg.order': 'conf', 'scan_cfg.ratio': 0.25, 'loss.class_weight': [0.7, 1, 1, 1, 1, 1, 1]})),
    ('m11_conf_bg05', dict(mode='mamba_only', **FAST, **{'scan_cfg.order': 'conf', 'scan_cfg.ratio': 0.25, 'loss.class_weight': [0.5, 1, 1, 1, 1, 1, 1]})),
    ('m15_mamba_conf_only',   dict(mode='mamba_only', **MID, **{'scan_cfg.order': 'conf', 'scan_cfg.ratio': 0.25})),
    ('m15_explore_only',      dict(mode='explore_only', **MID)),
]
GROUPS = {
    'base': ['exploit_only', 'gated', 'sparse25', 'explore_only', 'sum', 'sparse10'],
    'mamba': ['mamba_raster_dense', 'mamba_raster25', 'mamba_conf25', 'mamba_hybrid25', 'mamba_conf_dense', 'mamba_conf25_only'],
    'order': ['exploit_only', 'mamba_raster25', 'mamba_conf25', 'mamba_hybrid25'],      # does the scan order matter?
    'light': ['exploit_only', 'mamba_conf25', 'mamba_conf25_only'],
    # the explore / exploit story: exploit alone, exploit + attention explore, exploit + Mamba explore in three scan
    # orders, and Mamba explore without the class-center attention
    'bg': ['m11_mamba_conf_only', 'm11_conf_bg07', 'm11_conf_bg05'],
    'mid': ['m15_mamba_conf_only', 'm15_explore_only'],
    'fast': ['m11_mamba_conf_only', 'm11_mamba_landcover_legacy', 'm11_mamba_landcover_only', 'm11_mamba_raster_only',
             'm11_explore_only', 'm11_exploit_only'],
    'landcover_selector': ['m11_mamba_landcover_legacy', 'm11_mamba_landcover_only'],
    'landcover_budget': ['m11_mamba_landcover_only', 'm11_mamba_landcover50', 'm11_mamba_landcover_dense'],
    'ee': ['exploit_only', 'sparse25', 'mamba_raster25', 'mamba_conf25', 'mamba_hybrid25', 'mamba_conf25_only'],                      # is the class-center attention needed?
}

ECHO = re.compile(r'GPU memory|Traceback|Error|error:|NCCL|Timeout|out of memory')
NOISE = re.compile(r'frame #|c10::|terminate called|PossibleUserWarning|Config \(path')


def run_dir(name, seed):
    return name if seed is None else f'{name}_s{seed}'


def run(name, over, args, out_dir, seed=None):
    label = run_dir(name, seed)
    exp_dir = os.path.join(out_dir, label)
    os.makedirs(exp_dir, exist_ok=True)
    sets = [f'exp_name={exp_dir}', f'epoch={args.epochs}',
            f'optimizer_config.scheduler.max_epoch={args.epochs}',
            f'check_val_every_n_epoch={args.val_every}']
    if args.single_gpu:   # batch 8 x accumulate 2 = 16, the same effective batch as 2 GPUs
        sets += ['gpus=[0]', 'accumulate_grad_batches=2']
    if seed is not None:
        sets.append(f'seed={seed}')
    sets += [f'model_config.{k}={v}' if k.startswith('backbone.') else
             f'loss_config.{k[5:]}={v}' if k.startswith('loss.') else f'model_config.seghead.{k}={v}' for k, v in over.items()]
    cmd = [sys.executable, 'train.py', '-c', args.config, '--set'] + sets
    print(f'\n=== {label}: {" ".join(sets)}', flush=True)

    t0 = time.time()
    with open(os.path.join(exp_dir, 'train.log'), 'w') as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors='replace')
        for line in proc.stdout:
            log.write(line)
            if ECHO.search(line) and not NOISE.search(line):
                print('   ', line.strip()[:200], flush=True)
        proc.wait()
    minutes = (time.time() - t0) / 60

    rows = []
    jl = os.path.join(exp_dir, 'val_metrics.jsonl')
    if os.path.isfile(jl):
        rows = [json.loads(l) for l in open(jl) if l.strip()]
    if proc.returncode != 0 or not rows:
        print(f'=== {label} FAILED (exit {proc.returncode}); last lines of {exp_dir}/train.log:', flush=True)
        tail = [l.rstrip()[:220] for l in open(os.path.join(exp_dir, 'train.log'), errors='replace')
                if l.strip() and not NOISE.search(l)][-40:]
        print(chr(10).join('    ' + l for l in tail), flush=True)
        return None
    best = max(rows, key=lambda r: r['val_miou'])
    res = dict(name=name, seed=seed, best_miou=best['val_miou'], best_epoch=best['epoch'], last_miou=rows[-1]['val_miou'],
               iou=best['iou'], params_M=rows[-1]['params_M'], minutes=minutes)
    json.dump(res, open(os.path.join(exp_dir, 'result.json'), 'w'))
    print(f'=== {label} done in {minutes:.1f} min: best val_miou={best["val_miou"]:.4f} (epoch {best["epoch"]})',
          flush=True)
    return res


def summarize(out_dir, class_name):
    import statistics as st
    results = []
    for name, _ in EXPERIMENTS:
        for d in sorted(os.listdir(out_dir)):
            p = os.path.join(out_dir, d, 'result.json')
            if os.path.isfile(p):
                r = json.load(open(p))
                if r['name'] == name:
                    results.append(r)
    if not results:
        return
    head = ['run', 'params(M)', 'best mIoU', 'ep', 'last mIoU', 'min'] + list(class_name)
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for r in results:
        cells = [run_dir(r['name'], r.get('seed')), f"{r['params_M']:.2f}", f"{r['best_miou']*100:.2f}",
                 str(r['best_epoch']), f"{r['last_miou']*100:.2f}", f"{r['minutes']:.0f}"] + \
                [f'{v*100:.1f}' for v in r['iou']]
        lines.append('| ' + ' | '.join(cells) + ' |')

    # one row per method: mean and std over its seeds (std needs >= 2 seeds)
    by = {}
    for r in results:
        by.setdefault(r['name'], []).append(r)
    ahead = ['method', 'seeds', 'params(M)', 'best mIoU mean', 'std', 'min/run'] + list(class_name)
    agg = ['', '**Mean over seeds**', '', '| ' + ' | '.join(ahead) + ' |', '|' + '---|' * len(ahead)]
    for name, rs in by.items():
        m = [r['best_miou'] * 100 for r in rs]
        std = f'{st.stdev(m):.2f}' if len(m) > 1 else '-'
        cls = [sum(r['iou'][i] for r in rs) / len(rs) * 100 for i in range(len(class_name))]
        row = [name, str(len(rs)), f"{rs[0]['params_M']:.2f}", f'{sum(m)/len(m):.2f}', std,
               f"{sum(r['minutes'] for r in rs)/len(rs):.0f}"] + [f'{v:.1f}' for v in cls]
        agg.append('| ' + ' | '.join(row) + ' |')
    text = '\n'.join(lines + agg)
    open(os.path.join(out_dir, 'summary.md'), 'w').write(text + '\n')
    print('\n' + text, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', default='configs/loveda/gee_ablation.py')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--val_every', type=int, default=3, help='must divide --epochs so the last epoch is validated')
    ap.add_argument('--budget_hours', type=float, default=10.0, help='do not start a run that would end after this')
    ap.add_argument('--out', default='work_dirs/ablation')
    ap.add_argument('--only', nargs='+', default=None, help='subset of experiment names')
    ap.add_argument('--group', choices=sorted(GROUPS), default=None,
                    help='predefined subset: ' + ', '.join(f'{k} ({len(v)})' for k, v in GROUPS.items()))
    ap.add_argument('--seeds', nargs='+', type=int, default=None,
                    help='repeat every experiment for these seeds (result dirs <name>_s<seed>)')
    ap.add_argument('--single_gpu', action='store_true', help='use GPU 0 only (fallback if DDP does not work)')
    ap.add_argument('--no_profile', action='store_true', help='skip the per-mode head timing (tools/profile_head.py)')
    args = ap.parse_args()
    assert args.epochs % args.val_every == 0, '--epochs must be a multiple of --val_every'

    out_dir = os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)
    start = time.time()

    if not args.no_profile:
        print('=== profiling head modes (batch 2, fwd+bwd) ...', flush=True)
        p = subprocess.run([sys.executable, 'tools/profile_head.py', '-c', args.config, '--iters', '10',
                            '--backward', '--batch', '2'], cwd=ROOT, capture_output=True, text=True)
        text = p.stdout + (p.stderr if p.returncode else '')
        open(os.path.join(out_dir, 'profile_head.txt'), 'w').write(text)
        print(text[-2500:], flush=True)

    sys.path.insert(0, ROOT)
    from utils.config import Config
    class_name = Config.fromfile(os.path.join(ROOT, args.config)).class_name

    wanted = set(args.only or (GROUPS[args.group] if args.group else [n for n, _ in EXPERIMENTS]))
    # seed-major: every method at one seed before the next seed, so a cut-off budget still leaves a fair comparison
    plan = [(n, o, sd) for sd in (args.seeds or [None]) for n, o in EXPERIMENTS if n in wanted]
    done_minutes = []
    for name, over, seed in plan:
        label = run_dir(name, seed)
        rp = os.path.join(out_dir, label, 'result.json')
        if os.path.isfile(rp):
            print(f'=== {label}: already done, skipping', flush=True)
            done_minutes.append(json.load(open(rp))['minutes'])
            continue
        if done_minutes:
            est = max(done_minutes) * 1.15 / 60
            left = args.budget_hours - (time.time() - start) / 3600
            if est > left:
                print(f'=== stopping before {label}: needs ~{est:.1f} h, only {left:.1f} h of budget left '
                      f'(rerun the same command in a new session to continue)', flush=True)
                break
        res = run(name, over, args, out_dir, seed)
        if res:
            done_minutes.append(res['minutes'])
        elif not done_minutes:
            print('=== the first experiment failed, not starting the others (fix it, then rerun the same command)',
                  flush=True)
            break

    summarize(out_dir, class_name)


if __name__ == '__main__':
    main()
