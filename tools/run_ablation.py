"""Run several GEE_Head variants back to back inside one Kaggle session and tabulate them.

    python tools/run_ablation.py                       # all experiments, default budget
    python tools/run_ablation.py --epochs 10 --budget_hours 9
    python tools/run_ablation.py --only exploit_only gated sparse25

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

# ordered by importance: if the budget runs out, the last ones are the ones that get skipped
EXPERIMENTS = [
    ('exploit_only', dict(mode='exploit_only')),                 # LOGCAN++-style decoder = baseline
    ('gated',        dict(mode='gated')),                        # current design (soft entropy gate)
    ('sparse25',     dict(mode='sparse', sparse_ratio=0.25)),    # explore only on the 25% most uncertain pixels
    ('explore_only', dict(mode='explore_only')),
    ('sum',          dict(mode='sum')),
    ('sparse10',     dict(mode='sparse', sparse_ratio=0.10)),
]

ECHO = re.compile(r'GPU memory|Traceback|Error|error:|val_miou')


def run(name, over, args, out_dir):
    exp_dir = os.path.join(out_dir, name)
    os.makedirs(exp_dir, exist_ok=True)
    sets = [f'exp_name={exp_dir}', f'epoch={args.epochs}',
            f'optimizer_config.scheduler.max_epoch={args.epochs}',
            f'check_val_every_n_epoch={args.val_every}']
    sets += [f'model_config.seghead.{k}={v}' for k, v in over.items()]
    cmd = [sys.executable, 'train.py', '-c', args.config, '--set'] + sets
    print(f'\n=== {name}: {" ".join(sets)}', flush=True)

    t0 = time.time()
    with open(os.path.join(exp_dir, 'train.log'), 'w') as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors='replace')
        for line in proc.stdout:
            log.write(line)
            if ECHO.search(line):
                print('   ', line.strip()[:200], flush=True)
        proc.wait()
    minutes = (time.time() - t0) / 60

    rows = []
    jl = os.path.join(exp_dir, 'val_metrics.jsonl')
    if os.path.isfile(jl):
        rows = [json.loads(l) for l in open(jl) if l.strip()]
    if proc.returncode != 0 or not rows:
        print(f'=== {name} FAILED (exit {proc.returncode}); see {exp_dir}/train.log', flush=True)
        return None
    best = max(rows, key=lambda r: r['val_miou'])
    res = dict(name=name, best_miou=best['val_miou'], best_epoch=best['epoch'], last_miou=rows[-1]['val_miou'],
               iou=best['iou'], params_M=rows[-1]['params_M'], minutes=minutes)
    json.dump(res, open(os.path.join(exp_dir, 'result.json'), 'w'))
    print(f'=== {name} done in {minutes:.1f} min: best val_miou={best["val_miou"]:.4f} (epoch {best["epoch"]})',
          flush=True)
    return res


def summarize(out_dir, class_name):
    results = []
    for name, _ in EXPERIMENTS:
        p = os.path.join(out_dir, name, 'result.json')
        if os.path.isfile(p):
            results.append(json.load(open(p)))
    if not results:
        return
    head = ['run', 'params(M)', 'best mIoU', 'ep', 'last mIoU', 'min'] + list(class_name)
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for r in results:
        cells = [r['name'], f"{r['params_M']:.2f}", f"{r['best_miou']*100:.2f}", str(r['best_epoch']),
                 f"{r['last_miou']*100:.2f}", f"{r['minutes']:.0f}"] + [f'{v*100:.1f}' for v in r['iou']]
        lines.append('| ' + ' | '.join(cells) + ' |')
    text = '\n'.join(lines)
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

    done_minutes = []
    for name, over in EXPERIMENTS:
        if args.only and name not in args.only:
            continue
        rp = os.path.join(out_dir, name, 'result.json')
        if os.path.isfile(rp):
            print(f'=== {name}: already done, skipping', flush=True)
            done_minutes.append(json.load(open(rp))['minutes'])
            continue
        if done_minutes:
            est = max(done_minutes) * 1.15 / 60
            left = args.budget_hours - (time.time() - start) / 3600
            if est > left:
                print(f'=== stopping before {name}: needs ~{est:.1f} h, only {left:.1f} h of budget left '
                      f'(rerun the same command in a new session to continue)', flush=True)
                break
        res = run(name, over, args, out_dir)
        if res:
            done_minutes.append(res['minutes'])

    summarize(out_dir, class_name)


if __name__ == '__main__':
    main()
