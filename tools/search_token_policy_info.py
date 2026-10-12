"""Exact-quarter token-policy search with a constrained, multi-objective INFO variant.

The optimizer is deliberately OFFLINE.  A deployed model still performs one deterministic
score/top-k construction and scans exactly N/4 context tokens; no population is evaluated per
image.  INFO (weighted mean of vectors) is adapted in three ways for this problem:

1. conservation repair: ratio is not a search variable and is forced to 0.25 in every trial;
2. a Pareto archive keeps accuracy, difficult-class IoU and measured block latency;
3. cheap screening runs are followed by a higher-fidelity confirmation of the Pareto leaders.

Typical two-GPU Kaggle search (restartable):

    python tools/search_token_policy_info.py --gpus 0 1 --population 6 --generations 2

Quick plumbing check without training:

    python tools/search_token_policy_info.py --simulate --population 6 --generations 2

The final artifact is ``<out>/best_policy.json``.  It contains a ready-to-copy ``--set`` list
for a fresh 96-epoch training run.  Screening checkpoints are not used as final results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HARD_CLASSES = (4, 5)  # LoveDA barren, forest
EXACT_RATIO = 0.25


@dataclass(frozen=True)
class Axis:
    name: str
    low: float
    high: float
    kind: str = "float"       # float | int | choice
    choices: tuple = ()


# Keep the chromosome compact: every extra degree of freedom costs real training runs.
SPACE = (
    Axis("adapt_fraction", 0.08, 0.42),
    Axis("boundary_weight", 0.20, 1.40),
    Axis("entropy_weight", 0.10, 1.20),
    Axis("rarity_weight", 0.00, 0.80),
    Axis("spectral_temperature", 0.20, 1.10),
    Axis("spectral_self_loop", 0.40, 1.80),
    Axis("spectral_iters", 3, 8, "int"),
    Axis("d_state", 0, 2, "choice", (4, 8, 12)),
    Axis("chunk", 0, 2, "choice", (16, 32, 64)),
)

DEFAULT = {
    "adapt_fraction": 0.25,
    "boundary_weight": 0.75,
    "entropy_weight": 0.50,
    "rarity_weight": 0.25,
    "spectral_temperature": 0.50,
    "spectral_self_loop": 1.00,
    "spectral_iters": 6,
    "d_state": 8,
    "chunk": 32,
}


def repair(z):
    """Project an unconstrained normalized vector onto the legal mixed search space."""
    z = np.clip(np.asarray(z, dtype=np.float64), 0.0, 1.0)
    out = {}
    for value, axis in zip(z, SPACE):
        raw = axis.low + value * (axis.high - axis.low)
        if axis.kind == "int":
            out[axis.name] = int(round(raw))
        elif axis.kind == "choice":
            index = int(round(raw))
            out[axis.name] = axis.choices[max(0, min(index, len(axis.choices) - 1))]
        else:
            out[axis.name] = round(float(raw), 6)
    return out


def encode(policy):
    z = []
    for axis in SPACE:
        value = policy[axis.name]
        if axis.kind == "choice":
            raw = axis.choices.index(value)
        else:
            raw = value
        z.append((raw - axis.low) / (axis.high - axis.low))
    return np.asarray(z, dtype=np.float64)


def policy_key(policy, fidelity):
    payload = json.dumps({"policy": policy, "fidelity": fidelity}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def policy_overrides(policy):
    # ratio is a constant, never read from the chromosome.  This is the conservation repair.
    values = {"ratio": EXACT_RATIO, **policy}
    return [f"model_config.seghead.scan_cfg.{k}={v}" for k, v in values.items()]


def dominates(a, b):
    """Maximize both accuracies and minimize latency."""
    no_worse = (a["miou"] >= b["miou"] and a["hard_iou"] >= b["hard_iou"] and
                a["latency_ms"] <= b["latency_ms"])
    strict = (a["miou"] > b["miou"] or a["hard_iou"] > b["hard_iou"] or
              a["latency_ms"] < b["latency_ms"])
    return no_worse and strict


def pareto_front(rows):
    return [row for i, row in enumerate(rows)
            if not any(i != j and dominates(other, row) for j, other in enumerate(rows))]


def utilities(rows):
    """Rank scalar used only by the INFO update; final selection remains Pareto-based."""
    matrix = np.asarray([[r["miou"], r["hard_iou"], -r["latency_ms"]] for r in rows])
    lo, hi = matrix.min(0), matrix.max(0)
    norm = (matrix - lo) / np.maximum(hi - lo, 1e-12)
    return norm @ np.asarray([0.65, 0.20, 0.15])


def info_generation(population, rows, rng):
    """One bounded INFO-style weighted-mean/vector-combination update.

    Rank weights make the center robust to differently scaled objectives.  A differential term
    maintains diversity; a leader-local term supplies INFO's local search.  Greedy selection is
    performed after the offspring are evaluated by the caller.
    """
    score = utilities(rows)
    order = np.argsort(-score)
    rank = np.empty(len(score), dtype=int)
    rank[order] = np.arange(len(score))
    weights = np.exp(-rank / max(1.0, len(score) / 3.0))
    center = np.average(np.stack(population), axis=0, weights=weights)
    leader = population[int(order[0])]
    children = []
    n = len(population)
    for i, x in enumerate(population):
        pool = [j for j in range(n) if j != i]
        a, b, c = rng.choice(pool, size=3, replace=False)
        decay = 0.55 + 0.25 * rng.random()
        weighted_delta = weights[a] * (population[a] - population[b]) + weights[c] * (leader - population[c])
        weighted_delta /= weights[a] + weights[c] + 1e-12
        local = rng.normal(0.0, 0.06, size=x.shape) * (leader - x)
        proposal = center + decay * weighted_delta + local
        # Dimension-wise vector combining keeps some proven coordinates.
        mask = rng.random(x.shape) < 0.70
        child = np.where(mask, proposal, x)
        children.append(np.clip(child, 0.0, 1.0))
    return children


def parse_metrics(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"No validation metrics in {path}")
    best = max(rows, key=lambda row: row["val_miou"])
    return float(best["val_miou"]), float(np.mean([best["iou"][i] for i in HARD_CLASSES])), best


def measured_block_latency(policy, device, iters=20):
    """Measure the changed scan block, excluding training noise and dataloader time."""
    import torch
    from rsseg.models.basemodules.ssm_lite import AdaptiveCoverageScanBlock

    torch.cuda.set_device(device) if torch.cuda.is_available() else None
    dev = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    block = AdaptiveCoverageScanBlock(96, ratio=EXACT_RATIO, n_heads=4,
                                      **policy).to(device=dev, dtype=dtype).eval()
    feat = torch.randn(2, 96, 64, 64, device=dev, dtype=dtype)
    logits = torch.randn(2, 7, 64, 64, device=dev, dtype=dtype)
    with torch.no_grad():
        for _ in range(5):
            block(feat, logits)
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                block(feat, logits)
            end.record()
            torch.cuda.synchronize(device)
            latency = start.elapsed_time(end) / iters
        else:
            start = time.perf_counter()
            for _ in range(max(2, iters // 4)):
                block(feat, logits)
            latency = (time.perf_counter() - start) * 1000 / max(2, iters // 4)
    if block.last_context_tokens != (64 * 64) // 4:
        raise AssertionError(f"budget violation: {block.last_context_tokens} != 1024")
    return float(latency)


def simulate_result(policy):
    """Smooth synthetic landscape used only to test the optimizer plumbing."""
    z = encode(policy)
    target = np.asarray([.58, .52, .48, .36, .43, .44, .55, .50, .55])
    miou = .50 + .06 * math.exp(-float(np.square(z - target).sum()) / .35)
    hard = miou - .08 + .02 * policy["rarity_weight"]
    latency = 5.0 + .45 * policy["spectral_iters"] + .20 * policy["d_state"]
    return dict(miou=miou, hard_iou=hard, latency_ms=latency, best_epoch=0)


def run_trial(policy, fidelity, args, gpu, cache, lock):
    key = policy_key(policy, fidelity)
    with lock:
        if key in cache:
            return cache[key]
    if args.simulate:
        result = simulate_result(policy)
    else:
        exp = Path(args.out) / f"trial_{key}"
        exp.mkdir(parents=True, exist_ok=True)
        overrides = [f"exp_name={exp.as_posix()}", f"epoch={fidelity['epochs']}",
                     f"optimizer_config.scheduler.max_epoch={fidelity['epochs']}",
                     f"check_val_every_n_epoch={fidelity['val_every']}",
                     f"limit_train_batches={fidelity['train_fraction']}",
                     f"limit_val_batches={fidelity['val_fraction']}",
                     f"model_config.seghead.scan_cfg.warmup_epochs={fidelity['warmup']}",
                     f"model_config.seghead.scan_cfg.ramp_epochs={fidelity['ramp']}",
                     f"gpus=[{gpu}]", "save_top_k=1", "save_last=False"] + policy_overrides(policy)
        cmd = [sys.executable, "train.py", "-c", args.config, "--set", *overrides]
        with (exp / "command.txt").open("w") as f:
            f.write(" ".join(cmd) + "\n")
        start = time.time()
        with (exp / "train.log").open("w") as log:
            proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode:
            raise RuntimeError(f"Trial {key} failed; see {exp / 'train.log'}")
        miou, hard, best = parse_metrics(exp / "val_metrics.jsonl")
        latency = measured_block_latency(policy, gpu, args.latency_iters)
        result = dict(miou=miou, hard_iou=hard, latency_ms=latency,
                      best_epoch=best["epoch"], train_minutes=(time.time() - start) / 60,
                      exp=str(exp))
    result.update(policy=policy, fidelity=fidelity, key=key)
    with lock:
        cache[key] = result
        save_json(Path(args.out) / "cache.json", cache)
    return result


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True))
    temp.replace(path)


def evaluate_many(policies, fidelity, args, cache, lock):
    results = [None] * len(policies)
    # Submit one wave at a time.  A generic thread pool can otherwise start the next GPU-0
    # job when GPU 1 happens to finish first, colliding with the still-running GPU-0 job.
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        for begin in range(0, len(policies), len(args.gpus)):
            future_map = {}
            for offset, policy in enumerate(policies[begin:begin + len(args.gpus)]):
                i = begin + offset
                future_map[executor.submit(run_trial, policy, fidelity, args, args.gpus[offset], cache, lock)] = i
            for future in as_completed(future_map):
                i = future_map[future]
                results[i] = future.result()
                row = results[i]
                print(f"trial {row['key']}: mIoU={row['miou']*100:.2f} hard={row['hard_iou']*100:.2f} "
                      f"latency={row['latency_ms']:.2f} ms", flush=True)
    return results


def choose_compromise(front):
    score = utilities(front)
    return front[int(np.argmax(score))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", default="configs/loveda/gee_repvit_landcover_adaptive_spectral.py")
    parser.add_argument("--out", default="work_dirs/info_token_search")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--screen_epochs", type=int, default=12)
    parser.add_argument("--confirm_epochs", type=int, default=24)
    parser.add_argument("--train_fraction", type=float, default=0.35)
    parser.add_argument("--val_fraction", type=float, default=0.50)
    parser.add_argument("--confirm_top", type=int, default=3)
    parser.add_argument("--latency_iters", type=int, default=20)
    parser.add_argument("--simulate", action="store_true")
    args = parser.parse_args()
    if args.population < 4:
        parser.error("--population must be >= 4 (INFO needs three peers per candidate)")
    if not (0 < args.train_fraction <= 1 and 0 < args.val_fraction <= 1):
        parser.error("fractions must be in (0, 1]")

    os.chdir(ROOT)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.out) / "cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    lock = threading.Lock()
    rng = np.random.default_rng(args.seed)

    screen = dict(name="screen", epochs=args.screen_epochs,
                  val_every=max(1, args.screen_epochs // 3), train_fraction=args.train_fraction,
                  val_fraction=args.val_fraction, warmup=2, ramp=2)
    confirm = dict(name="confirm", epochs=args.confirm_epochs,
                   val_every=max(1, args.confirm_epochs // 4), train_fraction=1.0,
                   val_fraction=1.0, warmup=4, ramp=4)

    # Warm-start with C2 and use stratified random coverage for the remaining policies.
    population = [encode(DEFAULT)]
    for i in range(1, args.population):
        population.append(np.mod((i + rng.random(len(SPACE))) / args.population, 1.0))

    rows = evaluate_many([repair(z) for z in population], screen, args, cache, lock)
    for generation in range(args.generations):
        children = info_generation(population, rows, rng)
        child_rows = evaluate_many([repair(z) for z in children], screen, args, cache, lock)
        pool_z, pool_rows = population + children, rows + child_rows
        rank = np.argsort(-utilities(pool_rows))[:args.population]
        population = [pool_z[i] for i in rank]
        rows = [pool_rows[i] for i in rank]
        print(f"generation {generation + 1}: front={len(pareto_front(rows))} "
              f"best-screen={max(r['miou'] for r in rows)*100:.2f}", flush=True)

    front = pareto_front(rows)
    front_score = utilities(front)
    leaders = [front[i] for i in np.argsort(-front_score)[:args.confirm_top]]
    confirmed = evaluate_many([r["policy"] for r in leaders], confirm, args, cache, lock)
    confirmed_front = pareto_front(confirmed)
    best = choose_compromise(confirmed_front)
    final_overrides = policy_overrides(best["policy"])
    model_tag = "m15" if "m15" in Path(args.config).stem else "m11"
    final_exp = f"work_dirs/{model_tag}_mamba_info_final"
    artifact = dict(exact_ratio=EXACT_RATIO, algorithm="conservation-aware multi-fidelity Pareto INFO",
                    selected=best, pareto_front=confirmed_front, final_overrides=final_overrides,
                    final_command=[sys.executable, "train.py", "-c", args.config, "--set",
                                   f"exp_name={final_exp}", *final_overrides])
    save_json(Path(args.out) / "best_policy.json", artifact)
    print("\nSelected exact-25% policy:")
    print(json.dumps(best, indent=2))
    print("Final 96-epoch command:\n" + " ".join(artifact["final_command"]))


if __name__ == "__main__":
    main()
