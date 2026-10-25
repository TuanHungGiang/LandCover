from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from make_config import build_config


HERE = Path(__file__).resolve().parent
OOM_MARKERS = (
    "cuda out of memory",
    "cuda error: out of memory",
    "cudnn_status_alloc_failed",
    "defaultcpuallocator: can't allocate memory",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_summary(output: Path, rows: list[dict[str, object]]) -> None:
    write_json(output / "summary.json", rows)
    flat_rows = []
    for row in rows:
        flat = {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
        metrics = row.get("metrics", {})
        if isinstance(metrics, dict):
            flat.update(
                {f"metric_{key}": value for key, value in metrics.items() if not isinstance(value, (dict, list))}
            )
        flat_rows.append(flat)
    fieldnames = sorted({key for row in flat_rows for key in row})
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


def read_json(path: Path, default: object) -> object:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_files() -> set[Path]:
    roots = [
        Path(torch.hub.get_dir()) / "checkpoints",
        Path.home() / ".cache" / "openmmlab",
    ]
    return {
        path.resolve()
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file()
    }


def remove_downloaded_weights(before: set[Path], record_path: Path) -> None:
    after = cache_files()
    records = []
    for path in sorted(after - before):
        records.append({"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size})
        path.unlink()
    write_json(record_path, records)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def command_for(mmseg_root: Path, config: Path, world_size: int, test_checkpoint: Path | None = None) -> list[str]:
    script = mmseg_root / "tools" / ("test.py" if test_checkpoint else "train.py")
    arguments = [str(script), str(config)]
    if test_checkpoint:
        arguments.append(str(test_checkpoint))
    if world_size > 1:
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={world_size}",
            "--master_port=29517",
            *arguments,
            "--launcher",
            "pytorch",
        ]
    return [sys.executable, *arguments]


def last_line(path: Path) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        end = handle.tell()
        handle.seek(max(0, end - 8192))
        lines = handle.read().decode("utf-8", errors="replace").splitlines()
    for line in reversed(lines):
        cleaned = line.strip()
        if cleaned:
            return cleaned[-300:]
    return ""


def run_logged(command: list[str], cwd: Path, log_path: Path, heartbeat: int = 30) -> tuple[int, str]:
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write("\nCOMMAND: " + subprocess.list2cmdline(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )
        while process.poll() is None:
            time.sleep(heartbeat)
            line = last_line(log_path)
            if line:
                print(f"  {line}", flush=True)
        code = int(process.returncode)
    tail = log_path.read_text(encoding="utf-8", errors="replace")[-20000:]
    return code, tail


def is_oom(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in OOM_MARKERS)


def find_final_checkpoint(work_dir: Path) -> Path | None:
    exact = work_dir / "iter_80000.pth"
    if exact.is_file():
        return exact
    candidates = sorted(work_dir.glob("iter_*.pth"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def parse_metrics(work_dir: Path, test_log: Path) -> dict[str, object]:
    result: dict[str, object] = {}
    scalar_files = sorted(work_dir.rglob("scalars.json"), key=lambda p: p.stat().st_mtime)
    for scalar_file in scalar_files:
        for line in scalar_file.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            metric_keys = [key for key in row if key in {"mIoU", "mDice", "mFscore", "aAcc", "mAcc"}]
            if metric_keys:
                result.update({key: row[key] for key in metric_keys})
                result["iteration"] = row.get("iter") or row.get("step")
                result["scalar_source"] = str(scalar_file)

    class_names = {
        "background",
        "building",
        "road",
        "water",
        "barren",
        "forest",
        "agriculture",
        "agricultural",
    }
    classwise: dict[str, dict[str, float]] = {}
    if test_log.is_file():
        for line in test_log.read_text(encoding="utf-8", errors="replace").splitlines():
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if cells and cells[0].lower() in class_names:
                numbers = []
                for cell in cells[1:]:
                    match = re.search(r"-?\d+(?:\.\d+)?", cell)
                    if match:
                        numbers.append(float(match.group()))
                if numbers:
                    keys = ("IoU", "Acc", "Dice", "Fscore")[: len(numbers)]
                    classwise[cells[0].lower()] = dict(zip(keys, numbers))
    if classwise:
        result["classwise"] = classwise
    return result


def run_one(args: argparse.Namespace, model_name: str, manifest: dict[str, object]) -> dict[str, object]:
    model_root = args.output / model_name / f"seed_{args.seed}"
    status_path = model_root / "status.json"
    status = read_json(status_path, {})
    if (
        isinstance(status, dict)
        and status.get("state") == "complete"
        and status.get("initialization") == "backbone_pretrained"
    ):
        print(f"[{model_name}] already complete; skipping")
        return status

    recipe = manifest["models"].get(model_name)
    if recipe is None:
        return {"model": model_name, "state": "blocked", "reason": "unknown model"}
    if recipe["status"] != "runnable_backbone_pretrained":
        result = {
            "model": model_name,
            "state": "blocked",
            "fidelity": recipe["status"],
            "reason": recipe.get("reason") or recipe.get("limitation"),
            "updated_at": utc_now(),
        }
        write_json(status_path, result)
        print(f"[{model_name}] BLOCKED: {result['reason']}")
        return result

    world_size = min(args.gpus, torch.cuda.device_count())
    if world_size < 1:
        raise RuntimeError("A CUDA GPU is required for the paper-faithful queue")
    micro_candidates = [value for value in (16, 8, 4, 2, 1) if value * world_size <= 16]
    old_micro = status.get("micro_batch_per_gpu") if isinstance(status, dict) else None
    if old_micro in micro_candidates:
        micro_candidates.remove(old_micro)
        micro_candidates.insert(0, old_micro)

    cache_before = cache_files()
    final_result: dict[str, object] = {}
    try:
        for micro_batch in micro_candidates:
            physical = micro_batch * world_size
            if 16 % physical:
                continue
            run_dir = model_root / f"micro_{micro_batch}x{world_size}"
            config_path = run_dir / "generated_config.py"
            provenance = build_config(
                model_name,
                args.mmseg_root,
                args.data_root,
                config_path,
                run_dir,
                micro_batch,
                world_size,
                args.seed,
            )
            checkpoint = find_final_checkpoint(run_dir)
            if checkpoint is None or checkpoint.name != "iter_80000.pth":
                probe_config = run_dir / "probe_config.py"
                build_config(
                    model_name,
                    args.mmseg_root,
                    args.data_root,
                    probe_config,
                    run_dir / "probe",
                    micro_batch,
                    world_size,
                    args.seed,
                )
                probe_command = command_for(args.mmseg_root, probe_config, world_size)
                probe_command += [
                    "--cfg-options",
                    "train_cfg.max_iters=3",
                    "train_cfg.val_interval=999999",
                    "default_hooks.checkpoint.interval=999999",
                ]
                probe_code, probe_tail = run_logged(
                    probe_command, args.mmseg_root, run_dir / "probe.log", heartbeat=10
                )
                if probe_code != 0:
                    if is_oom(probe_tail):
                        print(f"[{model_name}] micro-batch {micro_batch} OOM; retrying smaller batch")
                        continue
                    final_result = {
                        "model": model_name,
                        "state": "failed",
                        "stage": "probe",
                        "micro_batch_per_gpu": micro_batch,
                        "log": str(run_dir / "probe.log"),
                        "updated_at": utc_now(),
                    }
                    write_json(status_path, final_result)
                    return final_result

                current = {
                    "model": model_name,
                    "state": "running",
                    "fidelity": recipe["status"],
                    "variant": recipe.get("variant"),
                    **provenance,
                    "started_at": utc_now(),
                }
                write_json(status_path, current)
                train_command = command_for(args.mmseg_root, config_path, world_size)
                if (run_dir / "last_checkpoint").is_file():
                    train_command.append("--resume")
                train_code, train_tail = run_logged(
                    train_command, args.mmseg_root, run_dir / "train.log"
                )
                if train_code != 0:
                    if is_oom(train_tail):
                        print(f"[{model_name}] training OOM at micro-batch {micro_batch}; retrying smaller batch")
                        continue
                    final_result = {
                        **current,
                        "state": "failed",
                        "stage": "train",
                        "log": str(run_dir / "train.log"),
                        "updated_at": utc_now(),
                    }
                    write_json(status_path, final_result)
                    return final_result
                checkpoint = find_final_checkpoint(run_dir)

            if checkpoint is None or checkpoint.name != "iter_80000.pth":
                final_result = {
                    "model": model_name,
                    "state": "failed",
                    "reason": "training ended without iter_80000.pth",
                    "work_dir": str(run_dir),
                    "updated_at": utc_now(),
                }
                write_json(status_path, final_result)
                return final_result

            test_log = run_dir / "test_val.log"
            test_code, _ = run_logged(
                command_for(args.mmseg_root, config_path, world_size, checkpoint),
                args.mmseg_root,
                test_log,
            )
            if test_code != 0:
                final_result = {
                    "model": model_name,
                    "state": "failed",
                    "stage": "val_test",
                    "checkpoint": str(checkpoint),
                    "log": str(test_log),
                    "updated_at": utc_now(),
                }
                write_json(status_path, final_result)
                return final_result

            final_result = {
                "model": model_name,
                "state": "complete",
                "fidelity": recipe["status"],
                "variant": recipe.get("variant"),
                "initialization": "backbone_pretrained",
                "backbone_pretrained_fields": provenance["backbone_pretrained_fields"],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "metrics": parse_metrics(run_dir, test_log),
                "provenance": str(run_dir / "provenance.json"),
                "test_log": str(test_log),
                "completed_at": utc_now(),
                "environment": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "gpus": [torch.cuda.get_device_name(index) for index in range(world_size)],
                },
            }
            write_json(status_path, final_result)
            print(f"[{model_name}] complete: {final_result['metrics']}")
            return final_result

        final_result = {
            "model": model_name,
            "state": "oom_untrainable",
            "reason": "Even micro-batch 1 per GPU failed; effective batch was never changed.",
            "updated_at": utc_now(),
        }
        write_json(status_path, final_result)
        return final_result
    finally:
        remove_downloaded_weights(cache_before, model_root / "deleted_pretrained_cache.json")


def main(args: argparse.Namespace) -> None:
    manifest = json.loads((HERE / "recipes.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for model_name in args.models:
        try:
            results.append(run_one(args, model_name, manifest))
        except Exception as error:
            result = {
                "model": model_name,
                "state": "failed",
                "reason": f"{type(error).__name__}: {error}",
                "updated_at": utc_now(),
            }
            write_json(args.output / model_name / f"seed_{args.seed}" / "status.json", result)
            results.append(result)
            print(f"[{model_name}] failed without stopping queue: {result['reason']}")
        finally:
            write_summary(args.output, results)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    write_summary(args.output, results)
    print(json.dumps(results, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail-closed, OOM-safe LoveDA baseline queue")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--mmseg-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--gpus", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

