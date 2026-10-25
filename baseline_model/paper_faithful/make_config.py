from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mmengine.config import Config, ConfigDict


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_num_classes(value: object) -> None:
    if isinstance(value, (dict, ConfigDict)):
        for key, child in value.items():
            if key == "num_classes":
                value[key] = 7
            else:
                set_num_classes(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            set_num_classes(child)


def set_scheduler_budget(schedulers: object, max_iters: int) -> None:
    if not isinstance(schedulers, (list, tuple)):
        return
    for scheduler in schedulers:
        if isinstance(scheduler, (dict, ConfigDict)) and scheduler.get("by_epoch") is False:
            if scheduler.get("type") not in {"LinearLR"}:
                scheduler["end"] = max_iters


def build_config(
    model_name: str,
    mmseg_root: Path,
    data_root: Path,
    output: Path,
    work_dir: Path,
    micro_batch: int,
    world_size: int,
    seed: int,
) -> dict[str, object]:
    manifest = json.loads((HERE / "recipes.json").read_text(encoding="utf-8"))
    recipe = manifest["models"].get(model_name)
    if recipe is None:
        raise KeyError(f"Unknown model: {model_name}")
    if recipe["status"] != "official_exact":
        raise RuntimeError(
            f"{model_name} is {recipe['status']}: {recipe.get('reason') or recipe.get('limitation')}"
        )
    if recipe.get("pretrained_scope") != "backbone_only":
        raise RuntimeError(f"{model_name} does not declare a backbone-only pretrained checkpoint")

    effective_batch = int(manifest["protocol"]["effective_global_batch"])
    physical_global_batch = micro_batch * world_size
    if effective_batch % physical_global_batch:
        raise ValueError(
            f"micro_batch({micro_batch}) * world_size({world_size}) must divide {effective_batch}"
        )
    accumulation = effective_batch // physical_global_batch

    source_config = mmseg_root / recipe["config"]
    if not source_config.is_file():
        raise FileNotFoundError(source_config)
    # Loading a config should not instantiate/import the model. The generated
    # config retains custom_imports and the training subprocess performs them
    # after all pinned runtime packages have been installed.
    cfg = Config.fromfile(source_config, import_custom_modules=False)

    crop = (512, 512)
    train_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="LoadAnnotations", reduce_zero_label=True),
        dict(
            type="RandomChoiceResize",
            scales=[512, 768, 1024, 1280, 1536],
            resize_type="ResizeShortestEdge",
            max_size=1536,
        ),
        dict(type="RandomCrop", crop_size=crop, cat_max_ratio=0.75),
        dict(type="RandomFlip", prob=0.5, direction="horizontal"),
        dict(type="RandomFlip", prob=0.5, direction="vertical"),
        dict(type="RandomRotate", prob=0.5, degree=0.2, seg_pad_val=255),
        dict(type="PackSegInputs"),
    ]
    val_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="LoadAnnotations", reduce_zero_label=True),
        dict(type="PackSegInputs"),
    ]
    dataset_type = "LoveDADataset"
    cfg.train_dataloader = dict(
        batch_size=micro_batch,
        num_workers=2,
        persistent_workers=True,
        sampler=dict(type="InfiniteSampler", shuffle=True),
        dataset=dict(
            type=dataset_type,
            data_root=str(data_root),
            data_prefix=dict(img_path="img_dir/train", seg_map_path="ann_dir/train"),
            pipeline=train_pipeline,
        ),
    )
    cfg.val_dataloader = dict(
        batch_size=1,
        num_workers=2,
        persistent_workers=True,
        sampler=dict(type="DefaultSampler", shuffle=False),
        dataset=dict(
            type=dataset_type,
            data_root=str(data_root),
            data_prefix=dict(img_path="img_dir/val", seg_map_path="ann_dir/val"),
            pipeline=val_pipeline,
        ),
    )
    cfg.test_dataloader = cfg.val_dataloader
    cfg.val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU", "mDice", "mFscore"])
    cfg.test_evaluator = cfg.val_evaluator

    if hasattr(cfg.model, "data_preprocessor"):
        cfg.model.data_preprocessor["size"] = crop
    set_num_classes(cfg.model)
    cfg.model["test_cfg"] = dict(mode="slide", crop_size=crop, stride=crop)

    max_iters = 80000
    cfg.train_cfg = dict(type="IterBasedTrainLoop", max_iters=max_iters, val_interval=max_iters)
    cfg.val_cfg = dict(type="ValLoop")
    cfg.test_cfg = dict(type="TestLoop")
    set_scheduler_budget(cfg.get("param_scheduler"), max_iters)
    cfg.optim_wrapper["accumulative_counts"] = accumulation
    cfg.train_dataloader["batch_size"] = micro_batch
    cfg.randomness = dict(seed=seed, deterministic=False)
    cfg.work_dir = str(work_dir)
    cfg.resume = False
    cfg.load_from = None
    if cfg.get("pretrained_ckpt_path"):
        raise RuntimeError("Task-level pretrained checkpoints are forbidden in strict mode")
    cfg.default_hooks["checkpoint"] = dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=5000,
        max_keep_ckpts=2,
        save_last=True,
    )
    cfg.default_hooks["logger"] = dict(
        type="LoggerHook", interval=50, log_metric_by_epoch=False
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    cfg.dump(output)
    provenance = {
        "model": model_name,
        "recipe": recipe,
        "source_config": str(source_config),
        "source_config_sha256": sha256(source_config),
        "mmseg_commit": manifest["implementation"]["commit"],
        "data_root": str(data_root),
        "micro_batch_per_gpu": micro_batch,
        "world_size": world_size,
        "gradient_accumulation": accumulation,
        "effective_global_batch": effective_batch,
        "max_iterations": max_iters,
        "seed": seed,
        "generated_config": str(output),
        "generated_config_sha256": sha256(output),
    }
    (output.parent / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mmseg-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--micro-batch", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = build_config(
        args.model,
        args.mmseg_root,
        args.data_root,
        args.output,
        args.work_dir,
        args.micro_batch,
        args.world_size,
        args.seed,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))

