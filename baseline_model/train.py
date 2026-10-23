from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline_suite.data import LOVEDA_CLASSES, LoveDADataset
from baseline_suite.metrics import SegmentationMetrics
from baseline_suite.models import MODEL_NAMES, build_model, model_info


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, classes: int = 7) -> torch.Tensor:
    valid = target != 255
    safe_target = target.masked_fill(~valid, 0)
    probs = torch.softmax(logits, 1) * valid[:, None]
    one_hot = F.one_hot(safe_target, classes).permute(0, 3, 1, 2).to(probs.dtype) * valid[:, None]
    intersection = (probs * one_hot).sum((0, 2, 3))
    denominator = probs.sum((0, 2, 3)) + one_hot.sum((0, 2, 3))
    return 1 - ((2 * intersection + 1) / (denominator + 1)).mean()


def make_loader(args: argparse.Namespace, split: str) -> DataLoader:
    training = split == "Train"
    dataset = LoveDADataset(args.data_root, split, args.crop_size, training, args.seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size if training else args.val_batch_size,
        shuffle=training,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=training,
        persistent_workers=args.workers > 0,
    )


def sliding_predict(model: torch.nn.Module, image: torch.Tensor, crop: int) -> torch.Tensor:
    """Non-overlapping tiled inference; LoveDA 1024 images divide exactly into 512 tiles."""
    b, _, h, w = image.shape
    if h == crop and w == crop:
        return model(image)
    logits = image.new_zeros((b, 7, h, w))
    counts = image.new_zeros((b, 1, h, w))
    for top in range(0, h, crop):
        for left in range(0, w, crop):
            bottom, right = min(top + crop, h), min(left + crop, w)
            tile = image[:, :, bottom - crop : bottom, right - crop : right]
            logits[:, :, bottom - crop : bottom, right - crop : right] += model(tile)
            counts[:, :, bottom - crop : bottom, right - crop : right] += 1
    return logits / counts.clamp_min(1)


@torch.inference_mode()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, amp: bool, crop: int) -> dict[str, object]:
    model.eval()
    metrics = SegmentationMetrics(7, 255)
    losses: list[float] = []
    for batch in tqdm(loader, desc="LoveDA Val", leave=False):
        image, target = batch["image"].to(device, non_blocking=True), batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            logits = sliding_predict(model, image, crop)
            loss = F.cross_entropy(logits, target, ignore_index=255)
        metrics.update(logits, target)
        losses.append(float(loss))
    result = metrics.compute()
    result["loss"] = float(np.mean(losses))
    result["class_names"] = list(LOVEDA_CLASSES)
    return result


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict) -> None:
    state_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    torch.save(
        {"epoch": epoch, "model": state_model.state_dict(), "optimizer": optimizer.state_dict(), "metrics": metrics},
        path,
    )


def load_checkpoint(path: Path, model: torch.nn.Module, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return checkpoint


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = args.output / args.model
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    model = build_model(args.model, num_classes=7, pretrained=args.pretrained, image_size=args.crop_size).to(device)
    if args.data_parallel and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    print(json.dumps({**model_info(args.model), "device": str(device)}, indent=2))

    val_loader = make_loader(args, "Val")
    if args.mode == "test":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required in test mode")
        load_checkpoint(args.checkpoint, model.module if isinstance(model, torch.nn.DataParallel) else model, device)
        result = evaluate(model, val_loader, device, args.amp, args.crop_size)
        (run_dir / "val_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return

    train_loader = make_loader(args, "Train")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    best_miou = -1.0
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, started = 0.0, time.time()
        bar = tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(bar, 1):
            image, target = batch["image"].to(device, non_blocking=True), batch["mask"].to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=args.amp and device.type == "cuda"):
                logits = model(image)
                loss = F.cross_entropy(logits, target, ignore_index=255) + args.dice_weight * dice_loss(logits, target)
                loss = loss / args.accumulate
            scaler.scale(loss).backward()
            if step % args.accumulate == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running += float(loss) * args.accumulate
            bar.set_postfix(loss=f"{running / step:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        scheduler.step()
        val = evaluate(model, val_loader, device, args.amp, args.crop_size)
        record = {"epoch": epoch, "train_loss": running / len(train_loader), "seconds": time.time() - started, **val}
        history.append(record)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, val)
        if float(val["miou"]) > best_miou:
            best_miou = float(val["miou"])
            save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, val)
        print(json.dumps(record, indent=2))

    best_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    load_checkpoint(run_dir / "best.pt", best_model, device)
    final = evaluate(model, val_loader, device, args.amp, args.crop_size)
    (run_dir / "val_metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(f"Best LoveDA Val mIoU: {final['miou']:.6f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified LoveDA Train -> Val semantic-segmentation runner")
    p.add_argument("--model", required=True, choices=MODEL_NAMES)
    p.add_argument("--mode", choices=("train", "test"), default="train")
    p.add_argument("--data-root", type=Path, default=Path(r"D:\landcover\data\loveda\extracted"))
    p.add_argument("--output", type=Path, default=Path("runs"))
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--val-batch-size", type=int, default=1)
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--workers", type=int, default=0, help="0 is the safest default on Windows")
    p.add_argument("--lr", type=float, default=6e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--dice-weight", type=float, default=0.0)
    p.add_argument("--accumulate", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--data-parallel", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
