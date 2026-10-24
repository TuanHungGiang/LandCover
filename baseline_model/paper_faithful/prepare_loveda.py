from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def paired_files(split_root: Path) -> list[tuple[Path, Path, str]]:
    pairs: list[tuple[Path, Path, str]] = []
    for image_dir in sorted(split_root.rglob("images_png")):
        mask_dir = image_dir.parent / "masks_png"
        if not mask_dir.is_dir():
            continue
        masks = {path.stem: path for path in mask_dir.glob("*.png")}
        relative_parent = image_dir.parent.relative_to(split_root)
        prefix = "__".join(relative_parent.parts).replace(" ", "_") or "root"
        for image in sorted(image_dir.glob("*.png")):
            mask = masks.get(image.stem)
            if mask is not None:
                pairs.append((image.resolve(), mask.resolve(), f"{prefix}__{image.stem}.png"))
    if not pairs:
        raise RuntimeError(f"No images_png/masks_png pairs found below {split_root}")
    names = [name for _, _, name in pairs]
    if len(names) != len(set(names)):
        raise RuntimeError(f"Generated duplicate sample ids below {split_root}")
    return pairs


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        return
    try:
        os.symlink(source, destination)
    except OSError:
        # Kaggle supports symlinks. This fallback is intentionally explicit for
        # environments that do not; it can consume several GB.
        import shutil

        shutil.copy2(source, destination)


def prepare(train_root: Path, val_root: Path, output: Path) -> dict[str, object]:
    counts: dict[str, int] = {}
    for source_root, split in ((train_root, "train"), (val_root, "val")):
        images_out = output / "img_dir" / split
        masks_out = output / "ann_dir" / split
        images_out.mkdir(parents=True, exist_ok=True)
        masks_out.mkdir(parents=True, exist_ok=True)
        pairs = paired_files(source_root)
        for image, mask, name in pairs:
            link_or_copy(image, images_out / name)
            link_or_copy(mask, masks_out / name)
        counts[split] = len(pairs)

    result = {
        "train_source": str(train_root.resolve()),
        "val_source": str(val_root.resolve()),
        "output": str(output.resolve()),
        "counts": counts,
    }
    (output / "prepared.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a zero-copy LoveDA layout for MMSegmentation")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(prepare(args.train, args.val, args.output), indent=2))

