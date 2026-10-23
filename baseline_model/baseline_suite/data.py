from __future__ import annotations

import random
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


LOVEDA_CLASSES = (
    "background",
    "building",
    "road",
    "water",
    "barren",
    "forest",
    "agriculture",
)


def prepare_loveda(zip_root: str | Path, output_root: str | Path) -> Path:
    """Extract only labeled Train and Val archives. Test.zip is never touched."""
    zip_root, output_root = Path(zip_root), Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for split in ("Train", "Val"):
        archive = zip_root / f"{split}.zip"
        target = output_root / split
        if target.exists() and any(target.rglob("*.png")):
            continue
        if not archive.is_file():
            raise FileNotFoundError(f"Missing LoveDA archive: {archive}")
        with zipfile.ZipFile(archive) as zf:
            bad = [n for n in zf.namelist() if Path(n).is_absolute() or ".." in Path(n).parts]
            if bad:
                raise RuntimeError(f"Unsafe paths in {archive}: {bad[:3]}")
            zf.extractall(output_root)
    return output_root


def _resolve_split(root: Path, split: str) -> Path:
    candidates = [root / split, root / "2021LoveDA" / split, root / "LoveDA" / split]
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(
        f"Cannot find extracted LoveDA/{split} below {root}. "
        "Run: python prepare_data.py"
    )


class LoveDADataset(Dataset):
    """Official LoveDA Train/Val loader with raw label ids 1..7 -> 0..6."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        crop_size: int = 512,
        training: bool = False,
        seed: int = 42,
    ) -> None:
        if split not in {"Train", "Val"}:
            raise ValueError("This suite intentionally permits only LoveDA Train and Val")
        self.root = _resolve_split(Path(root), split)
        self.crop_size = crop_size
        self.training = training
        self.seed = seed
        # Some Kaggle mirrors add an extra directory level (for example
        # Train/Train/Rural/...). Discover official images_png/masks_png pairs
        # recursively instead of assuming an exact nesting depth.
        samples: list[tuple[Path, Path]] = []
        image_dirs = sorted(
            path for path in self.root.rglob("*")
            if path.is_dir() and path.name.lower() == "images_png"
        )
        for image_dir in image_dirs:
            mask_dir = next(
                (
                    sibling
                    for sibling in image_dir.parent.iterdir()
                    if sibling.is_dir() and sibling.name.lower() == "masks_png"
                ),
                None,
            )
            if mask_dir is None:
                continue
            masks = {p.stem: p for p in mask_dir.glob("*.png")}
            samples.extend(
                (image_path, masks[image_path.stem])
                for image_path in image_dir.glob("*.png")
                if image_path.stem in masks
            )
        self.samples = sorted(samples, key=lambda pair: pair[0].as_posix())
        if not self.samples:
            discovered = sorted(
                str(path.relative_to(self.root))
                for path in self.root.rglob("*")
                if path.is_dir() and path.name.lower() in {"images_png", "masks_png"}
            )
            raise RuntimeError(
                f"No paired images/masks found in {self.root}. "
                f"Discovered candidate directories: {discovered[:12]}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _pad(image: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
        h, w = mask.shape
        ph, pw = max(0, size - h), max(0, size - w)
        if ph or pw:
            image = np.pad(image, ((0, ph), (0, pw), (0, 0)), mode="reflect")
            mask = np.pad(mask, ((0, ph), (0, pw)), constant_values=0)
        return image, mask

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path, mask_path = self.samples[index]
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        raw_mask = np.asarray(Image.open(mask_path), dtype=np.uint8)
        image, raw_mask = self._pad(image, raw_mask, self.crop_size)
        h, w = raw_mask.shape
        if self.training:
            rng = random.Random(self.seed + index + random.randrange(1 << 20))
            top = rng.randint(0, h - self.crop_size)
            left = rng.randint(0, w - self.crop_size)
            image = image[top : top + self.crop_size, left : left + self.crop_size]
            raw_mask = raw_mask[top : top + self.crop_size, left : left + self.crop_size]
            if rng.random() < 0.5:
                image, raw_mask = image[:, ::-1], raw_mask[:, ::-1]
            if rng.random() < 0.5:
                image, raw_mask = image[::-1], raw_mask[::-1]

        # Official masks: 0=no-data, 1..7=classes. Padding/no-data becomes ignore 255.
        mask = raw_mask.astype(np.int64) - 1
        mask[raw_mask == 0] = 255
        image_tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255)
        mean = image_tensor.new_tensor((0.485, 0.456, 0.406))[:, None, None]
        std = image_tensor.new_tensor((0.229, 0.224, 0.225))[:, None, None]
        return {
            "image": (image_tensor - mean) / std,
            "mask": torch.from_numpy(np.ascontiguousarray(mask)),
            "id": image_path.stem,
        }
