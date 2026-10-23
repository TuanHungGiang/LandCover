from pathlib import Path

import numpy as np
import torch
from PIL import Image

from baseline_suite.data import LoveDADataset
from baseline_suite.metrics import SegmentationMetrics
from baseline_suite.models import MODEL_NAMES, build_model


def test_registry_has_all_models():
    assert len(MODEL_NAMES) == 19


def test_loveda_mapping(tmp_path: Path):
    for split in ("Train", "Val"):
        image_dir = tmp_path / split / "Urban" / "images_png"
        mask_dir = tmp_path / split / "Urban" / "masks_png"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(image_dir / "1.png")
        raw = np.tile(np.arange(8, dtype=np.uint8), (8, 1))
        Image.fromarray(raw).save(mask_dir / "1.png")
    sample = LoveDADataset(tmp_path, "Val", crop_size=8)[0]
    assert sample["mask"][0, 0].item() == 255
    assert sample["mask"][0, 1:].tolist() == list(range(7))


def test_metric_perfect():
    target = torch.tensor([[[0, 1], [2, 255]]])
    logits = torch.full((1, 7, 2, 2), -9.0)
    for y, x, cls in ((0, 0, 0), (0, 1, 1), (1, 0, 2)):
        logits[0, cls, y, x] = 9.0
    metric = SegmentationMetrics()
    metric.update(logits, target)
    assert metric.compute()["miou"] == 1.0


def test_pspnet_forward():
    model = build_model("pspnet", pretrained=False, image_size=64).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 64, 64))
    assert output.shape == (1, 7, 64, 64)
