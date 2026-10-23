from __future__ import annotations

import torch


class SegmentationMetrics:
    def __init__(self, num_classes: int = 7, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = logits.argmax(1).detach().cpu().reshape(-1)
        target = target.detach().cpu().reshape(-1)
        valid = (target != self.ignore_index) & (target >= 0) & (target < self.num_classes)
        bins = target[valid] * self.num_classes + pred[valid]
        self.confusion += torch.bincount(bins, minlength=self.num_classes**2).reshape(
            self.num_classes, self.num_classes
        )

    def compute(self) -> dict[str, object]:
        cm = self.confusion.double()
        intersection = cm.diag()
        union = cm.sum(0) + cm.sum(1) - intersection
        iou = torch.where(union > 0, intersection / union, torch.nan)
        acc = intersection.sum() / cm.sum().clamp_min(1)
        return {
            "miou": float(torch.nanmean(iou)),
            "pixel_accuracy": float(acc),
            "class_iou": [None if torch.isnan(x) else float(x) for x in iou],
        }
