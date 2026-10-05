"""Shared full-image, sliding-window and TTA inference helpers.

The setting syntax is intentionally identical for validation and online test:

    full | tile512 | tile512s256
    full-lr | full-ms | full-d4
    tile512s256-lr | tile512s256-ms | tile512s256-d4

TTA is applied around the base predictor, so a transformed image is also evaluated
with sliding windows when a tiled setting is selected.
"""
import re

import torch
from torch import nn


def tile_starts(size, crop, stride):
    if size <= crop:
        return [0]
    starts = list(range(0, size - crop + 1, stride))
    if starts[-1] != size - crop:
        starts.append(size - crop)
    return starts


def parse_setting(name):
    """Return ``(sliding_config_or_None, tta_kind_or_None)``."""
    m = re.fullmatch(r'(full|tile(\d+)(?:s(\d+))?)(?:-(lr|ms|d4))?', name)
    if not m:
        raise ValueError(
            f'unknown inference setting {name!r}; examples: full, tile512s256, '
            'full-ms, tile512s256-ms'
        )
    base, crop_text, stride_text, tta_kind = m.groups()
    sliding = None
    if base != 'full':
        crop = int(crop_text)
        sliding = dict(crop=crop, stride=int(stride_text or crop))
    return sliding, tta_kind


def make_tta(kind):
    import ttach as tta

    if kind == 'lr':
        return tta.Compose([tta.HorizontalFlip(), tta.VerticalFlip()])
    if kind == 'ms':
        return tta.Compose([
            tta.HorizontalFlip(),
            tta.Scale(scales=[0.75, 1.0, 1.25], interpolation='bicubic', align_corners=False),
        ])
    if kind == 'd4':
        return tta.Compose([
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            tta.Rotate90(angles=[90]),
            tta.Scale(scales=[0.5, 0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False),
        ])
    raise ValueError(f'unknown TTA kind {kind!r}')


class LogitPredictor(nn.Module):
    """Expose one-logit-tensor inference, optionally stitched from overlapping tiles."""

    def __init__(self, model, sliding=None):
        super().__init__()
        self.model = model
        self.sliding = sliding

    def forward(self, image):
        if self.sliding is None:
            return self.model(image, True)

        crop = self.sliding['crop']
        stride = self.sliding['stride']
        batch, _, height, width = image.shape
        logits = count = None
        for y in tile_starts(height, crop, stride):
            for x in tile_starts(width, crop, stride):
                tile_logits = self.model(image[:, :, y:y + crop, x:x + crop], True)
                if logits is None:
                    logits = tile_logits.new_zeros(batch, tile_logits.shape[1], height, width)
                    count = tile_logits.new_zeros(1, 1, height, width)
                logits[:, :, y:y + crop, x:x + crop] += tile_logits
                count[:, :, y:y + crop, x:x + crop] += 1
        return logits / count.clamp_min_(1)


def build_predictor(model, setting):
    """Build the exact same prediction graph for validation and test export."""
    sliding, tta_kind = parse_setting(setting)
    predictor = LogitPredictor(model, sliding)
    if tta_kind:
        import ttach as tta
        predictor = tta.SegmentationTTAWrapper(predictor, make_tta(tta_kind))
    return predictor
