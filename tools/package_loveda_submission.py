"""Validate and package LoveDA semantic-segmentation test masks for Codabench.

The official supervised LoveDA masks use grayscale ids 1..7 (0 is no-data).
The archive contains PNG files at its root, without an enclosing directory.
"""
import argparse
import hashlib
import os
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--masks-dir', required=True)
    ap.add_argument('--data-root', default='data/2021LoveDA')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    masks_dir = Path(args.masks_dir)
    data_root = Path(args.data_root)
    output = Path(args.output)
    expected_paths = sorted(
        list((data_root / 'Test' / 'Urban' / 'images_png').glob('*.png'))
        + list((data_root / 'Test' / 'Rural' / 'images_png').glob('*.png'))
    )
    if not expected_paths:
        raise SystemExit(f'no LoveDA test images found below {data_root}')

    expected = {p.name: p for p in expected_paths}
    if len(expected) != len(expected_paths):
        raise SystemExit('Urban/Rural test image names are not unique; a flat submission would overwrite files')
    actual_paths = sorted(masks_dir.glob('*.png'))
    actual = {p.name: p for p in actual_paths}
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    nested = [p for p in masks_dir.rglob('*.png') if p.parent != masks_dir]
    if missing or extra or nested:
        raise SystemExit(
            f'invalid file set: expected={len(expected)} actual={len(actual)} '
            f'missing={missing[:10]} extra={extra[:10]} nested={len(nested)}'
        )

    hist = np.zeros(8, dtype=np.int64)
    for name in sorted(expected):
        with Image.open(actual[name]) as mask_image, Image.open(expected[name]) as source_image:
            mask = np.asarray(mask_image)
            if mask.ndim != 2:
                raise SystemExit(f'{name}: mask must be single-channel grayscale, got shape {mask.shape}')
            if mask.dtype != np.uint8:
                raise SystemExit(f'{name}: mask must be uint8, got {mask.dtype}')
            if mask_image.size != source_image.size:
                raise SystemExit(f'{name}: mask size {mask_image.size} != test image size {source_image.size}')
            values, counts = np.unique(mask, return_counts=True)
            if np.any((values < 1) | (values > 7)):
                raise SystemExit(f'{name}: invalid labels {values.tolist()}; supervised LoveDA submission requires 1..7')
            hist[values] += counts

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(expected):
            archive.write(actual[name], arcname=name)

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    with zipfile.ZipFile(output) as archive:
        members = archive.namelist()
        if len(members) != len(expected) or any('/' in name or '\\' in name for name in members):
            raise SystemExit('internal packaging error: ZIP members must be flat and complete')
    print(f'VALID LoveDA submission: masks={len(expected)} labels=1..7 hist={hist[1:].tolist()}')
    print(f'ZIP={output} bytes={output.stat().st_size} sha256={digest}')


if __name__ == '__main__':
    main()
