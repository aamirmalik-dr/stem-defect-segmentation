"""Bring-your-own real-image loader.

The models are trained purely on simulation, but the CLI and API can run on a
real micrograph. Real HAADF images come in many bit depths, often with a
burned-in scale bar and a column width that does not match the training
regime, so this module handles loading, contrast normalisation, optional
inversion (for bright-field or ADF-inverted data), cropping to a size the
U-Net can pool, and optional downsampling to bring the column width into the
trained range. There is no ground truth for real data, so any output is
qualitative.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_image(path: str | Path, invert: bool = False) -> np.ndarray:
    """Load a real image as a float32 array normalised to roughly [0, 1].

    Args:
        path: Path to a PNG/TIFF/JPEG grayscale or RGB image.
        invert: If True, invert contrast (dark features become bright).

    Returns:
        Float32 2D array.
    """
    from PIL import Image

    img = Image.open(path).convert("F")
    arr = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(arr, 1.0), np.percentile(arr, 99.0)
    if hi > lo:
        arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    if invert:
        arr = 1.0 - arr
    return arr


def crop_to_multiple(image: np.ndarray, multiple: int = 8) -> np.ndarray:
    """Centre-crop an image so both sides are a multiple of ``multiple``."""
    h, w = image.shape
    nh = (h // multiple) * multiple
    nw = (w // multiple) * multiple
    r0 = (h - nh) // 2
    c0 = (w - nw) // 2
    return image[r0 : r0 + nh, c0 : c0 + nw]


def downsample(image: np.ndarray, factor: int) -> np.ndarray:
    """Block-average downsample by an integer factor (>=1)."""
    if factor <= 1:
        return image
    h, w = image.shape
    nh, nw = h // factor, w // factor
    trimmed = image[: nh * factor, : nw * factor]
    return trimmed.reshape(nh, factor, nw, factor).mean(axis=(1, 3)).astype(np.float32)


def prepare_real(path: str | Path, invert: bool = False, downsample_factor: int = 1) -> np.ndarray:
    """Load, optionally downsample, and crop a real image for segmentation."""
    img = load_image(path, invert=invert)
    img = downsample(img, downsample_factor)
    return crop_to_multiple(img)
