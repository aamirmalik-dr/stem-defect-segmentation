"""Segmentation metrics: per-class IoU and Dice, confusion, boundary error.

The headline of the whole project is that pixel accuracy is a misleading
score when one class (background) dominates. Every function here is written
so the rare classes are visible: IoU and Dice are reported per class and
never silently averaged over the frequent ones, and the boundary error is a
geometric distance in pixels, not a pixel count.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

from stemseg.sim import CLASS_NAMES, NUM_CLASSES


def confusion_matrix(
    true: np.ndarray, pred: np.ndarray, num_classes: int = NUM_CLASSES
) -> np.ndarray:
    """Return the raw confusion matrix C[t, p] = count of true t predicted p.

    Args:
        true: Integer ground-truth label map.
        pred: Integer predicted label map of the same shape.
        num_classes: Number of classes.

    Returns:
        (num_classes, num_classes) integer array.
    """
    t = np.asarray(true).ravel()
    p = np.asarray(pred).ravel()
    index = t.astype(np.int64) * num_classes + p.astype(np.int64)
    counts = np.bincount(index, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def per_class_iou(cm: np.ndarray) -> np.ndarray:
    """Intersection-over-union per class from a confusion matrix.

    A class absent from both truth and prediction has an undefined IoU and is
    returned as NaN, so it never inflates or deflates a mean.
    """
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    union = tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, tp / union, np.nan)
    return iou


def per_class_dice(cm: np.ndarray) -> np.ndarray:
    """Dice (F1) per class from a confusion matrix; absent classes are NaN."""
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        dice = np.where(denom > 0, 2 * tp / denom, np.nan)
    return dice


def pixel_accuracy(cm: np.ndarray) -> float:
    """Overall fraction of correctly classified pixels."""
    total = cm.sum()
    return float(np.trace(cm) / total) if total else 0.0


@dataclass
class SegScore:
    """A full segmentation score for one image or a pooled set.

    Attributes:
        confusion: (C, C) confusion matrix.
        iou: (C,) per-class IoU (NaN where the class is absent).
        dice: (C,) per-class Dice.
        pixel_accuracy: Overall pixel accuracy.
        mean_iou: Mean IoU over classes present in truth or prediction.
        boundary_error_px: Mean symmetric boundary distance of the
            disordered-region footprint, in pixels (NaN if that region is
            absent from both truth and prediction).
    """

    confusion: np.ndarray
    iou: np.ndarray
    dice: np.ndarray
    pixel_accuracy: float
    mean_iou: float
    boundary_error_px: float

    def as_dict(self) -> dict:
        """Return a JSON-friendly dict keyed by class name."""
        return {
            "pixel_accuracy": self.pixel_accuracy,
            "mean_iou": self.mean_iou,
            "boundary_error_px": self.boundary_error_px,
            "iou": {name: _nan_to_none(v) for name, v in zip(CLASS_NAMES, self.iou)},
            "dice": {name: _nan_to_none(v) for name, v in zip(CLASS_NAMES, self.dice)},
        }


def _nan_to_none(value: float) -> float | None:
    return None if np.isnan(value) else float(value)


def _boundary_pixels(mask: np.ndarray) -> np.ndarray:
    """Return the inner-boundary pixels of a binary mask."""
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask, border_value=0)
    return mask & ~eroded


def symmetric_boundary_distance(true_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """Mean symmetric surface distance between two binary regions, in pixels.

    For every boundary pixel of one region, the distance to the nearest
    boundary pixel of the other is measured; the two directed means are
    averaged. This is the standard boundary-localisation error and it is the
    right score for "did we put the grain boundary in the right place".

    Returns:
        Mean symmetric boundary distance in pixels. NaN if both regions are
        empty; a large finite penalty (the image diagonal) if exactly one is
        empty, since a missed region is a total localisation failure.
    """
    tb = _boundary_pixels(true_mask)
    pb = _boundary_pixels(pred_mask)
    if not tb.any() and not pb.any():
        return float("nan")
    diag = float(np.hypot(*true_mask.shape))
    if not tb.any() or not pb.any():
        return diag

    dt_to_pred = distance_transform_edt(~pb)
    dt_to_true = distance_transform_edt(~tb)
    d_tp = dt_to_pred[tb].mean()
    d_pt = dt_to_true[pb].mean()
    return float(0.5 * (d_tp + d_pt))


def score_segmentation(true: np.ndarray, pred: np.ndarray, disorder_class: int = 4) -> SegScore:
    """Score one predicted label map against ground truth.

    Args:
        true: Integer ground-truth label map.
        pred: Integer predicted label map.
        disorder_class: Class code used for the boundary-error region.

    Returns:
        A SegScore.
    """
    cm = confusion_matrix(true, pred)
    iou = per_class_iou(cm)
    dice = per_class_dice(cm)
    boundary = symmetric_boundary_distance(true == disorder_class, pred == disorder_class)
    return SegScore(
        confusion=cm,
        iou=iou,
        dice=dice,
        pixel_accuracy=pixel_accuracy(cm),
        mean_iou=float(np.nanmean(iou)),
        boundary_error_px=boundary,
    )


def pool_scores(scores: list[SegScore]) -> SegScore:
    """Combine per-image scores by pooling confusions and averaging boundaries.

    IoU and Dice are recomputed from the summed confusion matrix (a
    pixel-weighted pool, which is the honest way to combine ratios), while the
    boundary error is averaged over the images where it is defined.
    """
    cm = np.sum([s.confusion for s in scores], axis=0)
    iou = per_class_iou(cm)
    dice = per_class_dice(cm)
    boundaries = [s.boundary_error_px for s in scores if not np.isnan(s.boundary_error_px)]
    return SegScore(
        confusion=cm,
        iou=iou,
        dice=dice,
        pixel_accuracy=pixel_accuracy(cm),
        mean_iou=float(np.nanmean(iou)),
        boundary_error_px=float(np.mean(boundaries)) if boundaries else float("nan"),
    )
