"""Tests for segmentation metrics."""

from __future__ import annotations

import numpy as np

from stemseg.metrics import (
    confusion_matrix,
    per_class_dice,
    per_class_iou,
    pixel_accuracy,
    pool_scores,
    score_segmentation,
    symmetric_boundary_distance,
)
from stemseg.sim import NUM_CLASSES


def test_perfect_prediction_scores_one():
    true = np.array([[0, 1, 2], [3, 4, 0]])
    score = score_segmentation(true, true.copy())
    assert score.pixel_accuracy == 1.0
    present = ~np.isnan(score.iou)
    assert np.allclose(score.iou[present], 1.0)
    assert np.allclose(score.dice[present], 1.0)


def test_absent_class_is_nan_not_zero():
    true = np.zeros((4, 4), dtype=int)  # only background present
    score = score_segmentation(true, true.copy())
    assert np.isnan(score.iou[1])
    assert not np.isnan(score.iou[0])


def test_iou_matches_hand_computation():
    # A 2x2 where one lattice pixel is predicted as background.
    true = np.array([[1, 1], [0, 0]])
    pred = np.array([[1, 0], [0, 0]])
    cm = confusion_matrix(true, pred)
    iou = per_class_iou(cm)
    # lattice: tp=1, fp=0, fn=1 -> 1/2. background: tp=2, fp=1, fn=0 -> 2/3.
    assert np.isclose(iou[1], 0.5)
    assert np.isclose(iou[0], 2 / 3)


def test_dice_relates_to_iou():
    true = np.array([[1, 1, 0], [0, 0, 1]])
    pred = np.array([[1, 0, 0], [0, 0, 1]])
    cm = confusion_matrix(true, pred)
    iou = per_class_iou(cm)
    dice = per_class_dice(cm)
    # Dice = 2*IoU / (1 + IoU) elementwise where defined.
    present = ~np.isnan(iou)
    assert np.allclose(dice[present], 2 * iou[present] / (1 + iou[present]))


def test_pixel_accuracy_inflated_by_imbalance():
    # 99% background, all correct; 1% lattice, all wrong. Accuracy stays high.
    true = np.zeros(100, dtype=int)
    true[0] = 1
    pred = np.zeros(100, dtype=int)  # predict all background
    cm = confusion_matrix(true.reshape(10, 10), pred.reshape(10, 10))
    assert pixel_accuracy(cm) == 0.99
    # But lattice IoU is zero, which is the honest signal.
    assert per_class_iou(cm)[1] == 0.0


def test_confusion_matrix_shape_and_counts():
    true = np.array([0, 1, 2, 3, 4])
    pred = np.array([0, 1, 2, 3, 4])
    cm = confusion_matrix(true, pred)
    assert cm.shape == (NUM_CLASSES, NUM_CLASSES)
    assert cm.sum() == 5
    assert np.array_equal(np.diag(cm), np.ones(NUM_CLASSES))


def test_boundary_distance_zero_for_identical_regions():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:12, 5:12] = True
    assert symmetric_boundary_distance(mask, mask) == 0.0


def test_boundary_distance_grows_with_shift():
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:20, 10:20] = True
    shift_small = np.roll(mask, 1, axis=0)
    shift_large = np.roll(mask, 5, axis=0)
    d_small = symmetric_boundary_distance(mask, shift_small)
    d_large = symmetric_boundary_distance(mask, shift_large)
    assert 0 < d_small < d_large


def test_boundary_distance_penalises_missing_region():
    mask = np.zeros((30, 30), dtype=bool)
    mask[5:15, 5:15] = True
    empty = np.zeros_like(mask)
    d = symmetric_boundary_distance(mask, empty)
    assert d == np.hypot(30, 30)


def test_boundary_distance_nan_when_both_empty():
    empty = np.zeros((10, 10), dtype=bool)
    assert np.isnan(symmetric_boundary_distance(empty, empty))


def test_pool_scores_equals_single_when_one_image():
    true = np.array([[0, 1], [2, 3]])
    pred = np.array([[0, 1], [2, 0]])
    single = score_segmentation(true, pred)
    pooled = pool_scores([single])
    assert np.array_equal(single.confusion, pooled.confusion)
    assert single.pixel_accuracy == pooled.pixel_accuracy
