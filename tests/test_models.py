"""Tests for features, classical baselines, the U-Net, and training helpers."""

from __future__ import annotations

import numpy as np
import torch

from stemseg.classical import (
    RandomForestPixelClassifier,
    ThresholdParams,
    majority_smooth,
    threshold_morphology,
)
from stemseg.features import design_matrix, feature_names, feature_stack
from stemseg.net import SegUNet, predict_labels, predict_proba, soft_dice_loss
from stemseg.sim import NUM_CLASSES, material_config, simulate_image
from stemseg.train import TrainSettings, class_weights, random_config, sample_pixels


def _sample(size=96, seed=0):
    return simulate_image(material_config("graphene", size=size), np.random.default_rng(seed))


def test_feature_stack_shape_matches_names():
    stack = feature_stack(_sample().image)
    assert stack.shape[-1] == len(feature_names())
    assert stack.dtype == np.float32


def test_design_matrix_is_flattened_stack():
    img = _sample(size=64).image
    dm = design_matrix(img)
    assert dm.shape == (64 * 64, len(feature_names()))


def test_threshold_morphology_returns_valid_labels():
    pred = threshold_morphology(_sample().image)
    assert pred.shape == (96, 96)
    assert pred.min() >= 0 and pred.max() < NUM_CLASSES


def test_threshold_params_change_output():
    img = _sample().image
    a = threshold_morphology(img, ThresholdParams(lattice_pct=60))
    b = threshold_morphology(img, ThresholdParams(lattice_pct=90))
    # A higher lattice percentile labels fewer pixels as a column.
    assert (a == 1).sum() > (b == 1).sum()


def test_majority_smooth_preserves_shape_and_labels():
    pred = threshold_morphology(_sample().image)
    smoothed = majority_smooth(pred)
    assert smoothed.shape == pred.shape
    assert set(np.unique(smoothed)).issubset(set(range(NUM_CLASSES)))


def test_random_forest_fit_predict():
    settings = TrainSettings(size=64)
    x, y = sample_pixels(settings, n_images=4, per_image=300, seed=0)
    rf = RandomForestPixelClassifier(n_estimators=20, seed=0).fit(x, y)
    pred = rf.predict(_sample(size=64).image)
    assert pred.shape == (64, 64)
    proba = rf.predict_proba_image(_sample(size=64).image)
    assert proba.shape == (64, 64, NUM_CLASSES)
    assert np.allclose(proba.sum(-1), 1.0, atol=1e-4)


def test_unet_forward_shape():
    model = SegUNet(base=8)
    x = torch.randn(2, 1, 64, 64)
    out = model(x)
    assert out.shape == (2, NUM_CLASSES, 64, 64)


def test_predict_labels_and_proba():
    model = SegUNet(base=8)
    img = _sample(size=64).image
    labels = predict_labels(model, img)
    proba = predict_proba(model, img)
    assert labels.shape == (64, 64)
    assert proba.shape == (64, 64, NUM_CLASSES)
    assert np.allclose(proba.sum(-1), 1.0, atol=1e-4)


def test_soft_dice_loss_in_unit_interval():
    logits = torch.randn(2, NUM_CLASSES, 32, 32)
    target = torch.randint(0, NUM_CLASSES, (2, 32, 32))
    loss = soft_dice_loss(logits, target)
    assert 0.0 <= float(loss) <= 1.0


def test_soft_dice_perfect_prediction_is_low():
    target = torch.randint(0, NUM_CLASSES, (1, 16, 16))
    onehot = torch.nn.functional.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
    logits = onehot * 20.0  # near-one-hot softmax
    assert float(soft_dice_loss(logits, target)) < 0.05


def test_class_weights_upweight_rare_classes():
    settings = TrainSettings(size=96)
    w = class_weights(np.random.default_rng(0), settings, n_images=6)
    assert w.shape == (NUM_CLASSES,)
    # Background (class 0) is the most common, so it gets the smallest weight.
    assert w[0] == w.min()


def test_random_config_respects_switches():
    rng = np.random.default_rng(0)
    off = TrainSettings(include_defects=False, include_disorder=False, randomize_dose=False)
    cfg = random_config(rng, off)
    assert cfg.vacancy_fraction == 0.0
    assert cfg.dopant_fraction == 0.0
    assert cfg.disorder_fraction == 0.0
    assert cfg.dose == off.fixed_dose
