"""Regression guard for the committed models, samples, and metrics.

These tests load the committed U-Net and random forest, run them on the
committed samples, and confirm they reproduce the committed
``results/metrics.json``. If a model file, a sample, or the scoring code drifts
out of sync with the published numbers, one of these fails, which is exactly
the class of mismatch a reader would otherwise only find by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from stemseg.benchmark import build_methods
from stemseg.io import load_sample
from stemseg.metrics import score_segmentation

ROOT = Path(__file__).resolve().parent.parent
MODELS = {"unet": ROOT / "models" / "unet.pt", "rf": ROOT / "models" / "rf.pkl"}
METRICS = ROOT / "results" / "metrics.json"

pytestmark = pytest.mark.skipif(
    not (MODELS["unet"].exists() and MODELS["rf"].exists() and METRICS.exists()),
    reason="committed models or metrics.json not present",
)


def test_committed_metrics_reproduce():
    """Each committed sample scores what results/metrics.json records."""
    recorded = json.loads(METRICS.read_text())["samples"]
    methods = build_methods(["threshold", "rf", "unet"], {k: str(v) for k, v in MODELS.items()})
    for name, per_method in recorded.items():
        sample = load_sample(ROOT / "data" / "sample" / f"{name}.npz")
        for method_name, expected in per_method.items():
            score = score_segmentation(sample.labels, methods[method_name].predict(sample.image))
            assert score.mean_iou == pytest.approx(expected["mean_iou"], abs=0.01), (
                f"{name}/{method_name}: mean IoU {score.mean_iou:.4f} "
                f"vs recorded {expected['mean_iou']}"
            )
            assert score.pixel_accuracy == pytest.approx(expected["pixel_accuracy"], abs=0.01)


def test_committed_unet_param_count():
    """The committed U-Net matches the architecture the model card describes."""
    import torch

    from stemseg.net import SegUNet

    model = SegUNet()
    model.load_state_dict(torch.load(MODELS["unet"], map_location="cpu", weights_only=True))
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 483221


def test_all_background_baseline_is_the_documented_trap():
    """Predicting all background scores high accuracy but near-zero mean IoU.

    This is the headline framing in the README and RESULTS; pin it so the claim
    cannot silently drift.
    """
    from stemseg.metrics import pool_scores
    from stemseg.sim import material_config, simulate_image

    cfgs = [
        simulate_image(
            material_config(
                "graphene",
                size=192,
                dose=100.0,
                vacancy_fraction=0.03,
                dopant_fraction=0.03,
                disorder_fraction=0.12,
                rotation_deg=15.0,
            ),
            np.random.default_rng(6151 * 3 + j),
        )
        for j in range(6)
    ]
    pooled = pool_scores([score_segmentation(c.labels, np.zeros_like(c.labels)) for c in cfgs])
    assert pooled.pixel_accuracy == pytest.approx(0.703, abs=0.01)
    assert pooled.mean_iou == pytest.approx(0.141, abs=0.01)
