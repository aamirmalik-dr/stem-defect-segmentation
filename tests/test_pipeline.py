"""Tests for io round-trips, the benchmark harness, and the real loader."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from stemseg.benchmark import build_methods, run_config
from stemseg.io import load_sample, save_sample
from stemseg.real import crop_to_multiple, downsample, prepare_real
from stemseg.sim import material_config, simulate_image


def test_sample_roundtrip(tmp_path):
    result = simulate_image(material_config("mos2", size=96), np.random.default_rng(0))
    path = tmp_path / "s.npz"
    save_sample(path, result)
    loaded = load_sample(path)
    assert np.array_equal(loaded.image, result.image)
    assert np.array_equal(loaded.labels, result.labels)
    assert loaded.config.material == "mos2"
    assert loaded.config.size == 96


def test_build_methods_threshold_only():
    methods = build_methods(["threshold"])
    assert "threshold" in methods
    img = simulate_image(material_config("graphene", size=64), np.random.default_rng(0)).image
    pred = methods["threshold"].predict(img)
    assert pred.shape == (64, 64)


def test_build_methods_unknown_raises():
    with pytest.raises(ValueError):
        build_methods(["nope"])


def test_run_sweep_threshold(tmp_path):
    config = {
        "name": "unit_sweep",
        "mode": "sweep",
        "seed": 0,
        "images_per_condition": 2,
        "methods": ["threshold"],
        "base_config": {"size": 64, "material": "graphene"},
        "sweep": {"parameter": "dose", "values": [20, 200]},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(config))
    payload = run_config(path, out_dir=tmp_path)
    assert payload["mode"] == "sweep"
    assert len(payload["rows"]) == 2
    for row in payload["rows"]:
        assert "pixel_accuracy" in row["threshold"]
        assert "iou" in row["threshold"]


def test_run_fair_tuning_threshold(tmp_path):
    config = {
        "name": "unit_fair",
        "mode": "fair_tuning",
        "seed": 0,
        "images_per_condition": 2,
        "methods": ["threshold"],
        "base_config": {"size": 64, "material": "graphene", "disorder_fraction": 0.15},
        "sweep": {"parameter": "dose", "values": [50]},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(config))
    payload = run_config(path, out_dir=tmp_path)
    row = payload["rows"][0]
    # The oracle is tuned to maximise rare IoU, so it cannot be worse.
    assert (
        row["threshold_oracle"]["rare_mean_iou"] >= row["threshold_default"]["rare_mean_iou"] - 1e-9
    )


def test_crop_to_multiple():
    img = np.zeros((70, 83), dtype=np.float32)
    cropped = crop_to_multiple(img, multiple=8)
    assert cropped.shape[0] % 8 == 0 and cropped.shape[1] % 8 == 0


def test_downsample_halves():
    img = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    small = downsample(img, 2)
    assert small.shape == (32, 32)


def test_prepare_real_from_png(tmp_path):
    from PIL import Image

    arr = (np.random.default_rng(0).random((100, 120)) * 255).astype(np.uint8)
    p = tmp_path / "img.png"
    Image.fromarray(arr).save(p)
    out = prepare_real(p, downsample_factor=2)
    assert out.ndim == 2
    assert out.shape[0] % 8 == 0 and out.shape[1] % 8 == 0
