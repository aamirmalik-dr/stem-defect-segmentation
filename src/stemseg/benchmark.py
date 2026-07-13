"""Config-driven segmentation benchmark harness.

Every benchmark is a YAML file (see configs/) with a mode and fixed seeds, so
each committed number regenerates bit-for-bit. Modes:

- "sweep": vary one SegConfig parameter (dose, defect fraction, disorder
  fraction) and score every method. Reports per-class IoU and Dice, pixel
  accuracy and boundary error, so the reader can watch pixel accuracy stay
  high while a rare class collapses.
- "materials": run every method on each preset and report per-class IoU. Also
  the source of the segmentation-overlay gallery.
- "fair_tuning": the honest check. At each condition it gives the classical
  threshold baseline the per-condition, ground-truth-optimal parameters (an
  oracle upper bound it could never reach in practice) and compares that,
  plus the balanced random forest, against the U-Net on the rare classes.
  If the U-Net still wins after the baseline is tuned this hard, the gap is
  real and not an artifact of an under-tuned baseline.
- "confusion": pooled confusion matrices per method at one condition.

Results go to results/<name>.json, figures to figures/<name>.png.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from stemseg.classical import RandomForestPixelClassifier, ThresholdParams, threshold_morphology
from stemseg.io import load_rf
from stemseg.metrics import CLASS_NAMES, SegScore, pool_scores, score_segmentation
from stemseg.net import SegUNet, predict_labels
from stemseg.sim import SegConfig, material_config, simulate_image

Predict = Callable[[np.ndarray], np.ndarray]


@dataclass
class Method:
    """One segmentation method exposed as ``predict(image) -> label_map``.

    Attributes:
        name: Registry key, e.g. "unet".
        label: Human-readable name for figures.
        predict: Callable mapping an image to an integer label map.
    """

    name: str
    label: str
    predict: Predict


def _load_unet(path: str | Path) -> SegUNet:
    model = SegUNet()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def build_methods(names: list[str], model_paths: dict[str, str] | None = None) -> dict[str, Method]:
    """Build the requested subset of the method registry.

    Args:
        names: Any of "threshold", "rf", "unet".
        model_paths: Paths to committed artifacts, keyed by method name.

    Returns:
        Ordered mapping of name to Method.
    """
    model_paths = model_paths or {}
    methods: dict[str, Method] = {}
    for name in names:
        if name == "threshold":
            methods[name] = Method(name, "Threshold + morphology", threshold_morphology)
        elif name == "rf":
            if name not in model_paths:
                raise ValueError("method 'rf' needs a model path")
            rf: RandomForestPixelClassifier = load_rf(model_paths[name])
            methods[name] = Method(name, "Random forest", rf.predict)
        elif name == "unet":
            if name not in model_paths:
                raise ValueError("method 'unet' needs a model path")
            model = _load_unet(model_paths[name])
            methods[name] = Method(name, "U-Net", lambda img, m=model: predict_labels(m, img))
        else:
            raise ValueError(f"unknown method: {name!r}")
    return methods


def _simulate_set(base: dict, overrides: dict, n_images: int, seed: int, cond_index: int) -> list:
    """Simulate a reproducible image set for one condition."""
    samples = []
    for k in range(n_images):
        config = SegConfig(**{**base, **overrides})
        rng = np.random.default_rng(seed + 6151 * cond_index + k)
        samples.append(simulate_image(config, rng))
    return samples


def _score_method(method: Method, samples: list) -> SegScore:
    """Score a method over an image set and return the pooled score."""
    scores = [score_segmentation(s.labels, method.predict(s.image)) for s in samples]
    return pool_scores(scores)


def _score_payload(score: SegScore) -> dict:
    """Convert a SegScore into a JSON-friendly dict."""
    return score.as_dict()


def _common(config: dict):
    base = dict(config.get("base_config", {}))
    n_images = int(config.get("images_per_condition", 6))
    seed = int(config.get("seed", 20))
    methods = build_methods(list(config.get("methods", [])), config.get("models"))
    return base, n_images, seed, methods


def run_sweep(config: dict) -> dict:
    """Vary one parameter and score every method at every value."""
    base, n_images, seed, methods = _common(config)
    parameter = config["sweep"]["parameter"]
    values = config["sweep"]["values"]

    rows = []
    for i, value in enumerate(values):
        samples = _simulate_set(base, {parameter: value}, n_images, seed, i)
        row: dict = {parameter: value}
        for name, method in methods.items():
            row[name] = _score_payload(_score_method(method, samples))
        rows.append(row)
        summary = "  ".join(
            f"{n} acc {row[n]['pixel_accuracy']:.3f} mIoU {row[n]['mean_iou']:.3f}" for n in methods
        )
        print(f"{parameter} {value:g}: {summary}")
    return {"mode": "sweep", "parameter": parameter, "values": list(values), "rows": rows}


def run_materials(config: dict) -> dict:
    """Score every method on each material preset."""
    base, n_images, seed, methods = _common(config)
    presets = list(config.get("presets", []))
    overrides = dict(config.get("preset_overrides", {}))

    rows = []
    for i, preset in enumerate(presets):
        preset_base = material_config(preset, **overrides).__dict__
        samples = _simulate_set(preset_base, {}, n_images, seed, i)
        row: dict = {"preset": preset}
        for name, method in methods.items():
            row[name] = _score_payload(_score_method(method, samples))
        rows.append(row)
        summary = "  ".join(f"{n} mIoU {row[n]['mean_iou']:.3f}" for n in methods)
        print(f"{preset}: {summary}")
    return {"mode": "materials", "rows": rows}


# Parameter grid the threshold baseline is allowed to tune over per condition.
_THRESHOLD_GRID = {
    "lattice_pct": (60.0, 68.0, 75.0),
    "dopant_pct": (96.0, 98.0, 99.0),
    "disorder_pct": (80.0, 86.0, 92.0),
    "vacancy_strength": (0.25, 0.35, 0.5),
}
RARE_CLASSES = ("vacancy", "dopant", "disordered")


def _rare_mean_iou(score: SegScore) -> float:
    """Mean IoU over the three rare classes, ignoring absent ones."""
    idx = [CLASS_NAMES.index(c) for c in RARE_CLASSES]
    vals = [score.iou[i] for i in idx if not np.isnan(score.iou[i])]
    return float(np.mean(vals)) if vals else float("nan")


def _oracle_threshold(samples: list) -> tuple[SegScore, dict]:
    """Return the threshold baseline's best score over its grid on this set.

    "Best" maximises the mean rare-class IoU, using ground-truth access. This
    is an upper bound the baseline could not reach without the labels.
    """
    keys = list(_THRESHOLD_GRID)
    best_score, best_params = None, None
    for combo in itertools.product(*[_THRESHOLD_GRID[k] for k in keys]):
        params = ThresholdParams(**dict(zip(keys, combo)))
        scores = [
            score_segmentation(s.labels, threshold_morphology(s.image, params)) for s in samples
        ]
        pooled = pool_scores(scores)
        if best_score is None or _rare_mean_iou(pooled) > _rare_mean_iou(best_score):
            best_score, best_params = pooled, dict(zip(keys, combo))
    return best_score, best_params


def run_fair_tuning(config: dict) -> dict:
    """The honest check: tune the classical baseline hard, then compare."""
    base, n_images, seed, methods = _common(config)
    parameter = config["sweep"]["parameter"]
    values = config["sweep"]["values"]

    rows = []
    for i, value in enumerate(values):
        samples = _simulate_set(base, {parameter: value}, n_images, seed, i)
        row: dict = {parameter: value}
        # Default classical, oracle-tuned classical, and every built method.
        default = pool_scores(
            [score_segmentation(s.labels, threshold_morphology(s.image)) for s in samples]
        )
        oracle, oracle_params = _oracle_threshold(samples)
        row["threshold_default"] = {
            **_score_payload(default),
            "rare_mean_iou": _rare_mean_iou(default),
        }
        row["threshold_oracle"] = {
            **_score_payload(oracle),
            "rare_mean_iou": _rare_mean_iou(oracle),
            "params": oracle_params,
        }
        for name, method in methods.items():
            score = _score_method(method, samples)
            row[name] = {**_score_payload(score), "rare_mean_iou": _rare_mean_iou(score)}
        rows.append(row)
        cls_default = row["threshold_default"]["rare_mean_iou"]
        cls_oracle = row["threshold_oracle"]["rare_mean_iou"]
        extra = "  ".join(f"{n} {row[n]['rare_mean_iou']:.3f}" for n in methods)
        print(
            f"{parameter} {value:g}: rare-IoU  thr {cls_default:.3f} "
            f"thr(oracle) {cls_oracle:.3f}  {extra}"
        )
    return {"mode": "fair_tuning", "parameter": parameter, "values": list(values), "rows": rows}


def run_confusion(config: dict) -> dict:
    """Pooled per-method confusion matrices at one condition."""
    base, n_images, seed, methods = _common(config)
    overrides = dict(config.get("condition", {}))
    samples = _simulate_set(base, overrides, n_images, seed, 0)
    out: dict = {"mode": "confusion", "condition": overrides, "class_names": list(CLASS_NAMES)}
    for name, method in methods.items():
        score = _score_method(method, samples)
        cm = score.confusion.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        normed = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
        out[name] = {
            "counts": cm.astype(int).tolist(),
            "row_normalized": normed.tolist(),
            "pixel_accuracy": score.pixel_accuracy,
            "iou": _score_payload(score)["iou"],
        }
        print(f"{name}: pixel acc {score.pixel_accuracy:.3f}")
    return out


MODES = {
    "sweep": run_sweep,
    "materials": run_materials,
    "fair_tuning": run_fair_tuning,
    "confusion": run_confusion,
}


def run_config(path: str | Path, out_dir: str | Path = "results") -> dict:
    """Run one YAML benchmark config and write results/<name>.json.

    Args:
        path: YAML config file.
        out_dir: Directory for the JSON result.

    Returns:
        The result payload (also written to disk).
    """
    path = Path(path)
    config = yaml.safe_load(path.read_text())
    mode = config.get("mode", "sweep")
    if mode not in MODES:
        raise ValueError(f"unknown benchmark mode: {mode!r}")
    print(f"== {path.stem} ({mode}) ==")
    payload = MODES[mode](config)
    payload["config"] = config
    out = Path(out_dir) / f"{config.get('name', path.stem)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"saved {out}")
    return payload
