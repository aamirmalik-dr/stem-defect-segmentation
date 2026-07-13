"""Regenerate the repository's hero figures from committed artifacts.

Run after the benchmarks so the results JSON files exist:

    stemseg benchmark configs/dose_sweep.yaml configs/defect_sweep.yaml \
        configs/imbalance_sweep.yaml configs/fair_tuning.yaml \
        configs/confusion.yaml
    python scripts/make_figures.py

It writes:
    figures/gallery.png            the segmentation-overlay gallery (hero 1)
    figures/iou_vs_dose.png        per-class IoU versus dose (hero 2)
    figures/sample_preview.png     one simulated image next to its labels
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stemseg.benchmark import build_methods
from stemseg.cli import DEFAULT_MODELS, SAMPLES
from stemseg.io import load_sample
from stemseg.plots import gallery, plot_iou_vs_dose

ROOT = Path(__file__).resolve().parent.parent


def make_gallery() -> None:
    predict = build_methods(["unet"], DEFAULT_MODELS)["unet"].predict
    rows = []
    for name, *_ in SAMPLES:
        sample = load_sample(ROOT / "data" / "sample" / f"{name}.npz")
        rows.append((name, sample.image, sample.labels, predict(sample.image)))
    gallery(rows, ROOT / "figures" / "gallery.png")
    print("saved figures/gallery.png")


def make_iou_vs_dose() -> None:
    payload = json.loads((ROOT / "results" / "dose_sweep.json").read_text())
    plot_iou_vs_dose(payload, ROOT / "figures" / "iou_vs_dose.png")
    print("saved figures/iou_vs_dose.png")


def make_sample_preview() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from stemseg.plots import show_image, show_labels
    from stemseg.sim import material_config, simulate_image

    result = simulate_image(
        material_config("mos2", size=192, dose=120.0, disorder_fraction=0.14),
        np.random.default_rng(2),
    )
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    show_image(axes[0], result.image)
    axes[0].set_title("simulated MoS2, dose 120")
    show_labels(axes[1], result.labels)
    axes[1].set_title("ground-truth labels")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "sample_preview.png", dpi=140)
    plt.close(fig)
    print("saved figures/sample_preview.png")


def main() -> None:
    make_sample_preview()
    make_gallery()
    make_iou_vs_dose()


if __name__ == "__main__":
    main()
