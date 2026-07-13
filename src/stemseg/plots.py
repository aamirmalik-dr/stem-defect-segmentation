"""Figure rendering for benchmarks, galleries and confusion matrices.

All figures share one class colour scheme so masks read the same everywhere.
The two hero figures are the segmentation-overlay gallery (image, ground
truth, prediction across materials) and the per-class IoU-versus-dose curve.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from stemseg.sim import CLASS_NAMES

# One fixed colour per class: background dark, lattice grey-blue, then three
# saturated colours for the rare classes so they pop against the lattice.
CLASS_COLORS = ["#10131a", "#5b7fa6", "#ffd23f", "#ff5c39", "#8f4fd1"]
CLASS_CMAP = ListedColormap(CLASS_COLORS)
_METHOD_COLORS = {
    "threshold": "#e07b39",
    "threshold_default": "#e07b39",
    "threshold_oracle": "#b0752f",
    "rf": "#3a9d6b",
    "unet": "#3f6fd1",
}


def _legend_handles() -> list[Patch]:
    return [
        Patch(facecolor=c, edgecolor="none", label=n) for c, n in zip(CLASS_COLORS, CLASS_NAMES)
    ]


def show_labels(ax: plt.Axes, labels: np.ndarray) -> None:
    """Render a label map with the fixed class colours on an axis."""
    ax.imshow(labels, cmap=CLASS_CMAP, vmin=0, vmax=len(CLASS_NAMES) - 1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])


def show_image(ax: plt.Axes, image: np.ndarray) -> None:
    """Render a STEM image with robust display scaling.

    The upper limit is the 97th percentile, so the bright dopant columns (a
    percent or two of pixels, and up to ~80x the host brightness for a heavy
    dopant in a light lattice) saturate to white instead of compressing the
    host lattice into the black floor. This is a display choice only; the
    models always see the raw image.
    """
    vmax = float(np.percentile(image, 97.0))
    vmin = float(np.percentile(image, 1.0))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    ax.imshow(image, cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])


def gallery(
    rows: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    path: str | Path,
    dpi: int = 110,
) -> None:
    """Save the segmentation-overlay gallery.

    Args:
        rows: One tuple per material: (title, image, true_labels, pred_labels).
        path: Output PNG path.
        dpi: Figure resolution.
    """
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(9.5, 3.15 * n))
    if n == 1:
        axes = axes[None, :]
    for r, (title, image, true, pred) in enumerate(rows):
        show_image(axes[r, 0], image)
        axes[r, 0].set_ylabel(title, fontsize=11)
        show_labels(axes[r, 1], true)
        show_labels(axes[r, 2], pred)
        if r == 0:
            axes[r, 0].set_title("STEM image", fontsize=11)
            axes[r, 1].set_title("ground truth", fontsize=11)
            axes[r, 2].set_title("U-Net prediction", fontsize=11)
    fig.legend(
        handles=_legend_handles(),
        loc="lower center",
        ncol=len(CLASS_NAMES),
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _iou_of(entry: dict, cls: str) -> float:
    val = entry["iou"].get(cls)
    return float("nan") if val is None else float(val)


def plot_sweep(payload: dict, path: str | Path) -> None:
    """Plot per-class IoU and pixel accuracy against the swept parameter."""
    parameter = payload["parameter"]
    rows = payload["rows"]
    values = [row[parameter] for row in rows]
    method_names = [k for k in rows[0] if k != parameter]

    rare = ["vacancy", "dopant", "disordered"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    for name in method_names:
        color = _METHOD_COLORS.get(name, None)
        ax.plot(
            values,
            [row[name]["pixel_accuracy"] for row in rows],
            "--",
            color=color,
            alpha=0.55,
            label=f"{name} pixel acc",
        )
        rare_iou = [np.nanmean([_iou_of(row[name], c) for c in rare]) for row in rows]
        ax.plot(values, rare_iou, "-o", color=color, ms=4, label=f"{name} rare mIoU")
    ax.set_xlabel(parameter)
    ax.set_ylabel("score")
    ax.set_title("Pixel accuracy hides the rare-class collapse")
    ax.set_ylim(0, 1.02)
    if _is_log_axis(parameter):
        ax.set_xscale("log")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25)

    ax = axes[1]
    best = method_names[-1]
    for cls in CLASS_NAMES:
        ax.plot(values, [_iou_of(row[best], cls) for row in rows], "-o", ms=4, label=cls)
    ax.set_xlabel(parameter)
    ax.set_ylabel("IoU")
    ax.set_title(f"Per-class IoU ({best})")
    ax.set_ylim(0, 1.02)
    if _is_log_axis(parameter):
        ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_iou_vs_dose(payload: dict, path: str | Path) -> None:
    """Hero curve: per-class IoU versus dose for every method, small multiples."""
    parameter = payload["parameter"]
    rows = payload["rows"]
    values = [row[parameter] for row in rows]
    method_names = [k for k in rows[0] if k != parameter]

    fig, axes = plt.subplots(
        1, len(method_names), figsize=(4.6 * len(method_names), 4.2), sharey=True
    )
    if len(method_names) == 1:
        axes = [axes]
    for ax, name in zip(axes, method_names):
        for cls in CLASS_NAMES:
            ax.plot(values, [_iou_of(row[name], cls) for row in rows], "-o", ms=4, label=cls)
        ax.set_title(name)
        ax.set_xlabel(parameter)
        if _is_log_axis(parameter):
            ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("IoU")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_fair_tuning(payload: dict, path: str | Path) -> None:
    """Rare-class IoU versus the swept parameter, including the oracle baseline."""
    parameter = payload["parameter"]
    rows = payload["rows"]
    values = [row[parameter] for row in rows]
    series = [k for k in rows[0] if k != parameter]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for name in series:
        color = _METHOD_COLORS.get(name, None)
        style = ":" if name == "threshold_oracle" else "-o"
        ax.plot(
            values,
            [row[name]["rare_mean_iou"] for row in rows],
            style,
            color=color,
            ms=4,
            label=name,
        )
    ax.set_xlabel(parameter)
    ax.set_ylabel("mean IoU over rare classes")
    ax.set_title("Rare-class IoU after tuning the baseline to an oracle")
    if _is_log_axis(parameter):
        ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion(payload: dict, path: str | Path) -> None:
    """Row-normalised confusion matrices, one panel per method."""
    names = [k for k in payload if k not in ("mode", "condition", "class_names", "config")]
    fig, axes = plt.subplots(1, len(names), figsize=(4.4 * len(names), 4.0))
    if len(names) == 1:
        axes = [axes]
    labels = payload["class_names"]
    for col, (ax, name) in enumerate(zip(axes, names)):
        cm = np.array(payload[name]["row_normalized"])
        im = ax.imshow(cm, cmap="magma", vmin=0, vmax=1)
        ax.set_title(f"{name}\npixel acc {payload[name]['pixel_accuracy']:.3f}", fontsize=10)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        # Only the leftmost panel gets y-tick labels and the "true" y-label, so
        # they never overlap the panel to their left.
        if col == 0:
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_ylabel("true")
        else:
            ax.set_yticklabels([])
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] < 0.6 else "black",
                    fontsize=7,
                )
        ax.set_xlabel("predicted")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _is_log_axis(parameter: str) -> bool:
    return parameter == "dose"


def plot_payload(payload: dict, path: str | Path) -> None:
    """Dispatch to the right figure for a benchmark payload's mode."""
    mode = payload.get("mode")
    if mode == "sweep":
        plot_sweep(payload, path)
    elif mode == "fair_tuning":
        plot_fair_tuning(payload, path)
    elif mode == "confusion":
        plot_confusion(payload, path)
    elif mode == "materials":
        return  # gallery is rendered separately from stored samples
    else:
        raise ValueError(f"no plotter for mode {mode!r}")
