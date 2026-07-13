"""Assemble and execute the tutorial notebook.

Building the notebook programmatically keeps its code in one reviewable place
and guarantees the committed outputs come from a real execution in this
environment. Run after the models are trained:

    python scripts/build_notebook.py

It writes an executed notebooks/tutorial.ipynb.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent


def md(text: str):
    return new_markdown_cell(text.strip())


def code(text: str):
    return new_code_cell(text.strip())


CELLS = [
    md("""
# Segmenting point defects in simulated STEM

This notebook walks the whole pipeline end to end: simulate an atomic-resolution
STEM image with an exact per-pixel label map, segment it with a classical
baseline and with a U-Net, and score both with per-class IoU, Dice and a
boundary-localisation error. The theme throughout is that **pixel accuracy is a
trap** on this problem, because one class (background) dominates every frame.

Everything runs on CPU in a minute or two.
"""),
    code("""
import numpy as np
import matplotlib.pyplot as plt

from stemseg import material_config, simulate_image, CLASS_NAMES
from stemseg.plots import show_labels, show_image, CLASS_COLORS

# stemseg.plots forces the Agg backend at import (for headless CLI use); switch
# back to the inline backend so figures render in the notebook.
%matplotlib inline

rng = np.random.default_rng(0)
print("classes:", CLASS_NAMES)
"""),
    md("""
## 1. Simulate an image and its ground truth

`simulate_image` returns the noisy image and the label map together. The label
map is built from the crystal geometry, never from the noisy pixels, so it is
exact by construction.
"""),
    code("""
cfg = material_config("graphene", size=192, dose=150.0,
                      vacancy_fraction=0.04, dopant_fraction=0.04,
                      disorder_fraction=0.14, rotation_deg=15.0)
sample = simulate_image(cfg, np.random.default_rng(0))

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
show_image(ax[0], sample.image)
ax[0].set_title("simulated STEM image (dose 150)")
show_labels(ax[1], sample.labels)
ax[1].set_title("ground-truth labels")
plt.tight_layout(); plt.show()
"""),
    md("""
## 2. The class imbalance, made explicit

Background and lattice fill the frame; the three defect classes are a tiny
sliver of it. This is exactly why a single accuracy number is misleading.
"""),
    code("""
frac = np.bincount(sample.labels.ravel(), minlength=5) / sample.labels.size
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.bar(CLASS_NAMES, frac, color=CLASS_COLORS)
ax.set_ylabel("fraction of pixels")
ax.set_title("class balance in one frame")
for i, f in enumerate(frac):
    ax.text(i, f + 0.005, f"{f:.3f}", ha="center", fontsize=9)
plt.tight_layout(); plt.show()

baseline_acc = frac[0]  # predicting 'background everywhere'
print(f"predicting background everywhere already scores {baseline_acc:.3f} pixel accuracy")
"""),
    md("""
## 3. Classical baseline: threshold + morphology

A hand-built pipeline: threshold for columns, a brighter threshold for dopants,
a local-texture test for the amorphous region, and a morphological "ordered
hole" test for vacancies. No learning.
"""),
    code("""
from stemseg import threshold_morphology, score_segmentation

pred_thr = threshold_morphology(sample.image)
score_thr = score_segmentation(sample.labels, pred_thr)

fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
show_image(ax[0], sample.image); ax[0].set_title("image")
show_labels(ax[1], sample.labels); ax[1].set_title("ground truth")
show_labels(ax[2], pred_thr); ax[2].set_title("threshold + morphology")
plt.tight_layout(); plt.show()

print(f"pixel accuracy {score_thr.pixel_accuracy:.3f}   mean IoU {score_thr.mean_iou:.3f}")
print("per-class IoU:", {n: (None if v is None else round(v, 3))
                          for n, v in score_thr.as_dict()["iou"].items()})
"""),
    md("""
## 4. Random forest on local features

The stronger classical baseline: a random forest over a multi-scale local
feature bank (Gaussians, Laplacians, gradients, local texture), trained on the
same simulator the U-Net uses. We load the committed model.
"""),
    code("""
from stemseg.io import load_rf

rf = load_rf("models/rf.pkl")
pred_rf = rf.predict(sample.image)
score_rf = score_segmentation(sample.labels, pred_rf)

fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
show_image(ax[0], sample.image); ax[0].set_title("image")
show_labels(ax[1], sample.labels); ax[1].set_title("ground truth")
show_labels(ax[2], pred_rf); ax[2].set_title("random forest")
plt.tight_layout(); plt.show()

print(f"pixel accuracy {score_rf.pixel_accuracy:.3f}   mean IoU {score_rf.mean_iou:.3f}")
print("per-class IoU:", {n: (None if v is None else round(v, 3))
                          for n, v in score_rf.as_dict()["iou"].items()})
"""),
    md("""
## 5. U-Net semantic segmenter

A compact multi-class U-Net (about 0.5 M parameters) trained with a
class-weighted cross-entropy plus soft Dice. We load the committed weights.
"""),
    code("""
import torch
from stemseg import SegUNet, predict_labels

model = SegUNet()
model.load_state_dict(torch.load("models/unet.pt", map_location="cpu", weights_only=True))

pred_unet = predict_labels(model, sample.image)
score_unet = score_segmentation(sample.labels, pred_unet)

fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
show_image(ax[0], sample.image); ax[0].set_title("image")
show_labels(ax[1], sample.labels); ax[1].set_title("ground truth")
show_labels(ax[2], pred_unet); ax[2].set_title("U-Net")
plt.tight_layout(); plt.show()

print(f"pixel accuracy {score_unet.pixel_accuracy:.3f}   mean IoU {score_unet.mean_iou:.3f}")
print("per-class IoU:", {n: (None if v is None else round(v, 3))
                          for n, v in score_unet.as_dict()["iou"].items()})
"""),
    md("""
## 6. Per-class IoU, side by side

Now the point lands. Read the background bar and every method looks similar.
Read the rare-class bars and the methods separate.
"""),
    code("""
methods = {"threshold": score_thr, "random forest": score_rf, "U-Net": score_unet}
x = np.arange(len(CLASS_NAMES))
width = 0.26
fig, ax = plt.subplots(figsize=(10, 4.2))
for k, (name, sc) in enumerate(methods.items()):
    vals = [0.0 if np.isnan(v) else v for v in sc.iou]
    ax.bar(x + (k - 1) * width, vals, width, label=name)
ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES)
ax.set_ylabel("IoU"); ax.set_ylim(0, 1)
ax.set_title("per-class IoU by method")
ax.legend(); plt.tight_layout(); plt.show()
"""),
    md("""
## 7. Boundary-localisation error

For the amorphous / grain-boundary region, "roughly right" is not enough: we
want the boundary in the right place. The symmetric boundary distance measures
exactly that, in pixels.
"""),
    code("""
for name, sc in methods.items():
    print(f"{name:14s} disordered IoU "
          f"{0.0 if np.isnan(sc.iou[4]) else sc.iou[4]:.3f}   "
          f"boundary error {sc.boundary_error_px:.2f} px")
"""),
    md("""
## 8. The final metric: rare-class IoU versus dose

The clearest summary of the whole project. As dose drops (more shot noise), the
rare classes get harder. We sweep dose and plot the mean IoU over the three
rare classes for each method. This is the curve the README leads with.
"""),
    code("""
from stemseg.metrics import pool_scores

def rare_iou(sc):
    vals = [sc.iou[i] for i in (2, 3, 4) if not np.isnan(sc.iou[i])]
    return float(np.mean(vals)) if vals else np.nan

doses = [15, 40, 100, 300]
curves = {"threshold": [], "random forest": [], "U-Net": []}
for dose in doses:
    imgs = [simulate_image(material_config("graphene", size=192, dose=dose,
             vacancy_fraction=0.04, dopant_fraction=0.04, disorder_fraction=0.12,
             rotation_deg=15.0), np.random.default_rng(900 + j)) for j in range(4)]
    curves["threshold"].append(rare_iou(pool_scores(
        [score_segmentation(s.labels, threshold_morphology(s.image)) for s in imgs])))
    curves["random forest"].append(rare_iou(pool_scores(
        [score_segmentation(s.labels, rf.predict(s.image)) for s in imgs])))
    curves["U-Net"].append(rare_iou(pool_scores(
        [score_segmentation(s.labels, predict_labels(model, s.image)) for s in imgs])))

fig, ax = plt.subplots(figsize=(7.5, 4.4))
for name, ys in curves.items():
    ax.plot(doses, ys, "-o", label=name)
ax.set_xscale("log"); ax.set_xlabel("dose (counts/peak)")
ax.set_ylabel("mean IoU over rare classes")
ax.set_title("where learning earns its keep")
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()

for name, ys in curves.items():
    print(name, [round(y, 3) for y in ys])
"""),
    md("""
## Takeaways

- Pixel accuracy is dominated by background and hides the rare-class behaviour
  that actually matters for defect analysis.
- The classical baselines segment the ordered lattice adequately but struggle
  on the rare classes, and the vacancy class (defined by *absence* of signal in
  an ordered context) is the hardest for a local classifier.
- The U-Net, with spatial context, is where the rare classes are recovered,
  and the gap widens as dose falls.

All of this is measured against exact synthetic ground truth. On real data,
expect a domain gap; see the model card. The full, seeded benchmarks live in
`configs/` and their numbers in `RESULTS.md`.
"""),
]


def main() -> None:
    nb = new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    client = NotebookClient(nb, timeout=1200, resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    out = ROOT / "notebooks" / "tutorial.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out)
    print(f"wrote executed {out}")


if __name__ == "__main__":
    main()
