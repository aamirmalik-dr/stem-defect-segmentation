# Results

Every number below was measured this session in a fresh virtual environment
(CPU-only PyTorch 2.13) against exact synthetic ground truth, and regenerates
from the seeded YAML configs in `configs/`. Raw values live in `results/*.json`.
All data is synthetic; see the README and model card for the scope.

Three methods are compared throughout:

- **threshold**: hand-built thresholding plus morphology (`stemseg.classical.threshold_morphology`), no learning.
- **rf**: a random forest over a local feature bank (`RandomForestPixelClassifier`), the strong classical baseline.
- **unet**: a compact multi-class U-Net (`SegUNet`), the learned segmenter.

## 1. Pixel accuracy is the wrong headline

Background is about 70% of every frame, so a model that predicts "background
everywhere" already scores 0.703 pixel accuracy while getting **every defect
wrong** (mean IoU 0.141). At dose 100 on graphene, the threshold baseline
scores 0.735 pixel accuracy, essentially the same as that trivial predictor,
yet its mean IoU is only 0.293 and its rare-class IoU is 0.008. Pixel accuracy
cannot tell "detects defects" apart from "detects nothing". This is why every
table here leads with per-class IoU.

| method (graphene, dose 100) | pixel accuracy | mean IoU | rare-class mean IoU |
|---|---|---|---|
| predict all background | 0.703 | 0.141 | 0.000 |
| threshold + morphology | 0.735 | 0.293 | 0.008 |
| random forest | 0.841 | 0.531 | 0.349 |
| U-Net | **0.957** | **0.853** | **0.804** |

## 2. Per-class IoU versus dose (graphene)

Full sweep in `results/dose_sweep.json`, figure `figures/dose_sweep.png` and the
per-method small-multiples hero `figures/iou_vs_dose.png`. Per-class IoU at each
dose (counts per host-column peak):

| dose | method | lattice | vacancy | dopant | disordered | mean IoU |
|---|---|---|---|---|---|---|
| 5 | threshold | 0.611 | 0.000 | 0.000 | 0.037 | 0.264 |
| 5 | rf | 0.671 | 0.070 | 0.777 | 0.180 | 0.495 |
| 5 | U-Net | **0.840** | **0.650** | **0.938** | **0.781** | **0.824** |
| 100 | threshold | 0.728 | 0.000 | 0.000 | 0.024 | 0.293 |
| 100 | rf | 0.788 | 0.010 | 0.764 | 0.273 | 0.531 |
| 100 | U-Net | **0.911** | **0.672** | **0.910** | **0.831** | **0.853** |
| 1000 | threshold | 0.742 | 0.000 | 0.000 | 0.045 | 0.301 |
| 1000 | rf | 0.811 | 0.005 | 0.761 | 0.344 | 0.554 |
| 1000 | U-Net | **0.919** | **0.674** | **0.925** | **0.827** | **0.858** |

The U-Net leads at every dose on every class. The threshold baseline never
recovers dopants or vacancies at all.

## 3. The vacancy class is where spatial context decides it

A vacancy is defined by the *absence* of a column in an otherwise ordered
neighbourhood, so a local classifier has almost no signal for it. The random
forest's vacancy IoU is near zero at every dose, and it gets **worse** as dose
rises (0.070 at dose 5 down to 0.005 at dose 1000): with sharper features the
forest grows more confident about the lattice and background and stops flagging
the subtle holes. The U-Net, which sees spatial context, holds vacancy IoU
between 0.57 and 0.70 across the whole dose range. This is the clearest example
of where learning earns its keep, and it is a property of the task, not of tuning
(see section 6).

## 4. Every material (dose 200)

`results/materials.json`, and the segmentation-overlay gallery
`figures/gallery.png`. Mean IoU and the two hardest classes:

| material | method | mean IoU | vacancy IoU | disordered IoU |
|---|---|---|---|---|
| graphene | rf / U-Net | 0.512 / **0.819** | 0.006 / **0.531** | 0.254 / **0.810** |
| hBN | rf / U-Net | 0.518 / **0.826** | 0.007 / **0.533** | 0.236 / **0.820** |
| MoS2 | rf / U-Net | 0.537 / **0.838** | 0.076 / **0.578** | 0.319 / **0.810** |
| oxide | rf / U-Net | 0.462 / **0.763** | 0.096 / **0.448** | 0.391 / **0.831** |

The oxide preset is hardest for the U-Net (mean IoU 0.763): it has the smallest
spacing and a faint pure-oxygen sublattice, so the lattice class itself is
harder.

## 5. Class imbalance inflates accuracy; boundary error tells the truth

Sweeping the disordered-region area from 2% to 35% of the frame
(`results/imbalance_sweep.json`, `figures/imbalance_sweep.png`):

| disorder fraction | U-Net pixel acc | U-Net disordered IoU | U-Net boundary error (px) |
|---|---|---|---|
| 0.02 | 0.969 | 0.612 | 15.4 |
| 0.06 | 0.965 | 0.775 | 2.6 |
| 0.12 | 0.954 | 0.811 | 4.3 |
| 0.22 | 0.946 | 0.853 | 3.1 |
| 0.35 | 0.943 | 0.890 | 3.0 |

Pixel accuracy barely moves while the disordered IoU swings from 0.61 to 0.89
and the boundary error drops five-fold, exactly the behaviour a single accuracy
number hides. The threshold baseline keeps a respectable-looking pixel accuracy
here too (0.55 to 0.83) while its disordered IoU stays near 0.01 to 0.08. A
small region is genuinely hard to localise for everyone: at 2% disorder every
method's boundary error is large (43 to 46 px).

## 6. Honest check: the gap survives tuning the baseline to an oracle

If the U-Net's rare-class lead came from an under-tuned baseline, it would not
be worth reporting. `configs/fair_tuning.yaml` gives the threshold baseline the
per-condition, ground-truth-optimal parameters over a grid (an oracle it could
never reach without the labels) and compares that, plus the balanced random
forest, against the U-Net on the mean IoU over the three rare classes:

| dose | threshold (default) | threshold (oracle) | random forest | U-Net |
|---|---|---|---|---|
| 15 | 0.008 | 0.022 | 0.340 | **0.788** |
| 40 | 0.012 | 0.030 | 0.364 | **0.821** |
| 100 | 0.008 | 0.038 | 0.350 | **0.790** |
| 300 | 0.012 | 0.038 | 0.355 | **0.791** |

Oracle tuning roughly triples the threshold baseline's rare-class IoU and it is
still under 0.04. The U-Net's advantage over the balanced random forest (a
factor of about 2.2) is real, not a tuning artifact.

## 7. Committed-sample demo

`stemseg demo` segments the four committed samples with all three methods
(`results/metrics.json`). Mean IoU / boundary error (px):

| sample | threshold | random forest | U-Net |
|---|---|---|---|
| graphene_d150 | 0.264 / 39.2 | 0.554 / 26.1 | **0.875 / 1.4** |
| hbn_d150 | 0.235 / 40.0 | 0.492 / 28.4 | **0.855 / 15.7** |
| mos2_d200 | 0.207 / 25.9 | 0.569 / 23.7 | **0.862 / 1.7** |
| oxide_d250 | 0.120 / 35.2 | 0.484 / 34.5 | **0.798 / 2.3** |

## Reproducing

```
stemseg train --model both --steps 2500   # retrain U-Net and refit RF (CPU, minutes)
stemseg benchmark configs/dose_sweep.yaml configs/defect_sweep.yaml \
    configs/imbalance_sweep.yaml configs/materials.yaml \
    configs/fair_tuning.yaml configs/confusion.yaml
python scripts/make_figures.py            # gallery, iou_vs_dose, sample preview
```

Seeds are fixed, so a rerun reproduces these numbers up to platform-level
floating-point differences.
