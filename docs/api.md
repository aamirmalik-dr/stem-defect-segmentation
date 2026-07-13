# Python API

`stemseg` is usable as a library, not only through the CLI. Everything below
runs on CPU. Import the top-level names from `stemseg`; internal modules
(`stemseg.sim`, `stemseg.metrics`, ...) are stable too.

## Simulate an image with ground truth

```python
import numpy as np
from stemseg import material_config, simulate_image

cfg = material_config("mos2", dose=200.0, disorder_fraction=0.15)
sample = simulate_image(cfg, np.random.default_rng(0))

sample.image        # float32 (H, W), the noisy STEM image
sample.labels       # int8 (H, W), classes 0..4
sample.disorder_mask  # bool (H, W), the amorphous-region footprint
sample.columns      # (N, 2) drawn column centres as (row, col)
sample.vacancy_sites  # (M, 2) removed-column sites
```

The five classes are `stemseg.CLASS_NAMES`:

```python
from stemseg import CLASS_NAMES
CLASS_NAMES  # ('background', 'lattice', 'vacancy', 'dopant', 'disordered')
```

`SegConfig` exposes every physical knob (dose, `vacancy_fraction`,
`dopant_fraction`, `disorder_fraction`, `probe_sigma`, `spacing`, ...). Use
`material_config(name, **overrides)` for sensible per-material defaults, or
build a `SegConfig` directly.

## Segment with a classical baseline

```python
from stemseg import threshold_morphology

pred = threshold_morphology(sample.image)   # int (H, W) label map
```

The random forest is a fitted object:

```python
from stemseg import RandomForestPixelClassifier
from stemseg.train import TrainSettings, sample_pixels

X, y = sample_pixels(TrainSettings(), n_images=30, per_image=1200)
rf = RandomForestPixelClassifier().fit(X, y)
pred_rf = rf.predict(sample.image)
proba = rf.predict_proba_image(sample.image)   # (H, W, 5)
```

## Segment with the U-Net

```python
import torch
from stemseg import SegUNet, predict_labels, predict_proba

model = SegUNet()
model.load_state_dict(torch.load("models/unet.pt", map_location="cpu", weights_only=True))

pred = predict_labels(model, sample.image)     # (H, W) argmax labels
proba = predict_proba(model, sample.image)     # (H, W, 5) softmax
```

Train a fresh one (a few minutes on CPU):

```python
from stemseg.train import TrainSettings, train_unet
model, loss_history = train_unet(TrainSettings(steps=2500, seed=0))
```

## Score a prediction

```python
from stemseg import score_segmentation

score = score_segmentation(sample.labels, pred)
score.pixel_accuracy      # overall fraction correct (inflated by imbalance!)
score.iou                 # (5,) per-class IoU, NaN where a class is absent
score.dice                # (5,) per-class Dice
score.mean_iou            # mean over present classes
score.boundary_error_px   # mean symmetric boundary distance of the
                          # disordered region, in pixels
score.as_dict()           # JSON-friendly, keyed by class name
```

Pool scores over several images the honest way (sum confusions, then form the
ratios) with `pool_scores`:

```python
from stemseg import pool_scores
pooled = pool_scores([score_segmentation(s.labels, p) for s, p in pairs])
```

## Why per-class, not just accuracy

Because background is roughly 70% of every frame, a model that predicts
"background everywhere" already scores about 0.70 pixel accuracy while getting
every defect wrong. `per_class_iou` and `per_class_dice` return NaN (not 0)
for absent classes, so a mean over them never launders a missed rare class
into a good-looking number. This is the whole reason the benchmark leads with
per-class IoU.

## Extending it

- **New material**: add a `Material` (cell vectors plus a `Sublattice` basis)
  to `stemseg.sim.MATERIALS` and a default entry in `material_config`.
- **New method**: expose any `predict(image) -> label_map` callable and add it
  to `stemseg.benchmark.build_methods`; the benchmark harness will score it.
- **New metric**: add it to `stemseg.metrics` and surface it in `SegScore`.
