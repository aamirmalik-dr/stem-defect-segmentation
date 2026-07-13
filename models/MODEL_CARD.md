# Model card: stemseg segmenters

Two models ship with this repository. Both consume a single-channel STEM image
and produce a five-class per-pixel segmentation (background, lattice, vacancy,
dopant, disordered). Both were trained only on synthetic data from
`stemseg.sim` and have never seen a real micrograph.

## Models

### U-Net (`models/unet.pt`)

- **Architecture**: three-level U-Net, 16 base channels, ~0.48 M parameters,
  softmax over five classes. Defined in `stemseg.net.SegUNet`.
- **Loss**: class-weighted cross-entropy (inverse-frequency weights, clipped)
  plus an equally-weighted macro soft-Dice term. The Dice term and the class
  weights are what keep the rare classes from being ignored.
- **Optimiser**: Adam, initial lr 1.5e-3, cosine annealing to 2% of that over
  the run.
- **Training data**: generated on the fly with full domain randomisation over
  material (graphene, hBN, MoS2, oxide), lattice spacing, rotation, probe
  width, dose (log-uniform, 5 to 1000 counts/peak), vacancy and dopant
  fractions, disordered-region area, and background level. One freshly
  simulated batch per step, so no image is ever repeated.
- **Checkpoint size**: ~1.9 MB.

### Random forest (`models/rf.pkl`)

- **Model**: scikit-learn RandomForestClassifier over a 15-channel local
  feature bank (multi-scale Gaussians, Laplacians, difference-of-Gaussians,
  gradient magnitude, and local mean/standard-deviation texture). Defined in
  `stemseg.classical.RandomForestPixelClassifier` and `stemseg.features`.
- **Settings**: 50 trees, max depth 12, min_samples_leaf 10,
  `class_weight="balanced_subsample"`, fit on natural-distribution pixels from
  the same simulator. This pairing was chosen because it gave the best mean IoU
  during tuning; a doubly-balanced variant (balanced sampling *and* balanced
  weights) over-predicts the rare classes and collapses pixel accuracy. The tree
  depth and count were then trimmed to keep the pickle small with negligible
  accuracy loss.
- **Checkpoint size**: ~5.0 MB.
- **Purpose**: the strong classical baseline, fair by construction because it
  learns from the same data as the U-Net. Only the model class differs.

## Intended use

Benchmarking and teaching: comparing classical and learned segmenters on a
controlled, physics-motivated defect-segmentation task with exact ground
truth. Not intended for quantitative analysis of real experimental images.

## Measured performance

Measured this session against synthetic ground truth; the numbers regenerate
from the seeded configs in `configs/`. Full tables in [RESULTS.md](../RESULTS.md).

- **Overall (graphene, dose 100)**: U-Net mean IoU 0.853 and pixel accuracy
  0.957, versus random forest 0.531 / 0.841 and threshold 0.293 / 0.735. The
  trivial "all background" predictor scores 0.703 pixel accuracy but 0.141 mean
  IoU, which is why per-class IoU is the reported metric.
- **Vacancy class**: the U-Net holds vacancy IoU 0.57 to 0.70 across doses and
  0.45 to 0.58 across the four materials, where the random forest stays near
  zero (0.005 to 0.10) and the threshold baseline gets 0.000. Vacancy detection
  needs spatial context a local classifier does not have.
- **Fair-tuning check**: with the threshold baseline oracle-tuned per condition,
  its rare-class IoU still only reaches 0.02 to 0.04, versus the random forest's
  0.34 to 0.36 and the U-Net's 0.79 to 0.82. The U-Net's advantage is real, not
  a tuning artifact.
- **Boundary localisation** (disordered region): on the four committed samples
  the U-Net's symmetric boundary error is 1.4, 15.7, 1.7 and 2.3 px for
  graphene, hBN, MoS2 and oxide. The hBN outlier is a genuine failure, not a
  small region: the model hallucinates a second small disordered patch away from
  the true one, and a stray patch inflates the symmetric boundary distance
  sharply. In the controlled imbalance sweep the error is 15 px when the region
  is a couple of percent of the frame and falls to about 3 px once the region is
  a tenth of the frame or more.

## Limitations and failure modes

- **Domain gap**: trained purely on incoherent Z-contrast with a Gaussian
  probe. Real instruments add aberrations, a detector transfer function,
  dynamical scattering and scan distortions the simulator omits. Expect
  degraded, qualitative-only behaviour on real data.
- **Vacancy class is the hardest**: a vacancy is defined by the *absence* of a
  column in an otherwise ordered neighbourhood. A local classifier has almost
  no signal for it; even the U-Net, with spatial context, recovers it only
  partially. Treat vacancy predictions with caution.
- **Faint sublattices**: pure-oxygen and light columns carry little contrast,
  so they are recovered less reliably than heavy columns.
- **Easy dopant class**: with the default `dopant_z`, a dopant is a heavy
  substitutional atom many times brighter than the host lattice, so the dopant
  class is the easiest of the three rare classes here. A subtler, iso-electronic
  dopant (set `dopant_z` closer to the host) would be genuinely hard; the
  committed numbers reflect the heavy-dopant setting.
- **Boundary precision**: the disordered-region boundary is localised to a few
  pixels at best; the reported boundary error quantifies this.
- **Fixed pixel scale**: both models assume a column width close to the
  training probe size. On real data, downsample to match (see
  `data/real/README.md`).

## Reproducing

```
stemseg train --model both --steps 2500     # retrain U-Net and refit RF
stemseg train --ablation                     # domain-randomisation ablation
```

Training is CPU-only and takes a few minutes. Seeds are fixed, so a rerun
reproduces the committed checkpoints up to platform-level floating-point
differences.
