"""Classical (non-deep) segmentation baselines.

Two baselines stand in for the two things a microscopist reaches for before
training a network:

- ``threshold_morphology``: a hand-built pipeline. Intensity thresholding
  finds bright columns (lattice), a brighter threshold flags dopant columns,
  a local-variance map flags the amorphous disordered region, and a
  morphological "hole in an otherwise ordered neighbourhood" test flags
  vacancy sites. It has knobs but no learned parameters.

- ``RandomForestPixelClassifier``: a random forest over the local feature
  bank in ``features.py``, trained on simulated pixels. This is the strong
  classical baseline and the fair comparison for the U-Net: same data, same
  supervision, only the model differs.

Both expose ``predict(image) -> label_map`` so the benchmark can treat them
and the U-Net through one interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, grey_closing, uniform_filter

from stemseg.features import design_matrix
from stemseg.net import normalize_image
from stemseg.sim import BACKGROUND, DISORDERED, DOPANT, LATTICE, NUM_CLASSES, VACANCY


@dataclass
class ThresholdParams:
    """Knobs for the threshold + morphology baseline.

    Attributes:
        smooth_sigma: Pre-smoothing applied before thresholding.
        lattice_pct: Intensity percentile above which a pixel is a column.
        dopant_pct: Percentile above which a bright column is called a dopant.
        texture_window: Window (px) for the local standard-deviation texture.
        disorder_pct: Local-std percentile above which a pixel is disordered.
        vacancy_gap: Closing size (px) for the "expected but missing" test.
        vacancy_strength: How much dimmer than the local closing a pixel must
            be to count as a vacancy site.
    """

    smooth_sigma: float = 1.0
    lattice_pct: float = 70.0
    dopant_pct: float = 98.0
    texture_window: int = 9
    disorder_pct: float = 88.0
    vacancy_gap: int = 7
    vacancy_strength: float = 0.35


def threshold_morphology(image: np.ndarray, params: ThresholdParams | None = None) -> np.ndarray:
    """Segment an image with the hand-built classical pipeline.

    Args:
        image: 2D STEM image.
        params: Pipeline knobs; defaults used if None.

    Returns:
        Int (H, W) label map with the five classes.
    """
    params = params or ThresholdParams()
    norm = normalize_image(image)
    smooth = gaussian_filter(norm, sigma=params.smooth_sigma)

    labels = np.full(norm.shape, BACKGROUND, dtype=np.int64)

    lattice_thresh = np.percentile(smooth, params.lattice_pct)
    is_column = smooth >= lattice_thresh
    labels[is_column] = LATTICE

    dopant_thresh = np.percentile(smooth, params.dopant_pct)
    labels[smooth >= dopant_thresh] = DOPANT

    # Texture: high local standard deviation marks the amorphous region.
    mean = uniform_filter(smooth, size=params.texture_window)
    sq = uniform_filter(smooth * smooth, size=params.texture_window)
    local_std = np.sqrt(np.clip(sq - mean * mean, 0.0, None))
    texture = gaussian_filter(local_std, sigma=2.0)
    disorder_thresh = np.percentile(texture, params.disorder_pct)
    labels[texture >= disorder_thresh] = DISORDERED

    # Vacancy: a dark spot sitting where a closing (fill of dark gaps between
    # bright columns) says a column should be, i.e. an ordered hole.
    closed = grey_closing(smooth, size=params.vacancy_gap)
    gap = closed - smooth
    expected = closed >= lattice_thresh
    is_vacancy = expected & (gap >= params.vacancy_strength) & (labels == BACKGROUND)
    labels[is_vacancy] = VACANCY

    return labels


class RandomForestPixelClassifier:
    """A random forest over local features, per pixel.

    The estimator is scikit-learn's RandomForestClassifier. Kept behind this
    thin wrapper so the benchmark can save/load it and call ``predict(image)``
    without touching sklearn directly.

    Args:
        n_estimators: Number of trees.
        max_depth: Maximum tree depth (None for full growth).
        min_samples_leaf: Minimum samples per leaf.
        class_weight: Passed to the forest; "balanced" up-weights rare classes
            and is the fair setting for this imbalanced problem.
        max_features: Feature subsampling per split.
        seed: RNG seed.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int | None = 12,
        min_samples_leaf: int = 10,
        class_weight: str | None = "balanced_subsample",
        max_features: str = "sqrt",
        seed: int = 0,
    ):
        from sklearn.ensemble import RandomForestClassifier

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            max_features=max_features,
            n_jobs=-1,
            random_state=seed,
        )
        self.classes_: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> RandomForestPixelClassifier:
        """Fit the forest on a (P, F) feature matrix and (P,) labels."""
        self.model.fit(features, labels)
        self.classes_ = self.model.classes_
        return self

    def predict_proba_image(self, image: np.ndarray) -> np.ndarray:
        """Return per-pixel class probabilities, shape (H, W, NUM_CLASSES).

        Classes never seen in training get a zero column so the output width
        is always NUM_CLASSES.
        """
        h, w = image.shape
        proba = self.model.predict_proba(design_matrix(image))
        full = np.zeros((h * w, NUM_CLASSES), dtype=np.float32)
        for j, c in enumerate(self.classes_):
            full[:, int(c)] = proba[:, j]
        return full.reshape(h, w, NUM_CLASSES)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Return the argmax label map for one image."""
        return self.predict_proba_image(image).argmax(-1).astype(np.int64)


def majority_smooth(labels: np.ndarray) -> np.ndarray:
    """A light 3x3 majority vote that removes isolated single-pixel flips.

    An optional label-map post-processor, used by no method in the benchmark.
    It cleans speckle but does not recover a class a method missed, so it does
    not close the rare-class gap; kept as a small utility for experimentation.
    """
    from scipy.stats import mode

    stack = np.stack(
        [np.roll(np.roll(labels, dr, 0), dc, 1) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
    )
    return mode(stack, axis=0, keepdims=False).mode.astype(np.int64)
