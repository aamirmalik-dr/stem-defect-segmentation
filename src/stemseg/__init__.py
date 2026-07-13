"""stemseg: pixel-level defect segmentation for simulated atomic-resolution STEM.

Public API, grouped by what you usually reach for:

Simulation and ground truth
    material_config, simulate_image, MATERIALS, CLASS_NAMES, SegConfig, SegResult

Methods
    threshold_morphology, RandomForestPixelClassifier   (classical baselines)
    SegUNet, predict_labels, predict_proba              (learned segmenter)

Scoring
    score_segmentation, pool_scores, per_class_iou, per_class_dice,
    symmetric_boundary_distance

Example:
    >>> import numpy as np
    >>> from stemseg import material_config, simulate_image, threshold_morphology
    >>> from stemseg import score_segmentation
    >>> cfg = material_config("graphene", dose=200.0)
    >>> sample = simulate_image(cfg, np.random.default_rng(0))
    >>> pred = threshold_morphology(sample.image)
    >>> score = score_segmentation(sample.labels, pred)
    >>> round(score.pixel_accuracy, 2)  # doctest: +SKIP
    0.9
"""

from stemseg.classical import (
    RandomForestPixelClassifier,
    ThresholdParams,
    threshold_morphology,
)
from stemseg.metrics import (
    SegScore,
    per_class_dice,
    per_class_iou,
    pool_scores,
    score_segmentation,
    symmetric_boundary_distance,
)
from stemseg.net import SegUNet, predict_labels, predict_proba
from stemseg.sim import (
    CLASS_NAMES,
    MATERIALS,
    NUM_CLASSES,
    SegConfig,
    SegResult,
    material_config,
    simulate_image,
)

__all__ = [
    "material_config",
    "simulate_image",
    "MATERIALS",
    "CLASS_NAMES",
    "NUM_CLASSES",
    "SegConfig",
    "SegResult",
    "threshold_morphology",
    "ThresholdParams",
    "RandomForestPixelClassifier",
    "SegUNet",
    "predict_labels",
    "predict_proba",
    "score_segmentation",
    "pool_scores",
    "per_class_iou",
    "per_class_dice",
    "symmetric_boundary_distance",
    "SegScore",
]

__version__ = "0.1.0"
