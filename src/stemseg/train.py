"""Training for the learned segmenter and the random-forest baseline.

Both models learn from the same simulator, so the comparison is fair: the
only difference is the model, not the data. Training images are generated on
the fly with randomised material, spacing, rotation, dose, defect rates and
disorder, so neither model can memorise a single scene.

The U-Net is trained with a class-weighted cross-entropy plus a soft-Dice
term. The random forest is fit on a class-balanced sample of pixels drawn
from the same distribution of images, using the local feature bank.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from stemseg.features import feature_stack
from stemseg.net import SegUNet, normalize_image, soft_dice_loss
from stemseg.sim import NUM_CLASSES, SegConfig, material_config, simulate_image

TRAIN_MATERIALS = ("graphene", "hbn", "mos2", "oxide")


@dataclass
class TrainSettings:
    """Configuration for U-Net training.

    Attributes:
        steps: Optimisation steps, one freshly simulated batch each.
        batch_size: Images per batch.
        lr: Adam learning rate.
        seed: Seed for simulation and initialisation.
        size: Training patch side (multiple of 8).
        dice_weight: Weight of the soft-Dice term added to cross-entropy.
        randomize_dose: Draw dose log-uniform in [dose_low, dose_high]; if
            False, fix it at ``fixed_dose``.
        randomize_geometry: Randomise material, spacing, rotation and probe
            size; if False, use one fixed graphene lattice.
        include_defects: Randomise vacancy and dopant fractions; if False,
            simulate defect-free crystals (so the model never sees those
            classes, which the ablation uses).
        include_disorder: Randomise the disordered-region area; if False,
            simulate fully ordered crystals.
        dose_low: Lower bound of the training dose range.
        dose_high: Upper bound of the training dose range.
        fixed_dose: Dose used when randomize_dose is False.
    """

    steps: int = 400
    batch_size: int = 6
    lr: float = 1.5e-3
    seed: int = 0
    size: int = 128
    dice_weight: float = 1.0
    randomize_dose: bool = True
    randomize_geometry: bool = True
    include_defects: bool = True
    include_disorder: bool = True
    dose_low: float = 5.0
    dose_high: float = 1000.0
    fixed_dose: float = 300.0


def random_config(
    rng: np.random.Generator, settings: TrainSettings, size: int | None = None
) -> SegConfig:
    """Draw one randomised simulation config for training."""
    size = size or settings.size
    if settings.randomize_geometry:
        material = str(rng.choice(TRAIN_MATERIALS))
        config = material_config(material, size=size)
        config.spacing *= float(rng.uniform(0.85, 1.15))
        config.probe_sigma = float(rng.uniform(2.0, 2.8))
        config.rotation_deg = None
    else:
        config = material_config("graphene", size=size, rotation_deg=15.0)

    if settings.include_defects:
        config.vacancy_fraction = float(rng.uniform(0.0, 0.06))
        config.dopant_fraction = float(rng.uniform(0.0, 0.06))
        config.displacement_sigma = float(rng.uniform(0.2, 0.4))
    else:
        config.vacancy_fraction = 0.0
        config.dopant_fraction = 0.0

    if settings.include_disorder:
        config.disorder_fraction = float(rng.uniform(0.0, 0.22))
    else:
        config.disorder_fraction = 0.0

    config.background = float(rng.uniform(0.04, 0.1))
    config.background_variation = float(rng.uniform(0.0, 0.04))
    config.dose = (
        float(np.exp(rng.uniform(np.log(settings.dose_low), np.log(settings.dose_high))))
        if settings.randomize_dose
        else settings.fixed_dose
    )
    return config


def make_batch(
    rng: np.random.Generator, settings: TrainSettings
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate one training batch of images and integer label maps."""
    images, labels = [], []
    for _ in range(settings.batch_size):
        result = simulate_image(random_config(rng, settings), rng)
        images.append(normalize_image(result.image))
        labels.append(result.labels.astype(np.int64))
    x = torch.from_numpy(np.stack(images))[:, None]
    y = torch.from_numpy(np.stack(labels))
    return x, y


def class_weights(
    rng: np.random.Generator, settings: TrainSettings, n_images: int = 12
) -> np.ndarray:
    """Estimate inverse-frequency class weights from a sample of images.

    The weights are the median class frequency divided by each class
    frequency, clipped so no class dominates the loss outright.
    """
    counts = np.ones(NUM_CLASSES, dtype=np.float64)
    for _ in range(n_images):
        result = simulate_image(random_config(rng, settings), rng)
        counts += np.bincount(result.labels.ravel(), minlength=NUM_CLASSES)
    freq = counts / counts.sum()
    weights = np.median(freq) / freq
    return np.clip(weights, 0.2, 30.0)


def train_unet(
    settings: TrainSettings | None = None, log_every: int = 50, **overrides
) -> tuple[SegUNet, list[float]]:
    """Train the segmentation U-Net on freshly simulated data.

    Args:
        settings: Training configuration; built from overrides if None.
        log_every: Print running loss every this many steps (0 = silent).
        **overrides: Convenience TrainSettings overrides.

    Returns:
        The trained model and the per-step loss history.
    """
    if settings is None:
        settings = TrainSettings(**overrides)

    torch.manual_seed(settings.seed)
    rng = np.random.default_rng(settings.seed)
    weights = torch.from_numpy(class_weights(rng, settings)).float()
    model = SegUNet()
    opt = torch.optim.Adam(model.parameters(), lr=settings.lr)
    # Cosine annealing to a small floor helps the rare classes settle in the
    # long tail of training instead of oscillating at the base rate.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=settings.steps, eta_min=settings.lr * 0.02
    )
    ce = nn.CrossEntropyLoss(weight=weights)

    model.train()
    history: list[float] = []
    for step in range(1, settings.steps + 1):
        x, y = make_batch(rng, settings)
        opt.zero_grad()
        logits = model(x)
        loss = ce(logits, y) + settings.dice_weight * soft_dice_loss(logits, y)
        loss.backward()
        opt.step()
        scheduler.step()
        history.append(float(loss.item()))
        if log_every and step % log_every == 0:
            recent = float(np.mean(history[-log_every:]))
            print(f"step {step:4d}/{settings.steps}  loss {recent:.4f}")
    return model, history


def sample_pixels(
    settings: TrainSettings,
    n_images: int = 30,
    per_image: int = 1200,
    seed: int = 1,
    balance: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a pixel-level training set for the random forest.

    Args:
        settings: Simulation settings (shared with the U-Net for fairness).
        n_images: Number of images to draw features from.
        per_image: Pixels sampled per image.
        seed: RNG seed.
        balance: If True, draw classes as evenly as availability allows. In
            practice this over-corrects when combined with the forest's own
            ``balanced_subsample`` weighting, so the default is natural
            sampling and the forest handles imbalance through its class
            weights (this pairing gave the best mean IoU when tuned).

    Returns:
        (X, y): feature matrix (P, F) and integer labels (P,).
    """
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for _ in range(n_images):
        result = simulate_image(random_config(rng, settings), rng)
        stack = feature_stack(result.image)
        labels = result.labels.ravel()
        feats = stack.reshape(-1, stack.shape[-1])
        if balance:
            idx = _balanced_indices(labels, per_image, rng)
        else:
            idx = rng.choice(len(labels), size=min(per_image, len(labels)), replace=False)
        xs.append(feats[idx])
        ys.append(labels[idx])
    return np.concatenate(xs), np.concatenate(ys)


def _balanced_indices(labels: np.ndarray, total: int, rng: np.random.Generator) -> np.ndarray:
    """Return indices sampling each present class as evenly as possible."""
    present = [c for c in range(NUM_CLASSES) if np.any(labels == c)]
    if not present:
        return rng.choice(len(labels), size=total, replace=False)
    per = max(total // len(present), 1)
    chosen = []
    for c in present:
        pool = np.flatnonzero(labels == c)
        take = min(per, len(pool))
        chosen.append(rng.choice(pool, size=take, replace=False))
    return np.concatenate(chosen)
