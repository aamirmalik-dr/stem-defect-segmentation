"""Local feature bank for the random-forest pixel classifier.

A per-pixel classifier can only be as good as the features it sees. This
bank is a compact, multi-scale filter set that gives a random forest enough
context to separate the five classes: raw intensity for the dopant/lattice
brightness difference, difference-of-Gaussians and Laplacian for blob
structure, gradient magnitude for edges, and local mean / standard deviation
at two scales to capture the "is this a clean lattice or an amorphous mess"
texture that distinguishes the disordered class.

``feature_stack`` returns an (H, W, F) array so it can be reshaped to a
per-pixel design matrix. The feature names are exposed for interpretability
and for the model card.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude, uniform_filter

# Scales in pixels, chosen to bracket the probe width and the column spacing.
SMALL_SCALES = (1.0, 2.0, 4.0)
TEXTURE_WINDOWS = (5, 11)


def _standardize(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    std = image.std()
    return (image - image.mean()) / (std if std > 0 else 1.0)


def feature_names() -> list[str]:
    """Return the ordered list of feature names produced by ``feature_stack``."""
    names = ["intensity"]
    for s in SMALL_SCALES:
        names.append(f"gauss_s{s:g}")
    for s in SMALL_SCALES:
        names.append(f"laplace_s{s:g}")
    for s in SMALL_SCALES[:-1]:
        names.append(f"dog_s{s:g}")
    for s in SMALL_SCALES:
        names.append(f"grad_s{s:g}")
    for w in TEXTURE_WINDOWS:
        names.append(f"localmean_w{w}")
        names.append(f"localstd_w{w}")
    return names


def feature_stack(image: np.ndarray) -> np.ndarray:
    """Compute the (H, W, F) feature stack for one image.

    Args:
        image: 2D image. It is standardised internally so the features do not
            depend on absolute intensity scale.

    Returns:
        Float32 array of shape (H, W, F) with F == len(feature_names()).
    """
    img = _standardize(image)
    channels: list[np.ndarray] = [img]

    smoothed = {s: gaussian_filter(img, sigma=s) for s in SMALL_SCALES}
    for s in SMALL_SCALES:
        channels.append(smoothed[s])
    for s in SMALL_SCALES:
        # Laplacian via second derivative sum, at scale s.
        lap = gaussian_filter(img, sigma=s, order=(2, 0)) + gaussian_filter(
            img, sigma=s, order=(0, 2)
        )
        channels.append(lap)
    for a, b in zip(SMALL_SCALES[:-1], SMALL_SCALES[1:]):
        channels.append(smoothed[a] - smoothed[b])
    for s in SMALL_SCALES:
        channels.append(gaussian_gradient_magnitude(img, sigma=s))
    for w in TEXTURE_WINDOWS:
        mean = uniform_filter(img, size=w)
        sq = uniform_filter(img * img, size=w)
        var = np.clip(sq - mean * mean, 0.0, None)
        channels.append(mean)
        channels.append(np.sqrt(var))

    return np.stack(channels, axis=-1).astype(np.float32)


def design_matrix(image: np.ndarray) -> np.ndarray:
    """Return the (H*W, F) per-pixel design matrix for one image."""
    stack = feature_stack(image)
    h, w, f = stack.shape
    return stack.reshape(h * w, f)
