"""Compact multi-class U-Net for STEM defect segmentation.

The network is small on purpose (three encoder stages, 16 base channels,
about 0.5 M parameters) so it trains in a few minutes on a CPU and the
checkpoint stays under a megabyte. It outputs one logit map per class and is
trained with a class-weighted cross-entropy plus a soft-Dice term, because
the rare classes (vacancy, dopant, disordered) would otherwise be drowned
out by the background.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from stemseg.sim import NUM_CLASSES


def _double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SegUNet(nn.Module):
    """Three-level U-Net returning per-pixel class logits.

    Args:
        num_classes: Number of output classes.
        base: Channel count at the top level; deeper levels double it.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, base: int = 16):
        super().__init__()
        self.enc1 = _double_conv(1, base)
        self.enc2 = _double_conv(base, base * 2)
        self.enc3 = _double_conv(base * 2, base * 4)
        self.bottom = _double_conv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = _double_conv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = _double_conv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = _double_conv(base * 2, base)
        self.head = nn.Conv2d(base, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, num_classes, H, W) logits for a (B, 1, H, W) input."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottom(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Standardise an image to zero mean and unit variance."""
    image = image.astype(np.float32)
    std = image.std()
    return (image - image.mean()) / (std if std > 0 else 1.0)


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Macro soft-Dice loss over classes, averaged equally.

    Averaging Dice equally over classes (rather than weighting by pixel count)
    is what pushes the network to care about the rare classes.

    Args:
        logits: (B, C, H, W) raw class logits.
        target: (B, H, W) integer labels.
        eps: Smoothing constant.

    Returns:
        Scalar loss.
    """
    num_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probs * onehot).sum(dims)
    cardinality = probs.sum(dims) + onehot.sum(dims)
    dice = (2 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


def predict_labels(model: nn.Module, image: np.ndarray) -> np.ndarray:
    """Run the network on one image and return the argmax label map.

    Args:
        model: A trained SegUNet.
        image: 2D image whose sides are multiples of 8 (three poolings).

    Returns:
        Int (H, W) predicted label map.
    """
    model.eval()
    x = torch.from_numpy(normalize_image(image))[None, None]
    with torch.no_grad():
        logits = model(x)[0]
    return logits.argmax(0).numpy().astype(np.int64)


def predict_proba(model: nn.Module, image: np.ndarray) -> np.ndarray:
    """Return the per-pixel softmax probabilities, shape (H, W, C)."""
    model.eval()
    x = torch.from_numpy(normalize_image(image))[None, None]
    with torch.no_grad():
        probs = F.softmax(model(x)[0], dim=0)
    return probs.permute(1, 2, 0).numpy().astype(np.float32)
