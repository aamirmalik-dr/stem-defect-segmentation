"""Synthetic atomic-resolution STEM simulator with per-pixel labels.

This module is the ground-truth engine for the whole project. Unlike a
point-detection simulator, it emits a *dense label map*: every pixel is
assigned one of five semantic classes, so a segmenter can be trained and
scored exactly. The classes are

    0 background   vacuum / interstitial matrix between columns
    1 lattice      a well-ordered host atomic column
    2 vacancy      the neighbourhood of a lattice site whose column is gone
    3 dopant       the neighbourhood of a substitutional foreign column
    4 disordered   an amorphous or grain-boundary region with no clean lattice

The imaging model is deliberately compact but physically motivated:

- Annular dark-field contrast is incoherent and roughly Z**1.7 per atom,
  so a column's brightness is the sum of Z**1.7 over its atoms.
- The probe is a Gaussian point-spread function of width ``probe_sigma``.
- A vacancy removes a whole column but its *site* is still labelled, because
  a microscopist identifies a vacancy by the ordered hole it leaves, not by
  any positive signal. This is what makes the vacancy class hard: the model
  must recognise absence in context.
- A dopant substitutes a column of a different atomic number, so it appears
  brighter or fainter than its neighbours.
- A disordered region is a smooth spatial blob inside which columns are
  heavily jittered and randomly thinned, mimicking an amorphous inclusion or
  a grain boundary. Its whole footprint is one class.
- Shot noise is Poisson, set by one ``dose`` parameter (mean electron counts
  at the brightest host, i.e. non-dopant, column peak; a heavy dopant is
  brighter still). An optional Gaussian read term (``read_noise``, off by
  default) can be added on top.

The label map is built geometrically from the exact column list, so it never
depends on the noisy image. Everything is float ``(row, col)`` in pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.ndimage import gaussian_filter

# Semantic class codes. Kept as plain module constants so downstream code
# (metrics, plots, the RF classifier) can index arrays without an enum import.
BACKGROUND = 0
LATTICE = 1
VACANCY = 2
DOPANT = 3
DISORDERED = 4
CLASS_NAMES: tuple[str, ...] = ("background", "lattice", "vacancy", "dopant", "disordered")
NUM_CLASSES = len(CLASS_NAMES)

Z_EXPONENT = 1.7


def z_contrast(*atomic_numbers: int) -> float:
    """Return the incoherent dark-field brightness of a column of atoms."""
    return float(sum(z**Z_EXPONENT for z in atomic_numbers))


@dataclass(frozen=True)
class Sublattice:
    """One column site in the projected unit cell.

    Attributes:
        offset: Fractional (u, v) position inside the cell spanned by a1, a2.
        z: Atomic number of the (single-atom) host column on this sublattice.
        atoms: Number of atoms stacked in the projected column (>=1).
        label: Short species tag used in figures, e.g. "Mo" or "N".
    """

    offset: tuple[float, float]
    z: int
    atoms: int = 1
    label: str = ""


@dataclass(frozen=True)
class Material:
    """A 2D projected crystal: two cell vectors and a sublattice basis.

    Cell vectors are given in units of the lattice parameter and scaled by
    ``spacing`` at simulation time. Coordinates are (row, col).

    Attributes:
        name: Preset key.
        a1: First cell vector (row, col), in units of the lattice parameter.
        a2: Second cell vector (row, col), in units of the lattice parameter.
        basis: The column sites in one cell.
        note: One-line description of the projection.
    """

    name: str
    a1: tuple[float, float]
    a2: tuple[float, float]
    basis: tuple[Sublattice, ...]
    note: str


_H = float(np.sqrt(3.0) / 2.0)  # honeycomb / hexagonal row height

MATERIALS: dict[str, Material] = {
    "graphene": Material(
        name="graphene",
        a1=(0.0, 1.0),
        a2=(_H, 0.5),
        basis=(
            Sublattice((0.0, 0.0), 6, 1, "C"),
            Sublattice((1.0 / 3.0, 1.0 / 3.0), 6, 1, "C"),
        ),
        note="Graphene honeycomb; both columns are single carbon atoms.",
    ),
    "hbn": Material(
        name="hbn",
        a1=(0.0, 1.0),
        a2=(_H, 0.5),
        basis=(
            Sublattice((0.0, 0.0), 7, 1, "N"),
            Sublattice((1.0 / 3.0, 1.0 / 3.0), 5, 1, "B"),
        ),
        note="Hexagonal boron nitride; the boron sublattice is markedly fainter.",
    ),
    "mos2": Material(
        name="mos2",
        a1=(0.0, 1.0),
        a2=(_H, 0.5),
        basis=(
            Sublattice((0.0, 0.0), 42, 1, "Mo"),
            Sublattice((1.0 / 3.0, 1.0 / 3.0), 16, 2, "S2"),
        ),
        note="MoS2 monolayer, plan view; bright Mo alternates with a two-atom S column.",
    ),
    "oxide": Material(
        name="oxide",
        a1=(0.0, 1.0),
        a2=(1.0, 0.0),
        basis=(
            Sublattice((0.0, 0.0), 56, 1, "M"),
            Sublattice((0.5, 0.5), 22, 1, "T"),
            Sublattice((0.0, 0.5), 8, 1, "O"),
            Sublattice((0.5, 0.0), 8, 1, "O"),
        ),
        note="Square perovskite-like oxide; heavy corner, medium centre, faint O edges.",
    ),
}


@dataclass
class SegConfig:
    """Configuration for one simulated image and its label map.

    Attributes:
        size: Square image side length in pixels.
        material: Preset key from MATERIALS.
        spacing: Lattice parameter in pixels (sets the cell size).
        rotation_deg: Lattice rotation; random in [0, 60) if None.
        probe_sigma: Gaussian probe standard deviation in pixels.
        label_radius: Disk radius (px) painted around a column centre for the
            lattice / vacancy / dopant classes.
        vacancy_fraction: Fraction of ordered columns removed (site labelled).
        dopant_fraction: Fraction of ordered columns substituted by a dopant.
        dopant_z: Atomic number of the substitutional dopant column.
        displacement_sigma: Random static column displacement (px) in the
            ordered region.
        disorder_fraction: Target area fraction of the disordered region. Zero
            disables it.
        disorder_jitter: Column displacement (px) inside the disordered region.
        disorder_thinning: Fraction of columns dropped inside the disordered
            region (it still scatters, just incoherently).
        background: Constant pedestal as a fraction of the brightest host
            column peak.
        background_variation: Amplitude of a smooth low-frequency background.
        read_noise: Gaussian read-noise standard deviation, in dose units.
            Off by default (0.0); not exercised by the shipped training or
            benchmark configs.
        dose: Mean electron counts at the brightest host (non-dopant) column
            peak, which sets the shot noise. A heavy dopant is much brighter
            and receives proportionally more counts. Lower dose is noisier.
    """

    size: int = 192
    material: str = "graphene"
    spacing: float = 16.0
    rotation_deg: float | None = None
    probe_sigma: float = 2.4
    label_radius: float = 3.0
    vacancy_fraction: float = 0.03
    dopant_fraction: float = 0.03
    dopant_z: int = 78
    displacement_sigma: float = 0.3
    disorder_fraction: float = 0.12
    disorder_jitter: float = 3.0
    disorder_thinning: float = 0.35
    background: float = 0.06
    background_variation: float = 0.0
    read_noise: float = 0.0
    dose: float = 300.0


@dataclass
class SegResult:
    """A simulated image with its exact per-pixel ground truth.

    Attributes:
        image: Float32 image, normalised so the brightest clean column peaks
            near 1 before noise.
        labels: Int8 (H, W) label map with values in 0..NUM_CLASSES-1.
        disorder_mask: Bool (H, W) footprint of the disordered region.
        columns: (N, 2) float centres of the columns that were actually drawn.
        column_class: (N,) int class of each drawn column (LATTICE or DOPANT).
        vacancy_sites: (M, 2) float centres of removed columns.
        config: The configuration that produced this result.
    """

    image: np.ndarray
    labels: np.ndarray
    disorder_mask: np.ndarray
    columns: np.ndarray
    column_class: np.ndarray
    vacancy_sites: np.ndarray
    config: SegConfig = field(repr=False)


def material_config(name: str, **overrides) -> SegConfig:
    """Return a SegConfig tuned for a material preset.

    Spacing and probe size are chosen so nearest-neighbour columns sit a
    dozen-ish pixels apart at the default resolution. Any field can be
    overridden.

    Args:
        name: Preset key from MATERIALS.
        **overrides: SegConfig fields to override.

    Returns:
        A ready-to-use SegConfig.
    """
    presets: dict[str, dict] = {
        "graphene": {"spacing": 18.0, "label_radius": 3.0},
        "hbn": {"spacing": 18.0, "label_radius": 3.0},
        "mos2": {"spacing": 19.0, "label_radius": 3.2},
        "oxide": {"spacing": 15.0, "label_radius": 2.6},
    }
    if name not in presets:
        raise ValueError(f"unknown material preset: {name!r}")
    config = SegConfig(material=name, **presets[name])
    return replace(config, **overrides) if overrides else config


def _cell_vectors(config: SegConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return the two rotated cell vectors in pixels."""
    material = MATERIALS[config.material]
    a1 = np.array(material.a1) * config.spacing
    a2 = np.array(material.a2) * config.spacing
    theta = (
        np.deg2rad(config.rotation_deg)
        if config.rotation_deg is not None
        else rng.uniform(0.0, np.pi / 3.0)
    )
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return rot @ a1, rot @ a2


def _tile_columns(
    config: SegConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tile the material over the frame.

    Returns:
        centres: (N, 2) column centres in pixels.
        brightness: (N,) per-column brightness (relative, not yet normalised).
        sub_index: (N,) index of the sublattice each column belongs to.
    """
    material = MATERIALS[config.material]
    a1, a2 = _cell_vectors(config, rng)
    origin = rng.uniform(0.0, config.spacing, size=2)

    reach = int(np.ceil(2.0 * config.size / config.spacing)) + 2
    ii, jj = np.meshgrid(np.arange(-reach, reach), np.arange(-reach, reach), indexing="ij")
    cells = ii.reshape(-1, 1) * a1 + jj.reshape(-1, 1) * a2

    centres, brightness, sub_index = [], [], []
    for k, site in enumerate(material.basis):
        pts = cells + site.offset[0] * a1 + site.offset[1] * a2 + origin
        centres.append(pts)
        brightness.append(np.full(len(pts), z_contrast(*([site.z] * site.atoms))))
        sub_index.append(np.full(len(pts), k))
    centres = np.concatenate(centres)
    brightness = np.concatenate(brightness)
    sub_index = np.concatenate(sub_index)

    pad = 3.0 * config.probe_sigma
    inside = np.all((centres >= -pad) & (centres <= config.size - 1 + pad), axis=1)
    return centres[inside], brightness[inside], sub_index[inside]


def _disorder_field(config: SegConfig, rng: np.random.Generator) -> np.ndarray:
    """Return a bool mask for the disordered region at the target area.

    A few random Gaussian bumps are summed into a smooth field and
    thresholded at the quantile that yields ``disorder_fraction`` of the
    frame, giving an organic blob whose size is controlled but whose shape
    varies with the seed.
    """
    if config.disorder_fraction <= 0.0:
        return np.zeros((config.size, config.size), dtype=bool)

    field = np.zeros((config.size, config.size), dtype=np.float64)
    n_bumps = int(rng.integers(1, 4))
    for _ in range(n_bumps):
        seed_img = np.zeros((config.size, config.size))
        r = int(rng.integers(0, config.size))
        c = int(rng.integers(0, config.size))
        seed_img[r, c] = 1.0
        width = rng.uniform(config.size / 10.0, config.size / 5.0)
        field += gaussian_filter(seed_img, sigma=width)

    if field.max() <= 0:
        return np.zeros((config.size, config.size), dtype=bool)
    threshold = np.quantile(field, 1.0 - config.disorder_fraction)
    return field >= threshold


def _paint_disk(labels: np.ndarray, centre: np.ndarray, radius: float, value: int) -> None:
    """Paint a filled disk of a class value into the label map in place."""
    h, w = labels.shape
    r0 = max(int(np.floor(centre[0] - radius)), 0)
    r1 = min(int(np.ceil(centre[0] + radius)) + 1, h)
    c0 = max(int(np.floor(centre[1] - radius)), 0)
    c1 = min(int(np.ceil(centre[1] + radius)) + 1, w)
    if r0 >= r1 or c0 >= c1:
        return
    rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
    inside = (rr - centre[0]) ** 2 + (cc - centre[1]) ** 2 <= radius**2
    block = labels[r0:r1, c0:c1]
    block[inside] = value


def _render(config: SegConfig, centres: np.ndarray, brightness: np.ndarray) -> np.ndarray:
    """Splat and blur the columns into a clean image, then add background."""
    size = config.size
    clean = np.zeros((size, size), dtype=np.float64)
    if len(centres):
        r, c = centres[:, 0], centres[:, 1]
        r0, c0 = np.floor(r).astype(int), np.floor(c).astype(int)
        fr, fc = r - r0, c - c0
        for dr, dc, w in (
            (0, 0, (1 - fr) * (1 - fc)),
            (0, 1, (1 - fr) * fc),
            (1, 0, fr * (1 - fc)),
            (1, 1, fr * fc),
        ):
            rr, cc = r0 + dr, c0 + dc
            ok = (rr >= 0) & (rr < size) & (cc >= 0) & (cc < size)
            np.add.at(clean, (rr[ok], cc[ok]), brightness[ok] * w[ok])
    # Peak of a splatted-then-blurred unit weight is 1/(2 pi sigma^2).
    clean *= 2.0 * np.pi * config.probe_sigma**2
    clean = gaussian_filter(clean, sigma=config.probe_sigma, mode="constant")

    clean += config.background
    return clean


def simulate_image(config: SegConfig, rng: np.random.Generator | None = None) -> SegResult:
    """Simulate one STEM image and its exact per-pixel label map.

    Args:
        config: Simulation parameters.
        rng: NumPy generator; a fixed generator gives a fixed sample.

    Returns:
        A SegResult with the noisy image, the label map, and the geometry.
    """
    if rng is None:
        rng = np.random.default_rng()

    centres, brightness, _ = _tile_columns(config, rng)
    disorder = _disorder_field(config, rng)

    def in_disorder(points: np.ndarray) -> np.ndarray:
        rr = np.clip(np.round(points[:, 0]).astype(int), 0, config.size - 1)
        cc = np.clip(np.round(points[:, 1]).astype(int), 0, config.size - 1)
        return disorder[rr, cc]

    inside_disorder = in_disorder(centres)
    ordered = centres[~inside_disorder]
    ordered_bright = brightness[~inside_disorder]
    dis_cols = centres[inside_disorder]
    dis_bright = brightness[inside_disorder]

    n = len(ordered)
    roll = rng.uniform(size=n)
    is_vacancy = roll < config.vacancy_fraction
    is_dopant = (roll >= config.vacancy_fraction) & (
        roll < config.vacancy_fraction + config.dopant_fraction
    )

    vacancy_sites = ordered[is_vacancy].copy()

    keep = ~is_vacancy
    kept_centres = ordered[keep].copy()
    kept_bright = ordered_bright[keep].copy()
    kept_dopant = is_dopant[keep].copy()
    kept_bright[kept_dopant] = z_contrast(config.dopant_z)
    # Static displacements in the ordered region.
    kept_centres += rng.normal(0.0, config.displacement_sigma, size=kept_centres.shape)

    # Disordered columns: heavy jitter and random thinning.
    if len(dis_cols):
        survive = rng.uniform(size=len(dis_cols)) >= config.disorder_thinning
        dis_cols = dis_cols[survive] + rng.normal(
            0.0, config.disorder_jitter, size=(int(survive.sum()), 2)
        )
        dis_bright = dis_bright[survive]

    all_centres = np.concatenate([kept_centres, dis_cols]) if len(dis_cols) else kept_centres
    all_bright = np.concatenate([kept_bright, dis_bright]) if len(dis_cols) else kept_bright
    # Normalise by the brightest HOST column, not the global max. A heavy
    # dopant is many times brighter than a light host lattice, so dividing by
    # its brightness would push the whole lattice below the background pedestal
    # and make it invisible. Referencing the host keeps the lattice near 1 and
    # lets the dopant simply saturate, which is what a real HAADF detector does.
    host_bright = np.concatenate(
        [kept_bright[~kept_dopant], dis_bright] if len(dis_cols) else [kept_bright[~kept_dopant]]
    )
    norm = host_bright.max() if len(host_bright) else (all_bright.max() if len(all_bright) else 1.0)
    all_bright = all_bright / norm

    clean = _render(config, all_centres, all_bright)
    if config.background_variation > 0:
        fluct = gaussian_filter(
            rng.normal(0.0, 1.0, size=(config.size, config.size)), sigma=config.size / 8.0
        )
        scale = fluct.std()
        if scale > 0:
            clean = clean + np.abs(fluct) * (config.background_variation / scale)

    scaled = np.clip(clean, 0.0, None) * config.dose
    noisy = rng.poisson(scaled).astype(np.float64)
    if config.read_noise > 0:
        noisy = noisy + rng.normal(0.0, config.read_noise, size=noisy.shape)
    noisy = noisy / config.dose

    # Build the label map from geometry (never from the noisy image).
    labels = np.full((config.size, config.size), BACKGROUND, dtype=np.int8)
    for site in vacancy_sites:
        _paint_disk(labels, site, config.label_radius, VACANCY)
    dopant_centres = kept_centres[kept_dopant]
    host_centres = kept_centres[~kept_dopant]
    for centre in host_centres:
        _paint_disk(labels, centre, config.label_radius, LATTICE)
    for centre in dopant_centres:
        _paint_disk(labels, centre, config.label_radius, DOPANT)
    # The disordered region wins over everything inside its footprint.
    labels[disorder] = DISORDERED

    column_class = np.where(kept_dopant, DOPANT, LATTICE).astype(np.int8)
    return SegResult(
        image=noisy.astype(np.float32),
        labels=labels,
        disorder_mask=disorder,
        columns=kept_centres,
        column_class=column_class,
        vacancy_sites=vacancy_sites,
        config=config,
    )
