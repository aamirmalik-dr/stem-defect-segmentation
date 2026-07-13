"""Tests for the labelled STEM simulator."""

from __future__ import annotations

import numpy as np
import pytest

from stemseg.sim import (
    BACKGROUND,
    CLASS_NAMES,
    DISORDERED,
    DOPANT,
    LATTICE,
    MATERIALS,
    NUM_CLASSES,
    VACANCY,
    SegConfig,
    material_config,
    simulate_image,
    z_contrast,
)


def test_class_codes_are_distinct_and_named():
    assert len({BACKGROUND, LATTICE, VACANCY, DOPANT, DISORDERED}) == NUM_CLASSES
    assert len(CLASS_NAMES) == NUM_CLASSES


def test_z_contrast_monotone_in_z():
    assert z_contrast(6) < z_contrast(42) < z_contrast(78)
    # A two-atom column outscatters a single atom of the same element.
    assert z_contrast(16, 16) == pytest.approx(2 * z_contrast(16))


def test_shapes_and_dtypes():
    result = simulate_image(SegConfig(size=96), np.random.default_rng(0))
    assert result.image.shape == (96, 96)
    assert result.image.dtype == np.float32
    assert result.labels.shape == (96, 96)
    assert result.labels.min() >= 0 and result.labels.max() < NUM_CLASSES


def test_determinism_same_seed():
    cfg = material_config("graphene", size=96, dose=100.0)
    a = simulate_image(cfg, np.random.default_rng(3))
    b = simulate_image(cfg, np.random.default_rng(3))
    assert np.array_equal(a.image, b.image)
    assert np.array_equal(a.labels, b.labels)


def test_different_seeds_differ():
    cfg = material_config("graphene", size=96)
    a = simulate_image(cfg, np.random.default_rng(1))
    b = simulate_image(cfg, np.random.default_rng(2))
    assert not np.array_equal(a.image, b.image)


def test_higher_dose_is_less_noisy():
    # Defect-free crystal so the background is a clean pedestal and the only
    # thing varying with dose is shot noise (a bright dopant's blurred tail
    # would otherwise dominate the interstitial variance).
    common = dict(size=96, dopant_fraction=0.0, vacancy_fraction=0.0, disorder_fraction=0.0)
    lo = simulate_image(material_config("graphene", dose=5.0, **common), np.random.default_rng(0))
    hi = simulate_image(
        material_config("graphene", dose=5000.0, **common), np.random.default_rng(0)
    )
    bg_lo = lo.image[lo.labels == BACKGROUND].std()
    bg_hi = hi.image[hi.labels == BACKGROUND].std()
    assert bg_hi < bg_lo


def test_no_defects_means_no_rare_defect_classes():
    cfg = material_config(
        "graphene", size=128, vacancy_fraction=0.0, dopant_fraction=0.0, disorder_fraction=0.0
    )
    result = simulate_image(cfg, np.random.default_rng(0))
    assert not np.any(result.labels == VACANCY)
    assert not np.any(result.labels == DOPANT)
    assert not np.any(result.labels == DISORDERED)


def test_vacancy_sites_are_labelled_and_dark():
    cfg = material_config(
        "graphene",
        size=160,
        vacancy_fraction=0.15,
        dopant_fraction=0.0,
        disorder_fraction=0.0,
        dose=4000.0,
    )
    result = simulate_image(cfg, np.random.default_rng(5))
    assert len(result.vacancy_sites) > 0
    assert np.any(result.labels == VACANCY)
    # Vacancy pixels are darker on average than lattice pixels.
    vac = result.image[result.labels == VACANCY].mean()
    lat = result.image[result.labels == LATTICE].mean()
    assert vac < lat


def test_dopant_columns_are_brighter_than_host():
    # Heavy dopant into a light carbon host at high dose.
    cfg = material_config(
        "graphene",
        size=160,
        dopant_fraction=0.15,
        vacancy_fraction=0.0,
        disorder_fraction=0.0,
        dopant_z=78,
        dose=4000.0,
    )
    result = simulate_image(cfg, np.random.default_rng(7))
    assert np.any(result.labels == DOPANT)
    dop = result.image[result.labels == DOPANT].mean()
    lat = result.image[result.labels == LATTICE].mean()
    assert dop > lat


def test_disorder_fraction_controls_area():
    small = simulate_image(
        material_config("graphene", size=160, disorder_fraction=0.05), np.random.default_rng(0)
    )
    large = simulate_image(
        material_config("graphene", size=160, disorder_fraction=0.30), np.random.default_rng(0)
    )
    assert large.disorder_mask.mean() > small.disorder_mask.mean()


def test_disorder_mask_matches_label():
    result = simulate_image(
        material_config("graphene", size=128, disorder_fraction=0.2), np.random.default_rng(1)
    )
    assert np.array_equal(result.disorder_mask, result.labels == DISORDERED)


@pytest.mark.parametrize("name", list(MATERIALS))
def test_every_material_simulates(name):
    result = simulate_image(material_config(name, size=96), np.random.default_rng(0))
    assert result.image.shape == (96, 96)
    assert np.any(result.labels == LATTICE)


def test_unknown_material_raises():
    with pytest.raises(ValueError):
        material_config("unobtainium")
