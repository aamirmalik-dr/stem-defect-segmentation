# Changelog

## Unreleased

- Package metadata completed: keywords, classifiers, and project URLs in `pyproject.toml`.
- Citation file (`CITATION.cff`) and this changelog added.
- Related repositories section in the README linking the six sibling electron-microscopy repositories.
- Test guarding the package `__version__` against the installed distribution metadata.

## 0.1.0 (2026-07-13)

- Labelled simulator for projected crystals with four presets (graphene, hBN, MoS2, a perovskite-like oxide), five pixel classes (background, lattice, vacancy, dopant, disordered) with exact geometry-derived label maps and Poisson noise set by one dose parameter.
- Three segmenters trained or tuned on the same simulator: a threshold-and-morphology pipeline, a random forest over a 15-channel local feature bank, and a three-level multi-class U-Net.
- Per-class IoU and Dice (NaN for absent classes so a mean never hides a missed rare class), pixel accuracy, confusion matrix, and a symmetric boundary-distance error for the disordered region.
- YAML benchmark harness with dose, defect-density, and class-imbalance sweeps, per-material scoring, a fair-tuning check for the classical baseline, and confusion analysis; the `stemseg` CLI, committed weights, results JSON, figures, model card, API docs, and executed tutorial.
- Committed-artifact regression guard in the test suite and a CI workflow.
- Maintenance after publication: confusion-matrix label overlap fixed, stale numbers corrected, ruff and black pinned to exact versions, README expanded and restructured, gallery re-exported at higher resolution.
