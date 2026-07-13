"""Reading and writing simulated samples and fitted models."""

from __future__ import annotations

import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np

from stemseg.sim import SegConfig, SegResult


def save_sample(path: str | Path, result: SegResult) -> None:
    """Save a SegResult (image, labels, geometry, config) to a .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = asdict(result.config)
    np.savez_compressed(
        path,
        image=result.image,
        labels=result.labels,
        disorder_mask=result.disorder_mask,
        columns=result.columns,
        column_class=result.column_class,
        vacancy_sites=result.vacancy_sites,
        config_keys=np.array(list(config.keys())),
        config_vals=np.array([repr(v) for v in config.values()]),
    )


def load_sample(path: str | Path) -> SegResult:
    """Load a SegResult previously written by ``save_sample``."""
    with np.load(path, allow_pickle=False) as data:
        keys = list(data["config_keys"])
        vals = [eval(v) for v in data["config_vals"]]  # noqa: S307 (values are our own reprs)
        config = SegConfig(**dict(zip(keys, vals)))
        return SegResult(
            image=data["image"],
            labels=data["labels"],
            disorder_mask=data["disorder_mask"],
            columns=data["columns"],
            column_class=data["column_class"],
            vacancy_sites=data["vacancy_sites"],
            config=config,
        )


def save_rf(path: str | Path, classifier) -> None:
    """Pickle a fitted RandomForestPixelClassifier to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(classifier, handle)


def load_rf(path: str | Path):
    """Load a pickled RandomForestPixelClassifier."""
    with open(path, "rb") as handle:
        return pickle.load(handle)  # noqa: S301 (only load repo-produced artifacts)
