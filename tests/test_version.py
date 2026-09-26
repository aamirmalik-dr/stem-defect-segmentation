"""Guard that the package version and the installed distribution metadata agree."""

from importlib.metadata import version

import stemseg


def test_version_matches_distribution_metadata() -> None:
    assert stemseg.__version__ == version("stemseg")
