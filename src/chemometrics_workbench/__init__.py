"""Open-source, local-first chemometrics workbench."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("chemometrics-workbench")
except PackageNotFoundError:  # pragma: no cover - a checkout that was never installed
    # `pyproject.toml` is the one place the number is written; a literal here
    # drifted from it for three releases and every experiment recorded the
    # stale one (#181).
    __version__ = "0.0.0"
