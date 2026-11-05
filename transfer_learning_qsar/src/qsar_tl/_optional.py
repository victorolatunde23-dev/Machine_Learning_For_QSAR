"""Lazy access to the heavy optional dependencies.

Importing ``qsar_tl`` must never require a chemistry toolkit. The data cleaning, split
construction, metric, calibration and threshold logic is pure numpy/pandas/sklearn and is
the part most worth testing, so it stays importable on a bare interpreter. Everything that
needs a molecule goes through :func:`require` and fails with a message naming the extra to
install rather than a bare ``ModuleNotFoundError`` six frames deep.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Iterable

__all__ = ["require", "have", "missing", "tqdm", "OptionalDependencyError"]


class OptionalDependencyError(ImportError):
    """Raised when a code path needs an optional dependency that is not installed."""


# module name -> (pip extra, what it is needed for)
_EXTRAS: Dict[str, "tuple[str, str]"] = {
    "rdkit": ("chem", "molecule parsing, fingerprints, descriptors, enumeration"),
    "lightgbm": ("boost", "the LightGBM classifier"),
    "torch": ("torch", "the multi-task neural network and ChEMBL pretraining"),
    "duckdb": ("bindingdb", "querying the BindingDB TSV dump"),
    "requests": ("bindingdb", "downloading the BindingDB release"),
    "matplotlib": ("plots", "figures"),
    "seaborn": ("plots", "figure styling"),
    "tqdm": ("progress", "progress bars"),
}

_cache: Dict[str, Any] = {}


def require(name: str) -> Any:
    """Import ``name`` or raise :class:`OptionalDependencyError` naming the extra.

    >>> require("numpy").__name__
    'numpy'
    """
    if name in _cache:
        return _cache[name]
    try:
        mod = importlib.import_module(name)
    except ImportError as exc:  # pragma: no cover - depends on the environment
        extra, purpose = _EXTRAS.get(name, (name, name))
        raise OptionalDependencyError(
            f"{name} is required for {purpose} but is not installed.\n"
            f"  pip install 'qsar-tl[{extra}]'\n"
            f"or install it directly: pip install {name}\n"
            f"Note: rdkit and lightgbm ship wheels for most platforms; torch is only "
            f"needed for the neural-network arm, which docs/RESULTS.md recommends "
            f"against for this dataset."
        ) from exc
    _cache[name] = mod
    return mod


def have(name: str) -> bool:
    """True if the optional dependency can be imported. Never raises."""
    try:
        require(name)
        return True
    except ImportError:
        return False


def missing(names: Iterable[str]) -> "list[str]":
    """The subset of ``names`` that is not installed, for a one-line capability report."""
    return [n for n in names if not have(n)]


def tqdm(iterable=None, **kwargs):
    """``tqdm`` when installed, a transparent pass-through when not.

    Progress bars are a convenience, so a missing tqdm must not change behaviour. The
    fallback supports the iterator, context-manager and manual-update forms the pipeline
    actually uses.
    """
    try:
        from tqdm.auto import tqdm as _tqdm  # type: ignore

        return _tqdm(iterable, **kwargs) if iterable is not None else _tqdm(**kwargs)
    except ImportError:
        return _NullBar(iterable)


class _NullBar:
    """Stand-in for a tqdm bar: iterates, enters, updates and closes, shows nothing."""

    def __init__(self, iterable=None):
        self._iterable = iterable
        self.n = 0

    def __iter__(self):
        return iter(() if self._iterable is None else self._iterable)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def update(self, n: int = 1) -> None:
        self.n += n

    def close(self) -> None:
        pass

    def set_postfix(self, *a, **k) -> None:
        pass

    def set_description(self, *a, **k) -> None:
        pass

    @staticmethod
    def write(*a, **k) -> None:
        print(*a)
