"""Lazy normalization of NumPy arrays and SciPy sparse matrices."""

from importlib.metadata import version as _version

from ._matrix import LazyMatrix as LazyMatrix
from ._matrix import standardize as standardize

__version__ = _version("lazymatrix")
__all__ = ["LazyMatrix", "standardize"]
