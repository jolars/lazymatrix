"""Lazy normalization of NumPy arrays and SciPy sparse matrices."""

from importlib.metadata import version as _version

from ._matrix import ColumnNormalization as ColumnNormalization
from ._matrix import ColumnNormalizationParams as ColumnNormalizationParams
from ._matrix import LazyMatrix as LazyMatrix
from ._matrix import RowNormalization as RowNormalization
from ._matrix import RowNormalizationParams as RowNormalizationParams
from ._matrix import normalize as normalize
from ._matrix import standardize as standardize

__version__ = _version("lazymatrix")
__all__ = [
    "ColumnNormalization",
    "ColumnNormalizationParams",
    "LazyMatrix",
    "RowNormalization",
    "RowNormalizationParams",
    "normalize",
    "standardize",
]
