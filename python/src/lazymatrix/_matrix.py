"""Public matrix interface backed by owned Rust storage."""

import operator
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

from ._core import Matrix


def _real_array(value: Any) -> NDArray[np.float64]:
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise TypeError("expected real numeric values")
    return np.asarray(array, dtype=np.float64)


class LazyMatrix:
    """An owned matrix with lazy column centering and scaling.

    Parameters
    ----------
    X : numpy.ndarray or scipy.sparse matrix or array
        Two-dimensional real numeric input. Sparse input must use CSR or CSC
        storage. Values are copied to float64 storage, and sparse copies are
        converted to canonical CSC. Later changes to X do not affect this object.
    center : bool, default=True
        Subtract each column's mean during matrix operations.
    scale : bool, default=True
        Divide each column by its population standard deviation. Exact zero
        scales are replaced with one.

    Notes
    -----
    Products and column access do not allocate a full normalized matrix.
    Dense input still requires a full raw-data copy. Multiple right-hand sides
    currently use repeated vector products. Nonfinite values follow the Rust
    core's IEEE arithmetic; missing values are not ignored.

    Use :meth:`toarray` for explicit dense conversion and
    :meth:`aslinearoperator` for SciPy solvers. General slicing, mutation,
    elementwise arithmetic, and implicit NumPy conversion are unsupported.

    Examples
    --------
    >>> import numpy as np
    >>> from lazymatrix import standardize
    >>> Z = standardize(np.array([[1.0, 0.0], [3.0, 4.0]]))
    >>> Z @ np.array([1.0, 2.0])
    array([-3.,  3.])
    >>> Z[:, 0]
    array([-1.,  1.])
    """

    __slots__ = ("_matrix", "_transposed", "_transpose")
    __array_priority__ = 1000

    def __init__(self, X: Any, *, center: bool = True, scale: bool = True) -> None:
        if not isinstance(center, bool) or not isinstance(scale, bool):
            raise TypeError("center and scale must be booleans")
        if sparse.issparse(X):
            if X.format not in ("csr", "csc"):
                raise TypeError("sparse input must use CSR or CSC storage")
            if X.ndim != 2:
                raise ValueError("input must be two-dimensional")
            if X.dtype.kind not in "biuf":
                raise TypeError("expected real numeric values")
            raw = X.copy()
            raw.check_format(full_check=True)
            raw = raw.astype(np.float64, copy=False).tocsc(copy=False)
            raw.sum_duplicates()
            raw.sort_indices()
            self._matrix = Matrix.from_csc(
                raw.shape,
                raw.data,
                np.asarray(raw.indices, dtype=np.int64),
                np.asarray(raw.indptr, dtype=np.int64),
                (center, scale),
            )
        else:
            if not isinstance(X, np.ndarray):
                raise TypeError("input must be a NumPy array or SciPy CSR/CSC matrix")
            raw = _real_array(X)
            if raw.ndim != 2:
                raise ValueError("input must be two-dimensional")
            self._matrix = Matrix.from_dense(raw, (center, scale))
        self._transposed = False
        self._transpose: LazyMatrix | None = None

    @property
    def shape(self) -> tuple[int, int]:
        """The logical number of rows and columns."""
        shape = self._matrix.shape
        return shape[::-1] if self._transposed else shape

    @property
    def ndim(self) -> int:
        """The number of dimensions, always two."""
        return 2

    @property
    def dtype(self) -> np.dtype[np.float64]:
        """The stored and output data type, always float64."""
        return np.dtype(np.float64)

    @property
    def centers(self) -> NDArray[np.float64] | None:
        """A copy of the original columns' fitted means, or None.

        Transposing the matrix retains these original column parameters.
        """
        return self._matrix.centers

    @property
    def scales(self) -> NDArray[np.float64] | None:
        """A copy of the original columns' fitted scales, or None.

        Transposing the matrix retains these original column parameters.
        """
        return self._matrix.scales

    @property
    def T(self) -> "LazyMatrix":
        """A transpose sharing the same raw storage and fitted parameters."""
        if self._transpose is None:
            result = object.__new__(LazyMatrix)
            result._matrix = self._matrix
            result._transposed = not self._transposed
            result._transpose = self
            self._transpose = result
        return self._transpose

    def __matmul__(self, other: Any) -> NDArray[np.float64]:
        if isinstance(other, LazyMatrix) or sparse.issparse(other):
            raise TypeError("products require a dense vector or matrix")
        values = _real_array(other)
        if values.ndim == 1:
            return self._matrix.matvec(values, self._transposed)
        if values.ndim == 2:
            return self._matrix.matmat(values, self._transposed)
        raise ValueError("products require a one- or two-dimensional operand")

    def __rmatmul__(self, other: Any) -> NDArray[np.float64]:
        result = self.T @ np.asarray(other).T
        return result.T

    def __getitem__(self, key: Any) -> NDArray[np.float64]:
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or not isinstance(key[0], slice)
            or key[0] != slice(None)
            or isinstance(key[1], (bool, np.bool_))
        ):
            raise TypeError("only column access Z[:, j] is supported")
        index = operator.index(key[1])
        if index < 0:
            index += self.shape[1]
        if not 0 <= index < self.shape[1]:
            raise IndexError("column index out of bounds")
        return self._matrix.column(index, self._transposed)

    def __array__(self, dtype: Any = None, copy: Any = None) -> NDArray[np.float64]:
        raise TypeError("use toarray() to explicitly materialize a LazyMatrix")

    def __array_ufunc__(
        self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any
    ) -> Any:
        if ufunc is np.matmul and method == "__call__" and not kwargs:
            if inputs[0] is self:
                return self.__matmul__(inputs[1])
            if inputs[1] is self:
                return self.__rmatmul__(inputs[0])
        return NotImplemented

    def __array_function__(self, func: Any, types: Any, args: Any, kwargs: Any) -> Any:
        return NotImplemented

    def aslinearoperator(self) -> LinearOperator:
        """Expose this matrix to SciPy's iterative linear algebra routines.

        Returns
        -------
        scipy.sparse.linalg.LinearOperator
            An operator sharing this matrix's storage and supporting forward,
            transpose, and multiple-right-hand-side products.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.sparse.linalg import lsqr
        >>> from lazymatrix import standardize
        >>> Z = standardize(np.array([[1.0], [3.0]]))
        >>> lsqr(Z.aslinearoperator(), np.array([-2.0, 2.0]))[0]
        array([2.])
        """
        return LinearOperator(
            shape=self.shape,
            dtype=self.dtype,
            matvec=lambda values: self @ values.reshape(-1),
            rmatvec=lambda values: self.T @ values.reshape(-1),
            matmat=lambda values: self @ values,
            rmatmat=lambda values: self.T @ values,
        )

    def toarray(self) -> NDArray[np.float64]:
        """Materialize the logical matrix as an independent dense array.

        Returns
        -------
        numpy.ndarray
            A float64 array with this matrix's shape. This operation allocates
            storage for every logical entry, including normalized sparse zeros.
        """
        return self._matrix.toarray(self._transposed)


def standardize(X: Any, *, center: bool = True, scale: bool = True) -> LazyMatrix:
    """Fit column means and population scales and return a lazy matrix.

    Parameters
    ----------
    X : numpy.ndarray or scipy.sparse matrix or array
        Two-dimensional real numeric input. Sparse input must use CSR or CSC.
        The returned matrix owns a copy of the raw values in float64 storage.
    center : bool, default=True
        Subtract each column's mean during operations.
    scale : bool, default=True
        Divide each column by its population standard deviation, replacing
        exact zero scales with one.

    Returns
    -------
    LazyMatrix
        A matrix retaining fitted column parameters without constructing a
        normalized copy. Statistics are fitted immediately.

    Raises
    ------
    TypeError
        If the input type, storage format, values, or options are unsupported.
    ValueError
        If the input is not two-dimensional or has no rows.

    Notes
    -----
    This first release fits each input independently. Reusing fitted
    normalization on new observations is not yet exposed by the Python API.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import sparse
    >>> from lazymatrix import standardize
    >>> Z = standardize(sparse.csr_matrix([[1.0, 0.0], [3.0, 4.0]]))
    >>> Z @ np.ones(2)
    array([-2.,  2.])
    """
    return LazyMatrix(X, center=center, scale=scale)
