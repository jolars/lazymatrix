"""Public matrix interface backed by owned Rust storage."""

import operator
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

from ._core import Matrix, Params


def _real_array(value: Any) -> NDArray[np.float64]:
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise TypeError("expected real numeric values")
    return np.asarray(array, dtype=np.float64)


Centering = bool | Literal["none", "mean", "min", "median"]
Scaling = bool | Literal["none", "sd", "range", "quantile_range", "maxabs", "l1", "l2"]
Specification = tuple[str, str, int, tuple[float, float]]


def _method(value: Any, name: str, default: str, choices: tuple[str, ...]) -> str:
    if isinstance(value, bool):
        return default if value else "none"
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a boolean or method name")
    if value not in choices:
        raise ValueError(f"unknown {name} method {value!r}; choose from {choices}")
    return value


class NormalizationParams:
    """Reusable float64 column centers and scales, independent of raw storage.

    Parameters
    ----------
    ncols : int
        Nonnegative number of columns, retained even when both components are None.
    centers : array-like or None, default=None
        One real value per column. Copied without changing nonfinite values.
    scales : array-like or None, default=None
        One real, nonzero value per column. Negative and nonfinite values are
        preserved. Positive and negative zero raise ValueError with the column
        index, including values that round to zero in float64.

    Notes
    -----
    Obtain fitted parameters with ``Z.normalization``. Reconstruct Rust's
    ``NormalizationParams::from_parts(ncols, centers, scales)`` using the same
    values and None components. Arrays returned by properties are independent
    copies. Parameters never update when transformed data changes.

    A portable checkpoint consists of an integer ncols and optional one-dimensional
    float64 centers and scales arrays in feature order. For example, save ncols
    and only active arrays with numpy.savez, and load with allow_pickle=False.
    Omitted arrays mean None; do not replace them with zeros or ones because
    inactive components have different IEEE semantics. This preserves NaN and
    infinity without nonstandard JSON values.
    """

    __slots__ = ("_params",)

    def __init__(self, ncols: int, *, centers: Any = None, scales: Any = None) -> None:
        ncols = operator.index(ncols)
        if ncols < 0:
            raise ValueError("ncols must be nonnegative")
        arrays = []
        for name, value in (("centers", centers), ("scales", scales)):
            array = None if value is None else _real_array(value)
            if array is not None and array.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional")
            arrays.append(array)
        self._params = Params(ncols, *arrays)

    @property
    def ncols(self) -> int:
        """The number of columns expected by these parameters."""
        return self._params.ncols

    @property
    def centers(self) -> NDArray[np.float64] | None:
        """An independent copy of column centers, or None when inactive."""
        return self._params.centers

    @property
    def scales(self) -> NDArray[np.float64] | None:
        """An independent copy of column scales, or None when inactive."""
        return self._params.scales

    def transform(self, X: Any) -> "LazyMatrix":
        """Wrap new raw data using these parameters without fitting statistics.

        X must have ncols columns. Input ownership, supported storage, and lazy
        operations follow LazyMatrix. A column-count mismatch raises ValueError.
        """
        return LazyMatrix.from_normalization(X, self)


class LazyMatrix:
    """An owned matrix with lazy column centering and scaling.

    Parameters
    ----------
    X : numpy.ndarray or scipy.sparse matrix or array
        Two-dimensional real numeric input. Sparse input must use CSR or CSC
        storage. Values are copied to float64 storage, and sparse copies are
        converted to canonical CSC. Later changes to X do not affect this object.
    center : bool or {"none", "mean", "min", "median"}, default=True
        Column center to subtract. True selects the mean; False disables centering.
    scale : bool or {"none", "sd", "range", "quantile_range", "maxabs", "l1", "l2"}, default=True
        Column scale to divide by. True selects standard deviation; False disables
        scaling. Norms and maximum absolute values use centered columns when
        centering is active. Exact zero scales become one.
    df : int, default=0
        Nonnegative degrees of freedom subtracted from the number of rows in
        the variance divisor. Use zero for population standard deviations or
        one for sample standard deviations. Columns with at most df rows have
        NaN scales. Used only for standard-deviation scaling.
    quantile_range : tuple of float, default=(0.25, 0.75)
        Lower and upper probabilities satisfying 0 <= lower < upper <= 1.
        Quantile-range scaling uses exact linear quantiles at (nrows - 1) * q,
        without standard-normal calibration. Bounds are validated for all methods.

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

    def __init__(
        self,
        X: Any,
        *,
        center: Centering = True,
        scale: Scaling = True,
        df: int = 0,
        quantile_range: tuple[float, float] = (0.25, 0.75),
    ) -> None:
        center = _method(center, "centering", "mean", ("none", "mean", "min", "median"))
        scale = _method(
            scale,
            "scaling",
            "sd",
            ("none", "sd", "range", "quantile_range", "maxabs", "l1", "l2"),
        )
        df = operator.index(df)
        if df < 0:
            raise ValueError("df must be nonnegative")
        bounds = _real_array(quantile_range)
        if bounds.shape != (2,):
            raise ValueError("quantile_range must contain two probabilities")
        lower, upper = map(float, bounds)
        if not (np.isfinite(bounds).all() and 0 <= lower < upper <= 1):
            raise ValueError("quantile bounds must satisfy 0 <= lower < upper <= 1")
        self._initialize(X, (center, scale, df, (lower, upper)), None)

    @classmethod
    def from_normalization(
        cls, X: Any, parameters: NormalizationParams
    ) -> "LazyMatrix":
        """Copy raw input and apply fitted parameters without recomputing them.

        The input must have parameters.ncols columns. Invalid parameter types
        raise TypeError, and column-count mismatches raise ValueError.
        """
        if not isinstance(parameters, NormalizationParams):
            raise TypeError("parameters must be NormalizationParams")
        result = cls.__new__(cls)
        result._initialize(X, None, parameters._params)
        return result

    def _initialize(
        self, X: Any, specification: Specification | None, parameters: Params | None
    ) -> None:
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
                specification,
                parameters,
            )
        else:
            if not isinstance(X, np.ndarray):
                raise TypeError("input must be a NumPy array or SciPy CSR/CSC matrix")
            raw = _real_array(X)
            if raw.ndim != 2:
                raise ValueError("input must be two-dimensional")
            self._matrix = Matrix.from_dense(raw, specification, parameters)
        self._transposed = False
        self._transpose: LazyMatrix | None = None

    @property
    def normalization(self) -> NormalizationParams:
        """A copy of the original columns' fitted parameters, including on T."""
        result = object.__new__(NormalizationParams)
        result._params = self._matrix.normalization
        return result

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
        """A copy of the original columns' fitted centers, or None.

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


def standardize(
    X: Any, *, center: bool = True, scale: bool = True, df: int = 0
) -> LazyMatrix:
    """Fit column means and standard deviations and return a lazy matrix.

    Parameters
    ----------
    X : numpy.ndarray or scipy.sparse matrix or array
        Two-dimensional real numeric input. Sparse input must use CSR or CSC.
        The returned matrix owns a copy of the raw values in float64 storage.
    center : bool, default=True
        Subtract each column's mean during operations.
    scale : bool, default=True
        Divide each column by its standard deviation, replacing
        exact zero scales with one.
    df : int, default=0
        Nonnegative degrees of freedom subtracted from the number of rows in
        the variance divisor. Use zero for population standard deviations or
        one for sample standard deviations. Columns with at most df rows have
        NaN scales. Ignored when scale is False.

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
        If the input is not two-dimensional, has no rows, or df is negative.

    Notes
    -----
    This function fits each input independently. Reuse training parameters with
    ``Z.normalization.transform(new_X)``. For other centering and scaling methods,
    construct LazyMatrix directly.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import sparse
    >>> from lazymatrix import standardize
    >>> Z = standardize(sparse.csr_matrix([[1.0, 0.0], [3.0, 4.0]]))
    >>> Z @ np.ones(2)
    array([-2.,  2.])
    """
    if not isinstance(center, bool) or not isinstance(scale, bool):
        raise TypeError("center and scale must be booleans")
    return LazyMatrix(X, center=center, scale=scale, df=df)
