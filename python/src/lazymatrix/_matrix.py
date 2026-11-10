"""Public matrix interface backed by owned Rust storage."""

import operator
from typing import Any, ClassVar, Literal, cast

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
NormalizationAxis = Literal[0, 1]
Specification = tuple[int, str, str, int, tuple[float, float], bool]


def _method(value: Any, name: str, default: str, choices: tuple[str, ...]) -> str:
    if isinstance(value, bool):
        return default if value else "none"
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a boolean or method name")
    if value not in choices:
        raise ValueError(f"unknown {name} method {value!r}; choose from {choices}")
    return value


def _specification(
    axis: NormalizationAxis,
    center: Centering,
    scale: Scaling,
    df: int,
    quantile_range: tuple[float, float],
    unit_variance: bool,
) -> Specification:
    if isinstance(axis, (bool, np.bool_)):
        raise TypeError("axis must be the integer 0 or 1")
    axis = operator.index(axis)
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")
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
    if not isinstance(unit_variance, (bool, np.bool_)):
        raise TypeError("unit_variance must be a boolean")
    if unit_variance:
        if scale != "quantile_range":
            raise ValueError('unit_variance requires scale="quantile_range"')
        if not (0 < lower < upper < 1):
            raise ValueError(
                "unit_variance requires quantile bounds strictly inside (0, 1)"
            )
    return axis, center, scale, df, (lower, upper), bool(unit_variance)


class _NormalizationParams:
    __slots__ = ("_params",)
    _axis: ClassVar[NormalizationAxis]

    def _initialize(self, dimension: int, centers: Any, scales: Any) -> None:
        dimension = operator.index(dimension)
        name = "ncols" if self._axis == 0 else "nrows"
        if dimension < 0:
            raise ValueError(f"{name} must be nonnegative")
        arrays = []
        for name, value in (("centers", centers), ("scales", scales)):
            array = None if value is None else _real_array(value)
            if array is not None and array.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional")
            arrays.append(array)
        self._params = Params(self._axis, dimension, *arrays)


class ColumnNormalizationParams(_NormalizationParams):
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
    ``ColumnNormalizationParams::from_parts(ncols, centers, scales)`` using the same
    values and None components. Arrays returned by properties are independent
    copies. Parameters never update when transformed data changes.

    A portable checkpoint consists of an integer ncols and optional one-dimensional
    float64 centers and scales arrays in feature order. For example, save ncols
    and only active arrays with numpy.savez, and load with allow_pickle=False.
    Omitted arrays mean None; do not replace them with zeros or ones because
    inactive components have different IEEE semantics. This preserves NaN and
    infinity without nonstandard JSON values.
    """

    __slots__ = ()
    _axis = 0

    def __init__(self, ncols: int, *, centers: Any = None, scales: Any = None) -> None:
        self._initialize(ncols, centers, scales)

    @property
    def ncols(self) -> int:
        """The number of columns expected by these parameters."""
        return self._params.dimension

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


class RowNormalizationParams(_NormalizationParams):
    """Float64 centers and scales for particular rows, independent of storage.

    Parameters
    ----------
    nrows : int
        Nonnegative number of rows, retained when both components are None.
    centers : array-like or None, default=None
        One real value per row. Copied without changing nonfinite values.
    scales : array-like or None, default=None
        One real, nonzero value per row. Negative and nonfinite values are
        preserved. Positive and negative zero raise ValueError with the row
        index, including values that round to zero in float64.

    Notes
    -----
    Obtain fitted values from an axis=1 matrix's ``normalization`` property or
    ``RowNormalization.fit``. Arrays returned by properties are independent
    copies. These parameters describe the fitted rows in their original order.
    Fit new samples independently with ``RowNormalization.fit_apply`` or a new
    axis=1 LazyMatrix, even when their row count matches the training data.

    Save nrows and active centers and scales arrays with numpy.savez, then
    reconstruct this class and use ``apply_frozen`` to restore the same rows.
    Omitted arrays mean None, preserving inactive components' IEEE semantics.
    """

    __slots__ = ()
    _axis = 1

    def __init__(self, nrows: int, *, centers: Any = None, scales: Any = None) -> None:
        self._initialize(nrows, centers, scales)

    @property
    def nrows(self) -> int:
        """The number of rows described by these parameters."""
        return self._params.dimension

    @property
    def centers(self) -> NDArray[np.float64] | None:
        """An independent copy of row centers, or None when inactive."""
        return self._params.centers

    @property
    def scales(self) -> NDArray[np.float64] | None:
        """An independent copy of row scales, or None when inactive."""
        return self._params.scales

    def apply_frozen(self, X: Any) -> "LazyMatrix":
        """Apply saved values unchanged to the same rows in the same order.

        X must have nrows rows; its column count may differ. The input is
        copied following LazyMatrix's storage rules. A row-count mismatch
        raises ValueError. This method performs no fitting.
        """
        return LazyMatrix.from_normalization(X, self)


class _Normalization:
    __slots__ = ("_center", "_scale", "_specification")
    _axis: ClassVar[NormalizationAxis]

    def __init__(
        self,
        *,
        center: Centering = True,
        scale: Scaling = True,
        df: int = 0,
        quantile_range: tuple[float, float] = (0.25, 0.75),
        unit_variance: bool = False,
    ) -> None:
        self._specification = _specification(
            self._axis, center, scale, df, quantile_range, unit_variance
        )
        self._center = center
        self._scale = scale

    @property
    def center(self) -> Centering:
        """The fixed centering option supplied to this recipe."""
        return self._center

    @property
    def scale(self) -> Scaling:
        """The fixed scaling option supplied to this recipe."""
        return self._scale

    @property
    def df(self) -> int:
        """Degrees of freedom subtracted from the fitted axis length."""
        return self._specification[3]

    @property
    def quantile_range(self) -> tuple[float, float]:
        """The fixed lower and upper quantile probabilities."""
        return self._specification[4]

    @property
    def unit_variance(self) -> bool:
        """Whether quantile ranges use standard-normal calibration."""
        return self._specification[5]

    def _fit_apply(self, X: Any) -> "LazyMatrix":
        result = LazyMatrix.__new__(LazyMatrix)
        result._initialize(X, self._specification, None)
        return result


class ColumnNormalization(_Normalization):
    """An immutable recipe for fitting column centers and scales.

    The keyword-only center, scale, df, quantile_range, and unit_variance options follow
    LazyMatrix with axis=0. Defaults select means and population standard
    deviations. ``fit`` returns ColumnNormalizationParams, and ``fit_apply``
    returns a LazyMatrix. Reuse fitted column parameters on new samples with
    ``params.transform(X_new)``. Options are available as read-only properties.
    """

    __slots__ = ()
    _axis = 0

    def __init__(
        self,
        *,
        center: Centering = True,
        scale: Scaling = True,
        df: int = 0,
        quantile_range: tuple[float, float] = (0.25, 0.75),
        unit_variance: bool = False,
    ) -> None:
        super().__init__(
            center=center,
            scale=scale,
            df=df,
            quantile_range=quantile_range,
            unit_variance=unit_variance,
        )

    def fit(self, X: Any) -> ColumnNormalizationParams:
        """Fit column parameters without retaining the input matrix.

        Input is copied temporarily using LazyMatrix's storage rules. Only
        fitted centers, scales, and ncols are retained. The recipe stays unchanged.
        """
        return cast(ColumnNormalizationParams, self.fit_apply(X).normalization)

    def fit_apply(self, X: Any) -> "LazyMatrix":
        """Fit column parameters and return a lazy matrix owning its input copy.

        Every call fits independently. Input storage and lazy operations follow
        LazyMatrix with axis=0. Reuse training column parameters on new samples
        with ``matrix.normalization.transform(X_new)``.
        """
        return self._fit_apply(X)


class RowNormalization(_Normalization):
    """An immutable recipe for fitting each row independently.

    The keyword-only center, scale, df, quantile_range, and unit_variance options follow
    LazyMatrix with axis=1. Defaults select means and population standard
    deviations. ``fit`` returns RowNormalizationParams, and ``fit_apply``
    returns a LazyMatrix. Fit new samples with another call to ``fit_apply``;
    saved row parameters apply only to the rows for which they were fitted.
    Options are available as read-only properties.

    Examples
    --------
    >>> recipe = RowNormalization(center=False, scale="l2")
    >>> train = recipe.fit_apply(np.array([[3.0, 4.0], [0.0, 0.0]]))
    >>> validation = recipe.fit_apply(np.array([[0.0, 2.0]]))
    >>> validation.toarray()
    array([[0., 1.]])
    """

    __slots__ = ()
    _axis = 1

    def __init__(
        self,
        *,
        center: Centering = True,
        scale: Scaling = True,
        df: int = 0,
        quantile_range: tuple[float, float] = (0.25, 0.75),
        unit_variance: bool = False,
    ) -> None:
        super().__init__(
            center=center,
            scale=scale,
            df=df,
            quantile_range=quantile_range,
            unit_variance=unit_variance,
        )

    def fit(self, X: Any) -> RowNormalizationParams:
        """Fit row parameters without retaining the input matrix.

        Input is copied temporarily using LazyMatrix's storage rules. Only
        fitted centers, scales, and nrows are retained. The recipe stays unchanged.
        """
        return cast(RowNormalizationParams, self.fit_apply(X).normalization)

    def fit_apply(self, X: Any) -> "LazyMatrix":
        """Fit each row independently and return a lazy matrix owning its input copy.

        Every call fits fresh row parameters, including calls on new samples.
        Products use the saved values without refitting. Input storage and
        lazy operations follow LazyMatrix with axis=1.
        """
        return self._fit_apply(X)


class LazyMatrix:
    """An owned matrix with lazy column or row centering and scaling.

    Parameters
    ----------
    X : numpy.ndarray or scipy.sparse matrix or array
        Two-dimensional real numeric input. Sparse input must use CSR or CSC
        storage. Values are copied to float64 storage, and sparse copies are
        canonicalized while retaining CSR or CSC orientation. Later changes to
        X do not affect this object.
    axis : {0, 1}, default=0
        Axis over which to compute statistics. Zero fits one center and scale
        per column; one fits each row independently. The fitted axis must have
        at least one entry: rows for axis=0, or columns for axis=1.
    center : bool or {"none", "mean", "min", "median"}, default=True
        Center to subtract. True selects the mean; False disables centering.
    scale : bool or {"none", "sd", "range", "quantile_range", "maxabs", "l1", "l2"}, default=True
        Scale to divide by. True selects standard deviation; False disables
        scaling. Norms and maximum absolute values use centered values when
        centering is active. Exact zero scales become one.
    df : int, default=0
        Nonnegative degrees of freedom subtracted from the fitted axis length
        in the variance divisor: nrows for axis=0, or ncols for axis=1. Use zero
        for population standard deviations or one for sample standard deviations.
        Scales are NaN when that length is at most df. Used only for sd scaling.
    quantile_range : tuple of float, default=(0.25, 0.75)
        Lower and upper probabilities satisfying 0 <= lower < upper <= 1.
        Quantile-range scaling uses exact linear quantiles at (n - 1) * q,
        where n is the fitted axis length. Bounds are validated for all methods.
    unit_variance : bool, default=False
        Calibrate quantile-range scales by the corresponding standard-normal
        quantile range (approximately 1.34898 for the interquartile range).
        Requires scale="quantile_range" and bounds strictly inside (0, 1).
        Targets unit variance for normally distributed data, without guaranteeing
        unit sample variance. Exact zero scales become one after calibration.

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
        axis: NormalizationAxis = 0,
        center: Centering = True,
        scale: Scaling = True,
        df: int = 0,
        quantile_range: tuple[float, float] = (0.25, 0.75),
        unit_variance: bool = False,
    ) -> None:
        self._initialize(
            X,
            _specification(axis, center, scale, df, quantile_range, unit_variance),
            None,
        )

    @classmethod
    def from_normalization(
        cls, X: Any, parameters: ColumnNormalizationParams | RowNormalizationParams
    ) -> "LazyMatrix":
        """Copy raw input and apply fitted parameters without recomputing them.

        Column parameters require parameters.ncols columns. Row parameters
        require parameters.nrows rows in the original order. Invalid parameter
        types raise TypeError, and dimension mismatches raise ValueError.
        This method always applies saved values unchanged.
        """
        if not isinstance(
            parameters, (ColumnNormalizationParams, RowNormalizationParams)
        ):
            raise TypeError(
                "parameters must be ColumnNormalizationParams or RowNormalizationParams"
            )
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
            raw = raw.astype(np.float64, copy=False)
            raw.sum_duplicates()
            raw.sort_indices()
            self._matrix = Matrix.from_sparse(
                raw.shape,
                raw.format == "csc",
                (
                    raw.data,
                    np.asarray(raw.indices, dtype=np.int64),
                    np.asarray(raw.indptr, dtype=np.int64),
                ),
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
    def normalization(self) -> ColumnNormalizationParams | RowNormalizationParams:
        """A copy of fitted parameters for the original axis, including on T."""
        params = self._matrix.normalization
        cls = ColumnNormalizationParams if params.axis == 0 else RowNormalizationParams
        result = object.__new__(cls)
        result._params = params
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
        """A copy of the original axis's fitted centers, or None.

        Transposing the matrix retains these original parameters.
        """
        return self._matrix.centers

    @property
    def scales(self) -> NDArray[np.float64] | None:
        """A copy of the original axis's fitted scales, or None.

        Transposing the matrix retains these original parameters.
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


def normalize(
    X: Any,
    *,
    axis: NormalizationAxis = 0,
    center: Centering = True,
    scale: Scaling = True,
    df: int = 0,
    quantile_range: tuple[float, float] = (0.25, 0.75),
    unit_variance: bool = False,
) -> LazyMatrix:
    """Fit centers and scales and return a lazy matrix.

    Parameters
    ----------
    X : numpy.ndarray or scipy.sparse matrix or array
        Two-dimensional real numeric input. Sparse input must use CSR or CSC.
        Values are copied to float64 storage, and sparse copies are canonicalized
        while retaining their orientation. Later changes to X do not affect
        the returned matrix.
    axis : {0, 1}, default=0
        Zero fits each column; one fits each row independently. The fitted axis
        must have at least one entry: rows for axis=0, or columns for axis=1.
    center : bool or {"none", "mean", "min", "median"}, default=True
        Center to subtract. True selects the mean; False or "none" disables
        centering.
    scale : bool or {"none", "sd", "range", "quantile_range", "maxabs", "l1", "l2"}, default=True
        Scale to divide by. True selects standard deviation; False or "none"
        disables scaling. Norms and maximum absolute values use centered values
        when centering is active, including implicit sparse zeros. Exact zero
        scales become one.
    df : int, default=0
        Nonnegative degrees of freedom subtracted from nrows for axis=0 or
        ncols for axis=1 in the variance divisor. Use zero for population
        standard deviations or one for sample standard deviations. Scales are
        NaN when that dimension is at most df. Used only for sd scaling.
    quantile_range : tuple of float, default=(0.25, 0.75)
        Lower and upper probabilities satisfying 0 <= lower < upper <= 1.
        Quantile-range scaling uses exact linear quantiles at (n - 1) * q,
        where n is the fitted axis length. Bounds are validated for all methods.
    unit_variance : bool, default=False
        Calibrate quantile-range scales by the corresponding standard-normal
        quantile range (approximately 1.34898 for the interquartile range).
        Requires scale="quantile_range" and bounds strictly inside (0, 1).
        Targets unit variance for normally distributed data, without guaranteeing
        unit sample variance. Exact zero scales become one after calibration.

    Returns
    -------
    LazyMatrix
        A matrix retaining fitted parameters without constructing a normalized
        copy. Statistics are fitted immediately. Defaults match standardize.

    Raises
    ------
    TypeError
        If the input type, storage format, values, or option types are unsupported.
    ValueError
        If the input is not two-dimensional, the fitted axis is empty, axis or
        method names are invalid, df is negative, or quantile bounds are invalid.

    Notes
    -----
    This function accepts the same options as LazyMatrix and fits each input
    independently. Products and column access do not materialize a normalized
    matrix. Nonfinite values follow the Rust core's IEEE arithmetic; missing
    values are not ignored.

    Reuse training column parameters with ``Z.normalization.transform(new_X)``.
    New samples need independently fitted row parameters; call this function
    again with axis=1 or reuse a RowNormalization recipe. Saved row parameters
    can be applied unchanged to the same rows with ``params.apply_frozen(X)``.

    Examples
    --------
    >>> import numpy as np
    >>> from lazymatrix import normalize
    >>> X = np.array([[1.0, 2.0], [3.0, 6.0]])
    >>> normalize(X, center="min", scale="range").toarray()
    array([[0., 0.],
           [1., 1.]])
    >>> normalize(np.array([[3.0, 4.0]]), axis=1, center=False, scale="l2").toarray()
    array([[0.6, 0.8]])
    """
    return LazyMatrix(
        X,
        axis=axis,
        center=center,
        scale=scale,
        df=df,
        quantile_range=quantile_range,
        unit_variance=unit_variance,
    )


def standardize(
    X: Any,
    *,
    axis: NormalizationAxis = 0,
    center: bool = True,
    scale: bool = True,
    df: int = 0,
) -> LazyMatrix:
    """Fit means and standard deviations and return a lazy matrix.

    Parameters
    ----------
    X : numpy.ndarray or scipy.sparse matrix or array
        Two-dimensional real numeric input. Sparse input must use CSR or CSC.
        The returned matrix owns a copy of the raw values in float64 storage.
    axis : {0, 1}, default=0
        Zero standardizes each column; one standardizes each row independently.
    center : bool, default=True
        Subtract the fitted column or row mean during operations.
    scale : bool, default=True
        Divide by the fitted column or row standard deviation, replacing
        exact zero scales with one.
    df : int, default=0
        Nonnegative degrees of freedom subtracted from nrows for axis=0 or
        ncols for axis=1 in the variance divisor. Use zero for population
        standard deviations or one for sample standard deviations. Scales are
        NaN when that dimension is at most df. Ignored when scale is False.

    Returns
    -------
    LazyMatrix
        A matrix retaining fitted parameters without constructing a
        normalized copy. Statistics are fitted immediately.

    Raises
    ------
    TypeError
        If the input type, storage format, values, or options are unsupported.
    ValueError
        If the input is not two-dimensional, the fitted axis is empty, axis
        is invalid, or df is negative.

    Notes
    -----
    This function fits each input independently. Reuse training parameters with
    ``Z.normalization.transform(new_X)`` for column normalization. New samples
    need independently fitted row parameters; call this function again with
    axis=1 or reuse a RowNormalization recipe. Use normalize for other centering
    and scaling methods.

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
    return normalize(X, axis=axis, center=center, scale=scale, df=df)
