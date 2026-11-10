import numpy as np
import pytest
from lazymatrix import (
    ColumnNormalization,
    ColumnNormalizationParams,
    LazyMatrix,
    RowNormalization,
    RowNormalizationParams,
    normalize,
    standardize,
)
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

CENTERS = [False, True, "none", "mean", "min", "median"]
SCALES = [False, True, "none", "sd", "range", "quantile_range", "maxabs", "l1", "l2"]
STORAGE = [
    np.asarray,
    sparse.csr_matrix,
    sparse.csc_matrix,
    sparse.csr_array,
    sparse.csc_array,
]


def reference(raw, center, scale, bounds=(0.25, 0.75), axis=0):
    if axis == 1:
        values, centers, scales = reference(raw.T, center, scale, bounds)
        return values.T, centers, scales
    centers = None
    if center is True or center == "mean":
        centers = raw.mean(axis=0)
    elif center == "min":
        centers = raw.min(axis=0)
    elif center == "median":
        centers = np.median(raw, axis=0)
    values = raw if centers is None else raw - centers
    scales = None
    if scale is True or scale == "sd":
        scales = raw.std(axis=0)
    elif scale == "range":
        scales = np.ptp(raw, axis=0)
    elif scale == "quantile_range":
        lower, upper = np.quantile(raw, bounds, axis=0, method="linear")
        scales = upper - lower
    elif scale == "maxabs":
        scales = np.max(np.abs(values), axis=0)
    elif scale == "l1":
        scales = np.sum(np.abs(values), axis=0)
    elif scale == "l2":
        scales = np.sqrt(np.sum(values**2, axis=0))
    if scales is not None:
        scales[scales == 0] = 1
        values = values / scales
    return values, centers, scales


def assert_optional(actual, expected):
    if expected is None:
        assert actual is None
    else:
        assert_allclose(actual, expected, atol=1e-14)


@pytest.mark.parametrize("storage", STORAGE)
@pytest.mark.parametrize("center", CENTERS)
@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("constructor", [LazyMatrix, normalize])
def test_methods_products_and_reuse(storage, center, scale, axis, constructor):
    raw = np.array(
        [
            [0.0, -6.0, 5.0, 0.0],
            [2.0, 0.0, 5.0, 0.0],
            [0.0, 3.0, 5.0, 0.0],
            [9.0, 0.0, 5.0, 0.0],
        ]
    )
    expected, centers, scales = reference(raw, center, scale, axis=axis)
    z = constructor(storage(raw), axis=axis, center=center, scale=scale)
    assert_optional(z.centers, centers)
    assert_optional(z.scales, scales)
    assert_allclose(z.toarray(), expected, atol=1e-14)
    v = np.array([0.5, -2.0, 1.0, 3.0])
    u = np.array([1.0, -0.25, 2.0, 0.0])
    assert_allclose(z @ v, expected @ v, atol=1e-14)
    assert_allclose(z.T @ u, expected.T @ u, atol=1e-14)
    assert_allclose(
        z @ np.column_stack([v, -v]), expected @ np.column_stack([v, -v]), atol=1e-14
    )
    assert_allclose(
        z.T @ np.column_stack([u, -u]),
        expected.T @ np.column_stack([u, -u]),
        atol=1e-14,
    )
    assert_allclose(u @ (z @ v), (z.T @ u) @ v, atol=1e-14)
    for j in range(raw.shape[1]):
        assert_allclose(z[:, j], expected[:, j], atol=1e-14)
        assert_allclose(z.T[:, j], expected.T[:, j], atol=1e-14)

    # Export Rust-fitted parameters, reconstruct through Python, and import into Rust.
    fitted = z.T.normalization
    cls = ColumnNormalizationParams if axis == 0 else RowNormalizationParams
    assert isinstance(fitted, cls)
    dimension = fitted.ncols if axis == 0 else fitted.nrows
    assert dimension == raw.shape[1 - axis]
    reconstructed = cls(dimension, centers=fitted.centers, scales=fitted.scales)
    new = np.array([[10.0, -8.0, 8.0, 1.0], [-1.0, 1.0, 4.0, 0.0]])
    prediction = new.copy()
    if centers is not None:
        prediction -= centers
    if scales is not None:
        prediction /= scales
    if axis == 1:
        new = new.T
        prediction = prediction.T
    reused = LazyMatrix.from_normalization(storage(new), reconstructed)
    assert_allclose(reused.toarray(), prediction, atol=1e-14)
    v = v[: new.shape[1]]
    u = np.arange(1.0, new.shape[0] + 1)
    assert_allclose(reused @ v, prediction @ v, atol=1e-14)
    assert_allclose(reused.T @ u, prediction.T @ u, atol=1e-14)
    assert_optional(reused.centers, centers)
    assert_optional(reused.scales, scales)


@pytest.mark.parametrize("bounds", [(0.0, 1.0), (0.1, 0.8), (0.4, 0.6)])
@pytest.mark.parametrize("storage", STORAGE)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("constructor", [LazyMatrix, normalize])
def test_custom_exact_quantile_ranges(storage, bounds, axis, constructor):
    raw = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 5.0], [4.0, 12.0], [10.0, 100.0]])
    z = constructor(
        storage(raw),
        axis=axis,
        center="median",
        scale="quantile_range",
        quantile_range=bounds,
    )
    expected, centers, scales = reference(raw, "median", "quantile_range", bounds, axis)
    assert_allclose(z.centers, centers)
    assert_allclose(z.scales, scales)
    assert_allclose(z.toarray(), expected)


@pytest.mark.parametrize("center", ["min", "median"])
@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("orientation", ["csr", "csc"])
@pytest.mark.parametrize("axis", [0, 1])
def test_duplicates_are_combined_before_nonlinear_statistics(
    center, scale, orientation, axis
):
    x = sparse.csc_matrix(
        ([3.0, 1.0, -1.0, 0.0, 4.0, -7.0, 2.0], [2, 0, 0, 1, 0, 0, 2], [0, 4, 7]),
        shape=(5, 2),
    )
    if orientation == "csr":
        # The transpose keeps the duplicate representation in CSR storage.
        x = x.T.tocsr()
    raw = x.toarray()
    before = (x.data.copy(), x.indices.copy(), x.indptr.copy())
    expected, centers, scales = reference(raw, center, scale, axis=axis)
    z = LazyMatrix(x, axis=axis, center=center, scale=scale)
    assert_allclose(z.toarray(), expected, atol=1e-14)
    assert_allclose(z.centers, centers)
    assert_optional(z.scales, scales)
    for actual, original in zip((x.data, x.indices, x.indptr), before, strict=True):
        assert_array_equal(actual, original)


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("storage", STORAGE[1:])
@pytest.mark.parametrize("axis", [0, 1])
def test_sparse_fitting_and_reuse_do_not_materialize(monkeypatch, scale, storage, axis):
    raw = np.array(
        [[0.0, 0.0, 3.0], [0.0, 2.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    x = storage(raw)
    expected, _, _ = reference(raw, "median", scale, axis=axis)

    def fail(*args, **kwargs):
        raise AssertionError("unexpected dense materialization")

    for cls in STORAGE[1:]:
        for method in ("toarray", "tocsc", "tocsr"):
            monkeypatch.setattr(cls, method, fail)
    monkeypatch.setattr(LazyMatrix, "toarray", fail)
    recipe = (ColumnNormalization if axis == 0 else RowNormalization)(
        center="median", scale=scale
    )
    z = recipe.fit_apply(x)
    params = recipe.fit(x)
    reused = LazyMatrix.from_normalization(x, params)
    normalized = normalize(x, axis=axis, center="median", scale=scale)
    for operator in (z, reused, normalized):
        assert_allclose(operator @ np.ones(3), expected @ np.ones(3), atol=1e-14)
        assert_allclose(operator.T @ np.ones(4), expected.T @ np.ones(4), atol=1e-14)
        assert_allclose(operator[:, 0], expected[:, 0])
        assert_allclose(operator.T[:, 0], expected.T[:, 0])


@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("scale", [False, True])
@pytest.mark.parametrize("axis", [0, 1])
def test_portable_checkpoint_round_trip(tmp_path, center, scale, axis):
    z = standardize(
        np.array([[1.0, 5.0], [3.0, 5.0]]), axis=axis, center=center, scale=scale
    )
    fitted = z.normalization
    dimension = "ncols" if axis == 0 else "nrows"
    cls = ColumnNormalizationParams if axis == 0 else RowNormalizationParams
    arrays = {dimension: getattr(fitted, dimension)}
    for name in ("centers", "scales"):
        value = getattr(fitted, name)
        if value is not None:
            arrays[name] = value
    checkpoint = tmp_path / "normalization.npz"
    np.savez(checkpoint, **arrays)
    with np.load(checkpoint, allow_pickle=False) as saved:
        restored = cls(
            int(saved[dimension]),
            centers=saved.get("centers"),
            scales=saved.get("scales"),
        )
    new = np.array([[7.0, 8.0]])
    expected = new.copy()
    if center:
        expected -= z.centers
    if scale:
        expected /= z.scales
    if axis == 1:
        new = new.T
        expected = expected.T
    assert_allclose(LazyMatrix.from_normalization(new, restored).toarray(), expected)


@pytest.mark.parametrize("storage", STORAGE)
@pytest.mark.parametrize("axis", [0, 1])
def test_explicit_parameters_preserve_ieee_values_and_ownership(
    storage, tmp_path, axis
):
    centers = np.array([np.nan, np.inf, -np.inf, -0.0, 2.0, 0.0])
    scales = np.array([np.nan, np.inf, -np.inf, -2.0, 1e-300, 3.0])
    cls = ColumnNormalizationParams if axis == 0 else RowNormalizationParams
    dimension = "ncols" if axis == 0 else "nrows"
    params = cls(6, centers=centers, scales=scales)
    centers[:] = 99
    scales[:] = 99
    expected_centers = np.array([np.nan, np.inf, -np.inf, -0.0, 2.0, 0.0])
    expected_scales = np.array([np.nan, np.inf, -np.inf, -2.0, 1e-300, 3.0])
    assert_array_equal(params.centers, expected_centers)
    assert_array_equal(params.scales, expected_scales)
    params.centers[:] = 99
    params.scales[:] = 99
    path = tmp_path / "ieee.npz"
    np.savez(path, **{dimension: 6, "centers": params.centers, "scales": params.scales})
    with np.load(path, allow_pickle=False) as saved:
        restored = cls(
            int(saved[dimension]), centers=saved["centers"], scales=saved["scales"]
        )
    raw = np.array([[1.0, 2.0, 3.0, 4.0, 2.0, 9.0], [0.0, 0.0, 0.0, 0.0, 2.0, 0.0]])
    with np.errstate(invalid="ignore"):
        expected = (raw - expected_centers) / expected_scales
    if axis == 1:
        raw = raw.T
        expected = expected.T
    z = LazyMatrix.from_normalization(storage(raw), restored)
    assert_array_equal(z.centers, expected_centers)
    assert_array_equal(z.scales, expected_scales)
    assert np.signbit(z.centers[3])
    assert_array_equal(z.toarray(), expected)
    assert_array_equal(z.T.toarray(), expected.T)
    for j in range(raw.shape[1]):
        assert_array_equal(z[:, j], expected[:, j])
    for j in range(raw.shape[0]):
        assert_array_equal(z.T[:, j], expected.T[:, j])


@pytest.mark.parametrize("center", ["min", "median"])
@pytest.mark.parametrize("scale", ["range", "quantile_range", "maxabs", "l1", "l2"])
def test_nan_statistics_propagate(center, scale):
    raw = np.array([[np.nan, 4.0], [1.0, 4.0], [0.0, 4.0]])
    for storage in STORAGE:
        z = LazyMatrix(storage(raw), center=center, scale=scale)
        assert np.isnan(z.centers[0])
        assert np.isnan(z.scales[0])
        assert z.scales[1] == 1
        assert_allclose(z.toarray()[:, 1], 0)
        reconstructed = z.normalization.transform(storage(raw))
        assert_array_equal(reconstructed.toarray(), z.toarray())


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("constructor", [LazyMatrix, normalize])
def test_zero_column_parameters(scale, constructor):
    for storage in STORAGE:
        x = storage(np.empty((3, 0)))
        z = constructor(x, center="median", scale=scale)
        assert z.normalization.ncols == 0
        assert z.normalization.transform(x).shape == (3, 0)


@pytest.mark.parametrize("storage", STORAGE)
def test_column_mismatch_is_recoverable(storage):
    for params in (
        ColumnNormalizationParams(3),
        ColumnNormalizationParams(3, scales=[1, 2, 3]),
    ):
        with pytest.raises(ValueError, match="column count"):
            params.transform(storage(np.ones((4, 2))))
        assert params.transform(storage(np.ones((4, 3)))).shape == (4, 3)


@pytest.mark.parametrize("zero", [0.0, -0.0])
@pytest.mark.parametrize(
    "cls, label",
    [(ColumnNormalizationParams, "column"), (RowNormalizationParams, "row")],
)
def test_explicit_zero_scales_are_rejected(zero, cls, label):
    with pytest.raises(ValueError, match=f"{label} 1"):
        cls(3, scales=[1, zero, 2])


@pytest.mark.parametrize("name", ["centers", "scales"])
@pytest.mark.parametrize("values", [[1], [1, 2, 3], [[1, 2]]])
@pytest.mark.parametrize("cls", [ColumnNormalizationParams, RowNormalizationParams])
def test_parameter_dimensions(name, values, cls):
    with pytest.raises(ValueError, match="length ncols|length nrows|one-dimensional"):
        cls(2, **{name: values})


@pytest.mark.parametrize("cls", [ColumnNormalizationParams, RowNormalizationParams])
def test_parameter_types(cls):
    with pytest.raises(ValueError, match="nonnegative"):
        cls(-1)
    for ncols in (1.5, "2", None):
        with pytest.raises(TypeError):
            cls(ncols)
    for value in ([1j], ["1"], [object()]):
        with pytest.raises(TypeError, match="real numeric"):
            cls(1, scales=value)
    with pytest.raises(TypeError, match="ColumnNormalizationParams"):
        LazyMatrix.from_normalization(np.ones((3, 2)), [1, 2])


@pytest.mark.parametrize("name", ["center", "scale"])
@pytest.mark.parametrize("constructor", [LazyMatrix, normalize])
def test_method_validation(name, constructor):
    for value in (1, None, ["mean"]):
        with pytest.raises(TypeError, match="boolean or method"):
            constructor(np.eye(2), **{name: value})
    with pytest.raises(ValueError, match="unknown"):
        constructor(np.eye(2), **{name: "unsupported"})


@pytest.mark.parametrize(
    "bounds",
    [
        (-0.1, 0.75),
        (0.25, 1.1),
        (0.5, 0.5),
        (0.9, 0.1),
        (np.nan, 0.75),
        (0.25, np.inf),
        (0.25,),
        (0, 0.5, 1),
        [[0, 1]],
    ],
)
@pytest.mark.parametrize("constructor", [LazyMatrix, normalize])
@pytest.mark.parametrize("scale", [True, "quantile_range"])
def test_quantile_validation(bounds, constructor, scale):
    with pytest.raises(ValueError, match="quantile"):
        constructor(np.eye(2), scale=scale, quantile_range=bounds)


@pytest.mark.parametrize("storage", STORAGE)
def test_frozen_rows_validate_row_count_and_preserve_inactive_dimension(storage):
    for params in (
        RowNormalizationParams(3),
        RowNormalizationParams(3, scales=[1, 2, 3]),
    ):
        with pytest.raises(ValueError, match="row count"):
            params.apply_frozen(storage(np.ones((4, 2))))
        z = params.apply_frozen(storage(np.ones((3, 2))))
        assert z.shape == (3, 2)
        assert z.normalization.nrows == 3
        assert_allclose(
            z.toarray(),
            np.ones((3, 2)) / np.array([1, 2, 3])[:, None]
            if params.scales is not None
            else np.ones((3, 2)),
        )


@pytest.mark.parametrize("storage", STORAGE)
@pytest.mark.parametrize("nrows", [1, 2, 3])
def test_row_recipe_fits_new_samples_independently(storage, nrows):
    recipe = RowNormalization(center=False, scale="l2")
    train = recipe.fit_apply(storage(np.array([[3.0, 4.0], [0.0, 0.0]])))
    raw = np.array([[0.0, 10.0], [6.0, 8.0], [-12.0, 5.0]])[:nrows]
    validation = recipe.fit_apply(storage(raw))
    assert isinstance(validation.normalization, RowNormalizationParams)
    assert validation.normalization.nrows == nrows
    assert_allclose(train.scales, [5.0, 1.0])
    assert_allclose(validation.scales, [10.0, 10.0, 13.0][:nrows])
    assert_allclose(validation.toarray(), raw / np.linalg.norm(raw, axis=1)[:, None])
    if nrows == 2:
        frozen = train.normalization.apply_frozen(storage(raw))
        assert_allclose(frozen.toarray(), raw / np.array([5.0, 1.0])[:, None])
        assert not np.allclose(frozen.toarray(), validation.toarray())


@pytest.mark.parametrize("storage", STORAGE)
@pytest.mark.parametrize(
    "cls, params_cls, axis",
    [
        (ColumnNormalization, ColumnNormalizationParams, 0),
        (RowNormalization, RowNormalizationParams, 1),
    ],
)
def test_recipes_fit_without_retaining_input(storage, cls, params_cls, axis):
    raw = np.array([[0.0, -1.0, 4.0], [2.0, 3.0, 0.0]])
    bounds = [0.1, 0.9]
    recipe = cls(center="median", scale="quantile_range", quantile_range=bounds)
    bounds[:] = [0.3, 0.4]
    assert recipe.quantile_range == (0.1, 0.9)
    with pytest.raises(AttributeError):
        recipe.center = False
    params = recipe.fit(storage(raw))
    assert isinstance(params, params_cls)
    expected, centers, scales = reference(
        raw, "median", "quantile_range", (0.1, 0.9), axis
    )
    raw[:] = 100
    assert_allclose(params.centers, centers)
    assert_allclose(params.scales, scales)
    source = storage(np.array([[0.0, -1.0, 4.0], [2.0, 3.0, 0.0]]))
    wrapped = params.transform(source) if axis == 0 else params.apply_frozen(source)
    assert_allclose(wrapped.toarray(), expected)


@pytest.mark.parametrize("cls", [ColumnNormalization, RowNormalization])
@pytest.mark.parametrize(
    "options, error",
    [
        ({"center": "invalid"}, ValueError),
        ({"scale": None}, TypeError),
        ({"df": -1}, ValueError),
        ({"quantile_range": (0.8, 0.2)}, ValueError),
    ],
)
def test_recipe_validation_is_immediate(cls, options, error):
    with pytest.raises(error):
        cls(**options)


@pytest.mark.parametrize("constructor", [LazyMatrix, normalize, standardize])
def test_axis_validation(constructor):
    for axis in (-1, 2):
        with pytest.raises(ValueError, match="axis"):
            constructor(np.eye(2), axis=axis)
    for axis in (True, np.bool_(False), 1.0, "rows", None, (0, 1)):
        with pytest.raises(TypeError):
            constructor(np.eye(2), axis=axis)
    assert isinstance(
        constructor(np.eye(2), axis=np.int64(1)).normalization, RowNormalizationParams
    )


@pytest.mark.parametrize("storage", STORAGE)
@pytest.mark.parametrize("constructor", [normalize, standardize])
def test_row_degrees_of_freedom_use_column_count(storage, constructor):
    raw = np.array([[1.0, 2.0, 3.0], [3.0, 5.0, 7.0]])
    z = constructor(storage(raw), axis=1, df=1)
    assert_allclose(z.scales, raw.std(axis=1, ddof=1))
    for df in (3, 4, np.iinfo(np.uintp).max):
        z = constructor(storage(raw), axis=1, df=df)
        assert np.isnan(z.scales).all()
        assert np.isnan(z.toarray()).all()
        unscaled = constructor(storage(raw), axis=1, scale=False, df=df)
        assert unscaled.scales is None
        assert_allclose(unscaled.toarray(), raw - raw.mean(axis=1)[:, None])


@pytest.mark.parametrize("storage", STORAGE)
@pytest.mark.parametrize("constructor", [normalize, standardize])
def test_empty_row_normalization_dimensions(storage, constructor):
    z = constructor(storage(np.empty((0, 3))), axis=1)
    assert z.shape == (0, 3)
    assert z.normalization.nrows == 0
    assert z.toarray().shape == (0, 3)
    assert_array_equal(z @ np.ones(3), np.empty(0))
    assert_array_equal(z.T @ np.empty(0), np.zeros(3))
    assert_array_equal(z[:, 0], np.empty(0))
    for shape in ((3, 0), (0, 0)):
        with pytest.raises(ValueError, match="column"):
            constructor(storage(np.empty(shape)), axis=1)
    rows = RowNormalizationParams(3).apply_frozen(storage(np.empty((3, 0))))
    assert rows.toarray().shape == (3, 0)
    columns = ColumnNormalizationParams(3).transform(storage(np.empty((0, 3))))
    assert columns.toarray().shape == (0, 3)
