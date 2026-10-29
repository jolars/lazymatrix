import numpy as np
import pytest
from lazymatrix import LazyMatrix, NormalizationParams, standardize
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


def reference(raw, center, scale, bounds=(0.25, 0.75)):
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
def test_methods_products_and_reuse(storage, center, scale):
    raw = np.array(
        [
            [0.0, -6.0, 5.0, 0.0],
            [2.0, 0.0, 5.0, 0.0],
            [0.0, 3.0, 5.0, 0.0],
            [9.0, 0.0, 5.0, 0.0],
        ]
    )
    expected, centers, scales = reference(raw, center, scale)
    z = LazyMatrix(storage(raw), center=center, scale=scale)
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
    assert fitted.ncols == raw.shape[1]
    reconstructed = NormalizationParams(
        fitted.ncols, centers=fitted.centers, scales=fitted.scales
    )
    new = np.array([[10.0, -8.0, 8.0, 1.0], [-1.0, 1.0, 4.0, 0.0]])
    prediction = new.copy()
    if centers is not None:
        prediction -= centers
    if scales is not None:
        prediction /= scales
    reused = reconstructed.transform(storage(new))
    assert_allclose(reused.toarray(), prediction, atol=1e-14)
    assert_allclose(reused @ v, prediction @ v, atol=1e-14)
    assert_allclose(
        reused.T @ np.array([1.0, -2.0]), prediction.T @ [1.0, -2.0], atol=1e-14
    )
    assert_optional(reused.centers, centers)
    assert_optional(reused.scales, scales)


@pytest.mark.parametrize("bounds", [(0.0, 1.0), (0.1, 0.8), (0.4, 0.6)])
@pytest.mark.parametrize("storage", STORAGE)
def test_custom_exact_quantile_ranges(storage, bounds):
    raw = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 5.0], [4.0, 12.0], [10.0, 100.0]])
    z = LazyMatrix(
        storage(raw), center="median", scale="quantile_range", quantile_range=bounds
    )
    expected, centers, scales = reference(raw, "median", "quantile_range", bounds)
    assert_allclose(z.centers, centers)
    assert_allclose(z.scales, scales)
    assert_allclose(z.toarray(), expected)


@pytest.mark.parametrize("center", ["min", "median"])
@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("orientation", ["csr", "csc"])
def test_duplicates_are_combined_before_nonlinear_statistics(
    center, scale, orientation
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
    expected, centers, scales = reference(raw, center, scale)
    z = LazyMatrix(x, center=center, scale=scale)
    assert_allclose(z.toarray(), expected, atol=1e-14)
    assert_allclose(z.centers, centers)
    assert_optional(z.scales, scales)
    for actual, original in zip((x.data, x.indices, x.indptr), before, strict=True):
        assert_array_equal(actual, original)


@pytest.mark.parametrize("scale", SCALES)
def test_sparse_fitting_and_reuse_do_not_materialize(monkeypatch, scale):
    x = sparse.csr_matrix(np.eye(5))
    expected, _, _ = reference(np.eye(5), "median", scale)

    def fail(*args, **kwargs):
        raise AssertionError("unexpected dense materialization")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", fail)
    monkeypatch.setattr(sparse.csc_matrix, "toarray", fail)
    monkeypatch.setattr(LazyMatrix, "toarray", fail)
    z = LazyMatrix(x, center="median", scale=scale)
    reused = z.normalization.transform(x)
    for operator in (z, reused):
        assert_allclose(operator @ np.ones(5), expected @ np.ones(5), atol=1e-14)
        assert_allclose(operator.T @ np.ones(5), expected.T @ np.ones(5), atol=1e-14)
        assert_allclose(operator[:, 0], expected[:, 0])


@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("scale", [False, True])
def test_portable_checkpoint_round_trip(tmp_path, center, scale):
    z = standardize(np.array([[1.0, 5.0], [3.0, 5.0]]), center=center, scale=scale)
    fitted = z.normalization
    arrays = {"ncols": fitted.ncols}
    for name in ("centers", "scales"):
        value = getattr(fitted, name)
        if value is not None:
            arrays[name] = value
    checkpoint = tmp_path / "normalization.npz"
    np.savez(checkpoint, **arrays)
    with np.load(checkpoint, allow_pickle=False) as saved:
        restored = NormalizationParams(
            int(saved["ncols"]),
            centers=saved.get("centers"),
            scales=saved.get("scales"),
        )
    new = np.array([[7.0, 8.0]])
    expected = new.copy()
    if center:
        expected -= z.centers
    if scale:
        expected /= z.scales
    assert_allclose(restored.transform(new).toarray(), expected)


@pytest.mark.parametrize("storage", STORAGE)
def test_explicit_parameters_preserve_ieee_values_and_ownership(storage, tmp_path):
    centers = np.array([np.nan, np.inf, -np.inf, -0.0, 2.0, 0.0])
    scales = np.array([np.nan, np.inf, -np.inf, -2.0, 1e-300, 3.0])
    params = NormalizationParams(6, centers=centers, scales=scales)
    centers[:] = 99
    scales[:] = 99
    expected_centers = np.array([np.nan, np.inf, -np.inf, -0.0, 2.0, 0.0])
    expected_scales = np.array([np.nan, np.inf, -np.inf, -2.0, 1e-300, 3.0])
    assert_array_equal(params.centers, expected_centers)
    assert_array_equal(params.scales, expected_scales)
    params.centers[:] = 99
    params.scales[:] = 99
    path = tmp_path / "ieee.npz"
    np.savez(path, ncols=params.ncols, centers=params.centers, scales=params.scales)
    with np.load(path, allow_pickle=False) as saved:
        restored = NormalizationParams(
            int(saved["ncols"]), centers=saved["centers"], scales=saved["scales"]
        )
    raw = np.array([[1.0, 2.0, 3.0, 4.0, 2.0, 9.0], [0.0, 0.0, 0.0, 0.0, 2.0, 0.0]])
    z = LazyMatrix.from_normalization(storage(raw), restored)
    with np.errstate(invalid="ignore"):
        expected = (raw - expected_centers) / expected_scales
    assert_array_equal(z.centers, expected_centers)
    assert_array_equal(z.scales, expected_scales)
    assert np.signbit(z.centers[3])
    assert_array_equal(z.toarray(), expected)
    assert_array_equal(z.T.toarray(), expected.T)
    for j in range(6):
        assert_array_equal(z[:, j], expected[:, j])


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
def test_zero_column_parameters(scale):
    for storage in STORAGE:
        x = storage(np.empty((3, 0)))
        z = LazyMatrix(x, center="median", scale=scale)
        assert z.normalization.ncols == 0
        assert z.normalization.transform(x).shape == (3, 0)


@pytest.mark.parametrize("storage", STORAGE)
def test_column_mismatch_is_recoverable(storage):
    for params in (NormalizationParams(3), NormalizationParams(3, scales=[1, 2, 3])):
        with pytest.raises(ValueError, match="column count"):
            params.transform(storage(np.ones((4, 2))))
        assert params.transform(storage(np.ones((4, 3)))).shape == (4, 3)


@pytest.mark.parametrize("zero", [0.0, -0.0])
def test_explicit_zero_scales_are_rejected(zero):
    with pytest.raises(ValueError, match="column 1"):
        NormalizationParams(3, scales=[1, zero, 2])


@pytest.mark.parametrize("name", ["centers", "scales"])
@pytest.mark.parametrize("values", [[1], [1, 2, 3], [[1, 2]]])
def test_parameter_dimensions(name, values):
    with pytest.raises(ValueError, match="length ncols|one-dimensional"):
        NormalizationParams(2, **{name: values})


def test_parameter_types():
    with pytest.raises(ValueError, match="nonnegative"):
        NormalizationParams(-1)
    for ncols in (1.5, "2", None):
        with pytest.raises(TypeError):
            NormalizationParams(ncols)
    for value in ([1j], ["1"], [object()]):
        with pytest.raises(TypeError, match="real numeric"):
            NormalizationParams(1, scales=value)
    with pytest.raises(TypeError, match="NormalizationParams"):
        LazyMatrix.from_normalization(np.ones((3, 2)), [1, 2])


@pytest.mark.parametrize("name", ["center", "scale"])
def test_method_validation(name):
    for value in (1, None, ["mean"]):
        with pytest.raises(TypeError, match="boolean or method"):
            LazyMatrix(np.eye(2), **{name: value})
    with pytest.raises(ValueError, match="unknown"):
        LazyMatrix(np.eye(2), **{name: "unsupported"})


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
def test_quantile_validation(bounds):
    with pytest.raises(ValueError, match="quantile"):
        LazyMatrix(np.eye(2), scale="quantile_range", quantile_range=bounds)
