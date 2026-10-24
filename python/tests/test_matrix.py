import importlib.metadata
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from lazymatrix import LazyMatrix, __version__, standardize
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from scipy.sparse.linalg import lsqr


@pytest.fixture(
    params=[
        np.asarray,
        sparse.csr_matrix,
        sparse.csc_matrix,
        sparse.csr_array,
        sparse.csc_array,
    ]
)
def matrix(request):
    return request.param(
        np.array([[1.0, 0.0, 5.0], [2.0, -3.0, 5.0], [0.0, 4.0, 5.0], [3.0, 0.0, 5.0]])
    )


def dense_reference(x, center=True, scale=True):
    raw = np.asarray(x.toarray() if sparse.issparse(x) else x, dtype=np.float64)
    scales = raw.std(axis=0) if scale else np.ones(raw.shape[1])
    scales[scales == 0] = 1
    centers = raw.mean(axis=0) if center else np.zeros(raw.shape[1])
    return (raw - centers) / scales


@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("scale", [False, True])
def test_normalization_and_products(matrix, center, scale):
    z = standardize(matrix, center=center, scale=scale)
    expected = dense_reference(matrix, center, scale)
    v = np.array([0.5, -2.0, 1.0])
    u = np.array([1.0, -0.25, 2.0, 0.0])
    b = np.column_stack([v, -v])
    a = np.vstack([u, -u])
    assert isinstance(z, LazyMatrix)
    assert z.shape == expected.shape
    assert z.T.shape == expected.T.shape
    assert z.ndim == 2
    assert z.dtype == np.dtype("float64")
    assert z.T.T is z
    assert_allclose(z.toarray(), expected)
    assert_allclose(z.T.toarray(), expected.T)
    assert_allclose(z @ v, expected @ v, atol=1e-14)
    assert_allclose(z.T @ u, expected.T @ u, atol=1e-14)
    assert_allclose(u @ z, u @ expected, atol=1e-14)
    assert_allclose(v @ z.T, v @ expected.T, atol=1e-14)
    assert_allclose(z @ b, expected @ b, atol=1e-14)
    assert_allclose(z.T @ a.T, expected.T @ a.T, atol=1e-14)
    assert_allclose(a @ z, a @ expected, atol=1e-14)
    assert_allclose(b.T @ z.T, b.T @ expected.T, atol=1e-14)
    assert_allclose(np.matmul(z, v), expected @ v, atol=1e-14)
    assert_allclose(np.matmul(u, z), u @ expected, atol=1e-14)
    assert_allclose(u @ (z @ v), (z.T @ u) @ v, atol=1e-14)
    for j in range(z.shape[1]):
        assert_allclose(z[:, j], expected[:, j])
    for j in range(z.shape[0]):
        assert_allclose(z.T[:, j], expected.T[:, j])
    assert_allclose(z[:, -1], expected[:, -1])
    assert (z.centers is None) == (not center)
    assert (z.scales is None) == (not scale)


def test_owned_storage_and_parameter_copies(matrix):
    z = standardize(matrix)
    expected = z.toarray()
    if sparse.issparse(matrix):
        matrix.data[:] = 100
        matrix.indices[:] = 0
    else:
        matrix[:] = 100
    centers = z.centers
    scales = z.scales
    centers[:] = 100
    scales[:] = 100
    materialized = z.toarray()
    materialized[:] = 100
    column = z[:, 0]
    column[:] = 100
    assert_allclose(z.toarray(), expected)


@pytest.mark.parametrize("order", ["C", "F", "strided", "reversed"])
def test_dense_storage_layouts(order):
    raw = np.arange(48.0).reshape(8, 6)
    x = {
        "C": raw,
        "F": np.asfortranarray(raw),
        "strided": raw[::2, ::2],
        "reversed": raw[::-1, ::-1],
    }[order]
    assert_allclose(standardize(x).toarray(), dense_reference(x))


@pytest.mark.parametrize(
    "dtype", [np.bool_, np.int32, np.int64, np.float32, np.float64]
)
def test_real_numeric_inputs_become_float64(dtype):
    x = np.array([[1, 0], [0, 1], [1, 1]], dtype=dtype)
    assert_allclose(standardize(x).toarray(), dense_reference(x))


def test_sparse_duplicates_unsorted_indices_and_explicit_zeros():
    x = sparse.csc_matrix(
        (
            np.array([3.0, 1.0, -1.0, 0.0, 4.0]),
            np.array([2, 0, 0, 1, 0]),
            np.array([0, 4, 5]),
        ),
        shape=(3, 2),
    )
    before = (x.data.copy(), x.indices.copy(), x.indptr.copy())
    expected = dense_reference(x)
    z = standardize(x)
    assert_allclose(z.toarray(), expected)
    assert_allclose(z @ np.ones(2), expected @ np.ones(2))
    for actual, original in zip((x.data, x.indices, x.indptr), before, strict=True):
        assert_array_equal(actual, original)


def test_sparse_products_do_not_materialize(monkeypatch):
    x = sparse.csr_matrix(np.eye(5))

    def fail(*args, **kwargs):
        raise AssertionError("unexpected dense materialization")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", fail)
    monkeypatch.setattr(sparse.csc_matrix, "toarray", fail)
    monkeypatch.setattr(LazyMatrix, "toarray", fail)
    z = standardize(x)
    assert_allclose(z @ np.ones(5), np.zeros(5), atol=1e-14)
    assert_allclose(z.T @ np.ones(5), np.zeros(5), atol=1e-14)
    assert_allclose(z[:, 0], np.array([2.0, -0.5, -0.5, -0.5, -0.5]))


def test_scipy_adapter_and_least_squares(matrix):
    z = standardize(matrix)
    expected = z.toarray()
    operator = z.aslinearoperator()
    rhs = np.array([1.0, -2.0, 0.5, 3.0])
    assert operator.shape == z.shape
    assert operator.dtype == z.dtype
    assert_allclose(operator @ np.ones(3), expected @ np.ones(3))
    assert_allclose(operator.matmat(np.ones((3, 1))), expected @ np.ones((3, 1)))
    assert_allclose(operator.matmat(np.eye(3)), expected)
    assert_allclose(operator.rmatmat(np.eye(4)), expected.T)
    assert_allclose(z.T.aslinearoperator() @ rhs, expected.T @ rhs)
    actual = lsqr(operator, rhs, atol=1e-12, btol=1e-12)[0]
    reference = np.linalg.lstsq(expected, rhs, rcond=None)[0]
    assert_allclose(actual, reference, atol=1e-12)


def test_empty_dimensions():
    for x in (np.empty((3, 0)), sparse.csc_matrix((3, 0))):
        z = standardize(x)
        assert z.toarray().shape == (3, 0)
        assert_array_equal(z @ np.empty(0), np.zeros(3))
        assert (z @ np.empty((0, 2))).shape == (3, 2)
        assert (np.empty((2, 3)) @ z).shape == (2, 0)
    with pytest.raises(ValueError, match="row"):
        standardize(np.empty((0, 2)))


def test_nonfinite_values_are_retained():
    x = np.array([[np.nan, np.inf, 1.0], [2.0, 3.0, 1.0]])
    for source in (x, sparse.csc_matrix(x)):
        z = standardize(source)
        assert np.isnan(z.centers[0])
        assert np.isinf(z.centers[1])
        assert np.isnan(z.scales[:2]).all()
        assert z.scales[2] == 1.0
        assert_allclose(z.toarray()[:, 2], np.zeros(2))


@pytest.mark.parametrize("x", [np.ones(3), np.ones((2, 2, 2))])
def test_invalid_matrix_dimensions(x):
    with pytest.raises(ValueError, match="two-dimensional"):
        standardize(x)


@pytest.mark.parametrize(
    "x",
    [
        np.ones((2, 2), dtype=complex),
        np.array([["1", "2"]]),
        np.array([[object()]], dtype=object),
        sparse.coo_matrix(np.eye(2)),
    ],
)
def test_unsupported_input_types(x):
    with pytest.raises(TypeError):
        standardize(x)


@pytest.mark.parametrize("index", [3, -4])
def test_column_bounds(index):
    with pytest.raises(IndexError):
        standardize(np.ones((4, 3)))[:, index]


@pytest.mark.parametrize(
    "index",
    [0, (0, 0), (slice(None), slice(None)), (slice(1, None), 0), (slice(None), True)],
)
def test_unsupported_indexing(index):
    with pytest.raises((TypeError, IndexError)):
        standardize(np.ones((4, 3)))[index]


def test_invalid_products_and_implicit_conversion():
    z = standardize(np.ones((4, 3)))
    for rhs in (np.ones(4), np.ones((4, 2)), np.ones((2, 2, 2)), np.array(1.0)):
        with pytest.raises(ValueError):
            z @ rhs
    with pytest.raises(ValueError):
        np.ones(3) @ z
    with pytest.raises(TypeError):
        z @ np.ones(3, dtype=complex)
    with pytest.raises(TypeError, match="toarray"):
        np.asarray(z)
    with pytest.raises(TypeError):
        np.sin(z)
    with pytest.raises(TypeError):
        np.sum(z)
    with pytest.raises(TypeError):
        z.T @ z
    with pytest.raises(TypeError):
        np.matmul(z, np.ones(3), out=np.empty(4))


def test_concurrent_products(matrix):
    z = standardize(matrix)
    v = np.array([1.0, -2.0, 0.5])
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: z @ v, range(12)))
    for result in results:
        assert_allclose(result, z.toarray() @ v)


def test_version_matches_installed_metadata():
    assert __version__ == importlib.metadata.version("lazymatrix")
