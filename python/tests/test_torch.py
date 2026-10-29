import importlib
import io
import os

import numpy as np
import pytest
from lazymatrix import LazyMatrix, NormalizationParams, standardize
from scipy import sparse

if os.environ.get("LAZYMATRIX_REQUIRE_TORCH") == "1":
    torch = importlib.import_module("torch")
else:
    torch = pytest.importorskip("torch", minversion="2.9")

NormalizedLinear = importlib.import_module("lazymatrix.torch").NormalizedLinear

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"),
    ),
]
LAYOUTS = ["dense", "coo", "csr"]


@pytest.fixture(autouse=True, scope="module")
def torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def input_tensor(dtype, device):
    return torch.tensor(
        [
            [0.0, 2.0, -1.0, 5.0, 0.0],
            [3.0, 0.0, 2.0, 5.0, 0.0],
            [-2.0, 0.0, 0.0, 5.0, 0.0],
            [0.0, 1.0, 3.0, 5.0, 0.0],
        ],
        dtype=dtype,
        device=device,
    )


def as_layout(values, layout):
    if layout == "coo":
        return values.to_sparse_coo()
    if layout == "csr":
        return values.to_sparse_csr()
    return values


def normalized(values, centers, scales):
    if centers is not None:
        values = values - centers
    if scales is not None:
        values = values / scales
    return values


def reference_layer(layer):
    reference = torch.nn.Linear(
        layer.in_features,
        layer.out_features,
        bias=layer.bias is not None,
        dtype=layer.weight.dtype,
        device=layer.weight.device,
    )
    with torch.no_grad():
        reference.weight.copy_(layer.weight)
        if layer.bias is not None:
            reference.bias.copy_(layer.bias)
    return reference


def assert_close(actual, expected):
    tolerance = (
        {"rtol": 2e-5, "atol": 2e-6}
        if actual.dtype == torch.float32
        else {"rtol": 1e-12, "atol": 1e-12}
    )
    torch.testing.assert_close(actual, expected, **tolerance)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("scale", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_predictions_gradients_and_training(device, dtype, layout, center, scale, bias):
    torch.manual_seed(1427)
    raw = input_tensor(dtype, device)
    layer = NormalizedLinear(
        5,
        3,
        bias,
        centers=[0.5, -1.0, 2.0, 5.0, 0.0] if center else None,
        scales=[1.5, -2.0, 0.5, 1.0, 1.0] if scale else None,
        dtype=dtype,
        device=device,
    )
    reference = reference_layer(layer)
    source = as_layout(raw, layout)
    dense = raw.clone()
    if layout == "dense":
        source.requires_grad_()
        dense.requires_grad_()
    actual_optimizer = torch.optim.SGD(layer.parameters(), lr=0.01, weight_decay=0.03)
    reference_optimizer = torch.optim.SGD(
        reference.parameters(), lr=0.01, weight_decay=0.03
    )
    target = torch.arange(12, dtype=dtype, device=device).reshape(4, 3) / 10
    for _ in range(2):
        actual_optimizer.zero_grad()
        reference_optimizer.zero_grad()
        if layout == "dense":
            source.grad = dense.grad = None
        actual = layer(source)
        expected = reference(normalized(dense, layer.centers, layer.scales))
        assert_close(actual, expected)
        (actual - target).square().mean().backward()
        (expected - target).square().mean().backward()
        assert_close(layer.weight.grad, reference.weight.grad)
        if bias:
            assert_close(layer.bias.grad, reference.bias.grad)
        if layout == "dense":
            assert_close(source.grad, dense.grad)
        actual_optimizer.step()
        reference_optimizer.step()
        assert_close(layer.weight, reference.weight)
        if bias:
            assert_close(layer.bias, reference.bias)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    "method", [None, "range", "quantile_range", "maxabs", "l1", "l2"]
)
def test_existing_fitted_parameters_are_reused_on_new_data(layout, method):
    raw = input_tensor(torch.float64, "cpu")
    source = sparse.csr_matrix(raw.numpy())
    fitted = (
        standardize(source, df=1)
        if method is None
        else LazyMatrix(source, center="median", scale=method)
    )
    params = fitted.normalization
    restored = NormalizationParams(
        params.ncols, centers=params.centers, scales=params.scales
    )
    layer = NormalizedLinear(
        5, 2, centers=restored.centers, scales=restored.scales, dtype=torch.float64
    )
    reference = reference_layer(layer)
    for values in (raw, raw * 2 + 1):
        actual = layer(as_layout(values, layout))
        expected = reference(normalized(values, layer.centers, layer.scales))
        assert_close(actual, expected)
    assert_close(layer.scales[-2:], torch.ones(2, dtype=torch.float64))


@pytest.mark.parametrize("device", DEVICES)
def test_duplicate_coo_entries_and_explicit_zeros(device):
    source = torch.sparse_coo_tensor(
        [[2, 0, 0, 1, 0], [1, 0, 0, 0, 1]],
        [4.0, 3.0, -1.0, 0.0, -2.0],
        (3, 2),
        dtype=torch.float64,
        device=device,
    )
    assert not source.is_coalesced()
    layer = NormalizedLinear(
        2, 3, centers=[2.0, -1.0], scales=[2.0, 0.5], dtype=torch.float64, device=device
    )
    reference = reference_layer(layer)
    dense = source.to_dense()
    actual = layer(source)
    expected = reference(normalized(dense, layer.centers, layer.scales))
    assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    assert_close(layer.weight.grad, reference.weight.grad)
    assert_close(layer.bias.grad, reference.bias.grad)
    assert not source.is_coalesced()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("shape", [(0, 5, 3), (3, 0, 2), (4, 5, 0), (4, 5, 3)])
def test_empty_dimensions_and_all_zero_inputs(device, layout, shape):
    nrows, ncols, outputs = shape
    source = torch.zeros(nrows, ncols, dtype=torch.float64, device=device)
    layer = NormalizedLinear(
        ncols,
        outputs,
        centers=torch.ones(ncols),
        scales=torch.full((ncols,), 2.0),
        dtype=torch.float64,
        device=device,
    )
    reference = reference_layer(layer)
    actual = layer(as_layout(source, layout))
    expected = reference(normalized(source, layer.centers, layer.scales))
    assert actual.shape == (nrows, outputs)
    assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    assert_close(layer.weight.grad, reference.weight.grad)
    assert_close(layer.bias.grad, reference.bias.grad)


@pytest.mark.parametrize("shape", [(5,), (2, 3, 5), (4, 0, 5)])
def test_dense_leading_dimensions_and_strides(shape):
    raw = torch.arange(np.prod(shape), dtype=torch.float64).reshape(shape)
    if len(shape) > 1:
        raw = raw.transpose(0, 1)
    layer = NormalizedLinear(
        5, 3, centers=torch.arange(5), scales=torch.arange(1, 6), dtype=torch.float64
    )
    expected = reference_layer(layer)(normalized(raw, layer.centers, layer.scales))
    assert_close(layer(raw), expected)


@pytest.mark.parametrize("layout", ["coo", "csr"])
def test_sparse_forward_and_backward_do_not_densify(monkeypatch, layout):
    source = as_layout(input_tensor(torch.float64, "cpu"), layout)
    layer = NormalizedLinear(
        5, 3, centers=torch.ones(5), scales=torch.full((5,), 2.0), dtype=torch.float64
    )

    def fail(*args, **kwargs):
        raise AssertionError("unexpected dense materialization")

    monkeypatch.setattr(torch.Tensor, "to_dense", fail)
    layer(source).square().sum().backward()
    assert layer.weight.grad is not None


@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("scale", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_checkpoint_round_trip(center, scale, bias):
    layer = NormalizedLinear(
        5,
        3,
        bias,
        centers=torch.arange(5) if center else None,
        scales=torch.arange(1, 6) if scale else None,
        dtype=torch.float64,
    )
    checkpoint = io.BytesIO()
    torch.save(layer.state_dict(), checkpoint)
    checkpoint.seek(0)
    state = torch.load(checkpoint, weights_only=True)
    restored = NormalizedLinear(
        5,
        3,
        bias,
        centers=state.get("centers"),
        scales=state.get("scales"),
        dtype=torch.float64,
    )
    restored.load_state_dict(state)
    raw = input_tensor(torch.float64, "cpu").to_sparse_csr()
    assert_close(layer(raw), restored(raw))
    assert set(state) == (
        {"weight"}
        | ({"bias"} if bias else set())
        | ({"centers"} if center else set())
        | ({"scales"} if scale else set())
    )


def test_buffers_are_fixed_owned_and_move_with_the_layer():
    centers = torch.arange(5, dtype=torch.float64, requires_grad=True)
    scales = np.arange(1, 6, dtype=np.float64)
    layer = NormalizedLinear(5, 3, centers=centers, scales=scales, dtype=torch.float32)
    raw = input_tensor(torch.float32, "cpu")
    expected = layer(raw).detach()
    with torch.no_grad():
        centers.fill_(100)
    scales[:] = 100
    assert_close(layer(raw), expected)
    assert set(dict(layer.named_parameters())) == {"weight", "bias"}
    assert set(dict(layer.named_buffers())) == {"centers", "scales"}
    assert not layer.centers.requires_grad
    assert not layer.scales.requires_grad
    saved = {name: value.clone() for name, value in layer.named_buffers()}
    for mode in (True, False):
        layer.train(mode)
        layer(raw).sum().backward()
    layer.reset_parameters()
    for name, value in layer.named_buffers():
        assert_close(value, saved[name])
    layer.double()
    assert (
        layer.centers.dtype == layer.scales.dtype == layer.weight.dtype == torch.float64
    )
    layer(raw.double())


def test_initialization_matches_linear_and_dtype_follows_pytorch():
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(205)
            reference = torch.nn.Linear(2, 3)
            torch.manual_seed(205)
            layer = NormalizedLinear(
                2, 3, centers=np.ones(2, dtype=np.float32), scales=[1, 2]
            )
        assert layer.weight.dtype == layer.centers.dtype == layer.scales.dtype
        assert layer.weight.dtype == torch.float64
        assert_close(layer.weight, reference.weight)
        assert_close(layer.bias, reference.bias)
        assert "center=True, scale=True" in repr(layer)
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("device", DEVICES)
def test_device_movement(device):
    layer = NormalizedLinear(5, 3, centers=np.arange(5), scales=np.ones(5)).to(device)
    raw = input_tensor(torch.float32, device)
    result = layer(raw.to_sparse_csr())
    assert result.device == layer.weight.device == layer.centers.device
    assert layer.scales.device == layer.weight.device
    if device == "cuda":
        with pytest.raises(ValueError, match="dtype and device"):
            layer(raw.cpu())
    assert_close(layer.cpu()(raw.cpu()), result.cpu())


@pytest.mark.parametrize("name", ["centers", "scales"])
@pytest.mark.parametrize("values", [1.0, [1.0], [[1.0, 2.0]]])
def test_parameter_shapes(name, values):
    with pytest.raises(ValueError, match="one entry"):
        NormalizedLinear(2, 3, **{name: values})


@pytest.mark.parametrize("values", [[1.0, 0.0], [-0.0, 1.0], [1.0, 1e-100]])
def test_zero_scales_in_the_layer_dtype_are_rejected(values):
    with pytest.raises(ValueError, match="column .* nonzero"):
        NormalizedLinear(2, 3, scales=np.array(values), dtype=torch.float32)


@pytest.mark.parametrize("name", ["centers", "scales"])
def test_parameters_require_dense_real_values(name):
    for values in ([1j, 2j], torch.ones(2).to_sparse()):
        with pytest.raises(TypeError, match="dense real"):
            NormalizedLinear(2, 3, **{name: values})


def test_negative_tiny_and_nonfinite_parameters_are_preserved():
    centers = torch.tensor([float("nan"), float("inf"), -float("inf"), 0.0, 1.0])
    scales = torch.tensor(
        [-2.0, float("inf"), -float("inf"), float("nan"), 1e-200], dtype=torch.float64
    )
    layer = NormalizedLinear(5, 3, centers=centers, scales=scales, dtype=torch.float64)
    torch.testing.assert_close(layer.centers, centers.double(), equal_nan=True)
    torch.testing.assert_close(layer.scales, scales, equal_nan=True)
    assert torch.isnan(layer(input_tensor(torch.float64, "cpu"))).all()


def test_disabled_centering_does_not_add_a_zero_correction():
    layer = NormalizedLinear(2, 1, bias=False, scales=[float("inf"), 1.0])
    with torch.no_grad():
        layer.weight.fill_(1.0)
    torch.testing.assert_close(layer(torch.tensor([[2.0, 3.0]])), torch.tensor([[3.0]]))
    layer = NormalizedLinear(1, 1, bias=False)
    with torch.no_grad():
        layer.weight.fill_(float("inf"))
    assert torch.isinf(layer(torch.ones(1, 1))).all()


@pytest.mark.parametrize(
    "dtype", [torch.int64, torch.float16, torch.bfloat16, torch.cfloat]
)
def test_unsupported_dtypes(dtype):
    with pytest.raises(TypeError, match="float32 and float64"):
        NormalizedLinear(2, 3, dtype=dtype)
    with pytest.raises(TypeError, match="float32 and float64"):
        NormalizedLinear(2, 3)(torch.ones(3, 2, dtype=dtype))


def test_invalid_inputs_and_devices():
    layer = NormalizedLinear(2, 3)
    with pytest.raises(TypeError, match="PyTorch tensor"):
        layer(np.ones((3, 2)))
    for values in (torch.tensor(1.0), torch.ones(3), torch.ones(3, 3)):
        with pytest.raises(ValueError, match="last dimension"):
            layer(values)
    with pytest.raises(ValueError, match="dtype and device"):
        layer(torch.ones(3, 2, dtype=torch.float64))
    with pytest.raises(ValueError, match="CPU and CUDA"):
        NormalizedLinear(2, 3, device="meta")
    for dims in ((-1, 2), (2, -1)):
        with pytest.raises(ValueError, match="nonnegative"):
            NormalizedLinear(*dims)
    source = torch.ones(3, 2)
    for values in (source.to_sparse_csc(), source.to_sparse_bsr((1, 1))):
        with pytest.raises(TypeError, match="COO.*CSR"):
            layer(values)
    with pytest.raises(TypeError, match="two sparse dimensions"):
        layer(source.to_sparse(1))
    with pytest.raises(ValueError, match="two-dimensional"):
        layer(torch.ones(2, 3, 2).to_sparse())


@pytest.mark.parametrize("layout", ["coo", "csr"])
def test_sparse_input_gradients_are_rejected(layout):
    source = as_layout(torch.ones(3, 2), layout).requires_grad_()
    with pytest.raises(ValueError, match="must not require gradients"):
        NormalizedLinear(2, 3)(source)
