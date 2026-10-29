# Python

LazyMatrix for Python applies column normalization during matrix operations.
It accepts NumPy arrays and SciPy CSR/CSC matrices and sparse arrays. Install it
with:

```sh
pip install lazymatrix
```

Python 3.11 or newer is required. Prebuilt wheels cover Linux x86_64, macOS
on Intel and Apple silicon, and Windows x86_64. Building from source requires
Rust 1.89 or newer.

## Normalize and multiply

```python
import numpy as np
from scipy import sparse
from lazymatrix import standardize

X = sparse.csr_matrix([[1.0, 0.0], [2.0, 3.0], [0.0, 4.0]])
Z = standardize(X)

beta = np.array([1.0, -1.0])
prediction = Z @ beta
gradient = Z.T @ prediction
column = Z[:, 0]
```

`standardize` fits column means and population standard deviations immediately.
Pass `df=1` for sample standard deviations; `df=0` is the default. The variance
divisor is `nrows - df`, and scales are `NaN` when `nrows <= df`. `df` must be a
nonnegative integer and affects scaling independently of centering.
Pass `center=False` or `scale=False` to disable either component. Exact zero
scales become one, so constant columns do not cause division by zero.
Nonfinite values follow ordinary IEEE arithmetic; missing values are not ignored.

`Z.shape`, `Z.ndim`, and `Z.dtype` describe the logical matrix. `Z.centers` and
`Z.scales` return copies of the fitted parameters, or `None` when the component
is disabled. These parameters describe the original columns even for `Z.T`.

Both `@` and `np.matmul` support dense vectors and matrices on either side.
Multiple right-hand sides currently use repeated vector products. `Z[:, j]`
returns one normalized column as an independent dense array, including support
for negative column indices. Other slicing and elementwise operations are not
part of the initial API.

## Use SciPy solvers

The adapter exposes forward and transpose products without materializing the
normalized input:

```python
from scipy.sparse.linalg import lsqr

y = np.array([1.0, -2.0, 1.0])
coefficients = lsqr(Z.aslinearoperator(), y)[0]
```

## Memory and conversion

LazyMatrix owns a copy of the raw input, so later changes to `X` do not change
`Z`. Dense inputs require a full raw-data copy. Sparse copies remain sparse and
are stored in canonical CSC form, with duplicates summed before normalization.
All values and results use float64. Complex and nonnumeric inputs are rejected.

Products, transposes, and column reads do not construct a full normalized matrix.
Call `Z.toarray()` when you explicitly want a dense copy:

```python
dense = Z.toarray()
```

Implicit conversion with `np.asarray(Z)` raises an error. Existing estimators
that require a concrete NumPy array or SciPy sparse matrix need an explicit
conversion or their own operator integration.

## Choose normalization methods

`LazyMatrix` exposes all Rust column normalization choices. Centering and
scaling are independent:

| Option   | Values                     | Meaning                                     |
| -------- | -------------------------- | ------------------------------------------- |
| `center` | `False` or `"none"`        | No centering                                |
| `center` | `True` or `"mean"`         | Column mean                                 |
| `center` | `"min"`, `"median"`        | Column minimum or median                    |
| `scale`  | `False` or `"none"`        | No scaling                                  |
| `scale`  | `True` or `"sd"`           | Standard deviation using `nrows - df`       |
| `scale`  | `"range"`                  | Maximum minus minimum                       |
| `scale`  | `"quantile_range"`         | Upper minus lower linear quantile           |
| `scale`  | `"maxabs"`, `"l1"`, `"l2"` | Maximum absolute value, L1 norm, or L2 norm |

```python
from lazymatrix import LazyMatrix

min_max = LazyMatrix(X, center="min", scale="range")
robust = LazyMatrix(X, center="median", scale="quantile_range")
custom_robust = LazyMatrix(
    X, center="median", scale="quantile_range", quantile_range=(0.1, 0.9)
)
unit_columns = LazyMatrix(X, center=False, scale="l2")
```

When centering is active, maximum-absolute, L1, and L2 scales use the **centered**
columns, including implicit sparse zeros. Standard deviation, range, and
quantile range are invariant to centering. `df` affects only standard deviations.
These operations normalize columns; they do not normalize rows.

Quantile bounds are probabilities satisfying `0 <= lower < upper <= 1` and
default to `(0.25, 0.75)`, the interquartile range. Quantiles use exact linear
interpolation at `(nrows - 1) * q`, including implicit zeros and summed duplicate
cells. Fitting in Python uses the Rust dense or CSC backend with O(nrows) working
storage for one column, without constructing a full normalized matrix. The
Python binding converts CSR to canonical CSC as described above. Quantile fitting
does not calibrate scales to standard-normal variance. All methods replace only
exact zero scales with one and propagate NaNs; they do not skip missing values
or detect near-constant columns using a tolerance.

## Reuse fitted parameters

Fit once on training data and reuse the fitted centers and scales unchanged:

```python
from lazymatrix import NormalizationParams

params = robust.normalization
X_validation = sparse.csr_matrix([[4.0, 1.0], [0.0, -2.0]])
validation = params.transform(X_validation)
# Equivalent: LazyMatrix.from_normalization(X_validation, params)

restored = NormalizationParams(
    params.ncols, centers=params.centers, scales=params.scales
)
prediction = restored.transform(X_validation) @ np.ones(params.ncols)
```

`params.ncols` records the required feature count even when both components are
inactive. The input's column count must match, and its feature order must match
the training data. Parameter vectors must be one-dimensional with exactly
`ncols` entries. Invalid dimensions and explicit positive or negative zero scales
raise `ValueError`. Negative, tiny nonzero, and nonfinite values are preserved
in float64. Parameters and transformed matrices own copies, and array properties
return independent copies. Transposes export the original columns' parameters.

A constant feature gets a fitted scale of one. With mean, minimum, or median
centering it becomes zero on training data, but new values still subtract the
training center. Reusing range normalization does not clip new observations to
`[0, 1]`. Calling `LazyMatrix` or `standardize` again fits new statistics instead
of reusing training parameters. No method updates fitted statistics online.

The portable representation is an integer `ncols` and optional one-dimensional
float64 `centers` and `scales` arrays in feature order. Missing arrays mean the
component is inactive. The same fields reconstruct Rust's
`ColumnNormalizationParams::from_parts(ncols, centers, scales)` and supply PyTorch
buffers. Keep application-specific scale exponents in consumer code and pass the
resulting explicit scales to `NormalizationParams`.

Use a NumPy checkpoint to retain IEEE values without pickling or nonstandard
JSON numbers:

```python
checkpoint = {"ncols": params.ncols}
for name in ("centers", "scales"):
    values = getattr(params, name)
    if values is not None:
        checkpoint[name] = values
np.savez("normalization.npz", **checkpoint)

with np.load("normalization.npz", allow_pickle=False) as saved:
    restored = NormalizationParams(
        int(saved["ncols"]),
        centers=saved.get("centers"),
        scales=saved.get("scales"),
    )
```

Preserve inactive components as missing arrays, including when moving parameters
between languages. Replacing `None` with a zero center or unit scale can change
IEEE arithmetic with nonfinite values. Keep feature names or other application
metadata alongside the checkpoint if needed to verify feature order.

## Use a PyTorch first layer

Install the optional adapter with:

```sh
pip install "lazymatrix[torch]"
```

PyTorch 2.9 or newer is required. Install a compatible PyTorch build using
[PyTorch's installation instructions](https://pytorch.org/get-started/locally/)
when you need a particular CUDA version. The ordinary LazyMatrix installation
does not require or import PyTorch. The extra is available on platforms
supported by PyTorch.

`NormalizedLinear` applies fixed column normalization as part of a trainable
first layer. Fit statistics on training data once, then pass raw training,
validation, and prediction tensors through the same layer:

```python
import torch
from lazymatrix.torch import NormalizedLinear

# X and Z are the raw training matrix and its fitted normalization from above.
raw = X.tocsr()
training = torch.sparse_csr_tensor(
    torch.tensor(raw.indptr, dtype=torch.int64),
    torch.tensor(raw.indices, dtype=torch.int64),
    torch.tensor(raw.data, dtype=torch.float32),
    size=raw.shape,
)
layer = NormalizedLinear(
    X.shape[1], 8, centers=Z.centers, scales=Z.scales, dtype=torch.float32
)
model = torch.nn.Sequential(layer, torch.nn.ReLU(), torch.nn.Linear(8, 1))
prediction = model(training)
prediction.square().mean().backward()
```

The layer accepts dense tensors with shape `(*, in_features)` and
two-dimensional sparse COO or CSR tensors. Convert other sparse layouts
explicitly before calling it. COO duplicates are summed; CSR tensors must
satisfy PyTorch's storage invariants, including sorted, distinct column indices
within each row. When constructing tensors from SciPy, call `sum_duplicates()`
and `sort_indices()` first if the input is not canonical.

Inputs and the layer must share their dtype and device. Float32 and float64
are supported on CPU and CUDA. For example, move both `model` and `training`
with `.to("cuda")` before training on a GPU. Sparse features are fixed data:
sparse tensors with `requires_grad=True` are rejected. Dense-input, weight,
and bias gradients are supported. Mixed precision, autocast, and other devices
are outside the supported contract.

Weights use the same shape and initialization as `torch.nn.Linear` and remain
in normalized coordinates during optimization. Every forward pass derives
`A = weight / scales` and computes `X @ A.T - A @ centers + bias`. This avoids
constructing a normalized design matrix and keeps all training operations in
PyTorch. Pass `centers=None` or `scales=None` to disable either component.
`bias=False` disables the learned intercept while retaining the centering
correction.

Centers and scales are copied and detached into persistent buffers using the
layer's dtype and device. The default dtype follows PyTorch rather than the
supplied statistics. Explicit zero scales, including values that round to zero
in the selected dtype, raise `ValueError`. Negative scales and nonfinite
parameters are preserved. The layer never fits statistics or updates them in
training mode.

Save and restore the normalization together with the trained parameters:

```python
torch.save(layer.state_dict(), "first-layer.pt")
state = torch.load("first-layer.pt", weights_only=True, map_location="cpu")
restored = NormalizedLinear(
    X.shape[1],
    8,
    centers=state.get("centers"),
    scales=state.get("scales"),
    dtype=state["weight"].dtype,
)
restored.load_state_dict(state)
```

Reconstruct the same feature counts and bias setting. Supplying the saved
buffers enables the same normalization components without refitting or retaining
the original training matrix. Calling `reset_parameters()` resets weights and
bias while preserving the fitted normalization.

Factoring normalization changes floating-point evaluation order. Large offsets
can lose centered variation, and nonfinite values and sparse implicit zeros can
produce different IEEE results from explicit dense normalization. Sparse storage
does not guarantee faster training; the
[PyTorch benchmark](https://github.com/jolars/lazymatrix/tree/main/benches/torch_adapter)
measures forward/backward throughput and peak memory against a dense first layer.

See the [Python API reference](/reference/python/) for public classes and methods.

## Development and releases

From the repository root:

```sh
task python:check
task python:torch:check
task python:package
```

The first command builds an editable extension and runs formatting, linting, and
correctness checks, including the optional PyTorch adapter. The second runs just
the adapter checks. CUDA tests run when hardware is available and report skips
otherwise. The third builds a wheel and source distribution, installs the wheel
in an isolated environment, rebuilds from the source distribution, and tests both
installations with and without the PyTorch extra. Distribution checks use CPU
PyTorch wheels. The development environment lives in `.devenv/python`.

Python and Rust use independent versions and share one Versionary release PR.
Rust releases also release Python bindings. Python-only releases leave the Rust
version unchanged. Python tags use `python-v<version>` and Rust tags use
`v<version>`.
