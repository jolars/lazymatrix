# Python

LazyMatrix lets you multiply by a centered and scaled matrix without storing
its normalized entries. This is particularly useful for sparse input:
subtracting column means would ordinarily turn implicit zeros into stored
values and make the matrix dense.

Use NumPy arrays or SciPy matrices and sparse arrays in CSR or CSC format.
The optional PyTorch adapter applies fitted normalization in a trainable first
layer.

## Installation

```sh
pip install lazymatrix
```

Python 3.11 or newer is required. Prebuilt wheels are available for Linux
x86_64, macOS on Intel and Apple silicon, and Windows x86_64. Installing from
source also requires Rust 1.89 or newer.

## Normalize and multiply

Arrange your data with observations in rows and features in columns. Fit column
means and standard deviations with `standardize`:

```python
import numpy as np
from lazymatrix import standardize
from scipy import sparse

X = sparse.csr_matrix([[1.0, 0.0], [2.0, 3.0], [0.0, 4.0]])
Z = standardize(X)

beta = np.array([1.0, -1.0])
prediction = Z @ beta
transpose_product = Z.T @ prediction
```

`Z` has the same shape as `X`. Its products behave as though each column had
been centered and scaled, while its raw data remains sparse. You can pass a
NumPy array in place of `X` without changing the calls.

Vector products return NumPy arrays. Both `@` and `np.matmul` also accept dense
matrices on either side, so you can multiply by several coefficient vectors at
once. `Z.T` shares the same raw storage and fitted normalization.

Standardization uses population standard deviations by default. Pass `df=1`
to use sample standard deviations, `center=False` to disable centering, or
`scale=False` to disable scaling.

## Use SciPy solvers

Expose the normalized matrix as a SciPy `LinearOperator` to use iterative
solvers. For least squares with an intercept, center the response, solve for
the coefficients, and add the response mean back to predictions:

```python
from scipy.sparse.linalg import lsqr

y = np.array([1.0, 3.0, 2.0])
intercept = y.mean()
coefficients = lsqr(Z.aslinearoperator(), y - intercept)[0]
prediction = intercept + Z @ coefficients
```

The solver uses forward and transpose products without creating a normalized
design matrix. The coefficients describe normalized features; keep the fitted
normalization alongside them for prediction.

## Reuse fitted parameters

Apply the training centers and scales to validation or prediction data:

```python
params = Z.normalization
X_validation = sparse.csr_matrix([[4.0, 1.0], [0.0, -2.0]])
validation = params.transform(X_validation)
validation_prediction = intercept + validation @ coefficients
```

`transform` creates a lazy operator using the saved parameters. It does not fit
statistics on the new data. The new input must have `params.ncols` columns in
the same feature order as the training input, but it can have a different
number of rows and use dense or sparse storage.

Keep using `params.transform` for new observations. Calling `standardize` or
`LazyMatrix` on them would fit new statistics. The parameter object owns its
centers and scales independently of the training matrix.

## Choose normalization methods

Use `LazyMatrix` to choose a different centering or scaling method:

```python
from lazymatrix import LazyMatrix

min_max = LazyMatrix(X, center="min", scale="range")
robust = LazyMatrix(X, center="median", scale="quantile_range")
unit_columns = LazyMatrix(X, center=False, scale="l2")
```

| Goal            | Centering | Scaling               |
| --------------- | --------- | --------------------- |
| Standardization | Mean      | Standard deviation    |
| Min-max scaling | Minimum   | Maximum minus minimum |
| Robust scaling  | Median    | Interquartile range   |
| Unit L2 norm    | None      | L2 norm               |

Centering and scaling are independent. Other scaling choices include the L1
norm (`"l1"`) and maximum absolute value (`"maxabs"`). When centering is active,
norms and maximum absolute values use the centered columns, including implicit
sparse zeros.

Robust scaling uses the 25th and 75th percentiles by default. Pass
`quantile_range=(0.1, 0.9)` for the 10th-to-90th percentile range. Bounds are
probabilities satisfying `0 <= lower < upper <= 1`. Quantiles use exact linear
interpolation, and quantile-range scaling does not target unit variance.

Min-max scaling maps finite, nonconstant training columns to `[0, 1]`. New
observations can fall outside that interval; reusing the parameters does not
clip them. See the [API reference](/reference/python/) for all method names and
constructor options.

## Memory and conversion

LazyMatrix owns a copy of the raw input, so later changes to `X` do not affect
`Z`. Dense input requires a full raw-data copy. Sparse input remains sparse and
is stored in canonical CSC form, with duplicate entries summed. Stored values
and results use float64.

Products and transposes do not create a full normalized matrix. To inspect one
column or explicitly materialize the matrix:

```python
column = Z[:, 0]
dense = Z.toarray()
```

Both results are independent NumPy arrays. Materializing a sparse input can
require much more memory than its stored data. `Z.centers` and `Z.scales` also
return independent arrays, or `None` when the component is disabled.

Implicit conversion with `np.asarray(Z)` raises an error. General slicing,
mutation, and elementwise NumPy operations are unsupported. An estimator that
requires a concrete array needs explicit conversion or its own operator
integration.

## Save fitted normalization

Save the feature count and active parameter vectors in a NumPy archive. This
keeps disabled components absent and preserves floating-point values without
pickling:

```python
from lazymatrix import NormalizationParams

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

Use `restored.transform` with new raw data. Keep feature names or other metadata
alongside the archive so you can check feature order when loading it.
`NormalizationParams` also accepts explicit centers and scales, with one entry
per feature. An omitted vector disables that component; explicit zero scales
raise `ValueError`.

## Use a PyTorch first layer

Install the optional adapter:

```sh
pip install "lazymatrix[torch]"
```

PyTorch 2.9 or newer is required. Use [PyTorch's installation
instructions](https://pytorch.org/get-started/locally/) when you need a particular
CUDA build. The base LazyMatrix package does not require or import PyTorch.

`NormalizedLinear` applies saved normalization as part of a trainable linear
layer. Pass raw feature tensors to the layer, using the centers and scales
fitted on training data. For the sparse matrix above:

```python
import torch
from lazymatrix.torch import NormalizedLinear

raw = X.tocsr(copy=True)
raw.sum_duplicates()
raw.sort_indices()
training = torch.sparse_csr_tensor(
    torch.tensor(raw.indptr, dtype=torch.int64),
    torch.tensor(raw.indices, dtype=torch.int64),
    torch.tensor(raw.data, dtype=torch.float32),
    size=raw.shape,
)
layer = NormalizedLinear(
    X.shape[1], 8, centers=params.centers, scales=params.scales, dtype=torch.float32
)
model = torch.nn.Sequential(layer, torch.nn.ReLU(), torch.nn.Linear(8, 1))
target = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
loss = torch.nn.functional.mse_loss(model(training), target)
loss.backward()
```

The layer also accepts dense tensors and sparse COO tensors. CSR input must
have sorted, distinct column indices within each row, as prepared above.
Sparse inputs must not require gradients; weights and biases remain trainable
in normalized coordinates. Centers and scales are fixed buffers and do not
change in training mode.

Float32 and float64 are supported on CPU and CUDA. Inputs and the model must
share their dtype and device. Move both with `.to("cuda")` for GPU training.
Mixed precision and autocast are unsupported.

### Save a layer

The layer's `state_dict` includes its normalization. Supply the saved buffers
when reconstructing the layer, then load its weights and bias:

```python
torch.save(layer.state_dict(), "first-layer.pt")
state = torch.load("first-layer.pt", weights_only=True, map_location="cpu")
restored_layer = NormalizedLinear(
    X.shape[1],
    8,
    centers=state.get("centers"),
    scales=state.get("scales"),
    dtype=state["weight"].dtype,
)
restored_layer.load_state_dict(state)
```

Reconstruct the same feature counts and bias setting. Normalization moves with
the layer and is restored without refitting or retaining the training matrix.

## Numerical conventions

Computed scales become one only when they are exactly zero. With centering,
a constant training feature therefore becomes zero. New observations still
subtract the saved training center.

Standard deviations use `nrows - df` as the variance divisor. Scales are `NaN`
when `nrows <= df`. Missing values are not ignored or imputed, and nonfinite
values follow IEEE arithmetic.

The PyTorch layer factors normalization into the linear operation. This changes
floating-point evaluation order, so large offsets, nonfinite values, and sparse
implicit zeros can produce different results from explicit dense normalization.
The [PyTorch benchmark](https://github.com/jolars/lazymatrix/tree/main/benches/torch_adapter)
compares training throughput and memory with a dense first layer.

See [how it works](/guide/how-it-works) for the normalization algebra and
[the Python API reference](/reference/python/) for the complete interface.
