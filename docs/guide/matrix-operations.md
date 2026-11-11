# Using normalized matrices

A fitted operator behaves like the normalized matrix during multiplication.
Start with an operator over a small matrix:

::: code-group

```rust [Rust]
use lazymatrix::{ColumnNormalization, LazyMatrix, MatTransposeVec, MatVec};
use ndarray::array;

let x = array![[1.0, 0.0], [2.0, 3.0], [0.0, 4.0]];
let lazy = LazyMatrix::new(x.view(), ColumnNormalization::standardize()).unwrap();
```

```python [Python]
import numpy as np
from lazymatrix import standardize
from scipy import sparse

X = sparse.csr_matrix([[1.0, 0.0], [2.0, 3.0], [0.0, 4.0]])
Z = standardize(X)
```

:::

## Forward and transpose products

A forward product maps one coefficient per column to one value per row. A
transpose product maps one value per row back to one value per column:

::: code-group

```rust [Rust]
let coefficients = array![1.0, -1.0];
let prediction = lazy.matvec(&coefficients).unwrap();
let gradient = lazy.mat_transpose_vec(&prediction).unwrap();
```

```python [Python]
beta = np.array([1.0, -1.0])
prediction = Z @ beta
gradient = Z.T @ prediction
```

:::

Rust operations return `Result` so storage backends can report read errors.
Python vector products return NumPy arrays. In Python, `@` and `np.matmul`
also accept dense matrices on either side for batched products; Rust exposes
batched products through [`MatMat`](/reference/rust/lazymatrix.MatMat) where
supported. Products and transposes do not create a normalized matrix copy.

## Use SciPy solvers

In Python, expose the operator as a SciPy `LinearOperator` for iterative solvers.
For least squares with an intercept, center the response, solve for coefficients,
and add the response mean back to predictions:

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

## Select a normalized submatrix

In Rust, select a contiguous rectangle after fitting:

```rust
let selected = lazy.submatrix(0..2, 0..1);
let y = selected.matvec(&array![1.0]).unwrap();
assert_eq!(selected.row_range(), 0..2);
assert_eq!(selected.column_range(), 0..1);
```

Selection borrows the parent and preserves its fitted transformation. In
particular, row normalization keeps the parameters fitted using all original
columns. To fit using only selected raw rows, apply `IndexedRows` before fitting.
See [`Submatrix`](/reference/rust/lazymatrix.Submatrix) for supported products
and borrowing behavior.

In Python, `Z[:, 0]` returns a normalized column as an independent NumPy array.
General slicing and mutation are unsupported.

## Choose lazy or eager storage

Keep normalization lazy when preserving sparse storage matters. Materialize
explicitly when you need a concrete dense array or when repeated operations
justify its memory cost:

::: code-group

```rust [Rust]
use ndarray::Array2;

let eager = lazy.to_eager::<Array2<f64>>().unwrap();
let y = eager.matvec(&array![1.0, -1.0]).unwrap();
```

```python [Python]
dense = Z.toarray()
prediction = dense @ np.array([1.0, -1.0])
```

:::

Rust's `to_eager` allocates dense normalized storage, `to_eager_into` fills a
reusable destination, and `into_eager` normalizes owned writable dense storage
in place. Eager operators retain fitted parameters for reuse and operate on
their already normalized values. See
[`LazyMatrix`](/reference/rust/lazymatrix.LazyMatrix) for conversion bounds and
metadata ownership.

Python's `toarray` returns an independent NumPy array. Implicit conversion with
`np.asarray(Z)` and elementwise NumPy operations are unsupported. The operator
owns a raw-data copy in float64; sparse input retains its CSR or CSC orientation,
with duplicates summed and indices sorted. Rust can borrow the original matrix
through a reference or backend view.

Read [how it works](/guide/how-it-works#choose-lazy-or-eager) and
[benchmarks](/benchmarks) for memory and runtime tradeoffs.

## Weighted operations

In Rust, use
[`WeightedColumnSumsInto`](/reference/rust/lazymatrix.WeightedColumnSumsInto)
and [`WeightedGramInto`](/reference/rust/lazymatrix.WeightedGramInto) for
weighted column sums and Gram matrices. They apply the saved normalization
before accumulating entries, including transformed implicit zeros. The caller
chooses the output backend. See the trait references for supported combinations
of normalization and storage.

## Reuse output and workspace

In Rust, reusable-output traits reduce allocation in repeated products:

```rust
use lazymatrix::MatVecInto;

let mut prediction = array![0.0, 0.0, 0.0];
lazy.matvec_into(&array![1.0, -1.0], &mut prediction).unwrap();
```

For repeated fused row products, provide product scratch:

```rust
use lazymatrix::RowNormalization;

let rows = LazyMatrix::new(x.view(), RowNormalization::standardize()).unwrap();
let mut scratch = vec![0.; rows.nrows()];
let mut out = vec![0.; rows.nrows()];
let coefficients = vec![1.; rows.ncols()];
rows.matvec_scaled_with_workspace(1., &coefficients, 0., &mut out, &mut scratch)
    .unwrap();
```

Forward scratch has length `nrows`; transpose scratch has length `ncols`.
Sparse traversal and chunk decoding may still allocate internal buffers. See
[`LazyMatrix`](/reference/rust/lazymatrix.LazyMatrix) for workspace contracts
and [how it works](/guide/how-it-works#row-and-ordered-normalization) for costs.

## Use a PyTorch first layer

Install the [optional PyTorch adapter](/guide/installation#pytorch-adapter) first.

`NormalizedLinear` applies saved normalization as part of a trainable linear
layer. Pass raw feature tensors to the layer, using the centers and scales
fitted on training data. For the sparse matrix above:

```python
import torch
from lazymatrix.torch import NormalizedLinear

params = Z.normalization
y = np.array([1.0, 3.0, 2.0])

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

The layer factors normalization into the linear operation, which changes
floating-point evaluation order. Large offsets, nonfinite values, and sparse
implicit zeros can produce different results from explicit dense normalization.
The [PyTorch benchmark](https://github.com/jolars/lazymatrix/tree/main/benches/torch_adapter)
compares training throughput and memory with a dense first layer. See the
[`NormalizedLinear` reference](/reference/python/lazymatrix.torch.NormalizedLinear)
for the complete interface.
