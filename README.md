# LazyMatrix <picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jolars/lazymatrix/main/assets/logo-dark.png" /><img src="https://raw.githubusercontent.com/jolars/lazymatrix/main/assets/logo.png" align="right" width="189" alt="LazyMatrix logo" /></picture>

[![CI](https://github.com/jolars/lazymatrix/actions/workflows/ci.yml/badge.svg)](https://github.com/jolars/lazymatrix/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/lazymatrix.svg)](https://crates.io/crates/lazymatrix)
[![PyPI](https://badge.fury.io/py/lazymatrix.svg)](https://badge.fury.io/py/lazymatrix)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23666450-blue)](https://doi.org/10.5281/zenodo.23666450)

LazyMatrix is a Rust library for normalization of design matrices. It applies
centering and scaling through matrix operations, preserving sparse storage
without creating a normalized copy. Column normalization supports dense and
sparse matrices, borrowed views, and chunked Zarr arrays. Row normalization
supports dense and sparse CSC/CSR matrices, including borrowed views, and
chunked Zarr arrays.

See the [documentation](https://lazymatrix.org) for guides, benchmarks, and
links to the API reference.

## Installation

Enable the feature for your matrix library and add a matching backend
dependency. For ndarray:

```sh
cargo add lazymatrix --features ndarray
cargo add ndarray@0.17
```

The core library requires Rust 1.84 or newer. Optional backends can require a
newer compiler. sprs 0.11.4 requires Rust 1.85, while sprs 0.11.5 and newer
require Rust 1.88. For Rust 1.85 through 1.87, select sprs 0.11.4 using
`cargo update --package sprs --precise 0.11.4`. The `nalgebra` feature requires
Rust 1.89. The `zarrs` feature requires Rust 1.91; `zarrs_v0_22` retains Rust
1.87 support.

## Usage

Fit column means and standard deviations, then multiply by the normalized
matrix:

```rust
use lazymatrix::{ColumnNormalization, LazyMatrix, MatVec};
use ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = array![[1.0, 0.0], [2.0, 3.0], [0.0, 4.0]];
    let normalization = ColumnNormalization::standardize();
    let lazy = LazyMatrix::new(x.view(), normalization)?;

    let y = lazy.matvec(&array![1.0, -1.0])?;
    println!("{y}");
    Ok(())
}
```

The operator borrows the original matrix and keeps the fitted centers and
scales. Use `ColumnNormalization::min_max()` for min-max scaling to `[0, 1]` on
fitted data or `ColumnNormalization::robust()` for median centering and
interquartile-range scaling. `ColumnNormalization::new(center, scale)` accepts
custom combinations, with centering and scaling independently optional.

Fit parameters independently of a matrix wrapper with `.fit(&data)`. The
parameters own their centers and scales and do not retain the fitting data:

```rust
let params = ColumnNormalization::standardize().fit(&x)?;
let training = params.view().apply(&x);
let validation_data = array![[3.0, 2.0]];
let validation = params.view().apply(&validation_data);
```

Both operators borrow the same fitted vectors without copying them. Application
checks the column count and uses the supplied values unchanged, without fitting.
Call `params.apply(data)` to move owned parameters into an operator, or
`recipe.fit_apply(data)?` to fit and apply in one call. Each operator stores the
supplied data handle, which can be an owned matrix, reference, or backend view.
Borrowed parameters must outlive their operators; owned parameters can leave the
scope where they were fitted.

Supply explicit vectors with `LazyMatrix::from_column_parts`; the column count
comes from the matrix:

```rust
let lazy = LazyMatrix::from_column_parts(
    x.view(),
    Some(vec![1.0, 2.0]),
    Some(vec![2.0, 4.0]),
);
```

Fit on selected raw rows without copying the training matrix:

```rust
use lazymatrix::IndexedRows;

let training_rows = [0, 2];
let held_out_rows = [1];
let training = LazyMatrix::new(
    IndexedRows::new(x.view(), &training_rows),
    ColumnNormalization::standardize(),
)?;
let validation = training.normalization().view().apply(
    IndexedRows::new(x.view(), &held_out_rows),
);
```

`IndexedRows` borrows strictly increasing, unique indices and accepts borrowed
or owned source storage. Empty selections are valid; invalid indices panic.
Products and statistics use local row indices and native dense, CSC, or CSR
traversal without copying the selected design. Sparse duplicates and implicit
zeros retain their logical values. In contrast, `LazyMatrix::submatrix`
restricts an already fitted operator without recomputing normalization. See the
[indexed-row measurements](benches/indexed_rows/README.md) for workspace and
scan costs.

Centered dense products normalize entries before accumulation to preserve small
variations around large column offsets. They use the same numerical policy for
allocating and reusable products without creating a normalized design copy.

The [examples](examples/) cover gradient descent, stochastic gradient descent,
coordinate descent, ridge regression through primal or dual systems, weighted
Gram matrices, and file-backed input. Run one from a checkout with:

```sh
cargo run --locked --example coordinate_descent --features faer
```

## Row normalization

Fit rows independently with `RowNormalization` and `LazyMatrix`:

```rust
use lazymatrix::{Centering, LazyMatrix, MatVec, RowNormalization, Scaling};
use ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = array![[3.0, 4.0], [0.0, 0.0]];
    let spec = RowNormalization::new(Centering::None, Scaling::L2);
    let normalized = LazyMatrix::new(x.view(), spec)?;

    let y = normalized.matvec(&array![1.0, 1.0])?;
    assert_eq!(y, array![1.4, 0.0]);
    Ok(())
}
```

Row normalization offers the same centering and scaling choices and common
recipes as column normalization. Each parameter vector has one entry per row.
Norm scaling leaves zero rows unchanged, and new samples need independently
fitted row parameters. Use `LazyMatrix::from_row_parts(data, centers, scales)`
for explicit vectors; the row count comes from the matrix. To fit row parameters
separately, call `spec.fit(&data)`. Reapply them unchanged to the same rows with
`params.view().apply_frozen(data)`, or move their vectors with
`params.apply_frozen(data)`. Use `spec.fit_apply(data)?` to fit and wrap each
new set of rows independently.

The parameter type identifies the axis and scalar:
`LazyMatrix<M, ColumnNormalizationParams<f64>>` or
`LazyMatrix<M, RowNormalizationParams<f64>>`. Column parameters with `f64` are
the default, so column-only annotations can remain `LazyMatrix<M>`. Read centers
and scales through `matrix.normalization()`. `matrix.into_parts()` returns the
raw data and the fitted parameter object without copying, and
`LazyMatrix::from_parts(data, params)` rebuilds the lazy operator. `EagerMatrix`
uses the same typed metadata API for its already normalized data.

Both parameter types have a storage parameter that defaults to `Vec<F>`. Their
`.view()` methods return `ColumnNormalizationParamsView<'a, F>` or
`RowNormalizationParamsView<'a, F>`, using borrowed slices. Owned parameters and
views share each axis's operator capabilities.

Parameter `into_parts()` returns `(dimension, centers, scales)`, retaining the
row or column count even when both vectors are absent. Generic code can borrow
parameters through the `NormalizationParams` trait's `View` associated type and
`view()` method, and fit recipes through `NormalizationRecipe<M, P>::fit`.

The row operator supports allocating and reusable forward and transpose vector
products across every dense, sparse, and Zarr backend release, including strided
dense views and borrowed CSC/CSR storage. Sparse fitting combines duplicate cells
and includes implicit zeros without converting storage or materializing the design.
CSR fitting buffers one row; CSC fitting streams column scans, repeating scans
for exact quantiles. Sparse fitting and products use O(nrows + ncols) workspace.
Zarr fitting streams chunks with O(nrows) state, and each product reads every
storage chunk once. Missing chunks retain their fill values, and scans exclude
edge padding. Chunk buffers and codec workspace add to the fitted vectors and
depend on the largest decoded storage chunk, including the outer shard.
Standardization takes two scans; exact row quantiles use repeated scans whose
count depends on the values, sharing scans for a median and quantile range.
`RowStats` supplies the fitting hook, and `RowNormalizationKernel` supplies the
products. Dense adapters can delegate to their provided dense helper methods.
Additional operator capabilities and Python bindings remain
in [the roadmap](TODO.md#row-normalization).

## Ordered normalization

Compose recipes with `.then()`, reading from left to right. For example, fit row
standardization before column standardization:

```rust
use lazymatrix::{ColumnNormalization, RowNormalization};
use ndarray::array;

let training = array![[1., 3., 2.], [7., 3., 8.], [2., 8., 5.]];
let validation = array![[100., 120., 90.]];
let recipe = RowNormalization::standardize()
    .then(ColumnNormalization::standardize());
let fitted = recipe.fit(&training).unwrap();
let train = fitted.view().apply_frozen(training.view());
let valid = fitted.view().for_new_samples(validation.view()).unwrap();
```

Every stage fits the logical output of its predecessors. Reversing the order
changes the transformation. Repeated axes are supported, and later stages can
change statistics established by earlier stages. Use
`LazyMatrix::new(data, recipe)?` or `recipe.fit_apply(data)?` to fit and wrap in
one call.

`apply_frozen` reuses every saved value for matching rows and columns.
`for_new_samples` fits new row parameters at each row stage while reusing
training column parameters. It owns the new row values and borrows the saved
column values when called on a view. Both methods consume their receiver; call
them on owned fitted parameters to move the column vectors instead.

Generic code can prepare new observations through `PrepareNewSamples<M>`. Custom
backends can implement `MatrixTraversal<F>` alongside `ColumnStats<F>` to fit
column-only chains. Its rectangular scans report source coordinates and coalesce
sparse duplicates without materializing the design. Sparse scans visit unique
cells in sorted native order.

`normalization()` exposes a `NormalizationChainParams` tree with `first()` and
`next()` accessors. Row stages retain their recipe and `RowNormalizationParams`
inside `FittedRowNormalization`; inspect them with `recipe()` and
`parameters()`. Column stages contain `ColumnNormalizationParams`.
`into_parts()` and `LazyMatrix::from_parts` preserve the fitted sequence without
refitting.

Mixed and row-only chains support dense backends and strided views. Column-only
chains also support sparse CSC and CSR storage and chunked Zarr arrays. Chains
currently expose allocating and reusable forward and transpose vector products.
They preserve storage and use bounded workspace, without materializing an
intermediate design matrix. See the
[guide](https://lazymatrix.org/guide/getting-started#compose-normalization-recipes)
for scan costs and explicit fitted stages.

## Select a normalized submatrix

`matrix.submatrix(rows, columns)` borrows a rectangular part of the logical
operator, retaining its fitted transformation. A row stage still uses the
parameters fitted from all of its original columns. Nested selections use local
coordinates and borrow the original parent without allocating or refitting.

`Submatrix<'a, O>` exposes its parent with `as_inner()` and its source ranges
with `row_range()` and `column_range()`. Vector products support column and row
normalization and supported chains. Column-normalized selections also retain
weighted operations, eager conversion, and borrowed logical views. Custom
operators can supply vector products through `RectangularProducts`.

## Features

- Mean, minimum, or median centering, with scaling by population or sample
  standard deviation, L1 or L2 norm, maximum absolute value, range, or
  configurable quantile range.
- Forward and transpose matrix–vector products, reusable output buffers, and
  weighted Gram matrices.
- Batched forward and transpose products with caller-selected dense outputs.
- Borrowed transpose operator views through `Transpose::new(&operator)`,
  preserving fitted normalization across vector and batched products.
- Borrowed logical columns and sparse column and row views for supported
  storage.
- Borrowed contiguous submatrices that retain the parent's fitted
  transformation, with weighted operations and bounded Zarr reads for column
  normalization.
- An implicit intercept column through `WithIntercept`.
- Explicit dense conversion through `to_eager` and `to_eager_into`, or in-place
  normalization of writable dense storage through `into_eager`.
- Chunked Zarr input for matrices larger than RAM.

## Matrix backends

No matrix backend is enabled by default. Choose a feature to use an existing
matrix library:

  | Feature    | Supported releases     | Storage                                 |
  | ---------- | ---------------------- | --------------------------------------- |
  | `faer`     | 0.22, 0.23, 0.24       | Dense, CSC, and CSR matrices            |
  | `nalgebra` | 0.32, 0.33, 0.34, 0.35 | Dense, CSC, and CSR matrices            |
  | `ndarray`  | 0.15, 0.16, 0.17       | Dense arrays and views                  |
  | `sprs`     | 0.11                   | CSC and CSR matrices                    |
  | `zarrs`    | 0.22, 0.23             | Synchronous two-dimensional Zarr arrays |

The short feature names select the newest supported release. Use a versioned
feature such as `nalgebra_v0_34` for an older release, and match your direct
backend dependency to it. Enable `parallel` alongside a backend for parallel
column statistics where supported.

Enable `ndarray-blas` for optional BLAS acceleration with ndarray 0.17, or use
`ndarray_v0_15-blas`, `ndarray_v0_16-blas`, or `ndarray_v0_17-blas` for a
specific release. Your application must select and link a BLAS provider, such as
LP64 OpenBLAS. LazyMatrix does not select one. See the [BLAS setup
guide](https://lazymatrix.org/guide/getting-started#optional-blas-acceleration).

BLAS can accelerate `f32` and `f64` uncentered vector and batched products,
eager products, and finite weighted Gram products. Centered lazy vector and
batched products use directly centered Rust kernels to preserve small variations
around large offsets. BLAS threading is controlled by the provider independently
of `parallel`.

Faer matrices support immutable and mutable borrowed views, including
`SparseColMatRef` and `SparseColMatMut` for CSC storage. Pass `matrix.as_ref()`
directly to `LazyMatrix::new` to fit normalization and use products or borrowed
columns without cloning the matrix. CSC operations respect occupied column
ranges, including views with nonzero storage offsets and spare capacity.

Custom backends can implement the crate's matrix and vector traits. See the [API
reference](https://docs.rs/lazymatrix) for capabilities and supported types.

## Python package

The companion Python package supports NumPy arrays and SciPy CSR/CSC input.
Install it with `pip install lazymatrix`:

```python
import numpy as np
from lazymatrix import standardize

x = np.array([[1.0, 0.0], [2.0, 3.0], [0.0, 4.0]])
z = standardize(x)
prediction = z @ np.array([1.0, -1.0])
gradient = z.T @ prediction
```

See the [Python guide](https://lazymatrix.org/guide/python) for installation,
ownership, and supported operations, and the [Python API
reference](https://lazymatrix.org/reference/python/) for details.

## Documentation

- [Getting started](https://lazymatrix.org/guide/getting-started)
- [How it works](https://lazymatrix.org/guide/how-it-works)
- [API reference](https://docs.rs/lazymatrix)
- [Python API reference](https://lazymatrix.org/reference/python/)
- [Rust API reference on the website](https://lazymatrix.org/reference/rust/)
- [Benchmarks](https://lazymatrix.org/benchmarks)

## Citation

If you use LazyMatrix in your research, please cite the release you used. Select
that release in the [Zenodo archive](https://doi.org/10.5281/zenodo.23666450)
and export its citation. To cite LazyMatrix generally, use the concept DOI
`10.5281/zenodo.23666450`.

Machine-readable metadata is available in [CITATION.cff](CITATION.cff).

## License

Licensed under either the [Apache License, Version 2.0](LICENSE-APACHE) or the
[MIT license](LICENSE-MIT), at your option.
