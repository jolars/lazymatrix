# Getting started

LazyMatrix applies column normalization through a matrix operator. Start with
your existing matrix backend, fit the centers and scales, and use the normalized
operator in your algorithm.

## Install a backend

The core depends only on `num-traits`. Enable a backend for ready-made matrix
implementations:

```sh
cargo add lazymatrix --features ndarray
cargo add ndarray@0.17
```

| Feature    | Storage                                 | Backend release selected |
| ---------- | --------------------------------------- | ------------------------ |
| `faer`     | Dense, CSC, and CSR sparse matrices     | faer 0.24                |
| `nalgebra` | Dense, CSC, and CSR sparse matrices     | nalgebra 0.35            |
| `ndarray`  | Dense arrays and views                  | ndarray 0.17             |
| `sprs`     | CSC and CSR sparse matrices             | sprs 0.11                |
| `zarrs`    | Synchronous two-dimensional Zarr arrays | zarrs 0.23               |

Match your direct backend dependency to the selected release. Versioned features
such as `nalgebra_v0_34` select an older supported release line. See the
[README](https://github.com/jolars/lazymatrix#matrix-backends) for supported
releases.

The core supports Rust 1.84. Backend dependencies can require a newer compiler.
sprs 0.11.4 requires Rust 1.85, while sprs 0.11.5 and newer require Rust 1.88.
For Rust 1.85 through 1.87, select sprs 0.11.4 using
`cargo update --package sprs --precise 0.11.4`. The `nalgebra` feature requires
Rust 1.89. The `zarrs` feature requires Rust 1.91; `zarrs_v0_22` retains Rust
1.87 support.

### Optional BLAS acceleration

Enable `ndarray-blas` for ndarray 0.17, or use `ndarray_v0_15-blas`,
`ndarray_v0_16-blas`, or `ndarray_v0_17-blas` to select a specific release:

```sh
cargo add lazymatrix --features ndarray-blas
cargo add ndarray@0.17
```

Your application must select and link a BLAS provider. LazyMatrix selects no
provider, so these features work with an application's existing BLAS setup.
For a system LP64 OpenBLAS installation, link the application with:

```sh
RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-l openblas" cargo run
```

For a provider in a nonstandard directory, add its library search path and
configure the runtime loader. In this checkout, the devenv environment supplies
`OPENBLAS_LP64_LIB`; `task test:blas` uses it to set both paths and tests every
supported ndarray release. Without that variable, the task links system
OpenBLAS. `task test` runs the backend suites without requiring a BLAS provider.
Running `cargo test --all-features` directly requires a provider at link time.

BLAS can accelerate `f32` and `f64` uncentered vector and batched products,
eager products, and finite weighted Gram products when memory layouts permit.
The Gram kernel centers bounded panels before multiplication. Centered lazy
vector and batched products use directly centered Rust kernels to preserve small
variations around large offsets. BLAS threading is controlled by the provider
independently of `parallel`; for OpenBLAS, set `OPENBLAS_NUM_THREADS` and
`OMP_NUM_THREADS` when controlling benchmark threads.

## Fit normalization and apply a product

```rust
use lazymatrix::{LazyMatrix, MatVec, ColumnNormalization};
use ndarray::array;

let x = array![[1.0, 0.0], [2.0, 3.0], [0.0, 4.0]];
let lazy = LazyMatrix::new(x.view(), ColumnNormalization::standardize()).unwrap();

let y = lazy.matvec(&array![1.0, -1.0]).unwrap();
```

Choose a named recipe or use `ColumnNormalization::new(center, scale)` for a custom
combination:

| Preset          | Centering | Scaling                       |
| --------------- | --------- | ----------------------------- |
| `standardize()` | Mean      | Population standard deviation |
| `min_max()`     | Minimum   | Range                         |
| `robust()`      | Median    | Interquartile range           |

Standardization uses `nrows` as the variance divisor by default. For sample
standard deviations, subtract one degree of freedom:

```rust
use lazymatrix::{Centering, Scaling};

let sample = ColumnNormalization::new(Centering::Mean, Scaling::Sd { df: 1 });
let lazy = LazyMatrix::new(x.view(), sample).unwrap();
```

`Scaling::Sd { df }` uses `nrows - df` as the variance divisor; `df = 0` selects
population standard deviations, and `df = 1` selects sample standard deviations.
`ColumnStats::col_sds(df)` uses the same convention. Scales are `NaN` when
`nrows <= df`, including constant columns with too few rows.

Min-max scaling maps finite, nonconstant columns to `[0, 1]` on the fitting data;
reusing fitted parameters
does not clip new observations to that range. Centering and scaling are
independently optional, and `ColumnNormalization::default()` performs neither.
Computed scales become one when they would otherwise be exactly zero.
Construction and products return `Result`, so storage backends can report read
errors. Dimension mismatches panic.

For repeated operations, reusable-output traits and workspace methods can reduce
allocation. See the [API documentation](https://docs.rs/lazymatrix) for the
capabilities supported by each backend.

## Fit robust normalization

Subtract column medians and divide by the interquartile range:

```rust
let robust = LazyMatrix::new(x.view(), ColumnNormalization::robust()).unwrap();
```

For the 10th-to-90th percentile range, specify custom quantile bounds:

```rust
use lazymatrix::{Centering, QuantileRange, Scaling};

let custom = ColumnNormalization::new(
    Centering::Median,
    Scaling::QuantileRange(QuantileRange::new(0.1, 0.9)),
);
let robust = LazyMatrix::new(x.view(), custom).unwrap();
```

Bounds are probabilities satisfying `0 <= lower < upper <= 1`; invalid bounds panic.
Median centering and quantile-range scaling can also be used independently or
combined with the existing options. Unlike standard-normal calibration, a raw
quantile range does not target unit variance.

Fitting uses exact quantiles with linear interpolation at `(nrows - 1) * q`.
Dense and CSC backends sort one working column at a time. CSR and Zarr use
O(ncols) selection state and repeated scans whose count depends on the values.
They preserve implicit zeros and sparse storage. See [how it works](/guide/how-it-works#quantile-fitting)
for memory, numerical behavior, and fitting costs.

## Normalize rows

Use the same constructor with a row recipe:

```rust
use lazymatrix::RowNormalization;

let rows = LazyMatrix::new(x.view(), RowNormalization::standardize()).unwrap();
let row_scales = rows.normalization().scales().unwrap();
```

Each row gets its own center and scale in `RowNormalizationParams<f64>`.
Row normalization works with dense matrices, strided views, sparse CSC/CSR
storage, and Zarr arrays. Use `MatVec`, `MatTransposeVec`, or their reusable-output
and fused variants. For repeated fused products, provide product scratch:

```rust
let mut scratch = vec![0.; rows.nrows()];
let mut out = vec![0.; rows.nrows()];
let coefficients = vec![1.; rows.ncols()];
rows.matvec_scaled_with_workspace(1., &coefficients, 0., &mut out, &mut scratch)
    .unwrap();
```

Forward scratch has length `nrows`; transpose scratch has length `ncols`.
Workspace calls reuse this vector while sparse traversal and chunk decoding
may still allocate internal buffers. Fit new samples' row parameters
independently, or compose row and column recipes as described below.

`LazyMatrix<M, P>` stores its fitted parameter object as `P`. The default is
`ColumnNormalizationParams<f64>`, so ordinary column-only type annotations can
use `LazyMatrix<M>`. Inspect centers and scales through `normalization()`.
Supply explicit vectors with `LazyMatrix::from_column_parts(data, centers, scales)`
or `LazyMatrix::from_row_parts(data, centers, scales)`. Each constructor infers
the dimension from the matrix. Centers and scales are independently optional:

```rust
let explicit = LazyMatrix::from_column_parts(
    x.view(),
    Some(vec![1.0, 2.0]),
    Some(vec![2.0, 4.0]),
);
let (data, params) = explicit.into_parts();
let restored = LazyMatrix::from_parts(data, params);
```

## Compose normalization recipes

Use `.then()` to apply recipes from left to right. Each stage fits the logical
output of its predecessors, so reversing the order changes the fitted
transformation. Repeated axes are supported. Later stages can change statistics
established by earlier stages.

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

Use `apply_frozen` to apply every saved stage unchanged to matching rows and
columns. Use `for_new_samples` to refit each row stage on the validation data
after applying its predecessors, while reusing training column parameters.
This distinction applies even when training and validation have the same number
of rows. Products never refit parameters.

Both methods consume their receiver. Calling them on `fitted.view()` borrows
training parameters without copying. New-sample preparation owns the new row
values, and its borrowed column values cannot outlive `fitted`. Calling either
method on owned fitted parameters moves the column vectors into the result.
`LazyMatrix::new(data, recipe)?` and `recipe.fit_apply(data)?` fit and wrap in one
call.

`normalization()` exposes a `NormalizationChainParams` tree in recipe order.
Inspect each prefix or suffix with `first()` and `next()`. Column stages contain
`ColumnNormalizationParams`. Row stages contain `FittedRowNormalization`, whose
`recipe()` and `parameters()` accessors expose the saved row recipe and
`RowNormalizationParams`.

Construct explicit stages without fitting:

```rust
use lazymatrix::{
    ColumnNormalizationParams, FittedRowNormalization, NormalizationChainParams,
    RowNormalization, RowNormalizationParams,
};

let rows = FittedRowNormalization::from_parts(
    RowNormalization::standardize(),
    RowNormalizationParams::from_parts(3, Some(vec![2., 6., 5.]), None),
);
let columns = ColumnNormalizationParams::from_parts(
    3, None, Some(vec![1., 2., 3.]),
);
let fitted = NormalizationChainParams::from_parts(rows, columns);
let train = fitted.view().apply_frozen(training.view());
let (raw, fitted) = train.into_parts();
let restored = LazyMatrix::from_parts(raw, fitted);
```

The explicit row recipe controls future new-sample fitting; it does not change
the supplied saved values. Dimension mismatches panic before reads or writes.

Mixed and row-only chains require dense storage for fitting and products.
Column-only chains also support sparse CSC/CSR storage and Zarr arrays. Products
execute the saved stages without refitting. Chains support allocating, reusable,
and fused forward and transpose vector products, including caller-owned product
workspace. Chains use the same product scratch lengths as row normalization.
See [how it works](/guide/how-it-works#row-and-ordered-normalization) for storage
and workspace costs.

## Select a normalized submatrix

Select a contiguous rectangle from the logical operator after fitting:

```rust
use lazymatrix::{LazyMatrix, MatVec, RowNormalization};
use ndarray::array;

let x = array![[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]];
let normalized = LazyMatrix::new(x.view(), RowNormalization::standardize()).unwrap();
let selected = normalized.submatrix(0..2, 1..3);
let y = selected.matvec(&array![1.0, -1.0]).unwrap();
assert_eq!(selected.row_range(), 0..2);
assert_eq!(selected.column_range(), 1..3);
```

Selection preserves every fitted stage. Row normalization uses the saved
parameters fitted with all of the original columns, even when the selection
excludes some columns. To fit using only selected raw rows, use `IndexedRows`
before fitting instead.

`Submatrix<'a, O>` borrows the logical parent `O`. Access that parent through
`as_inner()`, including its fitted parameters through
`selected.as_inner().normalization()`. The parent metadata still describes the
full matrix; use the selection ranges when inspecting a particular row or
column. Construction allocates nothing and reads no data. Nested selections
take local coordinates, combine their offsets, and retain the original parent
and lifetime.

Column and row normalization and supported chains supply allocating, reusable,
and fused forward and transpose vector products for their submatrices. Column
normalization also supplies weighted operations, eager conversion, and borrowed
logical views. Eager conversion copies the selected column parameters. Zarr
column selections read only intersecting chunks. Empty selections and
zero-alpha products skip storage reads.

Custom operators can implement `RectangularProducts` to support these vector
products. The rectangular methods receive source row and column ranges and
local input and output vectors. They must evaluate only the selected rectangle;
padding a vector and invoking a full product can read unselected data and
change IEEE nonfinite results.

## Materialize explicitly

Choose an eager representation when dense storage and repeated operations suit
your workload:

```rust
use ndarray::Array2;

let eager = lazy.to_eager::<Array2<f64>>().unwrap();
let y = eager.matvec(&array![1.0, -1.0]).unwrap();
```

`to_eager` allocates a normalized dense matrix. `to_eager_into` fills a reusable
dense destination. For owned writable dense inputs, `into_eager` normalizes the
existing storage in place. An eager operator retains the fitted parameters and
uses its normalized values directly.
`to_eager` and `to_eager_into` copy fitted metadata into owned parameter vectors,
including when the lazy operator borrows parameters. `into_eager` retains the
supplied parameter storage, including any borrows. Both wrappers expose metadata
through `normalization()`, and `into_parts()` returns `(data, params)` without
copying. Eager data already contains normalized values; lazy data remains raw.

## Reuse fitted parameters

Fit column parameters independently, then share them between training and
validation operators:

```rust
let recipe = ColumnNormalization::standardize();
let params = recipe.fit(&x).unwrap();
let training = params.view().apply(&x);
let x_valid = array![[2.0, 1.0], [0.0, 3.0]];
let validation = params.view().apply(&x_valid);
```

Fitting returns owned vectors without retaining the fitting data. `.view()`
borrows those vectors without allocation, and `.apply(data)` consumes the view
to construct a lazy operator. The operator retains the borrowed slices and the
supplied data handle. It checks the column count and performs no fitting or
parameter copies. A reference and a backend matrix view both borrow data without
copying it.

Use `params.apply(data)` to move owned parameters into an operator. Use
`recipe.fit_apply(data)` to fit and wrap in one call. Both forms let a function
return an operator owning its parameters without borrowing local fitted values.
The data handle can still borrow an input matrix.

The parameter types default to `Vec<F>` storage. Their view aliases are
`ColumnNormalizationParamsView<'a, F>` and `RowNormalizationParamsView<'a, F>`,
with `&'a [F]` storage. Both representations share each axis's capabilities.
Parameter `into_parts()` returns `(dimension, centers, scales)` so that the
parameters retain their row or column count even when both vectors are absent.
Pass owned parts back to the corresponding parameter type's
`from_parts()` constructor.

Row parameters use `.apply_frozen(data)` on owned values or views to emphasize
that they describe particular rows. Fit new samples independently, even when
their row count matches the training data:

```rust
let row_recipe = RowNormalization::standardize();
let row_params = row_recipe.fit(&x).unwrap();
let frozen = row_params.view().apply_frozen(&x);
let new_rows = row_recipe.fit_apply(&x_valid).unwrap();
```

Parameters retained by an existing lazy or eager operator can also be borrowed
for new observations:

```rust
let x_new = array![[2.0, 1.0], [0.0, 3.0]];
let prediction = eager.normalization().view().apply(&x_new);
let y_new = prediction.matvec(&array![1.0, -1.0]).unwrap();
```

Generic code can fit through `NormalizationRecipe<M, P>::fit`, borrow parameters
through `NormalizationParams::View` and `view()`, and prepare new observations
through `PrepareNewSamples<M>`. Chain `first()`, `next()`, and `into_parts()`
expose the saved stages without private trait bounds. Explicit chain fitting
uses `fit::<M, P>()`, where `P` is the fitted parameter type, and inference
usually supplies both types.

Custom backends can fit column-only chains by implementing
`MatrixTraversal<F>` and `ColumnStats<F>`. Traversal takes a source rectangle
and reports source coordinates. `TraversalLayout::Full` visits every selected
cell, including zeros and fill values. `SparseRows` and `SparseColumns` visit
unique cells in sorted native order, combine duplicate entries, preserve
explicit stored zeros, and treat omitted cells as zero. Empty rectangles
perform no reads. Dense indexed backends can use the provided dense traversal
helper. Mixed chains continue to use dense indexed access.

Read [how it works](/guide/how-it-works) for the algebra and
[benchmarks](/benchmarks) for measured tradeoffs between representations.
