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

## Normalize dense rows

Use the same constructor with a row recipe:

```rust
use lazymatrix::RowNormalization;

let rows = LazyMatrix::new(x.view(), RowNormalization::standardize()).unwrap();
let row_scales = rows.normalization().scales().unwrap();
```

The fitted type is `RowNormalizationParams<f64>`, with one center and scale per
row. Row normalization supports dense fitting and forward and transpose vector
products, including strided views. Fit new samples' row parameters independently.
Ordered composition with column normalization remains planned.

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
`EagerMatrix<M, P>` retains the same fitted parameter type. Both wrappers expose
it through `normalization()`, and `into_parts()` returns `(data, params)` without
copying. Eager data already contains normalized values; lazy data remains raw.

## Reuse fitted parameters

Apply training column parameters to new observations rather than fitting the prediction
data again:

```rust
let x_new = array![[2.0, 1.0], [0.0, 3.0]];
let prediction = LazyMatrix::from_parts(
    x_new.view(),
    eager.normalization().clone(),
);
let y_new = prediction.matvec(&array![1.0, -1.0]).unwrap();
```

Read [how it works](/guide/how-it-works) for the algebra and
[benchmarks](/benchmarks) for measured tradeoffs between representations.
