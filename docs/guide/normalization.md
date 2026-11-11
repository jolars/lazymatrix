# Normalization

Choose how to center and scale your data, fit columns or individual rows, and
reuse the fitted transformation for prediction. The examples use:

::: code-group

```rust [Rust]
use lazymatrix::{ColumnNormalization, LazyMatrix};
use ndarray::array;

let x = array![[1.0, 0.0], [2.0, 3.0], [0.0, 4.0]];
```

```python [Python]
import numpy as np
from lazymatrix import normalize, standardize
from scipy import sparse

X = sparse.csr_matrix([[1.0, 0.0], [2.0, 3.0], [0.0, 4.0]])
```

:::

## Choose centering and scaling

Centering and scaling are independently optional. Start with a named recipe or
select the two components yourself:

| Goal            | Centering | Scaling                       |
| --------------- | --------- | ----------------------------- |
| Standardization | Mean      | Population standard deviation |
| Min-max scaling | Minimum   | Range                         |
| Robust scaling  | Median    | Interquartile range           |

::: code-group

```rust [Rust]
let standardized = LazyMatrix::new(x.view(), ColumnNormalization::standardize()).unwrap();
let min_max = LazyMatrix::new(x.view(), ColumnNormalization::min_max()).unwrap();
let robust = LazyMatrix::new(x.view(), ColumnNormalization::robust()).unwrap();
```

```python [Python]
Z = standardize(X)
min_max = normalize(X, center="min", scale="range")
robust = normalize(X, center="median", scale="quantile_range")
unit_columns = normalize(X, center=False, scale="l2")
```

:::

For sample standard deviations, subtract one degree of freedom:

::: code-group

```rust [Rust]
use lazymatrix::{Centering, Scaling};

let sample = ColumnNormalization::new(Centering::Mean, Scaling::Sd { df: 1 });
let standardized = LazyMatrix::new(x.view(), sample).unwrap();
```

```python [Python]
standardized = standardize(X, df=1)
```

:::

Standard deviations use `nrows - df` for columns and `ncols - df` for rows.
Computed scales become one when they are exactly zero. Min-max scaling maps
finite, nonconstant fitting data to `[0, 1]`; it does not clip new observations
to that interval.

Rust's `ColumnNormalization::default()` disables both components. Python's
`normalize(X)` defaults to standardization; pass `center=False` or `scale=False`
to disable a component. Norm and maximum-absolute-value scaling use centered
values when centering is active, including implicit sparse zeros. See the
[Rust recipe reference](/reference/rust/lazymatrix.ColumnNormalization) and
[Python normalization reference](/reference/python/lazymatrix.normalize)
for the complete set of choices.

## Fit robust normalization

Robust normalization subtracts medians and divides by the interquartile range.
Choose different quantile bounds for a wider or narrower range:

::: code-group

```rust [Rust]
use lazymatrix::{Centering, QuantileRange, Scaling};

let recipe = ColumnNormalization::new(
    Centering::Median,
    Scaling::QuantileRange {
        range: QuantileRange::new(0.1, 0.9),
        unit_variance: false,
    },
);
let robust = LazyMatrix::new(x.view(), recipe).unwrap();
```

```python [Python]
robust = normalize(
    X, center="median", scale="quantile_range", quantile_range=(0.1, 0.9)
)
```

:::

Bounds are probabilities satisfying `0 <= lower < upper <= 1`. Set
`unit_variance` to `true` in Rust or `True` in Python to calibrate the scale for
a normal distribution. Calibrated bounds must be strictly inside `(0, 1)`.
Calibration targets unit variance for normally distributed data; finite samples
and other distributions can have different variances.

Exact quantiles can require repeated storage scans. Read
[quantile fitting](/guide/how-it-works#quantile-fitting) for numerical behavior
and costs on dense, sparse, and chunked inputs.

## Reuse fitted parameters

Fit column parameters on training data and apply those same values to validation
or prediction data. The new matrix must have the same features in the same
order, but it can have a different number of rows.

::: code-group

```rust [Rust]
let params = ColumnNormalization::standardize().fit(&x).unwrap();
let training = params.view().apply(x.view());
let x_valid = array![[2.0, 1.0], [0.0, 3.0]];
let validation = params.view().apply(x_valid.view());
```

```python [Python]
Z = standardize(X)
params = Z.normalization
X_validation = sparse.csr_matrix([[4.0, 1.0], [0.0, -2.0]])
validation = params.transform(X_validation)
```

:::

These application methods do not refit statistics. Calling a fitting constructor
on validation data would compute new centers and scales instead.

In Rust, fitting returns owned parameters independently of the training matrix.
`.view()` shares those values without copying; use `params.apply(data)` to move
them into an operator. An existing lazy or eager operator also exposes its
parameters through `normalization()`. See
[`ColumnNormalizationParams`](/reference/rust/lazymatrix.ColumnNormalizationParams)
for ownership and explicit construction.

In Python, `Z.normalization` returns an independent parameter object. Use
`ColumnNormalization(...).fit(X)` to fit separately from an operator, or
`fit_apply(X)` to fit and wrap in one call. See
[`ColumnNormalizationParams`](/reference/python/lazymatrix.ColumnNormalizationParams)
for explicit values and validation rules.

### Save fitted normalization in Python

Save the feature count and active vectors in a NumPy archive without pickling:

```python
from lazymatrix import ColumnNormalizationParams

checkpoint = {"ncols": params.ncols}
for name in ("centers", "scales"):
    values = getattr(params, name)
    if values is not None:
        checkpoint[name] = values
np.savez("normalization.npz", **checkpoint)

with np.load("normalization.npz", allow_pickle=False) as saved:
    restored = ColumnNormalizationParams(
        int(saved["ncols"]),
        centers=saved.get("centers"),
        scales=saved.get("scales"),
    )
```

Use `restored.transform` on new raw data. Save feature names alongside the
archive so you can check feature order when loading it.

## Normalize rows

Row normalization fits a separate center and scale for each sample:

::: code-group

```rust [Rust]
use lazymatrix::RowNormalization;

let rows = LazyMatrix::new(x.view(), RowNormalization::standardize()).unwrap();
```

```python [Python]
rows = standardize(X, axis=1)
unit_rows = normalize(X, axis=1, center=False, scale="l2")
```

:::

Row normalization supports dense matrices, sparse CSC/CSR storage, and, in Rust,
chunked Zarr arrays. It preserves sparse storage even when centering changes
the logical values of implicit zeros.

### Fit new samples independently

Each new sample needs its own row statistics. Reuse the recipe settings while
fitting fresh row parameters on each dataset:

::: code-group

```rust [Rust]
let recipe = RowNormalization::standardize();
let training = recipe.fit_apply(x.view()).unwrap();
let new_samples = array![[3.0, 4.0]];
let validation = recipe.fit_apply(new_samples.view()).unwrap();
```

```python [Python]
from lazymatrix import RowNormalization

recipe = RowNormalization(center=False, scale="l2")
training = recipe.fit_apply(X)
validation = recipe.fit_apply(X_validation)
```

:::

Use `apply_frozen` only when reapplying saved parameters to the same rows in
the same order. A matching row count alone does not make training row parameters
appropriate for new samples. See the
[Rust](/reference/rust/lazymatrix.RowNormalizationParams) and
[Python](/reference/python/lazymatrix.RowNormalizationParams) parameter
references for explicit or saved row values. To save row parameters in Python,
use the archive pattern above with `nrows` and `RowNormalizationParams`, and
retain the recipe separately for new samples.

## Compose normalization recipes

In Rust, use `.then()` to apply recipes from left to right. Each stage fits the
logical output of its predecessors, so reversing the order changes the fitted
transformation. Later stages can change statistics established by earlier ones.

```rust
let training = array![[1., 3., 2.], [7., 3., 8.], [2., 8., 5.]];
let validation = array![[100., 120., 90.]];
let recipe = RowNormalization::standardize()
    .then(ColumnNormalization::standardize());
let fitted = recipe.fit(&training).unwrap();
let train = fitted.view().apply_frozen(training.view());
let valid = fitted.view().for_new_samples(validation.view()).unwrap();
```

`apply_frozen` applies every saved stage unchanged to matching rows and columns.
`for_new_samples` refits each row stage on the validation data while reusing
training column parameters. Products never refit parameters.

Mixed and row-only chains require dense storage for fitting and products.
Column-only chains also support sparse CSC/CSR storage and Zarr arrays. See
[`NormalizationChainParams`](/reference/rust/lazymatrix.NormalizationChainParams)
for inspecting and constructing saved stages, and
[how it works](/guide/how-it-works#row-and-ordered-normalization) for costs.

For custom fitting methods, the
[`NormalizationRecipe` reference](/reference/rust/lazymatrix.NormalizationRecipe)
includes a complete implementation example and the required parameter contracts.
