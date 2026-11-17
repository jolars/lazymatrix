# How it works

Normalizing a matrix usually means subtracting a center from each column and
dividing by a scale. You might do this before fitting a model so that features
with different units are easier to work with.

LazyMatrix saves those centers and scales alongside the original data. When you
multiply by the matrix, it applies the normalization as part of the calculation.
You get the normalized result without first building a normalized matrix.

## Why zeros matter

Take a column with values `[0, 0, 3]`. If you subtract a center of `1` and divide
by a scale of `2`, you get `[-0.5, -0.5, 1]`.

A sparse matrix can leave the original zeros unstored. After centering, those
zeros become nonzero values. Across a large matrix, storing them can turn a
compact sparse representation into a full dense array.

LazyMatrix keeps the original sparse data and the two short vectors of centers
and scales. The normalized entries are still there mathematically, but they
don't have to take up space in memory.

## Matrix–vector products

When you write `Z @ beta` in Python or call `lazy.matvec(&beta)` in Rust, you
ask for a result vector. LazyMatrix computes that vector without storing a
normalized matrix first.

Think of `beta` as one weight for each column. The product combines the columns
using those weights and returns one value per row.

For column normalization, you can get that result in three steps:

1. Divide the coefficients by the column scales.
2. Multiply the original matrix by those scaled coefficients.
3. Subtract the contribution from the column centers.

The transpose product uses the same idea: multiply by the original transpose,
correct for centering, and apply the scales. This lets algorithms such as
iterative least-squares solvers work with a normalized matrix while keeping its
sparse storage.

::: details The algebra

Let $X$ have $n$ rows and $p$ columns. The vector $c$ contains one center per
column, and $S$ is a diagonal matrix containing the column scales. The vector
$\mathbf{1}$ contains $n$ ones, so $\mathbf{1}c^\top$ repeats the centers in
every row. The normalized matrix is

$$
  \widetilde{X} = (X - \mathbf{1}c^\top)S^{-1}.
$$

For a coefficient vector $v$ of length $p$,

$$
  \widetilde{X}v = X(S^{-1}v) - \mathbf{1}(c^\top S^{-1}v).
$$

The final term subtracts the same scalar from every output entry. For a vector
$u$ of length $n$, the transpose product is

$$
  \widetilde{X}^\top u = S^{-1}\left( X^\top u - c\sum_{i=1}^n u_i \right).
$$

These identities describe the transformation. LazyMatrix can choose a different
order of calculation when that helps preserve floating-point accuracy.

:::

## Fit once, apply many times

Fitting is the step that reads your data to choose centers and scales. For
example, standardization computes a mean and standard deviation for each
column. Later products reuse those values without fitting again.

For validation or prediction, reuse the parameters fitted on your training
data. That keeps new observations in the same coordinates as the data used to
train the model. The new matrix can have a different number of rows, but it
needs the same features in the same order.

The [normalization guide](/guide/normalization#reuse-fitted-parameters) shows how
to fit parameters separately, apply them to new data, and save them for later.

## Row and ordered normalization

You can also normalize each row independently. This is useful when each row is
a sample whose overall size should be removed, such as scaling every sample to
unit length. Each new sample needs its own fitted row parameters.

In Rust, you can combine recipes in a sequence. For example, you might normalize
each row, then standardize the columns. Each stage sees the output of the
previous one, so changing the order changes the result. LazyMatrix saves the
stages in that order and applies them during operations without storing an
intermediate normalized matrix.

For new samples, keep the training column parameters and fit fresh row
parameters. The [normalization guide](/guide/normalization#compose-normalization-recipes)
explains how to do this and which storage backends support each kind of chain.

## Quantile fitting

Robust normalization uses a median and a quantile range instead of a mean and
standard deviation. The default range runs from the 25th to the 75th percentile,
covering the middle half of the data. This makes the fitted values less
sensitive to extreme observations.

LazyMatrix computes exact quantiles, including the zeros that sparse storage
leaves unstored. When a requested quantile falls between two sorted values, it
interpolates between them.

Exact quantiles can cost more to fit than means and standard deviations. With
sparse matrices stored by row (CSR) or chunked Zarr arrays, fitting may read the
data several times. That can matter for large inputs or slow storage. Once
fitted, these parameters are reused just like any others.

See [robust normalization](/guide/normalization#fit-robust-normalization) to
choose quantile bounds or enable unit-variance calibration.

## Numerical behavior

If a fitted scale is exactly zero, LazyMatrix replaces it with one to avoid
division by zero. Small nonzero scales are kept as they are. NaNs and infinities
are not filtered out, and missing values are not filled in. They can propagate
into fitted parameters and results.

Floating-point arithmetic also affects how products are computed. If your
values are large but differ only slightly, subtracting two large totals can
lose those small differences. LazyMatrix centers dense values before adding up
their products to reduce this problem.

## Choose lazy or eager

Lazy normalization is useful when you want to preserve sparse storage or avoid
allocating another large matrix. Eager normalization stores the normalized
values in a dense matrix, ready for later operations.

| What you store or compute     | Lazy                                  | Eager                                |
| ----------------------------- | ------------------------------------- | ------------------------------------ |
| Matrix storage                | Original data                         | Normalized dense values              |
| Fitted parameters             | Retained for reuse                    | Retained for reuse                   |
| Work before the first product | Fit centers and scales                | Fit parameters and normalize entries |
| Work during each product      | Apply normalization while multiplying | Multiply already normalized values   |

For scale, a dense copy of a matrix with one million rows and 1,000 columns
needs about 8 GB just for `f64` values. The column centers and scales need only
2,000 numbers. A lazy operator still needs the original matrix and working
memory for its operations. Rust can borrow the input; Python keeps its own copy
of the raw data.

Eager normalization can pay off when you make enough repeated products, but
sparse products can remain faster than dense ones. In Rust, you can also
normalize writable dense storage in place. The [matrix operations
guide](/guide/matrix-operations#choose-lazy-or-eager-storage) shows the conversion
methods, and [benchmarks](/benchmarks) compare time and memory for a complete fit.

Lazy and eager results can differ slightly because floating-point operations
happen in a different order.
