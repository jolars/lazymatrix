# How it works

Let $X$ have $n$ rows and $p$ columns. Let $c$ contain the column centers,
and let $S$ be the diagonal matrix of column scales. The normalized design is

$$
\widetilde{X} = (X - \mathbf{1}c^\top)S^{-1}.
$$

Subtracting a nonzero center changes every implicit zero in a sparse column.
Storing those entries can require a full dense matrix. LazyMatrix instead keeps
the original storage and folds normalization into each operation.

## Matrix–vector products

For a coefficient vector $v$,

$$
\widetilde{X}v = X(S^{-1}v) - \mathbf{1}(c^\top S^{-1}v).
$$

Scale the coefficients, multiply the original matrix, and subtract a scalar
from each output entry. For a row-space vector $u$, the transpose product is

$$
\widetilde{X}^\top u
= S^{-1}\left(X^\top u - c\sum_{i=1}^n u_i\right).
$$

Both identities let the backend use its original dense, sparse, or chunked
storage. For centered dense matrices, LazyMatrix subtracts the center and
applies the scale at each entry before accumulating products. This avoids
cancellation when a column's offset is much larger than its variation.
Allocating, reusable-output, fused, and workspace products share this policy
across faer, nalgebra, and ndarray. Bounded stack panels follow the matrix's
strides while preserving each output's summation order. These kernels need no
heap scratch or normalized design copy, but can cost more time than a native
raw product.
Uncentered products retain the native multiplication path. Sparse and chunked
storage use the identities above to preserve their storage and access patterns.

Logical column and sparse row views also expose normalized values without
allocating an entire normalized matrix.

## Fit once, apply many times

Fitting computes column centers and scales. When both are enabled, scaling
statistics use the centered columns. Sparse statistics include the contribution
of implicit zeros. Exact zero fitted scales become one, and nonfinite values
retain their IEEE behavior.

`NormalizationParams` stores the fitted centers and scales independently of the
matrix backend. Reuse those parameters for prediction data or after converting
to an eager representation.

## Quantile fitting

Median centering and quantile-range scaling use exact order statistics. The
quantile at probability `q` interpolates the neighboring sorted values at
position `(nrows - 1) * q`, matching NumPy's default linear method for finite
data. The default range is `q = 0.25` to `q = 0.75`. Centers and scales remain
ordinary fitted parameters and can be reused on new matrices.

Dense fitting sorts a copied column. CSC fitting combines duplicate raw entries
at each cell and sorts the stored logical values, representing implicit zeros
by their count. Both reuse a working column and run serially, including with the
`parallel` feature. Sorting costs O(nrows log nrows) time per dense column and
O(nnz log nnz) per CSC column, with O(nrows) and O(nnz) scratch, respectively.
Combining unsorted duplicates can require additional sorting workspace.

CSR and Zarr fitting maintain O(ncols) selection state. Each pass counts values
around pivots and narrows the bounds to observed neighbors until every requested
rank is known. A median and a quantile range share passes when fitted together.
CSR scans rows, combines duplicate cells, and accounts for implicit zeros without
converting storage. Zarr reads chunks serially, including fill values and
excluding edge padding. Chunk and codec memory, including outer shards, remains
additional to the selection state.

The number of fitting passes depends on the values. Widely separated exponents
can require many passes, so exact quantiles may be expensive on storage-backed
inputs. Existing normalization choices without quantiles retain their fitting
paths and, for Zarr, the bound of two scans. [Quantile fitting measurements](https://github.com/jolars/lazymatrix/tree/main/benches/quantile_fitting)
record time, peak additional allocation, and storage scans.

Ranges use weighted gaps between order statistics to preserve small variation
around large offsets. Empty columns and columns containing NaN produce NaN;
missing values are not skipped. Infinities retain their order. Equal infinite
interpolation endpoints retain that infinity, while interpolation between
opposing infinities produces NaN. Subtraction of nonfinite quantiles follows
IEEE arithmetic. Only exact zero fitted scales become one, including when a
nonconstant column has zero IQR. These policies differ from scikit-learn's
missing-value and near-constant handling.

## Choose lazy or eager

| Consideration                 | Lazy                                   | Eager                                            |
| ----------------------------- | -------------------------------------- | ------------------------------------------------ |
| Normalized storage            | Original matrix plus fitted parameters | Normalized dense matrix plus fitted parameters   |
| Sparse input                  | Preserves sparse storage               | Materializes implicit entries into dense storage |
| Construction                  | Fits statistics                        | Fits statistics and normalizes entries           |
| Repeated operations           | Applies normalization during products  | Operates on already normalized values            |
| Existing writable dense input | Keeps normalization implicit           | Can normalize in place with `into_eager`         |

For a dense copy of an $n \times p$ matrix of `f64` values, the values alone
require $8np$ bytes. A borrowed lazy operator adds fitted parameters and operation
scratch without copying the design. Overall memory also depends on whether the
caller retains the source matrix.

Eager normalization can pay off over repeated operations, but sparse products
can remain faster than dense products. [Benchmarks](/benchmarks) compare the
construction and execution phases separately. Lazy and eager arithmetic can
also differ slightly because the floating-point operation order changes.
