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
storage. Logical column and sparse row views also expose normalized values
without allocating an entire normalized matrix.

## Fit once, apply many times

Fitting computes column centers and scales. When both are enabled, scaling
statistics use the centered columns. Sparse statistics include the contribution
of implicit zeros. Exact zero fitted scales become one, and nonfinite values
retain their IEEE behavior.

`NormalizationParams` stores the fitted centers and scales independently of the
matrix backend. Reuse those parameters for prediction data or after converting
to an eager representation.

## Choose lazy or eager

| Consideration                 | Lazy                                   | Eager                                            |
| ----------------------------- | -------------------------------------- | ------------------------------------------------ |
| Normalized storage            | Original matrix plus fitted parameters | Normalized dense matrix plus fitted parameters   |
| Sparse input                  | Preserves sparse storage               | Materializes implicit entries into dense storage |
| Construction                  | Fits statistics                        | Fits statistics and normalizes entries           |
| Repeated operations           | Applies normalization corrections      | Operates on already normalized values            |
| Existing writable dense input | Keeps normalization implicit           | Can normalize in place with `into_eager`         |

For a dense copy of an $n \times p$ matrix of `f64` values, the values alone
require $8np$ bytes. A borrowed lazy operator adds fitted parameters and operation
scratch without copying the design. Overall memory also depends on whether the
caller retains the source matrix.

Eager normalization can pay off over repeated operations, but sparse products
can remain faster than dense products. [Benchmarks](/benchmarks) compare the
construction and execution phases separately. Lazy and eager arithmetic can
also differ slightly because the floating-point operation order changes.
