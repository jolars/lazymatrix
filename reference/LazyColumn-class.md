# LazyColumn S4 class

An S4 class to represent a column vector as a subset of a
LazyMatrix-object

## Slots

- `data`:

  The underlying data column vector.

- `scale`:

  Numeric scalar containing column-scale parameter.

- `location`:

  Numeric scalar containing the column-location parameter.

## Examples

``` r
mat <- matrix(1:6, nrow = 2, ncol = 3)
lazy_mat <- LazyMatrix(mat, "sd", "mean")
lazy_column <- lazy_mat[, 2]
```
