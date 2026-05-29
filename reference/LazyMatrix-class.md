# LazyMatrix S4 class

An S4 class to represent a lazily transformed matrix with scaling and
location parameters.

## Slots

- `data`:

  The underlying matrix.

- `col_scales`:

  Numeric vector of column scales.

- `row_scales`:

  Numeric vector of row scales.

- `col_locations`:

  Numeric vector of column locations.

- `row_locations`:

  Numeric vector of row locations.

## Examples

``` r
mat <- matrix(1:6, nrow=2, ncol=3)
obj <- LazyMatrix(mat, "sd", "mean")
```
