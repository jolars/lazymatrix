# Vector Subtraction between regular vector and `LazyColumn`

Subtracts a `LazyColumn` vector from a base vector.

## Usage

``` r
# S4 method for class 'ANY,LazyColumn'
e1 - e2
```

## Arguments

- e1:

  A `LazyColumn` object.

- e2:

  A numeric vector.

## Value

A numeric vector.

## Examples

``` r
mat_a <- base::matrix(rnorm(12), nrow=3, ncol=4)
b <- rnorm(nrow(mat_a))
lazy_a <- LazyMatrix(mat_a, "sd", "mean")
lazy_col <- lazy_a[, 2]
b - lazy_col
#> [1]  3.0926947 -1.6888408 -0.4573915
```
