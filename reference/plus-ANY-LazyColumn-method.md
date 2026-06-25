# Vector Addition between regular vector and `LazyColumn`

Sums a regular base vector and a `LazyColumn` vector.

## Usage

``` r
# S4 method for class 'ANY,LazyColumn'
e1 + e2
```

## Arguments

- e1:

  A numeric vector.

- e2:

  A `LazyColumn` object.

## Value

A numeric vector.

## Examples

``` r
mat_a <- base::matrix(rnorm(12), nrow=3, ncol=4)
b <- rnorm(nrow(mat_a))
lazy_a <- LazyMatrix(mat_a, "sd", "mean")
lazy_col <- lazy_a[, 2]
b + lazy_col
#> [1] -1.5818428 -0.2352945  1.4523078
```
