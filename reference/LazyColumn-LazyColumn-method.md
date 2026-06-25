# Vector Subtraction between two `LazyColumn` vectors.

Subtracts a `LazyColumn` vector from another `LazyColumn` vector.

## Usage

``` r
# S4 method for class 'LazyColumn,LazyColumn'
e1 - e2
```

## Arguments

- e1:

  A `LazyColumn` object.

- e2:

  A `LazyColumn` object.

## Value

A numeric vector.

## Examples

``` r
mat_a <- base::matrix(rnorm(12), nrow = 3, ncol = 4)
lazy_a <- LazyMatrix(mat_a, "sd", "mean")
lazy_col <- lazy_a[, 2]
lazy_col_2 <- lazy_a[, 3]
lazy_col_2 - lazy_col
#> [1] -2.299519  1.038851  1.260668
```
