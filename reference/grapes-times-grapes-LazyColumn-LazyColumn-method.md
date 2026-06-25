# Perform the dot product between two LazyColumn vectors

Computes the inner product between two `LazyColumn` objects.

## Usage

``` r
# S4 method for class 'LazyColumn,LazyColumn'
x %*% y
```

## Arguments

- x:

  A `LazyColumn` object.

- y:

  A `LazyColumn` object.

## Value

A numeric value with the resulting scalar.

## Examples

``` r
mat_a <- base::matrix(rnorm(12), nrow=3, ncol=4)
X <- LazyMatrix(mat_a, "sd", "mean")
b <- rnorm(nrow(X))
lazy_col <- X[, 1]
lazy_col_2 <- X[, 2]
lazy_col %*% lazy_col_2
#>          [,1]
#> [1,] -0.10481
```
