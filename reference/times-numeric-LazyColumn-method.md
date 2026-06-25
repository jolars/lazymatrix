# Multiply a numeric scalar or vector by a LazyColumn

Computes the element-wise multiplication of a `LazyColumn` object with a
numeric value, preserving the lazy structure. If `e1` is a scalar,
scalar multiplication is performed. If `e1` is a vector of the same
length as the column, element-wise multiplication is performed.

## Usage

``` r
# S4 method for class 'numeric,LazyColumn'
e1 * e2
```

## Arguments

- e1:

  A numeric scalar or vector.

- e2:

  A `LazyColumn` object.

## Value

A numeric vector with the resulting vector.

## Examples

``` r
mat_a <- base::matrix(rnorm(12), nrow=3, ncol=4)
X <- LazyMatrix(mat_a, "sd", "mean")
lazy_col <- X[, 1]
2* lazy_col
#> [1]  2.097732 -1.885311 -0.212421
rnorm(nrow(X)) * lazy_col
#> [1] -0.58444438 -0.79645364  0.08307805
```
