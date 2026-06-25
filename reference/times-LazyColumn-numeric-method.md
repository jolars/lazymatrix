# Multiply a LazyColumn by a numeric scalar or vector

Computes the element-wise multiplication of a `LazyColumn` object with a
numeric value, preserving the lazy structure. If `e2` is a scalar,
scalar multiplication is performed. If `e2` is a vector of the same
length as the column, element-wise multiplication is performed.

## Usage

``` r
# S4 method for class 'LazyColumn,numeric'
e1 * e2
```

## Arguments

- e1:

  A `LazyColumn` object.

- e2:

  A numeric scalar or vector.

## Value

A numeric vector with the resulting vector.

## Examples

``` r
mat_a <- base::matrix(rnorm(12), nrow=3, ncol=4)
X <- LazyMatrix(mat_a, "sd", "mean")
lazy_col <- X[, 1]
lazy_col * 2
#> [1] -1.5710598 -0.6803584  2.2514181
lazy_col * rnorm(nrow(X))
#> [1] -1.5285243 -0.2724544  1.3117363
```
