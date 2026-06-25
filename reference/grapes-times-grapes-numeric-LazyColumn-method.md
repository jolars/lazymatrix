# Perform the dot product between a numeric vectorand a LazyColumn

Computes the inner product of a numeric vector and a `LazyColumn`
object. If `x` is a scalar, scalar multiplication is performed. If `x`
is a vector of the same length as the column, the dot product is
performed.

## Usage

``` r
# S4 method for class 'numeric,LazyColumn'
x %*% y
```

## Arguments

- x:

  A numeric scalar or vector.

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
b %*% lazy_col
#>          [,1]
#> [1,] 1.408863
```
