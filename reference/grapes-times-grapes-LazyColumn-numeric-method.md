# Perform the dot product between a LazyColumn and a numeric vector

Computes the inner product of a `LazyColumn` object and a numeric
vector. If `y` is a scalar, scalar multiplication is performed. If `y`
is a vector of the same length as the column, the dot product is
performed.

## Usage

``` r
# S4 method for class 'LazyColumn,numeric'
x %*% y
```

## Arguments

- x:

  A `LazyColumn` object.

- y:

  A numeric scalar or vector.

## Value

A numeric value with the resulting scalar.

## Examples

``` r
mat_a <- base::matrix(rnorm(12), nrow=3, ncol=4)
X <- LazyMatrix(mat_a, "sd", "mean")
b <- rnorm(nrow(X))
lazy_col <- X[, 1]
lazy_col %*% b
#>            [,1]
#> [1,] -0.9452133
```
