# Perform the norm of a LazyColumn vector

Computes the norm of a `LazyColumn` object.

## Usage

``` r
# S4 method for class 'LazyColumn'
norm(x, type = "2")
```

## Arguments

- x:

  A `LazyColumn` object.

- type:

  A character value defining type of norm. Default is Euclidean norm.

## Value

A numeric value with the resulting scalar.

## Examples

``` r
mat_a <- base::matrix(rnorm(12), nrow=3, ncol=4)
X <- LazyMatrix(mat_a, "sd", "mean")
b <- rnorm(nrow(X))
lazy_col <- X[, 1]
norm(lazy_col)
#> [1] 1.414214
```
