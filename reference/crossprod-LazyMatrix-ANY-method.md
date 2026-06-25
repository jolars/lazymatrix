# Crossproduct for `LazyMatrix`

Computes the crossproduct of a `LazyMatrix` object as it's Gram matrix
or computes the transposed matrix-vector multiplication.

## Usage

``` r
# S4 method for class 'LazyMatrix,ANY'
crossprod(x, y = NULL)
```

## Arguments

- x:

  A `LazyMatrix` object.

- y:

  An optional numeric vector or matrix. If NULL, computes the Gram
  matrix of x.

## Value

A matrix: the Gram matrix if y is NULL, otherwise the crossproduct
result.

## Examples

``` r
mat_a <- matrix(rep(1, 6), nrow=2, ncol=3)
b <- c(1, 2)
lazy_a <- LazyMatrix(mat_a, scale="sd", location="mean")
crossprod(lazy_a)
#> 3 x 3 Matrix of class "dgeMatrix"
#>      [,1] [,2] [,3]
#> [1,]  NaN  NaN  NaN
#> [2,]  NaN  NaN  NaN
#> [3,]  NaN  NaN  NaN
crossprod(lazy_a, b)
#>      [,1]
#> [1,]  NaN
#> [2,]  NaN
#> [3,]  NaN
```
