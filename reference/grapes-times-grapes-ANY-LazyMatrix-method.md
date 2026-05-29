# Matrix multiplication for vector and LazyMatrix

Multiplies a LazyMatrix object by a vector.

## Usage

``` r
# S4 method for class 'ANY,LazyMatrix'
x %*% y
```

## Arguments

- x:

  A numeric vector.

- y:

  A LazyMatrix object.

## Value

A Matrix object of class dgeMatrix.

## Examples

``` r
mat_a <- base::matrix(rep(1, 6), nrow=2, ncol=3)
b <- c(1, 2)
lazy_a <- LazyMatrix(mat_a, "sd", "mean")
b %*% lazy_a
#>      [,1] [,2] [,3]
#> [1,]  NaN  NaN  NaN
```
