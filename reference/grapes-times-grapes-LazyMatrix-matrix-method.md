# Matrix multiplication for LazyMatrix and matrix-object.

Multiplies a LazyMatrix object by a matrix

## Usage

``` r
# S4 method for class 'LazyMatrix,matrix'
x %*% y
```

## Arguments

- x:

  A LazyMatrix object.

- y:

  A matrix-object.

## Value

A matrix-object with the product of the lazy and non lazy object.

## Examples

``` r
mat_a <- matrix(rep(1, 6), nrow = 2, ncol = 3)
lazy_a <- LazyMatrix(mat_a, "sd", "mean")
set.seed(123)
m <- matrix(rnorm(6), nrow = 3, ncol = 2)
lazy_a %*% m
#>      [,1] [,2]
#> [1,]  NaN  NaN
#> [2,]  NaN  NaN
```
