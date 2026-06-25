# Matrix multiplication for `LazyMatrix` and vector

Multiplies a `LazyMatrix` object by a vector.

## Usage

``` r
# S4 method for class 'LazyMatrix,ANY'
x %*% y
```

## Arguments

- x:

  A `LazyMatrix` object.

- y:

  A numeric vector.

## Value

A numeric matrix.

## Examples

``` r
mat_a <- base::matrix(rep(1, 6), nrow=2, ncol=3)
b <- c(1, 2, 3)
lazy_a <- LazyMatrix(mat_a, "sd", "mean")
lazy_a %*% b
#>      [,1]
#> [1,]  NaN
#> [2,]  NaN
```
