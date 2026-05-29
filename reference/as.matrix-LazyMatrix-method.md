# Attempts to turn the data matrix to a scaled matrix-object.

Attempts to turn the data matrix to a scaled matrix-object.

## Usage

``` r
# S4 method for class 'LazyMatrix'
as.matrix(x)
```

## Arguments

- x:

  A LazyMatrix object.

## Value

A matrix-object with scaled entries.

## Examples

``` r
mat_a <- base::matrix(rep(1, 6), nrow=2, ncol=3)
lazy_a <- LazyMatrix(mat_a, "sd", "mean")
as.matrix(lazy_a)
#>      [,1] [,2] [,3]
#> [1,]  NaN  NaN  NaN
#> [2,]  NaN  NaN  NaN
```
