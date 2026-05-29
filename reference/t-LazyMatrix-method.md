# Given a LazyMatrix x, t returns the transpose of x.

Given a LazyMatrix x, t returns the transpose of x.

## Usage

``` r
# S4 method for class 'LazyMatrix'
t(x)
```

## Arguments

- x:

  A LazyMatrix object.

## Value

A LazyMatrix object with the transposed data matrix.

## Examples

``` r
mat_a <- base::matrix(rep(1, 6), nrow=2, ncol=3)
lazy_a <- LazyMatrix(mat_a, "sd", "mean")
lazy_t <- t(lazy_a)
```
