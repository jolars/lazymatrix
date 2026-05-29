# Computes the Frobenius norm of a LazyMatrix object.

Computes the Frobenius norm of a LazyMatrix object.

## Usage

``` r
norm(x)

# S4 method for class 'LazyMatrix'
norm(x)
```

## Arguments

- x:

  A LazyMatrix object.

## Value

A numeric scalar representing the Frobenius norm of the matrix.

## Examples

``` r
mat_a <- base::matrix(rnorm(50), nrow = 10, ncol = 5)
lazy_a <- LazyMatrix(mat_a, "sd", "mean")
norm(lazy_a)
#> [1] 6.708204
```
