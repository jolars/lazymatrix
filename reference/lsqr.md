# Performs least squares estimation on LazyMatrix object using the iterative lsqr algorithm.

Performs least squares estimation on LazyMatrix object using the
iterative lsqr algorithm.

## Usage

``` r
lsqr(x, y, ...)

# S4 method for class 'LazyMatrix'
lsqr(x, y)
```

## Arguments

- x:

  A LazyMatrix object.

- y:

  A response vector.

- ...:

  Additional arguments (currently unused).

## Value

A Matrix-object with the regression coefficients of the covariates.

## Examples

``` r
set.seed(123)
mat_a <- base::matrix(rnorm(500), nrow = 50, ncol = 10)
lazy_a <- LazyMatrix(mat_a, "sd", "mean")
response_vector <- rnorm(nrow(mat_a))
lsqr(lazy_a, response_vector)
#>              [,1]
#>  [1,]  0.08739095
#>  [2,]  0.01530533
#>  [3,]  0.19572144
#>  [4,] -0.14977447
#>  [5,]  0.07253498
#>  [6,] -0.09294621
#>  [7,] -0.12683084
#>  [8,]  0.11205506
#>  [9,]  0.12703560
#> [10,]  0.05902898
```
