# Multiply element-wise two LazyColumn vectors

Computes the element-wise multiplication of two `LazyColumn` objects ,
preserving the lazy structure.

## Usage

``` r
# S4 method for class 'LazyColumn,LazyColumn'
e1 * e2
```

## Arguments

- e1:

  A `LazyColumn` object.

- e2:

  A `LazyColumn` object.

## Value

A numeric vector with the resulting vector.

## Examples

``` r
mat_a <- base::matrix(rnorm(12), nrow=3, ncol=4)
X <- LazyMatrix(mat_a, "sd", "mean")
lazy_col <- X[, 1]
lazy_col_2 <- X[, 2]
lazy_col_2 * lazy_col
#> [1] -0.9881860  0.1076205 -0.7555702
```
