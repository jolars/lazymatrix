# Retrieve or set the row or column names of a LazyMatrix object.

Retrieve or set the row or column names of a LazyMatrix object.

Retrieve or set the row or column names of a LazyMatrix object.

## Usage

``` r
# S4 method for class 'LazyMatrix'
colnames(x)

# S4 method for class 'LazyMatrix'
colnames(x)
```

## Arguments

- x:

  A LazyMatrix object.

## Value

A character vector of column names, or NULL if the matrix has no column
names.

A character vector of column names, or NULL if the matrix has no column
names.

## Examples

``` r
mat_a <- base::matrix(rep(1, 6), nrow=2, ncol=3)
lazy_a <- LazyMatrix(mat_a, "sd", "mean")
colnames(lazy_a)
#> NULL
mat_a <- base::matrix(rep(1, 6), nrow=2, ncol=3)
lazy_a <- LazyMatrix(mat_a, "sd", "mean")
colnames(lazy_a)
#> NULL
```
