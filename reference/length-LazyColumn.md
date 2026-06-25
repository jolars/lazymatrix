# Get the length of a LazyColumn

Get the length of a LazyColumn

## Usage

``` r
# S4 method for class 'LazyColumn'
length(x)
```

## Arguments

- x:

  A LazyColumn object.

## Value

An integer value containing the number of elements within the vector.

## Examples

``` r
mat_a <- base::matrix(rep(1, 6), nrow=2, ncol=3)
lazy_a <- LazyMatrix(mat_a, "sd", "mean")
lazy_col <- lazy_a[, 2]
length(lazy_col)
#> [1] 2
```
