# Returns the number of rows of the data matrix

Returns the number of rows of the data matrix

## Usage

``` r
# S4 method for class 'LazyMatrix'
nrow(x)
```

## Arguments

- x:

  A LazyMatrix object.

## Value

an integer of length 1 or NULL.

## Examples

``` r
mat_a <- base::matrix(rep(1, 6), nrow=2, ncol=3)
lazy_a <- LazyMatrix(mat_a, "sd", "mean")
nrow(lazy_a)
#> [1] 2
```
