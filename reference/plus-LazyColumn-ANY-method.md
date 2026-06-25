# Vector Addition between `LazyColumn` and regular vector

Sums a `LazyColumn` vector and a regular base vector.

## Usage

``` r
# S4 method for class 'LazyColumn,ANY'
e1 + e2
```

## Arguments

- e1:

  A `LazyColumn` object.

- e2:

  A numeric vector.

## Value

A numeric vector.

## Examples

``` r
mat_a <- base::matrix(rnorm(12), nrow=3, ncol=4)
b <- rnorm(nrow(mat_a))
lazy_a <- LazyMatrix(mat_a, "sd", "mean")
lazy_col <- lazy_a[,2]
lazy_col + b
#> [1]  1.2747661  0.3230798 -1.2065788
```
