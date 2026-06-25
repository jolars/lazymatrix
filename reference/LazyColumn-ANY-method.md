# Vector subtraction between a `LazyColumn` vector and a regular R vector.

Subtracts a numeric R vector from a `LazyColumn` vector.

## Usage

``` r
# S4 method for class 'LazyColumn,ANY'
e1 - e2
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
lazy_col - b
#> [1]  0.6383079 -0.3267100  2.0435410
```
