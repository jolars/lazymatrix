# Set names for a LazyColumn

Assigns names to the underlying data of a `LazyColumn`, returning a new
`LazyColumn` with the same scale and location as the original. This
enables name-based subsetting via `[` on `LazyColumn` objects, mirroring
[`stats::setNames()`](https://rdrr.io/r/stats/setNames.html) for
ordinary vectors.

## Usage

``` r
# S4 method for class 'LazyColumn,character'
setNames(object = nm, nm)
```

## Arguments

- object:

  A `LazyColumn` object.

- nm:

  A character vector of names, with length equal to
  `length(object@data)`.

## Value

A new `LazyColumn` with named data, preserving the original scale and
location.

## Examples

``` r
set.seed(123)
mat_a <- matrix(rnorm(500), nrow = 50, ncol = 10)
lazy_a <- LazyMatrix(mat_a, "sd", "mean")
lazy_c <- lazy_a[, 2]
lazy_named <- setNames(lazy_c[1:26], letters[1:26])
lazy_named["a"]
#>         a 
#> 0.2096104 
```
