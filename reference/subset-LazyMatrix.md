# Subset a LazyMatrix by columns

Subsets a `LazyMatrix` object by columns, returning either a
`LazyColumn` or a new `LazyMatrix` depending on the number of columns
selected. Row subsetting is not yet supported.

## Usage

``` r
# S4 method for class 'LazyMatrix,ANY,ANY,ANY'
x[i, j, ..., drop = TRUE]
```

## Arguments

- x:

  A `LazyMatrix` object.

- i:

  Row index. Must be missing as row subsetting is not yet supported.

- j:

  Column index. Either a single integer returning a `LazyColumn`, or a
  vector of integers returning a `LazyMatrix`.

- ...:

  Additional arguments (ignored).

- drop:

  Logical. Currently ignored.

## Value

A `LazyColumn` if a single column is selected, or a `LazyMatrix` if
multiple columns are selected.

## Examples

``` r
A <- Matrix::sparseMatrix(i = c(1,2,3), j = c(1,2,3), x = c(1,2,3))
lazy_m <- LazyMatrix(A, "sd", "mean")

# Single column → LazyColumn
lazy_col <- lazy_m[, 2]

# Multiple columns → LazyMatrix
lazy_subset <- lazy_m[, 1:3]
```
