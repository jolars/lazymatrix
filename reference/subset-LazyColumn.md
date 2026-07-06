# Subset a LazyColumn

Subsets a `LazyColumn` object using standard R indexing rules: positive
integers, negative integers, logical vectors (with recycling), character
vectors (if named), zero, or missing (nothing).

## Usage

``` r
# S4 method for class 'LazyColumn,ANY,ANY,ANY'
x[i, j, ..., drop = TRUE]
```

## Arguments

- x:

  A `LazyColumn` object.

- i:

  Index: positive integers, negative integers, logical vector, character
  vector (if named), zero, or missing (nothing).

- j:

  Not used for `LazyColumn` (1-dimensional). Must be missing.

- ...:

  Additional arguments (ignored).

- drop:

  Logical. Currently ignored — single element extraction always returns
  a plain scaled numeric, consistent with vector subsetting.

## Value

A `LazyColumn` when multiple elements are selected, or a scaled numeric
value when a single element is selected.

## Examples

``` r
set.seed(123)
mat_a <- matrix(rnorm(500), nrow = 50, ncol = 10)
lazy_a <- LazyMatrix(mat_a, "sd", "mean")
lazy_c <- lazy_a[, 2]

# Single element → plain scaled numeric
lazy_c[2]
#> [1] -0.193225

# Multiple elements → LazyColumn
lazy_c[c(1, 3, 5)]
#> An object of class "LazyColumn"
#> Slot "data":
#> [1]  0.25331851 -0.04287046 -0.22577099
#> 
#> Slot "scale":
#> [1] 0.2417669
#> 
#> Slot "location":
#> [1] -0.005107643
#> 
```
