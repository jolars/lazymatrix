# Compute the norm of a LazyMatrix or LazyColumn

Dispatches to the appropriate method based on the class of `x`.

## Usage

``` r
norm(x, ...)
```

## Arguments

- x:

  A `LazyMatrix` or `LazyColumn` object.

- ...:

  Additional arguments passed to methods, such as `type`.

## Value

For `LazyColumn`, a numeric scalar containing the Euclidean norm. For
`LazyMatrix`, a numeric scalar containing the Frobenius norm.
