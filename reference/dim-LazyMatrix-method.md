# Returns the dimension of a LazyMarix Object.

Returns the dimension of a LazyMarix Object.

## Usage

``` r
# S4 method for class 'LazyMatrix'
dim(x)
```

## Arguments

- x:

  A LazyMatrix object.

## Value

For an array (and hence in particular, for a matrix) dim retrieves the
dim attribute of the object. It is NULL or a vector of mode integer. The
replacement method changes the "dim" attribute (provided the new value
is compatible) and removes any "dimnames" and "names" attributes.

## Examples

``` r
mat_a <- base::matrix(rep(1, 6), nrow=2, ncol=3)
lazy_a <- LazyMatrix(mat_a, "sd", "mean")
dim(lazy_a)
#> [1] 2 3
```
