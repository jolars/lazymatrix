# Performs a principal component analysis on the LazyMatrix object using irlba:s sparse svd.

Performs a principal component analysis on the LazyMatrix object using
irlba:s sparse svd.

## Usage

``` r
# S4 method for class 'LazyMatrix'
prcomp(x, retx = TRUE, tol = NULL, rank. = NULL, ...)
```

## Arguments

- x:

  a LazyMatrix object.

- retx:

  a logical value indicating whether the rotated variables should be
  returned.

- tol:

  a value indicating the magnitude below which components should be
  omitted. (Components are omitted if their standard deviations are less
  than or equal to tol times the standard deviation of the first
  component.) With the default null setting, no components are omitted
  (unless rank. is specified less than min(dim(x)).). Other settings for
  tol could be tol = 0 or tol = sqrt(.Machine\$double.eps), which would
  omit essentially constant components.

- rank.:

  optionally, a number specifying the maximal rank, i.e., maximal number
  of principal components to be used. Can be set as alternative or in
  addition to tol, useful notably when the desired rank is considerably
  smaller than the dimensions of the matrix.

- ...:

  Additional arguments passed to underlying methods.

## Value

A list of class `\"prcomp\"` containing:

- sdev:

  The standard deviations of the principal components (i.e., the square
  roots of the eigenvalues of the covariance/correlation matrix,
  calculated using the singular values of the data matrix).

- rotation:

  The matrix of variable loadings (columns are eigenvectors).

- x:

  If `retx` is TRUE, the value of the rotated data (centered and
  optionally scaled, multiplied by the rotation matrix).

- center:

  The centering used, or `FALSE`.

- scale:

  The scaling applied to the data, or `FALSE`

## Examples

``` r
set.seed(123)
mat_a <- matrix(rnorm(500), nrow=50, ncol=10)
lazy_a <- LazyMatrix(mat_a, "sd", "mean")
pca_lazy <- prcomp(lazy_a)
#> Warning: You're computing too large a percentage of total singular values, use a standard svd instead.
```
