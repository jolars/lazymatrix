# Singular Value decomposition for `LazyMatrix`.

Performs lazy SVD using irlba for partial Singular value decomposition
on sparse matrices.

## Usage

``` r
# S4 method for class 'LazyMatrix'
svd(x, nu = min(n, p), nv = min(n, p))
```

## Arguments

- x:

  A `LazyMatrix` object.

- nu:

  number of left singular vectors to estimate (defaults to nv).

- nv:

  number of right singular vectors to estimate.

## Value

A list with entries:

- d:

  max(nu, nv) approximate singular values

- u:

  nu approximate left singular vectors (only when right_only=FALSE)

- v:

  nv approximate right singular vectors

- iter:

  The number of Lanczos iterations carried out

- mprod:

  The total number of matrix vector products carried out

## Examples

``` r
set.seed(123)
mat_a <- matrix(rnorm(500), nrow = 50, ncol = 10)
lazy_a <-LazyMatrix(mat_a, scale = "sd", location = "mean")
S <- svd(lazy_a)
#> Warning: You're computing too large a percentage of total singular values, use a standard svd instead.
# Receive singular values with
S$d
#> [1] 9.132746 8.248143 7.827175 7.606301 7.062690
```
