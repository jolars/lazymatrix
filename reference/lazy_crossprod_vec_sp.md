# Fast crossprod for LazyMatrix (sparse case)

Fast crossprod for LazyMatrix (sparse case)

## Usage

``` r
lazy_crossprod_vec_sp(x, s, c, y)
```

## Arguments

- x:

  Sparse matrix (dgCMatrix)

- s:

  Column scale inverse vector (1 / col_scales)

- c:

  Column location vector (col_locations)

- y:

  Vector to multiply

## Value

Vector result of t(X_tilde) \* y
