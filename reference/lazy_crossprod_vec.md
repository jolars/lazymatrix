# Fast crossprod for LazyMatrix (dense case)

Fast crossprod for LazyMatrix (dense case)

## Usage

``` r
lazy_crossprod_vec(x, s, c, y)
```

## Arguments

- x:

  Dense matrix

- s:

  Column scale inverse vector (1 / col_scales)

- c:

  Column location vector (col_locations)

- y:

  Vector to multiply

## Value

Vector result of t(X_tilde) \* y
