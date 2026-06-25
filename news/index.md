# Changelog

## lazymatrix 0.1.0

*Initial CRAN submission.*

### New Features

- Introduced lazy evaluation framework for working with normalized
  sparse matrices
- Represent large sparse matrices symbolically with the S4 object
  `LazyMatrix`
- Fundamental matrix operations like matrix-vector multiplication and
  transpose matrix-vector multiplication supported for the `LazyMatrix`
  object
- Added support for subsetting methods, mirroring matrix objects from
  base R and `Matrix`. The symbolic equivalent of column vectors is
  called `LazyColumn`
- Implemented specific statistical algorithms, LSQR, truncated partial
  SVD, for showcasing how `lazymatrix` can be implemented
- Optimized internal C++ loops for sparse matrix operations using
  `RcppArmadillo`.

### Improvements

- Optimized matrix multiplication for lazy objects to avoid
  materialization
- Added vignette demonstrating typical workflow with large sparse
  datasets

### Bug Fixes

- No known bugs fixed in initial release

### Documentation

- Added comprehensive `roxygen2` documentation for all exported
  functions
- Included example code in each function’s documentation
- Wrote getting started vignette covering installation and basic usage

### Internal Changes

- Utilized Matrix package for underlying sparse matrix representation
- Implemented partial SVD using `irlba`
