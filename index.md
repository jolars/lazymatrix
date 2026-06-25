# lazymatrix

`lazymatrix` provides a framework for working with transformed sparse
data matrices using [lazy
evaluation](https://en.wikipedia.org/wiki/Lazy_evaluation). This
approach is particularly useful for handling large sparse matrices where
transformations (e.g., centering and scaling) would otherwise result in
dense matrices, consuming significant memory and computational
resources.

## Installation

You can install the development version of lazymatrix from
[GitHub](https://github.com/) with:

``` r

# install.packages("pak")
pak::pak("vsegersall/lazymatrix")
```

Note: Once the package is accepted to CRAN, you can also install it via:

``` r

install.packages("lazymatrix")
#> Installing package into 'C:/Users/vikto/AppData/Local/Temp/Rtmp4yD82T/temp_libpath969c781e6602'
#> (as 'lib' is unspecified)
#> Warning: package 'lazymatrix' is not available for this version of R
#> 
#> A version of this package for your version of R might be available elsewhere,
#> see the ideas at
#> https://cran.r-project.org/doc/manuals/r-patched/R-admin.html#Installing-packages
```

## Example

This is a basic example which shows you how to solve a common problem:

``` r

library(lazymatrix)
#> 
#> Attaching package: 'lazymatrix'
#> The following object is masked from 'package:base':
#> 
#>     norm
set.seed(123)
sparse_matrix <- Matrix::Matrix(0, 5, 3)
sparse_matrix[sample(length(sparse_matrix), 5)] <- rnorm(5)
b <- rnorm(3)
lazy_matrix <- LazyMatrix(sparse_matrix, scale = "sd", location = "mean")
print(lazy_matrix)
#> An object of class "LazyMatrix"
#> Slot "data":
#> 5 x 3 sparse Matrix of class "dgCMatrix"
#>                                      
#> [1,]  .         1.55870831  .        
#> [2,]  .         .           .        
#> [3,]  .         .           .        
#> [4,] -0.5604756 0.07050839 -0.2301775
#> [5,]  .         0.12928774  .        
#> 
#> Slot "col_scales":
#> [1] 0.2506523 0.6769030 0.1029385
#> 
#> Slot "row_scales":
#> [1] 0.89992066 0.00000000 0.00000000 0.31560781 0.07464431
#> 
#> Slot "col_locations":
#> [1] -0.1120951  0.3517009 -0.0460355
#> 
#> Slot "row_locations":
#> [1]  0.51956944  0.00000000  0.00000000 -0.24004825  0.04309591
lazy_matrix %*% b
#> 5 x 1 Matrix of class "dgeMatrix"
#>           [,1]
#> [1,] -1.751397
#> [2,]  2.225999
#> [3,]  2.225999
#> [4,] -4.596694
#> [5,]  1.896092
```

## Development Status

This package is currently in active development. While the core
functionality is stable, the API may evolve as the project matures.
Contributions are currently closed while the package is prepared for its
first CRAN release. However, bug reports and feature requests are
welcome via the GitHub Issue Tracker.
