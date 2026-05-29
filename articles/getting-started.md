# Getting started with LazyMatrix

## Why LazyMatrix?

When working with sparse matrices in R, a challenge is performing
transformations on the matrix without loosing sparsity. An operation
such as dividing the elements with it’s sample variance and subtracting
the sample mean removes sparsity and risks occupying a lot of memory. As
memory often is limited, creating copies of the transformed matrix,
which is no longer sparse, may not be desirable or even feasible. This
creates issues when we aim to work with a normalized version of the
design matrix for statistical algorithms such as principal component
analysis or linear regression. A solution is working with the
`LazyMatrix`-object, which never stores copies of the transformed matrix
but performs operations only when neccessary for the desired algorithm.

## Mathematical Background

Assume we want to perform the matrix operation
``` math
\tilde{A}b
```

where $`\tilde{A}`$ is a scaled version of the matrix $`A`$ being
$`n\times p`$, and the vector $`b\in R^p`$. Assume further that we want
to transform $`A`$ by subtracting some location parameter $`C`$ and
multiplying by the inverse of a diagonal scaling matrix $`S`$. This
operation is defined as
``` math
(A-C)S^{-1}b.
```

Now, assuming that $`A`$ is sparse, subtracting $`C`$ will result in a
non-sparse matrix as all zero, or close to zero, elements will now be
non-zero if $`c_i\neq0`$. The simple solution is basically to expand the
expression so that we get
``` math
(A-C)S^{-1}b=AS^{-1}b-CS^{-1}b
```

hence an equivalent formalization where we avoid materializing $`A`$.
This reasoning can be applied to all matrix operations but we will see
that just a few of them are neccessary to perform complex statistical
algorithms.

## Usage

The way of doing this in R is by creating the S4 class `LazyMatrix`,
which stores the original data matrix, location and scale parameters.

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
lazy_matrix
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
```

Mainly we are interested in the location and scale parameters of the
columns, as these usually represents the variables. However, storing the
location and scale for the rows as well allows for working with the
transpose of the matrix by just inverting the rows and columns.

``` r

lazy_transpose <- t(lazy_matrix)
```

As we aim for `LazyMatrix` to be as user-friendly as possible, the
operations are defined as is standard in R. Functions from `base` such
as `nrow`, `ncol` and `colnames` are designed to work equivalently. For
matrix operations, we design them to be as similar to the normal syntax
as possible. Below, we show examples of $`\tilde{A}b`$ or
$`\tilde{A}^Tc`$, where $`b\in R^p`$ and $`c\in R^n`$.

``` r

set.seed(123)
b <- rnorm(ncol(lazy_matrix))
c <- rnorm(nrow(lazy_matrix))
product <- lazy_matrix %*% b
cross_product <- crossprod(lazy_matrix, c)
product
#> 5 x 1 Matrix of class "dgeMatrix"
#>             [,1]
#> [1,]  0.03598638
#> [2,]  0.56601735
#> [3,]  0.56601735
#> [4,] -1.69007478
#> [5,]  0.52205370
cross_product
#>            [,1]
#> [1,] -0.5339126
#> [2,] -0.6083534
#> [3,] -0.5339126
```

## Design

This package has been designed carefully to align with the framework
`Matrix`. As `Matrix` already has efficient ways of handling sparse
matrices using `sparseMatrix`, `lazymatrix` should be seen as an
extension which allows for performing efficient lazy computations.
