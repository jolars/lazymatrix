# Performance of LazyMatrix

## Why Lazy Computation?

A legitimate question when implementing a new framework in a programming
language is how well it performs compared to already existing solutions.
The goal here is to show that using `lazymatrix` improves performance
and is especially effective the larger and sparser the original matrix
is. We compare the proposed method aganinst the existing solution such
as materializing the sparse matrix with
[`base::scale()`](https://rdrr.io/r/base/scale.html) and we will note
that this solution is not scaleable when sparsity and dimensionality is
high enough. Benchmarking is done using the package `bench`.

## Parametrized Benchmark

We start by defining a function which allows us to create sparse
matrices with different dimensions and sparsity, parameters being `n` as
the number of rows, `p` as the number of columns and `sparsity` as the
density of non-zero values. These values are generated from a standard
normal distribution.

``` r

library(lazymatrix)
#> 
#> Attaching package: 'lazymatrix'
#> The following object is masked from 'package:base':
#> 
#>     norm
create_sparsematrix <- function(n, p, sparsity) {
  n_nonzero <- round(sparsity * n * p)

  i <- sample(1:n, n_nonzero, replace = TRUE)
  j <- sample(1:p, n_nonzero, replace = TRUE)

  pairs <- unique(data.frame(i = i, j = j))
  i <- pairs$i
  j <- pairs$j

  x <- rnorm(length(i))
  A <- Matrix::sparseMatrix(i = i, j = j, x = x, dims = c(n, p))
  A
}
```

## Benchmarking: Lazy Multiplication

We start by comparing the matrix multiplication method `%*%`. We define
a function which takes in the sparse matrix as the first argument,
computes the scaled dense version `A` and defines the
`LazyMatrix`-object `X` of the same matrix. Then, we use
[`bench::mark()`](https://bench.r-lib.org/reference/mark.html) to
mesaure the performance of computing the product regularly or lazily.

``` r

library(bench)
bench_multiply <- function(sparse_matrix, b) {
  A <- scale(sparse_matrix, center = TRUE, scale = TRUE)
  X <- LazyMatrix(sparse_matrix, "sd", "mean")
  bm <- bench::mark(
    dense_product = {
      A %*% b
    },
    lazy_product = {
      X %*% b
    },
    check = FALSE,
    min_iterations = 20
  )
  bm
}
```

As we can see, the function outputs a `bench` object where the
parameters of interest are mainly the minimum time of the iterations,
the median time, how many iterations per seconds can be done and the
memory allocation of each of the operations.

For a regularly small matrix, let $`n=500`$ and $`p=50`$ with a density
of $`0.05`$, so $`5`$ % of the inputs $`a_{ij}`$ are non-zero. We also
generate a random normal vector $`b\in R^p`$.

``` r

set.seed(123)
m_small <- create_sparsematrix(n = 500, p = 50, sparsity = 0.05)
test_vector <- rnorm(ncol(m_small))

benchmark_small <- bench_multiply(m_small, test_vector)
benchmark_small[c("expression", "min", "median", "itr/sec", "mem_alloc")]
#> # A tibble: 2 × 5
#>   expression         min   median `itr/sec` mem_alloc
#>   <bch:expr>    <bch:tm> <bch:tm>     <dbl> <bch:byt>
#> 1 dense_product   13.2µs   15.7µs    62451.    3.95KB
#> 2 lazy_product    38.9µs   42.4µs    21960.  122.82KB
```

For small matrices, we note that the lazy computation does not perform
better than the dense computation, it is even slower. Moreover, it shows
that the overhead from lazy evaluation remains modest even when it
doesn’t provide a speedup compared to using a dense matrix. The real
strength of lazy computation, however, is shown when the matrices gets
bigger and sparsity more widespread.

Hence, let $`A\in R^{1000\times 100}`$ be a sparse matrix with $`1`$ %
of $`a_{ij}\neq 0`$.

``` r

set.seed(123)
m_medium <- create_sparsematrix(n = 1000, p = 100, sparsity = 0.01)
test_vector <- rnorm(ncol(m_medium))

benchmark_medium <- bench_multiply(m_medium, test_vector)
benchmark_medium[c("expression", "min", "median", "itr/sec", "mem_alloc")]
#> # A tibble: 2 × 5
#>   expression         min   median `itr/sec` mem_alloc
#>   <bch:expr>    <bch:tm> <bch:tm>     <dbl> <bch:byt>
#> 1 dense_product     49µs     54µs    18061.    7.86KB
#> 2 lazy_product    39.6µs     43µs    20722.    18.2KB
```

We note that the computation time for the lazy computation remains more
or less the same as for the smaller matrix, while the computation time
for the dense computation rises dramatically. The lazy computation is on
median $`3.5`$ times faster than working with dense matrices.

Lastly, we will test a large matrix, $`A\in R^{1000\times 700}`$ with
only $`0.1`$ % of elements $`a_{ij}\neq 0`$ at this point.

``` r

set.seed(123)
m_large<- create_sparsematrix(n = 1000, p = 700, sparsity = 0.001)
test_vector <- rnorm(ncol(m_large))

benchmark_large <- bench_multiply(m_large, test_vector)
benchmark_large[c("expression", "min", "median", "itr/sec", "mem_alloc")]
#> # A tibble: 2 × 5
#>   expression         min   median `itr/sec` mem_alloc
#>   <bch:expr>    <bch:tm> <bch:tm>     <dbl> <bch:byt>
#> 1 dense_product  639.8µs  652.4µs     1534.    7.86KB
#> 2 lazy_product    43.9µs   47.6µs    20190.   32.27KB
```

Now, the computation time is approximately $`20`$ times faster on the
lazy object, allowing for performing around $`18,500`$ iterations per
second rather than around $`1,300`$ for the dense computation. These
performance differences are clearly shown in the plots below.

``` r

if (requireNamespace("ggplot2", quietly = TRUE) &&
    requireNamespace("ggbeeswarm", quietly = TRUE) &&
    requireNamespace("gridExtra", quietly = TRUE)) {

  p_m <- plot(benchmark_medium)
  p_l <- plot(benchmark_large)

  gridExtra::grid.arrange(
  p_m, p_l,
  ncol = 2,
  top = "Microbenchmark for Medium(n=1000, p=100) and Large(n=1000, p=700) Sparse Matrices"
)
}
```

In conclusion, we can say that lazy computation gets better the larger
and sparser the matrices we work with are. While lazy evaluation has a
slight overhead for small matrices, this becomes negligible as matrix
size grows—maintaining consistent computation time regardless of
dimensions, while dense operations degrade heavily. The trade-off
between the added overhead in `lazymatrix` and computation time reduced
tells us that when we work with large sparse matrices, lazy computation
is the superior approach. While $`131.7`$ microseconds may seem small,
we need to recall that for many statistical algorithm, for example
iterative least square computations, the operation `A%*%b` may have to
be perfomed hundreds or even thousands of times, implying that speed is
crucial. Once a matrix gets large and sparse enough, it will not even be
feasible to scale it, resulting in an advantage of the
`lazymatrix`-approach.

## Optimization of Matrix Transpose Multiplication

While many of `lazymatrx`’s methods depends on operations already
optimized or in `base R` or in `Matrix`, the method for computing the
product of the transpose of a `LazyMatrix` with a vector,
i.e. `t(X) %*% b`, is implemented from scratch. This is done to ensure
lazy computation of the product, and involves looping over the elements
in the matrix. Loops in R are generally not as efficient as vectorized
operations, so what we did was writing a method for
[`crossprod()`](https://rdrr.io/r/base/crossprod.html) using `Rcpp` and
`RcppArmadillo`. This allows us to compute the product of the transpose
of a `LazyMatrix` with a vector much faster than if we were to use a
loop in R. The method is implemented in C++ and is optimized for
performance, allowing us to compute the product efficiently even for
large matrices. We can compare the performance of this method against a
naive implementation using loops in R, and we will see that the C++
implementation is significantly faster.

To avoid premature optimization, the issue of the
[`lazymatrix::crossprod()`](https://rdrr.io/r/base/crossprod.html)
method was identified using the `profvis` package. We profiled
particularly some of the statistical algorithms and found that the
current implementation was a clear bottleneck in the performance of the
computation.

### Microbenchmarking `crossprod()`

Here we will present a microbenchmark for the old implementation of the
matrix transpose multiplication compared to the current one. This
showcases that especially when working with large sparse matrices, the
obvious use-case of `lazymatrix`, the current implementation is much
stronger. We start by defining the function that was used in R.

``` r

crossprod_R_impl <- function(x, y) {
    # t(X) %*% y
    s <- 1 / x@col_scales
    c_vec <- x@col_locations

    x_tb <- numeric(ncol(x@data))
    sum_y <- base::sum(y)

    for (j in seq_len(ncol(x@data))) {
      x_tb[j] <- s[j] * base::sum(x@data[, j] * y) -
                 s[j] * c_vec[j] * sum_y
    }
    return(matrix(x_tb, ncol = 1))
}
```

The arguments for the function is the `LazyMatrix` object `x` and the
vector `y` which we will use to perform the operation

``` math
x^Ty.
```

Moreover, we need also to create a function for the benchmarking among
the two methods. Note that both the old and the new implementations
takes only the `LazyMatrix` object as argument and the benchmarking is
therefore done only on the `LazyMatrix`. Compared to the previous
benchmarking, the dimensions are changed as $`X^T\in R^{p\times n}`$ and
hence we need $`b\in R^n`$.

``` r

bench_crossprod <- function(sparse_matrix, b) {
  X <- LazyMatrix(sparse_matrix, "sd", "mean")
  bm <- bench::mark(
    r_cross_product = {
      crossprod_R_impl(X, b)
    },
    rcpp_cross_product = {
      crossprod(X, b)
    },
    check = FALSE,
    min_iterations = 20
  )
  bm
}
```

We perform similar tests as the previous ones, starting with the smaller
matrix defined above and then aumenting the dimensionality with further
testing.

``` r

set.seed(123)
test_vector <- rnorm(nrow(m_small))
bm_cp_small <- bench_crossprod(m_small, test_vector)
bm_cp_small[c("expression", "min", "median", "itr/sec", "mem_alloc")]
#> # A tibble: 2 × 5
#>   expression              min   median `itr/sec` mem_alloc
#>   <bch:expr>         <bch:tm> <bch:tm>     <dbl> <bch:byt>
#> 1 r_cross_product      4.17ms   5.65ms      193.  632.67KB
#> 2 rcpp_cross_product  16.77µs   18.5µs    49795.    8.27KB
```

Already at this stage, we see that the implementation with `Rcpp` is
around $`65`$ times faster than the previous old method. It has also
smaller memory allocation.

As many of the sparse algorithms implemented in `lazymatrix` are
iterative due to the nature of sparse data, the
[`crossprod()`](https://rdrr.io/r/base/crossprod.html) method is many
times done many times for an algorithm. This shows the importance of
having efficient methods for these basic operations and we can compute
an even larger matrix to acknowledge the power of optimization.

``` r

set.seed(123)
large_vector <- rnorm(nrow(m_large))
bm_cp_large <- bench_crossprod(m_large, large_vector)
bm_cp_large[c("expression", "min", "median", "itr/sec", "mem_alloc")]
#> # A tibble: 2 × 5
#>   expression              min   median `itr/sec` mem_alloc
#>   <bch:expr>         <bch:tm> <bch:tm>     <dbl> <bch:byt>
#> 1 r_cross_product      59.7ms   60.9ms      16.4    10.9MB
#> 2 rcpp_cross_product   21.3µs   22.9µs   41824.       11KB
```

While computation with the old method with R gets slower the higher
dimensionality of $`X`$, the `Rcpp`method remains stable with around the
same computation time, giving us a higher performance of about $`600`$
times faster than the original. These differences are shown in the plots
below.

``` r

if (requireNamespace("ggplot2", quietly = TRUE) &&
    requireNamespace("ggbeeswarm", quietly = TRUE) &&
    requireNamespace("gridExtra", quietly = TRUE)) {

  p1 <- plot(bm_cp_small)
  p2 <- plot(bm_cp_large)

  gridExtra::grid.arrange(
  p1, p2,
  ncol = 2,
  top = "Microbenchmark for Small(n=500, p=50) and Large(n=1000, p=700) Sparse Matrices"
)
}
```

### Some Notes on Optimization

The performance improvement here is not due entirely to asymptotic time
complexity, but part of it can be described more rigorously. Namely, we
note that the operation

``` math
X^T b
```

where $`X^T \in R^{p \times n}`$ and $`b \in R^n`$ requires

``` math
O(np)
```

in time complexity. However, the R implementation performs additional
allocations within its loop; each iteration involves operations over
vectors of length $`n`$, leading to $`O(n)`$ work per iteration. The
`Rcpp` implementation instead can be expressed as having a single output
vector of size

``` math
O(p)
```

and avoids repeated intermediate allocations during the computation.
Hence, the improvement here is not fully due to time complexity but
reduced constant factors and allocations within the R implementation. In
particular, the R implementation allocates temporary vectors within each
iteration of the loop, while the Rcpp implementation performs the
computation in a single pass over memory without repeated allocation
overhead. This leads to significantly improved performance in practice,
especially for large-scale sparse matrices where the operation is
invoked repeatedly within iterative algorithms.
