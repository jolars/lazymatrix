# Use Cases for LazyMatrix: Statistical Algorithms

## Linear Regression with LSQR

### Introduction

Several algorithms for iterative least squares algorithms have been
proposed when working with sparse matrices. We have chosen to work with
the method proposed by Paige and Saunders (1982) that is called LSQR.
This iterative algorithm allows us to use custom methods for matrix
operations which allows for our lazy computations. We will see that the
operations `%*%`, `crossprod` and `norm` are enough for getting stable
estimates of the regression coefficients.

### Notation

The authors proposes an algorithm as a set of initalisation parameters
and the design matrix $`A`$. As mentioned by the authors, the matrix
$`A`$ is used only to compute products of the form $`Av`$ and $`A^Tu`$.
This object will be our `LazyMatrix` objects and hence, all computations
will be done lazily at every step of the iteration. For clarity,
matrices will be denoted by capital letters $`A, B,\cdots`$, vectors by
$`v,w,\cdots`$ and scalars by greek letters $`\alpha, \beta,\cdots`$.
For vectors, the norm is the Euclidean norm and for matrices we assume
the Frobenius norm $`||A||_F=\sqrt{\sum |a_{ij}|^2}`$.

### Mathematical Background

We will make use of the [Lanczos
process](https://en.wikipedia.org/wiki/Lanczos_algorithm), for which we
will not go into further detail. The main idea is that with the help of
a set of initialization parameters, we can with the help of the design
matrix $`A`$ update the regression coefficients for every step of the
algorithm. The algorithm is explained as

1.  Initialization:
    ``` math
    \begin{align*}
       \beta_1 &= \sqrt{\sum b^2}, \\
       \mathbf{u}_1 &= \frac{\mathbf{b}}{\beta_1}, \\
       \mathbf{A}^\top \mathbf{u}_1 &= \mathbf{A}^\top \mathbf{u}_1, \\
       \alpha_1 &= \sqrt{\sum (\mathbf{A}^\top \mathbf{u}_1)^2}, \\
       \mathbf{v}_1 &= \frac{\mathbf{A}^\top \mathbf{u}_1}{\alpha_1}, \\
       \mathbf{w}_1 &= \mathbf{v}_1, \\
       \mathbf{x}_0 &= \mathbf{0}, \\
       \bar{\phi_1} &= \beta_1, \\
       \bar{\rho_1} &= \alpha_1.
    \end{align*}
    ```

2.  For $`i = 1, 2, \dots`$, repeat steps 3–6:

    3.  Continue the bidiagonalization:
        ``` math
        \begin{align*}
        \mathbf{\beta}_u &= \mathbf{A} \mathbf{v}_1 - \alpha_1 \mathbf{u}_1, \\
        \beta_2 &= \sqrt{\sum \mathbf{\beta}_u^2}, \\
        \mathbf{u}_2 &= \frac{\mathbf{\beta}_u}{\beta_2}, \\
        \mathbf{\alpha}_v &= \mathbf{A}^\top \mathbf{u}_2 - \beta_2 \mathbf{v}_1, \\
        \alpha_2 &= \sqrt{\sum \mathbf{\alpha}_v^2}, \\
        \mathbf{v}_2 &= \frac{\mathbf{\alpha}_v}{\alpha_2}.
        \end{align*}
        ```

    4.  Construct and apply the next orthogonal transformation:
        ``` math
        \begin{align*}
        \rho_1 &= \sqrt{(\bar{\rho_1})^2 + \beta_2^2}, \\
        c_1 &= \frac{\bar{\rho_1}}{\rho_1}, \\
        s_1 &= \frac{\beta_2}{\rho_1}, \\
        \theta_2 &= s_1 \alpha_2, \\
        \bar{\rho_2} &= -c_1 \alpha_2, \\
        \phi_1 &= c_1\bar{\phi_1}, \\
        \bar{\phi_2} &= s_1 \bar{\phi_1}.
        \end{align*}
        ```

    5.  Update $`\mathbf{x}`$ and $`\mathbf{w}`$:
        ``` math
        \begin{align*}
        \mathbf{x}_1 &= \mathbf{x}_0 + \frac{\phi_1}{\rho_1} \mathbf{w}_1, \\
        \mathbf{w}_2 &= \mathbf{v}_2 - \frac{\theta_2}{\rho_1} \mathbf{w}_1.
        \end{align*}
        ```

    6.  Reset the loop variables:
        ``` math
        \begin{align*}
        \beta_1 &= \beta_2, \\
        \mathbf{u}_1 &= \mathbf{u}_2, \\
        \alpha_1 &= \alpha_2, \\
        \mathbf{v}_1 &= \mathbf{v}_2, \\
        \bar{\rho_1} &= \bar{\rho_2}, \\
        \bar{\phi_1} &= \bar{\phi_2}, \\
        \mathbf{x}_0 &= \mathbf{x}_1, \\
        \mathbf{w}_1 &= \mathbf{w}_2.
        \end{align*}
        ```

    7.  Check for convergence.

At this point, the authors do not propose a criterion for when we should
stop iterating. However, we have chosen to use the criterion that [LSQR
in
scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html)
(Virtanen et al. 2020; developers 2025) uses. First, compute the
residual between the response vector $`y`$ and the estimate as

``` math
r = y-Ax_0
```

and then we stop iterating when

``` math
\|r\| \leq \text{tol}\cdot \|A\| \cdot \|x_0\| + \text{tol} \cdot \|b\|.
```

`tol` is a parameter for tolerance level of the difference that we
accept. In our algorithm, it is by default $`1e-6`$.

### Implementation

Assuming we want to fit regression coefficient vector $`\beta`$ on the
scaled matrix $`X`$ with response vector $`y`$. We can compute these
estimates using lazy computation. Assuming the original sparse matrix
`A`, create an instance of `LazyMatrix` and run the method
[`lazymatrix::lsqr`](https://vsegersall.github.io/lazymatrix/reference/lsqr.md).

``` r

library(lazymatrix)
#> 
#> Attaching package: 'lazymatrix'
#> The following object is masked from 'package:base':
#> 
#>     norm
 # 1. Define sparseMatrix
base::set.seed(123)
n_row <- 50
n_col <- 10
i <- c(
  1:50,
  base::sample(1:50, 20, replace = TRUE),
  base::sample(1:50, 15, replace = TRUE)
)
j <- c(
  base::rep_len(1:10, 50),
  base::sample(1:10, 20, replace = TRUE),
  base::sample(1:10, 15, replace = TRUE)
)
pairs <- base::unique(data.frame(i = i, j = j))
i <- pairs$i
j <- pairs$j
x <- stats::rnorm(length(i))
A <- Matrix::sparseMatrix(i = i, j = j, x = x, dims = c(n_row, n_col))

# 2. Define response vector
y <- stats::rnorm(nrow(A))

# 3. Define LazyMatrix
X <- LazyMatrix(A, "sd", "mean")

# 4. Perform lsqr
lazy_beta <- lazymatrix::lsqr(X, y)
```

Now,the non-lazy approach would be to scale `A` using
[`base::scale()`](https://rdrr.io/r/base/scale.html)and then use
something like [`stats::lm.fit()`](https://rdrr.io/r/stats/lmfit.html)
to get the estimates. This would imply creating a copy of `A` with
non-zero elements taking up unneccessary memory and being much slower
generally.

``` r

scaled_a <- base::scale(A)
non_lazy_beta <- stats::lm.fit(scaled_a, y)$coefficients
```

The estimates are equivalent as is shown below.

``` r

isTRUE(all.equal(as.vector(lazy_beta), as.vector(non_lazy_beta)))
#> [1] TRUE
```

## Gradient Descent

### Introduction

In statistical modeling, the classical gradient descent algorithm
remains a fundamental part in any kind of modern machine learning
approach. As an example of the power of `lazymatrix`, we propose The
approach we propose the simplest form of gradient descent on a scaled
matrix. As many of our implementations show, it is sufficient to use a
small set of matrix operations in order to perform this algorithm.

### Mathematical Background

The algorithm follows the approach as explained by Ng et al. (2025).
Assume the cost function $`J(\theta)`$ defined as

``` math
J(\theta)=\frac{1}{2}\sum_{i=1}^n(h_\theta(x^{(i)})-y^{(i)})^2
```

where $`h_\theta(x)=\theta^Tx`$, $`x`$ being the elements of the design
matrix and $`\theta`$ the parameters. In gradient descent, we usually
talk about two types of parameters, weights and biases. In the notation
above, they are both included within the parameter vector $`\theta`$.
This is done by letting $`x_0=1`$ so that the first term in
$`\sum_{i=0}^n\theta_ix_i`$ includes the additive bias term. $`y`$ is
the response vector, so we recognize $`J`$ to be the cost function
representing the residual sum of squares. Hence, the goal is to choose
$`\theta`$ to minimize $`J`$. We start with an initial guess of the
parameter and repeatedly update it to make the cost function smaller.
This is done with the gradient descent algorithm defined as

``` math
\theta_j:=\theta_j - \alpha\frac{\partial}{\partial\theta_j}J(\theta)
```

where the partial derivative is the derivative of the cost function with
regards to the parameter. $`\alpha`$ is the learning rate that usually
takes a small value to control how fast the algorithm converges.
Computing this derivative is straightforward and we get the expression

``` math
\frac{\partial}{\partial\theta_j}J(\theta)=(h_\theta(x)-y)x_j,
```

providing us with the update rule

``` math
\theta_j:=\theta_j+\alpha(y^{(i)}-h_\theta(x^{(i)}))x_j^{(i)}.
```

### Implementation

The following gradient descent algorithm works as helper function for
which we will compare the lazy-approach to the non-lazy.

``` r

gradient_descent_helper <- function(
  x,
  y,
  w_init,
  b_init,
  learning_rate,
  n_epochs
) {
  w <- w_init
  b <- Matrix::Matrix(b_init, nrow = nrow(x))
  n <- nrow(x)
  for (epoch in 1:n_epochs) {
    y_pred <- x %*% w + b
    error <- y_pred - y
    w <- w - (learning_rate * crossprod(x, error)) / n
    b <- b - (learning_rate * sum(error)) / n
  }
  list(w = w, b = b)
}
```

Compared to the definition above, in this algorithm we include the bias
term as `b`. This results in a list of the two parameter vectors,
weights and biases. `n_epochs` is the number of time we choose to
iterate. As the result of `LazyMatrix %*% vector` results in a
`Matrix`-object, the addition of the bias term can be handled completely
within the `Matrix` framework. The key of `lazymatrix` is to handle the
heavy matrix operations lazily which is done in this case using `%*%`
and `crossprod`. Now, we use the sparse matrix `A` from section \#1 and
it’s `LazyMatrix` `X` and set initial values randomly for the weights
and biases and the response vector `y_true`.

``` r

  set.seed(4567)
  p <- ncol(X)
  w_init <- stats::rnorm(p)
  b_init <- stats::rnorm(1)
  y_true <- stats::rnorm(nrow(A))
  learning_rate <- 0.01
  n_epochs <- 10
```

Note that the function returns the parameters from the iterative
algorithm.

``` r

pars_lazy <- gradient_descent_helper(
   x = X,
   y = y_true,
   w_init = w_init,
   b_init = b_init,
   learning_rate = learning_rate,
   n_epochs = n_epochs
  )
print("First 5 parameter weights and the bias term.")
#> [1] "First 5 parameter weights and the bias term."
print(pars_lazy$w[1:5])
#> [1] -0.6537508 -0.7836177  0.2716844  0.5295553  1.2169584
print(pars_lazy$b[1])
#> [1] 1.381793
```

These parameters can now be used for predictions and are comparible with
the non-lazy way of using `scale(A)`and then running gradient descent.

### Discussion

Nonetheless there exists much more complex versions of the gradient
descent algorithm, this example rather proves that with a few matrix
operations being performed lazily, we can avoid storing heavy copies of
large matrices and perform operations solely when they are needed. The
idea is to implement methods to `lazymatrix` which allows for more
complex algorithms such as Stochastic gradient descent.

## Principal Component Analysis

### Introduction

A fundamental statistical algorithm for dimension reduction is principal
component analysis (PCA). In base R, the function used to perform PCA is
[`stats::prcomp()`](https://rdrr.io/r/stats/prcomp.html), which does so
by breaking up the matrix `A` into it’s singular value decomposition
(SVD) using [`base::svd()`](https://rdrr.io/r/base/svd.html). This
implies that we require a lazy implementation of svd in order for
principal component analysis to work. Moreover, many methods have been
proposed to perform svd on sparse matrices and in `lazymatrix` we have
chosen to follow the iterative algorithm proposed by Baglama and Reichel
(2005), which the authors have also implemented in the R package
`irlba`. As this method uses iterative methods, we can easily include
`lazymatrix`’s methods for `%*%` and `crossprod` in order to get a
working algorithm for svd. However, as the authors themselves say in the
paper, the `irlba` function will not compute all singular values and
proposes the user to use [`base::svd`](https://rdrr.io/r/base/svd.html)
if one wishes to compute all singular values. This would however involve
materializing `A` breaking the lazy computation in the process.
Therefore, we work solely with the algorithm proposed by `irlba`.

### Mathematical Background

#### Partial Singular Value Decomposition with `irlba`

Assume the matrix $`\tilde{A}`$ being the scaled version of `A` being
$`n\times p`$. We wish to perform a partial singular value
decomposition, where the first step is using a Lanczos process which can
be described as

1.  Choose a random unit vector $`p_1\in R^p`$

2.  Compute $`\tilde{A}p_1=z\in R^n`$, transforming $`p_1`$ from $`R^p`$
    to $`R^n`$

3.  Normalize the vector $`q_1=\frac{z}{\lVert z \rVert}`$ and let
    $`\alpha_1=\lVert z \rVert`$. This builds up the matrix $`B`$ as
    it’s first diagonal entry.

4.  Compute the residual vector by first returning to $`R^p`$ with
    $`\tilde{A}^Tq_1=z\in R^p`$ and let

    ``` math
    r=z-\alpha_1p_1 \in R^p.
    ```

5.  Compute the length of the residual vector
    $`\beta_1=\lVert r \rVert`$ and let

    ``` math
    p_2 = \frac{r}{\beta_1}.
    ```

6.  Continue to build up $`B`$ with $`\beta_1`$ as

    ``` math
    B=\begin{pmatrix}
    \alpha_1 & \beta_1 &
    \end{pmatrix}
    ```

At the next iteration, $`p_2`$ is used as input vector and we start over
until we reach convergence after $`m`$ steps, resulting in the
bidiagonal form of $`B\in R^{m\times m}`$. At every point when matrix
multiplication is being computed with $`\tilde{A}`$ or $`\tilde{A}^T`$,
lazy computation through `lazymatrix` is being implemented. Once we have
performed every iteration in the Lanczos process, we can stack the right
basis vectors $`p_1, p_2, \ldots, p_m`$ into the matrix
$`P_m\in R^{p\times m}`$ and the left basis vectors $`q_1, q_2,...,q_m`$
into $`Q_m\in R^{n\times m}`$ resulting in the equations

``` math
\begin{align*}
\tilde{A}P_m &= Q_mB_m\\
\tilde{A}^TQ_m &= P_mB_m^T+r_me_m^T
\end{align*}
```

where we know that $`B_m`$ is bidiagonal and since $`m \ll \min(n,p)`$,
this is a smaller matrix and the decomposition can be written as

``` math
\tilde{A}P_m=Q_mB_m
```

where we know that $`B_m`$ is bidiagonal and since $`m<<\min(n,p)`$,
this is a smaller matrix and the decomposition can be written as

``` math
B_m=U_B\Sigma_B V_B^T
```

where

\$\$ \begin{align\*} \Sigma_B&=\text{diag}(\sigma_1^B,
\sigma_2^B,\ldots,\sigma_m^B)\newline U_B&\in R^{m\times m} \\ V_B&\in
R^{m\times m}. \end{align\*} \$\$

This decompoisition can be used to get approximate solutions to the svd
of $`\tilde{A}`$ with the relationships

``` math
\begin{align*}
\tilde{\sigma}_j&=\sigma_j^B\\
\tilde{u}_j &= Q_mu_j^B \in R^n\\
\tilde{v}_j &= P_mv_j^B \in R^p.\\
\end{align*}
```

These equations can be validated easily by looking first at the left
singular vectors

``` math
\tilde{A}\tilde{v}_j=\tilde{A}P_mv_j^B=Q_mB_mv_j^B=\sigma_j^BQ_mu_j^B=\tilde{\sigma}_j\tilde{u}_j
```

which implies that $`(\tilde{\sigma}_j,\tilde{u}_j, \tilde{v}_j)`$
satisfies the left singular vector equation. Instead, the second
equation $`\tilde{u}_j = Q_mu_j^B`$, has a residual error if we look at

``` math
\tilde{A}^TQ_m=P_mB_m^T+r_me_m^T
```

where it can be shown that
``` math
\tilde{A}^T\tilde{u_j}=\tilde{\sigma}_j\tilde{v_j}+(e_m^Tu_j^B)r_m.
```

This error term is proportional to the last computation of $`u_j^B`$ and
residual $`r_m`$.

Now, we are not neccessarily interested in all $`m`$ iterations, but it
is sufficient to choose $`k<m`$ as the number of triplets
$`(\tilde{\sigma_j}, \tilde{u}_j,\tilde{v}_j)`$ for which we aim to
compute. This is also a parameter in the `svd` call for `lazymatrix`.
What we do is after $`m`$ iterations, we choose the $`k`$ largest
singular values $`\sigma^B_1, \sigma^B_2,\ldots,\sigma^B_k`$ and their
corresponding left and right vectors. We see whether these have
converged based on the criterion

``` math
|\beta_m(e_m^Tu_j^B)|<\text{tol}\cdot \tilde{\sigma}_j
```

and $`\beta=\lVert r_m \rVert`$ and `tol` being a tolerance parameter.
If all $`k`$ have converged, the algorithm is done. If instead we do not
have convergence, we start the Lanczos process again using the subspace
spanned by the $`k`$ right singular vectors
$`\tilde{v}_1, \tilde{v}_2,\ldots, \tilde{v}_k`$ and from there we run
another $`m-k`$ runs of the process. This ensures that the previously
computed singular values $`\tilde{\sigma}_1, \ldots, \tilde{\sigma}_k`$
are carried over into the new $`B_m`$​, rather than restarting from
scratch.

#### Principal Component Analysis with Partial SVD

As the regular [`stats::prcomp()`](https://rdrr.io/r/stats/prcomp.html)
function uses [`base::svd()`](https://rdrr.io/r/base/svd.html) for
dimension reduction, the equivalent method for principal component
analysis in `lazymatrix` is identical but with the custom method using
partial SVD with irlba.

### Implementation

As before, we define a sparse matrix for which we will perform singular
value decomposition and thereafter principal component analysis. We use
the same sparse matrix from above (`A`) and use it do define the lazy
matrix `X`.

``` r

#3. SVD
svd_lazy <- svd(X)

# 4. Print singular values
print("The singular values are: ")
#> [1] "The singular values are: "
svd_lazy$d
#> [1] 9.104398 7.842210 7.653368 7.277682 6.978369

print("The first 6 rows of the left singular vectors are: ")
#> [1] "The first 6 rows of the left singular vectors are: "
head(svd_lazy$u)
#>             [,1]         [,2]         [,3]        [,4]         [,5]
#> [1,]  0.01333517 -0.004771403  0.019689873 0.068220394 -0.173749037
#> [2,]  0.01350000 -0.003359592  0.006420480 0.001814066  0.011252310
#> [3,] -0.02927655  0.001950910  0.009877735 0.007683850  0.008493306
#> [4,]  0.21819519 -0.028828949 -0.100755039 0.111868855  0.071569385
#> [5,]  0.02331452 -0.149512123  0.051400116 0.289542383  0.169000315
#> [6,] -0.07461649 -0.073651823 -0.207556881 0.145466353  0.087293952

print("The frist 6 rows of the right singular vectors are: ")
#> [1] "The frist 6 rows of the right singular vectors are: "
head(svd_lazy$v)
#>               [,1]        [,2]        [,3]        [,4]         [,5]
#> [1,]  0.0005177039 -0.02539160  0.08076583  0.36481720 -0.889025448
#> [2,] -0.0349727147  0.39383668 -0.21740956 -0.59873418 -0.308986839
#> [3,] -0.5573476051  0.18356325  0.19092437  0.01085593  0.083917203
#> [4,]  0.4681343605 -0.05644394 -0.20230136  0.21057716  0.110572231
#> [5,]  0.0512182125 -0.37439995  0.62396360  0.06067131  0.002115646
#> [6,] -0.2623446370 -0.18904462 -0.53246233  0.35541508  0.180506467
```

This function has been incorporated within the S4 method for the
`LazyMatrix` object if we call
[`lazymatrix::prcomp()`](https://rdrr.io/r/stats/prcomp.html). Hence,
working with the `LazyMatrix` is equivalent to working with a regular
matrix object.

``` r

pca_lazy <- prcomp(X)
print("The first 6 rows of the principal components are: ")
#> [1] "The first 6 rows of the principal components are: "
head(pca_lazy$rotation)
#>                PC1         PC2         PC3         PC4          PC5         PC6
#> [1,]  0.0005177039 -0.02539160 -0.08076583 -0.36481720 -0.889025448 -0.22840881
#> [2,] -0.0349727147  0.39383668  0.21740956  0.59873418 -0.308986839 -0.11636259
#> [3,] -0.5573476051  0.18356325 -0.19092437 -0.01085593  0.083917203 -0.03513912
#> [4,]  0.4681343605 -0.05644394  0.20230136 -0.21057716  0.110572231 -0.27175921
#> [5,]  0.0512182125 -0.37439995 -0.62396360 -0.06067131  0.002115646  0.16468595
#> [6,] -0.2623446370 -0.18904462  0.53246233 -0.35541508  0.180506467 -0.30508329
#>              PC7         PC8         PC9
#> [1,] -0.05356839  0.07535334  0.01491230
#> [2,]  0.22989075 -0.51995123 -0.06941254
#> [3,] -0.23435083 -0.02269329 -0.51967863
#> [4,] -0.43806174 -0.35281652 -0.51250510
#> [5,]  0.08626780 -0.63852801  0.11599341
#> [6,]  0.15492293 -0.34646067  0.34329372
```

### Conclusion

What we encounter when using iterative methods for performing sparse
SVD, is that we will not compute all singular values as is the case when
calling [`base::svd()`](https://rdrr.io/r/base/svd.html). One could
argue that the method should, if possible, compute all of them using a
regular [`base::svd()`](https://rdrr.io/r/base/svd.html) call. However,
this breaks sparsity and the lazy structure, materializing the matrix
and hence loosing the lazy computation. Since the use case for
`lazymatrix` is mainly for working with large sparse matrices, where
computation of all singular values may not even be feasible, we
recommend not to use this framework for small matrices where sparsity is
not an issue and all singular values may be computed using
[`base::svd()`](https://rdrr.io/r/base/svd.html).

## References

Baglama, James, and Lothar Reichel. 2005. “Augmented Implicitly
Restarted Lanczos Bidiagonalization Methods.” *SIAM Journal on
Scientific Computing* 27 (1): 19–42.

developers, SciPy. 2025. *scipy.sparse.linalg.lsqr — SciPy V1.17.0
Documentation*. Released.
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html>.

Ng, Andrew et al. 2025. *CS229 Lecture Notes*.
<https://cs229.stanford.edu/notes2022fall/main_notes.pdf>.

Paige, Christopher C., and Michael A. Saunders. 1982. “LSQR: An
Algorithm for Sparse Linear Equations and Sparse Least Squares.” *ACM
Transactions on Mathematical Software* 8 (2): 43–71.

Virtanen, Pauli, Ralf Gommers, Travis E. Oliphant, et al. 2020. “SciPy
1.0: Fundamental Algorithms for Scientific Computing in Python.” *Nature
Methods* 17: 261–72.
