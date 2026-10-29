# Python

LazyMatrix for Python applies column normalization during matrix operations.
It accepts NumPy arrays and SciPy CSR/CSC matrices and sparse arrays. Install it
with:

```sh
pip install lazymatrix
```

Python 3.11 or newer is required. Prebuilt wheels cover Linux x86_64, macOS
on Intel and Apple silicon, and Windows x86_64. Building from source requires
Rust 1.89 or newer.

## Normalize and multiply

```python
import numpy as np
from scipy import sparse
from lazymatrix import standardize

X = sparse.csr_matrix([[1.0, 0.0], [2.0, 3.0], [0.0, 4.0]])
Z = standardize(X)

beta = np.array([1.0, -1.0])
prediction = Z @ beta
gradient = Z.T @ prediction
column = Z[:, 0]
```

`standardize` fits column means and population standard deviations immediately.
Pass `df=1` for sample standard deviations; `df=0` is the default. The variance
divisor is `nrows - df`, and scales are `NaN` when `nrows <= df`. `df` must be a
nonnegative integer and affects scaling independently of centering.
Pass `center=False` or `scale=False` to disable either component. Exact zero
scales become one, so constant columns do not cause division by zero.
Nonfinite values follow ordinary IEEE arithmetic; missing values are not ignored.

`Z.shape`, `Z.ndim`, and `Z.dtype` describe the logical matrix. `Z.centers` and
`Z.scales` return copies of the fitted parameters, or `None` when the component
is disabled. These parameters describe the original columns even for `Z.T`.

Both `@` and `np.matmul` support dense vectors and matrices on either side.
Multiple right-hand sides currently use repeated vector products. `Z[:, j]`
returns one normalized column as an independent dense array, including support
for negative column indices. Other slicing and elementwise operations are not
part of the initial API.

## Use SciPy solvers

The adapter exposes forward and transpose products without materializing the
normalized input:

```python
from scipy.sparse.linalg import lsqr

y = np.array([1.0, -2.0, 1.0])
coefficients = lsqr(Z.aslinearoperator(), y)[0]
```

## Memory and conversion

LazyMatrix owns a copy of the raw input, so later changes to `X` do not change
`Z`. Dense inputs require a full raw-data copy. Sparse copies remain sparse and
are stored in canonical CSC form, with duplicates summed before normalization.
All values and results use float64. Complex and nonnumeric inputs are rejected.

Products, transposes, and column reads do not construct a full normalized matrix.
Call `Z.toarray()` when you explicitly want a dense copy:

```python
dense = Z.toarray()
```

Implicit conversion with `np.asarray(Z)` raises an error. Existing estimators
that require a concrete NumPy array or SciPy sparse matrix need an explicit
conversion or their own operator integration.

The initial Python API fits each input independently. Reusing training
normalization on new observations is planned; calling `standardize` on validation
data computes new statistics.

See the [Python API reference](/reference/python/) for public classes and methods.

## Development and releases

From the repository root:

```sh
task python:check
task python:package
```

The first command builds an editable extension and runs formatting, linting, and
correctness checks. The second builds a wheel and source distribution, installs
the wheel in an isolated environment, rebuilds from the source distribution,
and tests both installations. The environment lives in `.devenv/python`.

Python and Rust use independent versions and share one Versionary release PR.
Rust releases also release Python bindings. Python-only releases leave the Rust
version unchanged. Python tags use `python-v<version>` and Rust tags use
`v<version>`.
