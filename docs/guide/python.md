# Getting started with Python

LazyMatrix centers and scales NumPy arrays and SciPy sparse matrices during
matrix operations without creating a normalized copy. For Rust, see the
[Rust quick start](/guide/getting-started).

## Install

```sh
pip install lazymatrix
```

Python 3.11 or newer is required. See
[backend setup](/guide/installation#python-installation) for supported platforms
and source installation.

## Normalize and multiply

Arrange observations in rows and features in columns. Fit column means and
standard deviations, then multiply by the normalized matrix:

```python
import numpy as np
from lazymatrix import standardize
from scipy import sparse

X = sparse.csr_matrix([[1.0, 0.0], [2.0, 3.0], [0.0, 4.0]])
Z = standardize(X)

beta = np.array([1.0, -1.0])
prediction = Z @ beta
transpose_product = Z.T @ prediction
```

`Z` owns a copy of the raw input and stores fitted centers and scales. Sparse
input remains sparse. You can pass a NumPy array in place of `X` without changing
the calls. Products return NumPy arrays and use the normalized values.
Standardization uses population standard deviations.

## Reuse fitted parameters

Apply the training parameters to validation or prediction data:

```python
X_validation = sparse.csr_matrix([[4.0, 1.0], [0.0, -2.0]])
validation = Z.normalization.transform(X_validation)
validation_prediction = validation @ beta
```

`transform` reuses the fitted values without computing new statistics. The new
input must have the same features in the same order. See
[normalization](/guide/normalization#reuse-fitted-parameters) for fitting
separately, normalizing rows, and saving fitted values.

## Next steps

- [Normalization](/guide/normalization): choose centering and scaling methods,
  normalize rows, and reuse fitted parameters.
- [Using normalized matrices](/guide/matrix-operations): products, SciPy solvers,
  dense conversion, and a PyTorch first layer.
- [How it works](/guide/how-it-works): the normalization algebra and memory costs.
- [Python API reference](/reference/python/): public classes and methods.
