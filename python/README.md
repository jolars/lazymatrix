# LazyMatrix for Python

Center and scale NumPy arrays and SciPy CSR/CSC matrices without constructing
a normalized matrix. LazyMatrix owns a copy of the raw input and applies fitted
column normalization during products.

```python
import numpy as np
from lazymatrix import standardize

X = np.array([[1.0, 0.0], [2.0, 3.0], [0.0, 4.0]])
Z = standardize(X)
prediction = Z @ np.array([1.0, -1.0])
gradient = Z.T @ prediction
column = Z[:, 0]
```

Install with `pip install lazymatrix`. Python 3.11 or newer is required.
See the [Python guide](https://lazymatrix.org/guide/python) and
[API reference](https://lazymatrix.org/reference/python/).

`Z.aslinearoperator()` integrates with SciPy's iterative solvers. `Z.toarray()`
explicitly creates a normalized dense array. Other NumPy operations and implicit
dense conversion are unsupported in this first release.

Dense inputs require a full raw-data copy. Sparse inputs remain sparse and are
copied into canonical CSC storage. Products accept dense vectors and matrices;
multiple right-hand sides currently use repeated vector products. All stored
values and results use float64. Normalization uses population standard deviations
and replaces exact zero scales with one.

From the repository root, use `task python:check` to build and test the package.
Python and Rust have independent versions and share one release PR.

PyPI trusted publishing must name `jolars/lazymatrix`, `publish-python.yml`, and
the `release` environment. The existing Versionary release token must retain
permission to create tags that trigger publishing workflows.
