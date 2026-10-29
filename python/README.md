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
by default; pass `df=1` for sample standard deviations. The variance divisor is
`nrows - df`; scales are `NaN` when `nrows <= df`. Exact zero scales become one.

For neural first layers, install `pip install "lazymatrix[torch]"` and use
`lazymatrix.torch.NormalizedLinear`. It accepts supplied centers and scales,
including `Z.centers` and `Z.scales`, and performs lazy normalization entirely
in PyTorch. Weights remain trainable in normalized coordinates. The layer
supports dense, sparse COO, and sparse CSR tensors in float32 or float64 on
CPU and CUDA. Sparse inputs must not require gradients. PyTorch 2.9 or newer
is required and is imported only when the adapter is used. See the
[PyTorch guide](https://lazymatrix.org/guide/python#use-a-pytorch-first-layer).

From the repository root, use `task python:check` to build and test the package.
This includes the PyTorch adapter and runs CUDA cases when hardware is available.
Python and Rust have independent versions and share one release PR.

PyPI trusted publishing must name `jolars/lazymatrix`, `publish-python.yml`, and
the `release` environment. The existing Versionary release token must retain
permission to create tags that trigger publishing workflows.
