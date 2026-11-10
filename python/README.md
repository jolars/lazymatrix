# LazyMatrix for Python <img src="https://raw.githubusercontent.com/jolars/lazymatrix/main/assets/logo.png" align="right" width="189" alt="LazyMatrix logo" />

[![CI](https://github.com/jolars/lazymatrix/actions/workflows/ci.yml/badge.svg)](https://github.com/jolars/lazymatrix/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/lazymatrix.svg)](https://pypi.org/project/lazymatrix/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23666450-blue)](https://doi.org/10.5281/zenodo.23666450)

LazyMatrix centers and scales columns or rows of NumPy arrays and SciPy CSR and
CSC matrices during matrix operations, without constructing a normalized copy.
It supports matrix products, SciPy's iterative solvers, and an optional PyTorch
first layer.

See the [Python guide](https://lazymatrix.org/guide/python) for supported
operations and the [API reference](https://lazymatrix.org/reference/python/) for
public classes and methods.

## Installation

Install [lazymatrix](https://pypi.org/project/lazymatrix/) from PyPI:

```sh
pip install lazymatrix
```

Python 3.11 or newer is required.

## Usage

Fit column means and standard deviations, then multiply by the normalized
matrix:

```python
import numpy as np
from lazymatrix import standardize

X = np.array([[1.0, 0.0], [2.0, 3.0], [0.0, 4.0]])
Z = standardize(X)
prediction = Z @ np.array([1.0, -1.0])
gradient = Z.T @ prediction
column = Z[:, 0]
```

The operator owns a copy of the raw input and stores fitted column means and
scales. Sparse inputs remain sparse and retain their CSR or CSC orientation,
with duplicate entries summed. Stored values and results use float64.
Standardization uses population standard
deviations by default; pass `df=1` for sample standard deviations.

`Z.aslinearoperator()` integrates with SciPy's iterative solvers. Call
`Z.toarray()` to explicitly create a normalized dense array. Implicit dense
conversion and elementwise NumPy operations are unsupported.

## Normalization

Use `normalize` for other normalization recipes:

```python
from lazymatrix import normalize

min_max = normalize(X, center="min", scale="range")
robust = normalize(X, center="median", scale="quantile_range")
unit_columns = normalize(X, center=False, scale="l2")
```

`normalize(X)` defaults to mean centering and population standard-deviation
scaling, like `standardize(X)`. Centering and scaling are independently optional,
and exact zero scales become one. Both factories return a `LazyMatrix`; direct
`LazyMatrix` construction accepts the same options as `normalize`.

Reuse training normalization without refitting with
`Z.normalization.transform(X_validation)`. The input must have the same features
in the same order. See the [normalization
guide](https://lazymatrix.org/guide/python#choose-normalization-methods) for all
methods and [fitted
parameters](https://lazymatrix.org/guide/python#reuse-fitted-parameters) for
saving and restoring normalization.

Pass `axis=1` to `standardize` or `normalize` to normalize each row independently:

```python
from lazymatrix import RowNormalization

rows = standardize(X, axis=1)
unit_rows = normalize(X, axis=1, center=False, scale="l2")
recipe = RowNormalization(center=False, scale="l2")
training = recipe.fit_apply(X)
validation = recipe.fit_apply(np.array([[3.0, 4.0]]))
```

Every call fits fresh row statistics. `recipe.fit(X)` returns
`RowNormalizationParams` without retaining the input matrix. Use
`params.apply_frozen(X)` to reapply saved values to the same rows in the same
order. Explicit row parameters take `nrows`, `centers`, and `scales`.
Column recipes and parameters are named `ColumnNormalization` and
`ColumnNormalizationParams`; the latter replaces the former
`NormalizationParams` name. See the [row normalization
guide](https://lazymatrix.org/guide/python#normalize-rows) for details.

## PyTorch

Install the optional adapter for a neural network's first layer:

```sh
pip install "lazymatrix[torch]"
```

`lazymatrix.torch.NormalizedLinear` applies supplied centers and scales during
training, with weights in normalized coordinates. It supports dense tensors and
sparse COO and CSR tensors on CPU and CUDA. PyTorch 2.9 or newer is required.
See the [PyTorch
guide](https://lazymatrix.org/guide/python#use-a-pytorch-first-layer) for an
example and supported dtypes.

## Citation

If you use LazyMatrix in your research, please cite the release you used. See
the [citation guide](https://lazymatrix.org/guide/citation) for the Zenodo
archive and citation metadata.

## Contributing

From a [repository checkout](https://github.com/jolars/lazymatrix), run
`task python:check` at the repository root to build, lint, and test the package.
See the [contribution
guide](https://github.com/jolars/lazymatrix/blob/main/CONTRIBUTING.md) for
setup, packaging checks, and pull request guidance.

## License

Licensed under either the [Apache License, Version
2.0](https://github.com/jolars/lazymatrix/blob/main/LICENSE-APACHE) or the [MIT
license](https://github.com/jolars/lazymatrix/blob/main/LICENSE-MIT), at your
option.
