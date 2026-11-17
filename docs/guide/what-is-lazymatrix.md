---
description: Learn what LazyMatrix does, why lazy normalization helps, and where to start in Rust or Python.
---

# What is LazyMatrix?

LazyMatrix is a library for centering and scaling matrices in Rust and Python.
It lets you work with normalized data without creating and storing a normalized
copy of the whole matrix.

When you fit a model, you often arrange your data in a _design matrix_: each row
is an observation, and each column is a feature. Features can have different
units and ranges. Centering subtracts a value, such as the mean, from each
feature. Scaling divides by a value, such as the standard deviation, to put
features on a comparable scale.

## Why keep normalization lazy?

Sparse matrices save memory by leaving most zeros unstored. Centering can turn
those zeros into nonzero values, so storing the normalized matrix can require
far more memory than storing the original data.

LazyMatrix keeps the original data together with the fitted centers and scales.
When you multiply by the matrix, it applies normalization during the calculation.
You get the result without first storing every normalized entry. This also
helps with dense matrices when a second large array would be costly.

Use LazyMatrix when your algorithm mainly needs matrix products, such as an
iterative regression solver. You can:

- **Choose how to normalize.** Center and scale columns or individual rows,
  using standardization, min-max scaling, or robust statistics.
- **Reuse fitted values.** Fit column centers and scales on training data and
  apply them to validation or prediction data without refitting.
- **Materialize when needed.** Create a dense normalized matrix explicitly
  when another library needs one or repeated operations justify its memory cost.

The [benchmarks](/benchmarks) compare time and memory for lazy and eager
normalization, which stores the normalized values in advance.

## Use your matrix library

- **Rust:** use faer, nalgebra, ndarray, or sprs matrices, including borrowed
  views where supported. Chunked Zarr arrays let you work with data larger than
  memory. See [backend setup](/guide/installation) to choose a backend.
- **Python:** use NumPy arrays or SciPy sparse matrices. The normalized operator
  supports multiplication with `@` and integrates with SciPy's iterative
  solvers. An optional PyTorch adapter applies fitted normalization in a
  trainable first layer.

The available operations depend on your backend and normalization choices.

## Where to go next

Start with [Getting started](/guide/getting-started) to install LazyMatrix,
compute your first normalized product, and reuse fitted parameters. It includes
examples for both Rust and Python.

Then follow the guide for your task:

- [Normalization](/guide/normalization): choose centering and scaling methods,
  normalize rows, and prepare new samples.
- [Using normalized matrices](/guide/matrix-operations): multiply, use SciPy
  solvers, select submatrices, convert to dense storage, or use PyTorch.
- [How it works](/guide/how-it-works): understand the normalization algebra,
  sparse storage, and memory tradeoffs.

For individual types and methods, see the [Rust API reference](/reference/rust/)
or [Python API reference](/reference/python/).
