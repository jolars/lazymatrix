# Getting started

LazyMatrix centers and scales matrices during matrix operations without creating
a normalized copy. Follow the Rust or Python examples to install it, multiply
by a normalized matrix, and reuse fitted parameters on new data.

## Install

::: code-group

```sh [Rust]
cargo add lazymatrix --features ndarray
cargo add ndarray@0.17
```

```sh [Python]
pip install lazymatrix
```

:::

The Rust examples use ndarray. Python requires version 3.11 or newer. See
[backend setup](/guide/installation) for other Rust backends, compiler
requirements, optional BLAS acceleration, and Python installation details.

## Normalize and multiply

Arrange observations in rows and features in columns. Fit column means and
standard deviations, then multiply by the normalized matrix:

::: code-group

```rust [Rust]
use lazymatrix::{ColumnNormalization, LazyMatrix, MatVec};
use ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = array![[1.0, 0.0], [2.0, 3.0], [0.0, 4.0]];
    let lazy = LazyMatrix::new(x.view(), ColumnNormalization::standardize())?;

    let y = lazy.matvec(&array![1.0, -1.0])?;
    println!("{y}");
    Ok(())
}
```

```python [Python]
import numpy as np
from lazymatrix import standardize
from scipy import sparse

X = sparse.csr_matrix([[1.0, 0.0], [2.0, 3.0], [0.0, 4.0]])
Z = standardize(X)

beta = np.array([1.0, -1.0])
prediction = Z @ beta
```

:::

Both operators store fitted column centers and scales. Products use the
normalized values without creating a normalized copy. Standardization uses
population standard deviations.

The Rust operator borrows `x`. The Python operator owns a copy of the raw input,
and sparse input remains sparse. You can use a NumPy array in place of `X`.

## Reuse fitted parameters

For validation or prediction, apply the training parameters to new observations.
The new input must have the same features in the same order. Using the fitted
operator from above, continue inside `main` in Rust or the same Python session:

::: code-group

```rust [Rust]
let x_valid = array![[2.0, 1.0], [0.0, 3.0]];
let validation = lazy.normalization().view().apply(x_valid.view());
let y_valid = validation.matvec(&array![1.0, -1.0]).unwrap();
```

```python [Python]
X_validation = sparse.csr_matrix([[4.0, 1.0], [0.0, -2.0]])
validation = Z.normalization.transform(X_validation)
validation_prediction = validation @ beta
```

:::

In Rust, `.view()` borrows the fitted parameters and `.apply()` wraps the new
data. In Python, `transform` applies the fitted values. Neither method refits
statistics. See
[normalization](/guide/normalization#reuse-fitted-parameters) for fitting
separately, preparing new samples, and saving fitted values.

## Next steps

- [Normalization](/guide/normalization): choose centering and scaling methods,
  normalize rows, and compose stages.
- [Using normalized matrices](/guide/matrix-operations): forward and transpose
  products, SciPy solvers, submatrices, eager conversion, and PyTorch integration.
- [How it works](/guide/how-it-works): the normalization algebra and memory
  costs.
- [Rust API reference](/reference/rust/): public types, traits, and backend
  capabilities.
- [Python API reference](/reference/python/): public classes and methods.

## Citation

If you find LazyMatrix useful in your academic work, please see [citing
LazyMatrix](/guide/citation) for a text citation and BibLaTeX and BibTeX
entries.
