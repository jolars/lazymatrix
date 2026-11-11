# Getting started with Rust

LazyMatrix centers and scales matrices during matrix operations without creating
a normalized copy. This walkthrough uses ndarray. For Python, see the [Python
quick start](/guide/python).

## Install a backend

Add LazyMatrix and ndarray to your Rust project:

```sh
cargo add lazymatrix --features ndarray
cargo add ndarray@0.17
```

For other backends, compiler requirements, and optional BLAS acceleration, see
[backend setup](/guide/installation).

## Normalize and multiply

Fit column means and standard deviations, then multiply by the normalized
matrix:

```rust
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

The operator borrows `x` and stores its fitted centers and scales. `matvec`
multiplies by the normalized values while leaving the original matrix unchanged.
Standardization uses population standard deviations.

Use `ColumnNormalization::min_max()` for min-max scaling or
`ColumnNormalization::robust()` for median centering and interquartile-range
scaling. See [normalization](/guide/normalization) for custom recipes, row
normalization, and composing stages.

## Reuse fitted parameters

For validation or prediction, apply the training parameters to new observations.
Using `lazy` from the example above:

```rust
let x_valid = array![[2.0, 1.0], [0.0, 3.0]];
let validation = lazy.normalization().view().apply(x_valid.view());
let y_valid = validation.matvec(&array![1.0, -1.0]).unwrap();
```

`.view()` borrows the fitted parameters, and `.apply()` wraps the new data
without refitting. See
[normalization](/guide/normalization#reuse-fitted-parameters) for fitting
separately and preparing new samples.

## Next steps

- [Using normalized matrices](/guide/matrix-operations): forward and transpose
  products, submatrices, and eager conversion.
- [How it works](/guide/how-it-works): the normalization algebra and memory
  costs.
- [Rust API reference](/reference/rust/): public types, traits, and backend
  capabilities.
