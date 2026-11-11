# LazyMatrix <picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jolars/lazymatrix/main/assets/logo-dark.png" /><img src="https://raw.githubusercontent.com/jolars/lazymatrix/main/assets/logo.png" align="right" width="189" alt="LazyMatrix logo" /></picture>

[![CI](https://github.com/jolars/lazymatrix/actions/workflows/ci.yml/badge.svg)](https://github.com/jolars/lazymatrix/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/lazymatrix.svg)](https://crates.io/crates/lazymatrix)
[![docs.rs](https://img.shields.io/docsrs/lazymatrix)](https://docs.rs/lazymatrix)
[![PyPI](https://img.shields.io/pypi/v/lazymatrix.svg)](https://pypi.org/project/lazymatrix/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23666450-blue)](https://doi.org/10.5281/zenodo.23666450)

LazyMatrix is a Rust library for centering and scaling design matrices without
creating a normalized copy. It applies normalization during matrix operations,
preserving sparse storage. It supports dense and sparse matrices, borrowed
views, and chunked Zarr arrays. A companion Python package supports NumPy,
SciPy, and PyTorch.

See the [documentation](https://lazymatrix.org) for guides, benchmarks, and API
references.

## Installation

Add [lazymatrix](https://crates.io/crates/lazymatrix) and enable the feature for
your matrix library. For ndarray:

```sh
cargo add lazymatrix --features ndarray
cargo add ndarray@0.17
```

The core library requires Rust 1.84 or newer. Optional backends can require a
newer compiler; see [backend setup](https://lazymatrix.org/guide/installation).

For Python, run `pip install lazymatrix` and see the [Python
README](python/README.md) for a quick start.

## Usage

Fit column means and standard deviations, then multiply by the normalized
matrix:

```rust
use lazymatrix::{ColumnNormalization, LazyMatrix, MatVec};
use ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = array![[1.0, 0.0], [2.0, 3.0], [0.0, 4.0]];
    let normalization = ColumnNormalization::standardize();
    let lazy = LazyMatrix::new(x.view(), normalization)?;

    let y = lazy.matvec(&array![1.0, -1.0])?;
    println!("{y}");
    Ok(())
}
```

The operator borrows the original matrix and stores its fitted centers and
scales. Standardization uses population standard deviations. Use
`ColumnNormalization::min_max()` for min-max scaling or
`ColumnNormalization::robust()` for median centering and interquartile-range
scaling. Centering and scaling are independently optional.

Fit parameters once and reuse them for validation or prediction without
refitting. The [normalization
guide](https://lazymatrix.org/guide/normalization#reuse-fitted-parameters)
shows how to share fitted parameters between operators.

Implement `NormalizationRecipe<M, P>` for custom fitting methods that return
the existing parameter types. Custom recipes work with `LazyMatrix::new`,
`.fit()`, and `.fit_apply()` and can report their own fitting errors.

The [examples](examples/) cover gradient descent, coordinate descent, ridge
regression, weighted Gram matrices, and file-backed input. Run one from a
checkout with:

```sh
cargo run --locked --example coordinate_descent --features faer
```

## Features

- **Row and column normalization** with mean, minimum, or median centering and
  scaling by standard deviation, L1 or L2 norm, maximum absolute value, range,
  or quantile range.
- **Reusable fitted parameters** for training and validation, and ordered
  normalization recipes composed with `.then()`.
- **Matrix products** including forward and transpose products, batched
  products, reusable output buffers, and weighted Gram matrices.
- **Borrowed views** of logical rows, columns, and submatrices for supported
  storage, plus an implicit intercept column through `WithIntercept`.
- **Explicit eager conversion**, including in-place normalization of writable
  dense storage.
- **Chunked Zarr input** for matrices larger than RAM.

Capabilities vary by normalization and backend. See the [Rust API
reference](https://docs.rs/lazymatrix) for supported combinations and traits for
custom backends.

## Matrix backends

No matrix backend is enabled by default. Match your matrix dependency to one of
these features:

  | Feature    | Supported releases     | Storage                                 |
  | ---------- | ---------------------- | --------------------------------------- |
  | `faer`     | 0.22, 0.23, 0.24       | Dense, CSC, and CSR matrices            |
  | `nalgebra` | 0.32, 0.33, 0.34, 0.35 | Dense, CSC, and CSR matrices            |
  | `ndarray`  | 0.15, 0.16, 0.17       | Dense arrays and views                  |
  | `sprs`     | 0.11                   | CSC and CSR matrices                    |
  | `zarrs`    | 0.22, 0.23             | Synchronous two-dimensional Zarr arrays |

The short feature names select the newest supported release. Use a versioned
feature such as `nalgebra_v0_34` for an older release, and match your direct
backend dependency to it. Enable `parallel` alongside a backend for parallel
column statistics where supported.

Optional BLAS acceleration is available for ndarray. See the [BLAS setup
guide](https://lazymatrix.org/guide/installation#optional-blas-acceleration)
for features and provider configuration.

## Documentation

- [Getting started with Rust](https://lazymatrix.org/guide/getting-started):
  installation, a first product, and parameter reuse.
- [Normalization](https://lazymatrix.org/guide/normalization): choose recipes,
  normalize rows, and reuse fitted parameters.
- [Using normalized matrices](https://lazymatrix.org/guide/matrix-operations):
  products, submatrices, and eager conversion.
- [How it works](https://lazymatrix.org/guide/how-it-works): the algebra and
  memory tradeoffs.
- [Rust API reference](https://docs.rs/lazymatrix): public types and backend
  capabilities.
- [Getting started with Python](https://lazymatrix.org/guide/python) and [API
  reference](https://lazymatrix.org/reference/python/): NumPy, SciPy, and
  PyTorch integrations.
- [Benchmarks](https://lazymatrix.org/benchmarks): runtime and memory
  comparisons.

## Citation

If you use LazyMatrix in your research, please cite the release you used. Select
that release in the [Zenodo archive](https://doi.org/10.5281/zenodo.23666450)
and export its citation. To cite LazyMatrix generally, use the concept DOI
`10.5281/zenodo.23666450`.

Machine-readable metadata is available in [CITATION.cff](CITATION.cff).

## Contributing

Bug reports, documentation improvements, and code contributions are welcome. See
[CONTRIBUTING.md](CONTRIBUTING.md) for setup, checks, and pull request guidance
for Rust, Python, and the website.

## License

Licensed under either the [Apache License, Version 2.0](LICENSE-APACHE) or the
[MIT license](LICENSE-MIT), at your option.
