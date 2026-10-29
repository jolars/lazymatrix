# LazyMatrix <picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jolars/lazymatrix/main/assets/logo-dark.png" /><img src="https://raw.githubusercontent.com/jolars/lazymatrix/main/assets/logo.png" align="right" width="189" alt="LazyMatrix logo" /></picture>

[![CI](https://github.com/jolars/lazymatrix/actions/workflows/ci.yml/badge.svg)](https://github.com/jolars/lazymatrix/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/lazymatrix.svg)](https://crates.io/crates/lazymatrix)
[![PyPI](https://badge.fury.io/py/lazymatrix.svg)](https://badge.fury.io/py/lazymatrix)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23666450-blue)](https://doi.org/10.5281/zenodo.23666450)

LazyMatrix is a Rust library for column normalization of design matrices. It
applies centering and scaling through matrix operations, preserving sparse
storage without creating a normalized copy. It supports dense and sparse
matrices, borrowed views, and chunked Zarr arrays.

See the [documentation](https://lazymatrix.org) for guides, benchmarks, and
links to the API reference.

## Installation

Enable the feature for your matrix library and add a matching backend
dependency. For ndarray:

```sh
cargo add lazymatrix --features ndarray
cargo add ndarray@0.17
```

The core library requires Rust 1.84 or newer. Optional backends can require a
newer compiler. sprs 0.11.4 requires Rust 1.85, while sprs 0.11.5 and newer
require Rust 1.88. For Rust 1.85 through 1.87, select sprs 0.11.4 using
`cargo update --package sprs --precise 0.11.4`. The `nalgebra` feature requires
Rust 1.89. The `zarrs` feature requires Rust 1.91; `zarrs_v0_22` retains Rust
1.87 support.

## Usage

Fit column means and standard deviations, then multiply by the normalized
matrix:

```rust
use lazymatrix::{LazyMatrix, MatVec, Normalization};
use ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = array![[1.0, 0.0], [2.0, 3.0], [0.0, 4.0]];
    let normalization = Normalization::standardize();
    let lazy = LazyMatrix::new(x.view(), normalization)?;

    let y = lazy.matvec(&array![1.0, -1.0])?;
    println!("{y}");
    Ok(())
}
```

The operator borrows the original matrix and keeps the fitted centers and
scales. Use `Normalization::min_max()` for min-max scaling to `[0, 1]` on fitted
data or `Normalization::robust()` for median centering and interquartile-range
scaling. `Normalization::new(center, scale)` accepts custom combinations, with
centering and scaling independently optional. Reuse fitted parameters on new
observations with `LazyMatrix::from_normalization`.

Centered dense products normalize entries before accumulation to preserve small
variations around large column offsets. They use the same numerical policy for
allocating and reusable products without creating a normalized design copy.

The [examples](examples/) cover gradient descent, stochastic gradient descent,
coordinate descent, ridge regression through primal or dual systems, weighted
Gram matrices, and file-backed input. Run one from
a checkout with:

```sh
cargo run --locked --example coordinate_descent --features faer
```

## Features

- Mean, minimum, or median centering, with scaling by population or sample
  standard deviation, L1 or L2 norm, maximum absolute value, range, or
  configurable quantile range.
- Forward and transpose matrix–vector products, reusable output buffers, and
  weighted Gram matrices.
- Batched forward and transpose products with caller-selected dense outputs.
- Borrowed transpose operator views through `Transpose::new(&operator)`,
  preserving fitted normalization across vector and batched products.
- Borrowed logical columns and sparse column and row views for supported
  storage.
- Borrowed contiguous submatrices that retain fitted normalization parameters,
  with weighted operations and bounded Zarr reads.
- An implicit intercept column through `WithIntercept`.
- Explicit dense conversion through `to_eager` and `to_eager_into`, or in-place
  normalization of writable dense storage through `into_eager`.
- Chunked Zarr input for matrices larger than RAM.

## Matrix backends

No matrix backend is enabled by default. Choose a feature to use an existing
matrix library:

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

Custom backends can implement the crate's matrix and vector traits. See the [API
reference](https://docs.rs/lazymatrix) for capabilities and supported types.

## Python package

The companion Python package supports NumPy arrays and SciPy CSR/CSC input.
Install it with `pip install lazymatrix`:

```python
import numpy as np
from lazymatrix import standardize

x = np.array([[1.0, 0.0], [2.0, 3.0], [0.0, 4.0]])
z = standardize(x)
prediction = z @ np.array([1.0, -1.0])
gradient = z.T @ prediction
```

See the [Python guide](https://lazymatrix.org/guide/python) for installation,
ownership, and supported operations, and the [Python API
reference](https://lazymatrix.org/reference/python/) for details.

## Documentation

- [Getting started](https://lazymatrix.org/guide/getting-started)
- [How it works](https://lazymatrix.org/guide/how-it-works)
- [API reference](https://docs.rs/lazymatrix)
- [Python API reference](https://lazymatrix.org/reference/python/)
- [Rust API reference on the website](https://lazymatrix.org/reference/rust/)
- [Benchmarks](https://lazymatrix.org/benchmarks)

## Citation

If you use LazyMatrix in your research, please cite the release you used. Select
that release in the [Zenodo archive](https://doi.org/10.5281/zenodo.23666450)
and export its citation. To cite LazyMatrix generally, use the concept DOI
`10.5281/zenodo.23666450`.

Machine-readable metadata is available in [CITATION.cff](CITATION.cff).

## License

Licensed under either the [Apache License, Version 2.0](LICENSE-APACHE) or the
[MIT license](LICENSE-MIT), at your option.
