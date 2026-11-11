# Backend setup

## Rust backends

The core depends only on `num-traits`. Enable a backend for ready-made matrix
implementations:

```sh
cargo add lazymatrix --features ndarray
cargo add ndarray@0.17
```

| Feature    | Storage                                 | Backend release selected |
| ---------- | --------------------------------------- | ------------------------ |
| `faer`     | Dense, CSC, and CSR sparse matrices     | faer 0.24                |
| `nalgebra` | Dense, CSC, and CSR sparse matrices     | nalgebra 0.35            |
| `ndarray`  | Dense arrays and views                  | ndarray 0.17             |
| `sprs`     | CSC and CSR sparse matrices             | sprs 0.11                |
| `zarrs`    | Synchronous two-dimensional Zarr arrays | zarrs 0.23               |

Match your direct backend dependency to the selected release. Versioned features
such as `nalgebra_v0_34` select an older supported release line. See the
[README](https://github.com/jolars/lazymatrix#matrix-backends) for supported
releases.

## Compiler requirements

The core supports Rust 1.84. Backend dependencies can require a newer compiler.
sprs 0.11.4 requires Rust 1.85, while sprs 0.11.5 and newer require Rust 1.88.
For Rust 1.85 through 1.87, select sprs 0.11.4 using
`cargo update --package sprs --precise 0.11.4`. The `nalgebra` feature requires
Rust 1.89. The `zarrs` feature requires Rust 1.91; `zarrs_v0_22` retains Rust
1.87 support.

## Optional BLAS acceleration

Enable `ndarray-blas` for ndarray 0.17, or use `ndarray_v0_15-blas`,
`ndarray_v0_16-blas`, or `ndarray_v0_17-blas` to select a specific release:

```sh
cargo add lazymatrix --features ndarray-blas
cargo add ndarray@0.17
```

Your application must select and link a BLAS provider. LazyMatrix selects no
provider, so these features work with an application's existing BLAS setup. For
a system LP64 OpenBLAS installation, link the application with:

```sh
RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-l openblas" cargo run
```

For a provider in a nonstandard directory, add its library search path and
configure the runtime loader. In this checkout, the devenv environment supplies
`OPENBLAS_LP64_LIB`; `task test:blas` uses it to set both paths and tests every
supported ndarray release. Without that variable, the task links system
OpenBLAS. `task test` runs the backend suites without requiring a BLAS provider.
Running `cargo test --all-features` directly requires a provider at link time.

BLAS threading is controlled by the provider independently of `parallel`.
For OpenBLAS, set `OPENBLAS_NUM_THREADS` and `OMP_NUM_THREADS` when controlling
threads. See [how it works](/guide/how-it-works#matrix–vector-products) for
which products use BLAS and the numerical policy for centered data.

## Python installation

```sh
pip install lazymatrix
```

Python 3.11 or newer is required. Prebuilt wheels are available for Linux
x86_64, macOS on Intel and Apple silicon, and Windows x86_64. Installing from
source also requires Rust 1.89 or newer. The Python package includes its native
extension; Cargo backend features apply to Rust consumers.

### PyTorch adapter

```sh
pip install "lazymatrix[torch]"
```

PyTorch 2.9 or newer is required. Use [PyTorch's installation
instructions](https://pytorch.org/get-started/locally/) when you need a particular
CUDA build. The base LazyMatrix package does not require or import PyTorch.
See [using normalized matrices](/guide/matrix-operations#use-a-pytorch-first-layer)
for a training example.
