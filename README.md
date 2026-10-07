# lazymatrix

[![CI](https://github.com/jolars/lazymatrix-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/jolars/lazymatrix-rs/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/lazymatrix.svg)](https://crates.io/crates/lazymatrix)
[![docs.rs](https://img.shields.io/docsrs/lazymatrix)](https://docs.rs/lazymatrix)

Lazy column normalization for design matrices in Rust. `lazymatrix` presents

```text
X̃ = (X − 1cᵀ) S⁻¹
```

as a linear operator without materializing the centered matrix. This matters for
sparse matrices, where subtracting a column center would turn structural zeros
into nonzeros. Matrix–vector products instead use the original matrix:

```text
X̃v  = X(S⁻¹v) − 1(cᵀS⁻¹v)
X̃ᵀu = S⁻¹(Xᵀu − c Σu)
```

Centering and scaling are independently optional. The crate also provides
borrowed logical column views and sparse column and row access for algorithms
that work directly with stored entries.

## Install

The core trait and operator API has no linear algebra dependency beyond
`num-traits`. Enable a backend for ready-made matrix implementations:

```sh
cargo add lazymatrix --features faer
# or
cargo add lazymatrix --features nalgebra
# or, for dense arrays
cargo add lazymatrix --features ndarray
# or, for sprs sparse matrices
cargo add lazymatrix --features sprs
# or, for chunked Zarr arrays
cargo add lazymatrix --features zarrs
```

Unversioned features select the newest supported release. Use a versioned
feature to stay on a particular release line:

| Backend | Versioned features | Unversioned feature selects |
| --- | --- | --- |
| faer | `faer_v0_22`, `faer_v0_23`, `faer_v0_24` | 0.24 |
| nalgebra | `nalgebra_v0_32`, `nalgebra_v0_33`, `nalgebra_v0_34`, `nalgebra_v0_35` | 0.35 |
| ndarray | `ndarray_v0_15`, `ndarray_v0_16`, `ndarray_v0_17` | 0.17 |
| sprs | `sprs_v0_11` | 0.11 |
| zarrs | `zarrs_v0_22` | 0.22 |

For example, `cargo add lazymatrix --features nalgebra_v0_34` enables nalgebra
0.34 and nalgebra-sparse 0.11. Set your direct backend dependency to the same
release line. If Cargo enables multiple releases of one backend, lazymatrix
implements traits independently for every enabled release. Another dependency
enabling a newer adapter does not remove support for your existing types. The
`*_all` features are internal markers, not entry points for selecting a backend.

The core and older backends support Rust 1.87. The `nalgebra` feature now selects
nalgebra 0.35, which requires Rust 1.89. To retain the previous release and Rust
1.87 support, replace `nalgebra` with `nalgebra_v0_34` in your feature list.
The sprs backend supports Rust 1.87 with sprs 0.11.4 (used in the lockfile);
sprs 0.11.5 requires Rust 1.88.

## Example

```rust
use lazymatrix::{
    Centering, LazyMatrix, MatVec, Normalization, Scaling,
};
use nalgebra::{DMatrix, DVector};

let x = DMatrix::from_row_slice(
    3,
    2,
    &[1.0, 0.0, 2.0, 3.0, 0.0, 4.0],
);
let x = LazyMatrix::new(
    x,
    Normalization::new(Centering::Mean, Scaling::Sd),
).unwrap();

let y = x.matvec(&DVector::from_vec(vec![1.0, -1.0])).unwrap();
```

The same interface works with faer and nalgebra dense matrices, their borrowed
views, and CSC sparse matrices. The ndarray backends support dense arrays,
including borrowed, transposed, and strided views:

```rust
use lazymatrix::{Centering, LazyMatrix, MatVec, Normalization, Scaling};
use ndarray::array;

let x = array![[1.0, 0.0], [2.0, 3.0], [0.0, 4.0]];
let lazy = LazyMatrix::new(
    x.view(),
    Normalization::new(Centering::Mean, Scaling::Sd),
).unwrap();
let y = lazy.matvec(&array![1.0, -1.0]).unwrap();
```

Allocating ndarray products use `Array1` vectors. Reusable-output products can
write into mutable, strided vector views. A lazy forward product needs an owned
input because it may clone and scale that input; call `to_owned()` on an
immutable input view first. Logical-column operations and reusable transpose
products accept immutable vector views directly. Borrowed column access preserves
the original strides and takes O(1) time; dense logical-column operations take
O(nrows) time.

`WeightedGramInto` computes `Aᵀ diag(weights) A` into a reusable dense matrix:

```rust
use lazymatrix::{LazyMatrix, WeightedGramInto};
use ndarray::{Array2, array};

let x = array![[1.0, 0.0], [2.0, 3.0], [0.0, 4.0]];
let lazy = LazyMatrix::with_centers(x.view(), vec![1.0, 2.0]);
let mut gram = Array2::zeros((2, 2));
lazy.weighted_gram_into(&array![1.0, 0.5, 2.0], &mut gram).unwrap();
```

Inputs can be dense ndarray arrays or views, faer or nalgebra CSC matrices,
or checked `SprsCsc` matrices or views with `usize` row indices. Output can be
an owned or mutable-view ndarray, faer, or nalgebra dense matrix, independent
of the input backend. Weights may be signed, zero, or nonfinite. The operation
overwrites both triangles and supports strided weights and outputs.

Kernels center values before multiplying, preserving small variations around
large offsets. Dense kernels use bounded panels. CSC kernels choose sparse pair
accumulation or bounded panels according to density; the sparse pair path uses
O(nrows + ncols) scratch with centering, in addition to the O(ncols²) output.
Neither creates a full normalized or weighted design matrix. Unsorted or
duplicate sparse indices and
nonfinite arithmetic use a slower direct fallback with at most two working
columns. Scratch is allocated internally on each call. See the trait documentation
for complexity and error contracts, and `examples/weighted_gram.rs` for a dense
and sparse demonstration.
The [weighted Gram benchmarks](benches/weighted_gram.md) compare both kernels
with repeated operator products and dense matrix multiplication, including BLAS.

`WithIntercept` adds a leading constant column after predictor normalization:

```rust
use lazymatrix::{LazyMatrix, MatVec, WeightedGramInto, WithIntercept};
use ndarray::{Array2, array};

let x = array![[1.0, 0.0], [2.0, 3.0], [0.0, 4.0]];
let predictors = LazyMatrix::with_centers(x.view(), vec![1.0, 2.0]);
let design = WithIntercept::new(&predictors);
let y = design.matvec(&array![2.0, 1.0, -1.0]).unwrap();
let mut gram = Array2::zeros((3, 3));
design.weighted_gram_into(&array![1.0, 0.5, 2.0], &mut gram).unwrap();
```

The intercept occupies coefficient zero and stays equal to one when predictors
are centered. The wrapper also accepts unnormalized operators. Products preserve
the underlying error type and storage access pattern, including chunked reads.
Reusable products allocate coefficient scratch; no column of ones or design
matrix is created. `as_inner()` exposes predictor normalization metadata, and
`into_inner()` recovers the predictors. Fitting, penalty exclusions, and
coefficient transformations remain with the caller.

Intercept Gram products reuse the predictor output block and compute the cross
terms in a separate pass through `WeightedColumnSumsInto`. Its kernels apply
centering before accumulation, including sparse implicit zeros, to preserve
small variations around large offsets. These capabilities cover the existing
dense ndarray and CSC Gram inputs; adding an intercept does not add Gram support
to other storage types. Ordinary forward and transpose products retain the
underlying operator's factored normalization arithmetic.

The sprs backend supports owned CSC and CSR matrices and borrowed views. Use
`Vec` for allocating products, or enable an ndarray feature to use that
release's `Array1`. Reusable-output products accept any supported dense vector
view, including strided ndarray inputs and destinations. The `sprs` feature
does not select an ndarray backend version.

Wrap CSC storage in `SprsCsc` to borrow logical or sparse columns:

```rust
use lazymatrix::{Centering, LazyMatrix, MatVec, Normalization, Scaling, SprsCsc};
use sprs::CsMat;

let x = CsMat::new_csc(
    (3, 2),
    vec![0, 2, 3],
    vec![0, 2, 1],
    vec![1.0, 3.0, 2.0],
);
let csc = SprsCsc::try_new(x.view()).unwrap();
let lazy = LazyMatrix::new(csc, Normalization::new(Centering::Mean, Scaling::Sd)).unwrap();
let y = lazy.matvec(&vec![1.0, -1.0]).unwrap();
let column = lazy.sparse_column(0);
assert_eq!(column.row_indices(), &[0, 2]);
```

`SprsCsc::try_new` checks orientation in O(1) time and returns a CSR input
unchanged as `Err`. Convert CSR explicitly with `to_csc()` or `into_csc()` when
column access is needed. Products and statistics also work directly on
`CsMat` or `CsMatView` in either orientation. They visit stored entries without
materializing the normalized matrix. CSC statistics take O(ncols + nnz) time;
CSR statistics take O(nrows + ncols + nnz) time and O(ncols) workspace.

Logical columns borrow sprs vector views for any supported index type.
`SparseColumns` requires `usize` row indices so it can return slices without
copying. A centered column dot takes O(nrows + nnz_column); `dot_with_sum`
uses a caller-supplied vector sum to take O(nnz_column).

Weighted column norms accumulate normalized entries directly in
O(nrows + nnz_column) time, including implicit zeros. A cached total weight
cannot preserve small implicit-row weights when subtracting a much larger
stored-weight sum, so `weighted_norm_squared_with_sum` uses the same scan.
Sorted, unique columns need constant scratch space; unsorted or duplicate
indices require one working column.

`SparseRows` borrows raw column-index and value slices from CSR storage in O(1)
time, including explicitly stored zeros. It supports faer's `SparseRowMat`,
`SparseRowMatRef`, and `SparseRowMatMut` with `usize` indices, and
nalgebra-sparse's `CsrMatrix`. These CSR types provide shape and borrowed
row access; their operator and column-statistics implementations remain future
work.

For sprs, use the checked `SprsCsr` wrapper:

```rust
use lazymatrix::{LazyMatrix, SparseRows, SprsCsr};
use sprs::CsMat;

let x = CsMat::new(
    (2, 3),
    vec![0, 2, 3],
    vec![0, 2, 1],
    vec![1.0, 0.0, 2.0],
);
let csr = SprsCsr::try_new(x.view()).unwrap();
let (columns, values) = csr.sparse_row(0);
assert_eq!(columns, &[0, 2]);
assert_eq!(values, &[1.0, 0.0]);

let lazy = LazyMatrix::from_parts(csr, Some(vec![0.5, -1.0, 2.0]), Some(vec![2.0; 3]));
let row = lazy.row(0);
assert_eq!(row.len(), 3);
assert_eq!(row.column_indices(), &[0, 2]);
assert_eq!(row.values(), &[1.0, 0.0]);
assert_eq!(row.implicit_value(1), 0.5);
assert_eq!(row.stored_corrections().collect::<Vec<_>>(), vec![(0, 0.5), (2, 0.0)]);
```

`SprsCsr::try_new` checks orientation without copying and returns a CSC input
unchanged as `Err`. The wrapper forwards products and statistics, so it can
also be passed to `LazyMatrix::new`. Row borrowing requires `usize` column
indices; pointer indices may use any supported width. The returned slices
describe the original matrix, before normalization.

`LazyMatrix::row(i)` requires `SparseRows` and borrows raw storage and the full
optional center and scale slices without allocating. `centers()` and `scales()`
expose those slices; `center(j)` and `scale(j)` return effective parameters of
zero and one when normalization is inactive. Explicit parameters let faer and
nalgebra CSR matrices use row views without column-statistics implementations.

A logical entry is `(raw[j] - center(j)) / scale(j)`. Sum duplicate raw entries
before normalizing. The affine representation has a background
`-center(j) / scale(j)` and stored corrections `raw_value / scale(j)`. Centering
can therefore make a logical row dense even when its stored entries are empty.
Consuming the corrections takes O(nnz_row) time and allocates nothing. Adding
the affine parts can differ from direct normalization because of floating-point
rounding or nonfinite arithmetic.

Enable `parallel` alongside a backend to compute column statistics with Rayon.
For sprs, CSC columns run independently in parallel; CSR statistics scan rows
serially to accumulate columns without converting storage.
See [`examples/`](examples/) for complete solver examples that consume the
operator.

## Fallible operations

Matrix products, column statistics, and `LazyMatrix::new` now return `Result`.
This is a breaking API change: use `?` to propagate errors, or unwrap results
when using an in-memory backend. Existing in-memory backends use
`std::convert::Infallible`; storage backends can return read and decoding errors.

Custom backends implement `MatrixErrorType` once and use its associated `Error`
type across all four operator traits and `ColumnStats`. Borrowing a matrix or
wrapping it in `LazyMatrix` preserves that error type. Shape queries, borrowed
column and row operations, and explicit-parameter construction remain infallible.
Dimension mismatches still panic. After a failed reusable-output product, the
output may be partial and must be discarded or overwritten by a successful call.

`ColumnStats::normalization_stats` computes the optional centers and raw scales
as a `NormalizationStats<F>` pair. Its default dispatches to individual
statistics. Storage backends can override it to share scans; `LazyMatrix::new`
then replaces exact zero scales with one, preserving nonfinite values.

`LazyMatrix::from_parts` and `LazyMatrix::with_scales` reject explicit zero scales
with a panic that reports the column index. Both positive and negative zero are
rejected. Explicit parameters are otherwise preserved unchanged, including
negative scales, tiny nonzero scales, and nonfinite centers or scales. These
constructors do not guarantee finite arithmetic results.

## Out-of-core matrices

A borrowed ndarray view can refer to a memory-mapped file. The `ndarray_mmap`
example creates a private temporary `.npy` file in column-major order, fills it
through a writable mapping, and normalizes a read-only `ArrayView2` without
allocating an owned matrix:

```sh
cargo run --locked --release --example ndarray_mmap --features ndarray -- 10000 32
```

This example requires ndarray 0.17. Memory mappings depend on the backing file
remaining unmodified while views exist, including by other processes. Mapping a
file lets the OS manage page residency; it does not impose a resident-memory
limit. Column-major storage keeps this backend's column-statistics scans
contiguous.

The `zarrs` feature supplies `ZarrMatrix` for synchronous, two-dimensional Zarr
arrays with `f32` or `f64` elements. Enable zarrs 0.22 in your direct dependency
as well. The adapter supports Rust 1.87 and does not select an ndarray backend.

```rust
use std::sync::Arc;
use lazymatrix::{Centering, LazyMatrix, MatVec, Normalization, Scaling, ZarrMatrix};
use zarrs::{array::Array, filesystem::FilesystemStore};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let store = Arc::new(FilesystemStore::new("data.zarr")?);
    let array = Array::open(store, "/matrix")?;
    let matrix = ZarrMatrix::<_, f64>::try_new(array)?;
    let lazy = LazyMatrix::new(matrix, Normalization::new(Centering::Mean, Scaling::Sd))?;
    let result = lazy.matvec(&vec![1.0; lazy.ncols()])?;
    println!("{} rows", result.len());
    Ok(())
}
```

Each product reads one chunk at a time. Normalization uses zero scans when
inactive, one when sufficient, and at most two otherwise. Missing chunks retain
the configured fill value, including nonzero or nonfinite values. Partial edge
chunks contribute only entries inside the array. The array must remain unchanged
throughout normalization and subsequent use; the adapter does not provide
snapshot isolation.

Allocating products use `Vec<F>`. Reusable-output products accept the existing
vector-view capabilities, including enabled backend vector types. The adapter
provides no borrowed columns or rows. Working vectors and column statistics stay
in RAM, together with chunk buffers and codec workspaces. Choose chunks that fit
in memory; for sharded arrays, the relevant bound is the outer storage chunk.
There is no byte budget, cache, prefetching, or parallel chunk scanning.

Filesystem and gzip support are enabled by this crate. Additional codecs can be
enabled through your direct zarrs dependency. An unsupported codec is reported
when zarrs opens the array.

The self-contained filesystem example creates data chunk by chunk:

```sh
cargo run --locked --release --example zarrs_chunked --features zarrs -- 10000 32
```

Both examples accept optional row and column counts and print operation timings.
For larger-than-RAM measurements, first build the examples, then run the binaries
with GNU time installed:

```sh
cargo build --locked --release --examples --features ndarray,zarrs
env time -v target/release/ndarray_mmap 1000000 256
env time -v target/release/zarrs_chunked 1000000 256
```

Increase dimensions so `rows * cols * 8` exceeds available RAM, while vectors and
chunks still fit. Set `TMPDIR` to a directory on disk and account for temporary
disk space. Record peak RSS and timings separately from correctness tests, and
distinguish page-cache effects from disk throughput: these examples generate
their own files before reading them, so a run on data smaller than RAM is not a
cold-I/O benchmark. CI uses small fixtures and verifies chunk-read counts without
allocating a larger-than-RAM dataset.

## License

Licensed under either the [Apache License, Version 2.0](LICENSE-APACHE) or the
[MIT license](LICENSE-MIT), at your option.
