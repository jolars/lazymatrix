//! Lazy normalized design matrices.
//!
//! A [`LazyMatrix`] wraps an underlying (typically sparse) matrix `X` together
//! with optional column **centers** `c` and column **scales** `s`, and presents
//! the *normalized* matrix
//!
//! ```text
//! X̃ = (X − 1cᵀ) S⁻¹,   S = diag(s)
//! ```
//!
//! as a linear operator — **without ever materializing `X − 1cᵀ`**. Centering a
//! sparse matrix turns its structural zeros into nonzeros, destroying sparsity;
//! factoring the normalization into the matrix–vector products avoids that:
//!
//! ```text
//! X̃ v  = X (S⁻¹ v) − 1 · (cᵀ S⁻¹ v)
//! X̃ᵀ u = S⁻¹ (Xᵀ u − c · Σu)
//! ```
//!
//! Both centering and scaling are independently optional, giving the four
//! combinations handled by the `if let Some` branches in the operator impls.
//!
//! # Eager normalization
//!
//! [`LazyMatrix::to_eager`] explicitly materializes normalized dense data from
//! a dense, sparse, or Zarr input. The caller chooses the destination backend.
//! [`LazyMatrix::to_eager_into`] fills existing dense storage and returns an
//! [`EagerMatrix`] borrowing it. [`LazyMatrix::into_eager`] consumes writable
//! dense data and normalizes it in its existing allocation.
//!
//! Eager operations use the already normalized backend data. Original fitted
//! centers and scales remain available through [`EagerMatrix::normalization`]
//! and can be reused with [`LazyMatrix::from_normalization`] for prediction.
//! Conversion does not recompute statistics. Sparse and Zarr materialization
//! requires O(nrows * ncols) dense output storage; the lazy path remains the
//! default. Arithmetic order changes, so rounding and nonfinite product results
//! can differ from lazy products.
//!
//! ```
//! use lazymatrix::{ColumnStats, EagerMatrix, LazyMatrix, MaterializeDense,
//!                  MatrixOwned, Normalization};
//!
//! fn eager_design<M, D>(x: &M, spec: Normalization)
//!     -> Result<EagerMatrix<D>, M::Error>
//! where
//!     M: ColumnStats<f64> + MaterializeDense<f64>,
//!     D: MatrixOwned<f64>,
//! {
//!     LazyMatrix::new(x, spec)?.to_eager()
//! }
//! ```
//!
//! # Backends
//!
//! The core is generic over the backend matrix `M` and scalar `F` and pulls in
//! no linear-algebra dependency by itself. Concrete implementations are provided
//! behind feature flags:
//!
//! * `faer` — `faer::Mat` and `faer::sparse::SparseColMat` over `faer::Col`.
//! * `nalgebra` — `nalgebra::DMatrix` and `nalgebra_sparse::CscMatrix` over
//!   `nalgebra::DVector`.
//! * `ndarray` — `ndarray::Array2` and borrowed, strided matrix views over
//!   `ndarray::Array1`.
//! * `sprs` — CSC and CSR `sprs::CsMat` matrices and borrowed views over `Vec`.
//!   Supports `Array1` vectors from every enabled ndarray release.
//!   `SprsCsc` checks CSC orientation for borrowed columns; `SprsCsr` checks
//!   CSR orientation for borrowed rows.
//! * `zarrs` — synchronous chunked `ZarrMatrix` arrays over `Vec`, with
//!   fallible products and statistics. Supports `f32` and `f64`.
//! * `parallel` — parallel column statistics through Rayon; also enables the
//!   enabled faer releases' Rayon support.
//!
//! Unversioned features select the newest supported release. Use a versioned
//! feature to stay on a particular release line:
//!
//! | Backend | Versioned features | Unversioned feature selects |
//! | --- | --- | --- |
//! | faer | `faer_v0_22`, `faer_v0_23`, `faer_v0_24` | 0.24 |
//! | nalgebra | `nalgebra_v0_32`, `nalgebra_v0_33`, `nalgebra_v0_34`, `nalgebra_v0_35` | 0.35 |
//! | ndarray | `ndarray_v0_15`, `ndarray_v0_16`, `ndarray_v0_17` | 0.17 |
//! | sprs | `sprs_v0_11` | 0.11 |
//! | zarrs | `zarrs_v0_22` | 0.22 |
//!
//! If Cargo enables several releases of one backend, every enabled release
//! receives its own trait implementations. Another dependency enabling a newer
//! adapter does not remove support for existing types. Select a matching version
//! feature for each direct backend dependency. The `*_all` features are internal
//! markers and cannot be enabled without a version feature.
//!
//! The core requires Rust 1.85. Other backends retain their dependency MSRVs. The `nalgebra` and
//! `nalgebra_v0_35` features require Rust 1.89. The `nalgebra` feature previously
//! selected 0.34; use `nalgebra_v0_34` to retain that release and Rust 1.87 support.
//! The sprs backend supports Rust 1.85 with sprs 0.11.4, as locked in this
//! repository. sprs 0.11.5 requires Rust 1.88.
//!
//! Any type implementing the [`traits`] surface (a dense matrix, say) works too.
//!
//! # Example
//!
//! ```ignore
//! use lazymatrix::{LazyMatrix, MatVec, Normalization, Centering, Scaling};
//!
//! // `x` is some backend matrix implementing `MatVec`, `MatTransposeVec`,
//! // `ColumnStats`, `MatrixShape`; `v` a backend vector.
//! let spec = Normalization::new(Centering::Mean, Scaling::Sd);
//! let lazy = LazyMatrix::new(x, spec).unwrap();
//! let y = lazy.matvec(&v).unwrap(); // == ((X − 1cᵀ)S⁻¹) v, sparsity preserved
//! ```
//!
//! With any ndarray version feature, a matrix view borrows the original array:
//!
//! ```
//! # #[cfg(feature = "ndarray_all")]
//! # {
//! # #[cfg(all(feature = "ndarray_v0_15", not(any(feature = "ndarray_v0_16", feature = "ndarray_v0_17"))))]
//! # use ndarray_0_15 as ndarray;
//! # #[cfg(all(feature = "ndarray_v0_16", not(feature = "ndarray_v0_17")))]
//! # use ndarray_0_16 as ndarray;
//! use lazymatrix::{Centering, LazyMatrix, MatVec, Normalization, Scaling};
//! use ndarray::array;
//!
//! let x = array![[1.0, 0.0], [2.0, 3.0], [0.0, 4.0]];
//! let lazy = LazyMatrix::new(
//!     x.view(),
//!     Normalization::new(Centering::Mean, Scaling::Sd),
//! ).unwrap();
//! let y = lazy.matvec(&array![1.0, -1.0]).unwrap();
//! assert_eq!(y.len(), 3);
//! # }
//! ```
//!
//! Allocating ndarray products use owned `Array1` vectors. Reusable-output
//! products also support mutable, strided destinations. Lazy forward products
//! require a clonable, mutable input, so convert immutable input views with
//! `to_owned()` first. Logical-column operations and reusable transpose products
//! accept immutable vector views directly. Raw columns borrow in O(1) time;
//! dense logical-column operations take O(nrows) time.
//!
//! [`MatVecScaledInto`] and [`MatTransposeVecScaledInto`] fuse a product with
//! output scaling for all supported matrix backends, including normalized
//! [`LazyMatrix`] and [`WithIntercept`] wrappers. Exact zero `alpha` skips the
//! product; exact zero `beta` ignores previous output values. Scaling a normalized
//! forward product needs O(ncols) coefficient scratch. The optional
//! [`LazyMatrix::matvec_scaled_with_workspace`] and
//! [`LazyMatrix::mat_transpose_vec_scaled_with_workspace`] methods let callers
//! reuse that storage across calls.
//! [`WithIntercept::matvec_scaled_with_workspace`] and
//! [`WithIntercept::mat_transpose_vec_scaled_with_workspace`] reuse the wrapper's
//! predictor buffer, whose length excludes the intercept. Inner normalization
//! may still allocate its own scratch. Fusing an operation alone does not
//! guarantee fewer allocations or faster products; see the consumer benchmarks.
//!
//! sprs products and statistics work directly on either CSC or CSR storage.
//! They visit stored entries without copying or materializing a normalized
//! matrix. CSC statistics take O(ncols + nnz) time and support `parallel`;
//! CSR statistics scan rows serially in O(nrows + ncols + nnz) time using
//! O(ncols) workspace. `sprs` does not select an ndarray backend version.
//!
//! To borrow columns, pass CSC storage or a view to `SprsCsc::try_new` before
//! constructing a `LazyMatrix`. The wrapper checks orientation in O(1) time
//! without copying, and returns CSR inputs unchanged as `Err`. Logical columns
//! accept any sprs index type; `SparseColumns` requires `usize` row indices.
//! Raw columns borrow in O(1) time. A centered column dot takes
//! O(nrows + nnz_column), or O(nnz_column) with `dot_with_sum`.
//!
//! [`SparseRows`] borrows raw column-index and value slices from CSR storage in
//! O(1) time, preserving explicitly stored zeros. It is available for faer's
//! `SparseRowMat`, `SparseRowMatRef`, and `SparseRowMatMut` with `usize` indices,
//! and nalgebra-sparse's `CsrMatrix`. These CSR types provide shape and borrowed
//! row access; their operator and column-statistics implementations remain future
//! work. With sprs, wrap a CSR matrix or view in `SprsCsr::try_new`. The wrapper
//! returns CSC inputs unchanged as `Err` and forwards existing products and
//! statistics. Borrowing rows requires `usize` column indices, while pointer
//! indices may use any supported width. The slices describe the original
//! matrix, before normalization.
//!
//! [`LazyMatrix::row`] requires [`SparseRows`] and returns a borrowed [`LazyRow`]
//! in O(1) time without allocation. The view exposes raw storage and the full
//! optional center and scale slices. Its logical length is the number of columns,
//! even for an empty raw row. Centering generally makes the logical row dense:
//! [`LazyRow::implicit_value`] exposes the background at a column, and
//! [`LazyRow::stored_corrections`] iterates over the scaled raw entries.

//! # An implicit intercept
//!
//! [`WithIntercept`] represents `[1, predictors]`. Normalize predictors first,
//! then wrap them so the intercept remains one. It also accepts unnormalized
//! operators and borrowed inputs. Coefficient zero is always the intercept.
//! Products retain backend errors and use coefficient scratch without allocating
//! a column of ones. Fitting and coefficient transformations stay downstream.
//!
//! ```
//! # #[cfg(feature = "ndarray_all")]
//! # {
//! # #[cfg(all(feature = "ndarray_v0_15", not(any(feature = "ndarray_v0_16", feature = "ndarray_v0_17"))))]
//! # use ndarray_0_15 as ndarray;
//! # #[cfg(all(feature = "ndarray_v0_16", not(feature = "ndarray_v0_17")))]
//! # use ndarray_0_16 as ndarray;
//! use lazymatrix::{LazyMatrix, MatVec, WithIntercept};
//! use ndarray::array;
//!
//! let x = array![[1.0], [3.0]];
//! let predictors = LazyMatrix::with_centers(x.view(), vec![2.0]);
//! let design = WithIntercept::new(&predictors);
//! assert_eq!(design.matvec(&array![3.0, 2.0]).unwrap(), array![1.0, 5.0]);
//! # }
//! ```
//!
//! Intercept Gram products require [`WeightedGramInto`] and
//! [`WeightedColumnSumsInto`] on the predictors. Cross terms use a separate pass
//! with direct centering, preserving the existing Gram kernels' numerical policy.
//! [`WeightedColumnSumsKernel`] supplies explicit backend normalization, and
//! [`VectorOwned`] supplies owned scratch compatible with backend vector views.
//!
//! # Operational errors and out-of-core storage
//!
//! [`WeightedGramInto`] computes `Aᵀ diag(weights) A` for dense ndarray and
//! `usize`-index CSC inputs from faer, nalgebra, and `SprsCsc` (when enabled),
//! as well as CSR inputs through `SprsCsr`. CSR kernels use bounded row panels
//! with O(nrows × ncols²) arithmetic and no storage conversion.
//! [`MatrixWrite`] destinations include owned and mutable-view dense matrices
//! from ndarray, faer, and nalgebra. The output backend is independent of the
//! input backend. Both triangles are overwritten, and weights can be signed,
//! zero, or nonfinite. Kernels center values before accumulation, avoiding
//! cancellation from subtracting large raw moments. Bounded dense panels or
//! sparse working vectors avoid materializing the full logical design matrix.
//! See `examples/weighted_gram.rs` for a borrowed-input demonstration.
//!
//! Products, [`ColumnStats`] methods, and [`LazyMatrix::new`] return `Result`.
//! [`MatrixErrorType`] gives each backend one shared error type. In-memory
//! backends use [`std::convert::Infallible`]; storage backends propagate read
//! and decoding errors. Dimension mismatches still panic. After a failed
//! reusable-output product, discard the partial output or overwrite it with a
//! successful product. Borrowed views and explicit normalization parameters do
//! not require I/O and retain their infallible APIs.
//!
//! [`LazyMatrix::from_parts`] and [`LazyMatrix::with_scales`] panic on explicit
//! zero scales, including negative zero. Explicit parameters are otherwise
//! preserved unchanged, including negative scales and nonfinite centers or
//! scales. Computed normalization through [`LazyMatrix::new`] replaces exact
//! zero scales with one while preserving nonfinite statistics.
//!
//! With `zarrs`, `ZarrMatrix` wraps an opened synchronous array without reading
//! its chunks. Each product scans chunks serially, and normalization shares work
//! through [`ColumnStats::normalization_stats`] to need at most two scans. Working
//! vectors stay in RAM. Memory for data and codecs depends on chunk size, including
//! the outer shard for sharded storage; no strict byte budget is imposed. The
//! backing array must remain unchanged throughout normalization and use.
//! Filesystem and gzip support are enabled; additional codecs can be selected
//! through a direct zarrs dependency. No ndarray backend is selected by `zarrs`.
//!
//! ```
//! # #[cfg(feature = "zarrs_all")]
//! # {
//! use std::sync::Arc;
//! use lazymatrix::{Centering, LazyMatrix, MatVec, Normalization, Scaling, ZarrMatrix};
//! use zarrs::{array::{ArrayBuilder, DataType}, storage::store::MemoryStore};
//!
//! let array = ArrayBuilder::new(vec![3, 2], vec![2, 2], DataType::Float64, 1.0f64)
//!     .build(Arc::new(MemoryStore::new()), "/matrix")?;
//! let matrix = ZarrMatrix::<_, f64>::try_new(array)?;
//! let lazy = LazyMatrix::new(matrix, Normalization::new(Centering::Mean, Scaling::Sd))?;
//! assert_eq!(lazy.matvec(&vec![1.0, 2.0])?, vec![0.0; 3]);
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Borrowed ndarray views can also wrap memory-mapped `.npy` data. The
//! `ndarray_mmap` example uses ndarray 0.17 and a private, immutable backing file.
//! The `zarrs_chunked` example creates a filesystem array chunk by chunk. Both
//! accept row and column counts and print normalization and product timings.

// Each enabled release keeps its own crate identity and implementations.
#[cfg(feature = "faer_v0_22")]
extern crate faer_0_22;

#[cfg(feature = "faer_v0_22")]
extern crate faer_traits_0_22;

#[cfg(feature = "faer_v0_23")]
extern crate faer_0_23;

#[cfg(feature = "faer_v0_23")]
extern crate faer_traits_0_23;

#[cfg(feature = "faer_v0_24")]
extern crate faer as faer_0_24;

#[cfg(feature = "faer_v0_24")]
extern crate faer_traits as faer_traits_0_24;

#[cfg(feature = "nalgebra_v0_32")]
extern crate nalgebra_0_32;

#[cfg(feature = "nalgebra_v0_32")]
extern crate nalgebra_sparse_0_9;

#[cfg(feature = "nalgebra_v0_33")]
extern crate nalgebra_0_33;

#[cfg(feature = "nalgebra_v0_33")]
extern crate nalgebra_sparse_0_10;

#[cfg(feature = "nalgebra_v0_34")]
extern crate nalgebra_0_34;

#[cfg(feature = "nalgebra_v0_34")]
extern crate nalgebra_sparse_0_11;

#[cfg(feature = "nalgebra_v0_35")]
extern crate nalgebra as nalgebra_0_35;

#[cfg(feature = "nalgebra_v0_35")]
extern crate nalgebra_sparse as nalgebra_sparse_0_12;

#[cfg(feature = "ndarray_v0_15")]
extern crate ndarray_0_15;

#[cfg(feature = "ndarray_v0_16")]
extern crate ndarray_0_16;

#[cfg(feature = "ndarray_v0_17")]
extern crate ndarray as ndarray_0_17;

#[cfg(feature = "sprs_v0_11")]
extern crate sprs;

#[cfg(feature = "zarrs_v0_22")]
extern crate zarrs;

#[cfg(all(
    feature = "faer_all",
    not(any(feature = "faer_v0_22", feature = "faer_v0_23", feature = "faer_v0_24"))
))]
compile_error!("`faer_all` is internal; enable `faer` or a `faer_v*` feature");

#[cfg(all(
    feature = "nalgebra_all",
    not(any(
        feature = "nalgebra_v0_32",
        feature = "nalgebra_v0_33",
        feature = "nalgebra_v0_34",
        feature = "nalgebra_v0_35"
    ))
))]
compile_error!("`nalgebra_all` is internal; enable `nalgebra` or a `nalgebra_v*` feature");

#[cfg(all(
    feature = "ndarray_all",
    not(any(
        feature = "ndarray_v0_15",
        feature = "ndarray_v0_16",
        feature = "ndarray_v0_17"
    ))
))]
compile_error!("`ndarray_all` is internal; enable `ndarray` or a `ndarray_v*` feature");

#[cfg(all(feature = "sprs_all", not(feature = "sprs_v0_11")))]
compile_error!("`sprs_all` is internal; enable `sprs` or a `sprs_v*` feature");

#[cfg(all(feature = "zarrs_all", not(feature = "zarrs_v0_22")))]
compile_error!("`zarrs_all` is internal; enable `zarrs` or a `zarrs_v*` feature");

mod backends;
mod column;
mod eager;
mod gram;
mod intercept;
mod materialize;
mod matrix;
mod normalization;
mod row;
pub mod traits;
mod weighted_sums;

#[cfg(feature = "sprs_all")]
pub use backends::sprs::{SprsCsc, SprsCsr};
#[cfg(feature = "zarrs_all")]
pub use backends::zarrs::{ZarrMatrix, ZarrMatrixError};
pub use column::{LazyColumn, LazySparseColumn, SparseColumnRef};
pub use eager::EagerMatrix;
pub use intercept::WithIntercept;
pub use matrix::LazyMatrix;
pub use normalization::{
    Centering, Normalization, NormalizationParams, NormalizationStats, Scaling,
};
pub use row::LazyRow;
pub use traits::{
    ColumnStats, Columns, DenseBlock, DenseNormalize, DotProduct, DotSlice, ElemDivAssign, L2Norm,
    LogicalColumn, MatTransposeVec, MatTransposeVecInto, MatTransposeVecScaledInto, MatVec,
    MatVecInto, MatVecScaledInto, MaterializeDense, MatrixErrorType, MatrixOwned, MatrixShape,
    MatrixWrite, RawColumn, RawColumns, ReadBlock, Scalar, ScaleAssign, ScaledAddAssign,
    ScaledSubSlice, SparseColumns, SparseRows, SubScalarAssign, SumEntries, VectorOwned,
    VectorView, VectorViewMut, WeightedColumnSumsInto, WeightedColumnSumsKernel, WeightedGramInto,
    WeightedGramKernel,
};
