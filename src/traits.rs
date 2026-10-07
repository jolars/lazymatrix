//! Backend-agnostic trait surface for [`LazyMatrix`](crate::LazyMatrix).
//!
//! The traits separate numeric operations from matrix capabilities:
//!
//! * [`Scalar`] — the numeric element type, a blanket-implemented bundle of
//!   `num-traits` bounds.
//! * [`MatrixErrorType`] — the shared operational error type for a backend.
//! * [`MatrixShape`], [`MatVec`] / [`MatTransposeVec`], and their reusable-output
//!   [`MatVecInto`] / [`MatTransposeVecInto`] counterparts — the matrix-free
//!   linear-operator interface, implemented both by concrete backend matrices
//!   and by [`LazyMatrix`](crate::LazyMatrix) itself. Products return `Result`.
//!   [`MatVecScaledInto`] and [`MatTransposeVecScaledInto`] provide fused
//!   `alpha`/`beta` products for all supported backends and [`crate::WithIntercept`].
//! * Solver-facing vector algebra ([`DotProduct`], [`L2Norm`],
//!   [`ScaledAddAssign`], and [`ScaleAssign`]).
//! * [`WeightedGramInto`] computes dense coefficient-space products into
//!   [`MatrixWrite`] storage. [`WeightedGramKernel`] lets backends apply
//!   normalization during accumulation to preserve small centered variations.
//! * [`WeightedColumnSumsInto`] and [`WeightedColumnSumsKernel`] compute stable
//!   weighted sums for intercept Gram cross terms.
//! * [`VectorOwned`] constructs backend-compatible coefficient scratch for lazy
//!   operator components.
//! * The five normalization-specific *vector* traits ([`ElemDivAssign`], [`DotSlice`],
//!   [`SubScalarAssign`], [`SumEntries`], [`ScaledSubSlice`]) — the elementwise
//!   primitives that fold the lazy normalization into a backend vector. They are
//!   phrased as a backend vector against a coefficient slice `&[F]`, which is
//!   exactly the shape the centering/scaling math needs.
//! * [`ColumnStats`] — column statistics computed directly over a (possibly
//!   sparse) backend matrix, used by [`LazyMatrix::new`](crate::LazyMatrix::new).
//! * [`VectorView`] / [`VectorViewMut`] — storage-independent borrowed vector
//!   access, including strided backend views.
//! * [`RawColumn`] / [`RawColumns`] and [`LogicalColumn`] / [`Columns`] — the
//!   backend and normalized sides of storage-independent column access.
//! * [`SparseColumns`] — the stronger borrowed access capability for
//!   contiguous sparse columns.
//! * [`SparseRows`] — borrowed access to contiguous sparse rows.

/// Numeric scalar element type.
///
/// This is a blanket-implemented alias for the bound bundle the crate relies on,
/// so any floating-point type that satisfies the underlying `num-traits` bounds
/// (notably `f32` and `f64`) is a `Scalar` automatically.
pub trait Scalar:
    num_traits::Float + num_traits::FromPrimitive + std::iter::Sum + std::fmt::Debug + Default + 'static
{
}

mod columns;
mod gram;
mod operator;
mod rows;
mod stats;
mod vectors;
mod weighted_sums;

pub(crate) use operator::scale_output;

pub use crate::normalization::{Centering, Normalization, NormalizationStats, Scaling};
pub use columns::{Columns, LogicalColumn, RawColumn, RawColumns, SparseColumns};
pub use gram::{MatrixWrite, WeightedGramInto, WeightedGramKernel};
pub use operator::{
    MatTransposeVec, MatTransposeVecInto, MatTransposeVecScaledInto, MatVec, MatVecInto,
    MatVecScaledInto, MatrixErrorType, MatrixShape,
};
pub use rows::SparseRows;
pub use stats::ColumnStats;
pub use vectors::{
    DotProduct, DotSlice, ElemDivAssign, L2Norm, ScaleAssign, ScaledAddAssign, ScaledSubSlice,
    SubScalarAssign, SumEntries, VectorOwned, VectorView, VectorViewMut,
};

pub use weighted_sums::{WeightedColumnSumsInto, WeightedColumnSumsKernel};
