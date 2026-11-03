use std::ops::Range;

use super::{MatrixErrorType, MatrixShape, Scalar, VectorView, VectorViewMut};

/// Vector products restricted to a rectangle of a logical operator.
///
/// Ranges refer to the parent operator, while input and output indices are
/// local to the rectangle. Implementations preserve the parent's fitted
/// transformation, including stage order, without fitting new statistics.
/// They visit only selected dense entries, selected sparse outer storage, or
/// intersecting storage chunks, using bounded workspace.
///
/// Validate ranges and dimensions before reading data or modifying outputs.
/// Zero `alpha` skips all source reads, and zero `beta` ignores previous output
/// values. Empty inner dimensions represent empty sums. On a backend error,
/// output may be partial; normalization corrections follow successful reads.
pub trait RectangularProducts: MatrixShape + MatrixErrorType {
    /// Scalar of the logical entries and product vectors.
    type Scalar: Scalar;

    /// Compute `out = alpha * self[rows, columns] * x + beta * out`.
    fn rectangle_matvec_into<X, Y>(
        &self,
        rows: Range<usize>,
        columns: Range<usize>,
        alpha: Self::Scalar,
        x: &X,
        beta: Self::Scalar,
        out: &mut Y,
    ) -> Result<(), Self::Error>
    where
        X: VectorView<Self::Scalar> + ?Sized,
        Y: VectorViewMut<Self::Scalar> + ?Sized;

    /// Compute `out = alpha * self[rows, columns]^T * x + beta * out`.
    fn rectangle_transpose_into<X, Y>(
        &self,
        rows: Range<usize>,
        columns: Range<usize>,
        alpha: Self::Scalar,
        x: &X,
        beta: Self::Scalar,
        out: &mut Y,
    ) -> Result<(), Self::Error>
    where
        X: VectorView<Self::Scalar> + ?Sized,
        Y: VectorViewMut<Self::Scalar> + ?Sized;
}
