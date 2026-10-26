use std::ops::Range;

use super::{MatrixErrorType, MatrixShape, MatrixWrite, Scalar, VectorView, VectorViewMut};

/// Rectangular kernels over raw storage, with optional fitted normalization.
///
/// Ranges are relative to this source. Parameter slices describe only the
/// selected columns. Implementations must validate ranges, dimensions, and
/// parameters before reading storage or writing output. Empty rectangles need
/// no storage reads. No kernel may materialize a full source matrix.
///
/// Products preserve the fused product contracts: zero `alpha` skips source
/// reads, zero `beta` ignores previous output, and dimensions are always checked.
/// On an operational error, output may be partial. Materialization combines raw
/// duplicates before normalization and includes implicit zeros.
#[allow(clippy::too_many_arguments)]
pub trait SubmatrixKernel<F: Scalar>: MatrixShape + MatrixErrorType {
    /// Compute a normalized rectangular forward product into reusable storage.
    fn submatrix_matvec_into<X: VectorView<F> + ?Sized, Y: VectorViewMut<F> + ?Sized>(
        &self,
        rows: Range<usize>,
        columns: Range<usize>,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        alpha: F,
        x: &X,
        beta: F,
        out: &mut Y,
    ) -> Result<(), Self::Error>;

    /// Compute a normalized rectangular transpose product into reusable storage.
    fn submatrix_transpose_into<X: VectorView<F> + ?Sized, Y: VectorViewMut<F> + ?Sized>(
        &self,
        rows: Range<usize>,
        columns: Range<usize>,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        alpha: F,
        x: &X,
        beta: F,
        out: &mut Y,
    ) -> Result<(), Self::Error>;

    /// Write normalized rectangular entries into caller-selected dense storage.
    fn materialize_submatrix_into<O: MatrixWrite<F> + ?Sized>(
        &self,
        rows: Range<usize>,
        columns: Range<usize>,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut O,
    ) -> Result<(), Self::Error>;
}

macro_rules! forward {
    ($reference:ty) => {
        impl<M: SubmatrixKernel<F> + ?Sized, F: Scalar> SubmatrixKernel<F> for $reference {
            fn submatrix_matvec_into<X: VectorView<F> + ?Sized, Y: VectorViewMut<F> + ?Sized>(
                &self,
                rows: Range<usize>,
                columns: Range<usize>,
                centers: Option<&[F]>,
                scales: Option<&[F]>,
                alpha: F,
                x: &X,
                beta: F,
                out: &mut Y,
            ) -> Result<(), Self::Error> {
                (**self).submatrix_matvec_into(rows, columns, centers, scales, alpha, x, beta, out)
            }
            fn submatrix_transpose_into<X: VectorView<F> + ?Sized, Y: VectorViewMut<F> + ?Sized>(
                &self,
                rows: Range<usize>,
                columns: Range<usize>,
                centers: Option<&[F]>,
                scales: Option<&[F]>,
                alpha: F,
                x: &X,
                beta: F,
                out: &mut Y,
            ) -> Result<(), Self::Error> {
                (**self)
                    .submatrix_transpose_into(rows, columns, centers, scales, alpha, x, beta, out)
            }
            fn materialize_submatrix_into<O: MatrixWrite<F> + ?Sized>(
                &self,
                rows: Range<usize>,
                columns: Range<usize>,
                centers: Option<&[F]>,
                scales: Option<&[F]>,
                out: &mut O,
            ) -> Result<(), Self::Error> {
                (**self).materialize_submatrix_into(rows, columns, centers, scales, out)
            }
        }
    };
}
forward!(&M);
forward!(&mut M);
