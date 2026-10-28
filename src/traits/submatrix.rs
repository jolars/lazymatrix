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

    /// Accumulate weighted logical-column sums over a rectangle.
    ///
    /// Weights use local row indices, and output uses local column indices.
    /// The validation, nonfinite, and error contracts are those of
    /// [`super::WeightedColumnSumsKernel`]. The default reads normalized panels
    /// of at most 32 rows and 32 columns through `materialize_submatrix_into`.
    /// It may revisit stored entries or decode a chunk more than once.
    fn submatrix_weighted_column_sums_into<W, O>(
        &self,
        rows: Range<usize>,
        columns: Range<usize>,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        weights: &W,
        out: &mut O,
    ) -> Result<(), Self::Error>
    where
        W: VectorView<F> + ?Sized,
        O: VectorViewMut<F> + ?Sized,
    {
        crate::submatrix_weighted::sums(self, rows, columns, centers, scales, weights, out)
    }

    /// Accumulate a symmetric weighted Gram product over a rectangle.
    ///
    /// Weights use local row indices. Output has shape `(columns.len(),
    /// columns.len())`. The validation, nonfinite, and error contracts are those
    /// of [`super::WeightedGramKernel`]. The default centers entries before
    /// multiplication using two normalized panels and one accumulator, each
    /// at most 32 by 32. It revisits source panels for each column-panel pair.
    /// Backend chunk and codec workspace is additional to these fixed buffers.
    fn submatrix_weighted_gram_into<W, O>(
        &self,
        rows: Range<usize>,
        columns: Range<usize>,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        weights: &W,
        out: &mut O,
    ) -> Result<(), Self::Error>
    where
        W: VectorView<F> + ?Sized,
        O: MatrixWrite<F> + ?Sized,
    {
        crate::submatrix_weighted::gram(self, rows, columns, centers, scales, weights, out)
    }
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
            fn submatrix_weighted_column_sums_into<W, O>(
                &self,
                rows: Range<usize>,
                columns: Range<usize>,
                centers: Option<&[F]>,
                scales: Option<&[F]>,
                weights: &W,
                out: &mut O,
            ) -> Result<(), Self::Error>
            where
                W: VectorView<F> + ?Sized,
                O: VectorViewMut<F> + ?Sized,
            {
                (**self).submatrix_weighted_column_sums_into(
                    rows, columns, centers, scales, weights, out,
                )
            }
            fn submatrix_weighted_gram_into<W, O>(
                &self,
                rows: Range<usize>,
                columns: Range<usize>,
                centers: Option<&[F]>,
                scales: Option<&[F]>,
                weights: &W,
                out: &mut O,
            ) -> Result<(), Self::Error>
            where
                W: VectorView<F> + ?Sized,
                O: MatrixWrite<F> + ?Sized,
            {
                (**self).submatrix_weighted_gram_into(rows, columns, centers, scales, weights, out)
            }
        }
    };
}
forward!(&M);
forward!(&mut M);
