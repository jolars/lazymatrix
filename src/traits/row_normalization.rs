use super::{MatrixErrorType, MatrixRead, MatrixShape, Scalar, VectorView, VectorViewMut};
use crate::{RowNormalization, RowNormalizationStats};

/// Row statistics used to fit independently normalized samples.
///
/// Implementations sum duplicate cells before nonlinear statistics and include
/// implicit zeros. Present vectors have length `nrows`; inactive components are
/// `None`. Raw zero scales remain zero here. Fitting replaces them with one.
pub trait RowStats<F: Scalar>: MatrixShape + MatrixErrorType {
    /// Compute the centers and raw scales requested by `spec`.
    ///
    /// Backends can share scans through this hook. Returns the first storage
    /// error encountered while computing the statistics.
    fn row_normalization_stats(
        &self,
        spec: RowNormalization,
    ) -> Result<RowNormalizationStats<F>, Self::Error>;

    /// Compute raw row statistics from dense indexed storage.
    ///
    /// Dense adapters can delegate their fitting hook to this implementation.
    /// It uses O(nrows + ncols) storage, including one working row for quantiles.
    fn dense_row_normalization_stats(&self, spec: RowNormalization) -> RowNormalizationStats<F>
    where
        Self: MatrixRead<F> + Sized,
    {
        crate::normalization::row::fit(self, spec)
    }
}

/// Vector products over raw storage with optional fitted row parameters.
///
/// Kernels normalize logical entries before multiplication, combine duplicates,
/// and include the normalized background of implicit zeros. They preserve
/// storage orientation and use bounded workspace. Dimensions, parameter lengths,
/// and explicit zero scales must be checked before reads or output writes.
/// Output may be partial after a storage error.
pub trait RowNormalizationKernel<F: Scalar>: MatrixShape + MatrixErrorType {
    /// Overwrite `out` with the row-normalized forward product.
    fn row_normalized_matvec_into<X: VectorView<F> + ?Sized, Y: VectorViewMut<F> + ?Sized>(
        &self,
        x: &X,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut Y,
    ) -> Result<(), Self::Error>;

    /// Overwrite `out` with the row-normalized transpose product.
    fn row_normalized_transpose_into<X: VectorView<F> + ?Sized, Y: VectorViewMut<F> + ?Sized>(
        &self,
        x: &X,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut Y,
    ) -> Result<(), Self::Error>;

    /// Compute either product directly from dense indexed storage.
    ///
    /// Dense adapters can delegate both kernels here, selecting the transpose
    /// with `transpose`. Entries are centered and scaled before accumulation.
    /// The kernel needs no workspace beyond `out`.
    fn dense_row_normalized_product_into<X: VectorView<F> + ?Sized, Y: VectorViewMut<F> + ?Sized>(
        &self,
        transpose: bool,
        x: &X,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut Y,
    ) -> Result<(), Self::Error>
    where
        Self: MatrixRead<F>,
    {
        crate::row_normalization::dense_product(self, transpose, x, centers, scales, out);
        Ok(())
    }
}

macro_rules! forward {
    ($reference:ty) => {
        impl<M: RowStats<F> + ?Sized, F: Scalar> RowStats<F> for $reference {
            fn row_normalization_stats(
                &self,
                spec: RowNormalization,
            ) -> Result<RowNormalizationStats<F>, Self::Error> {
                (**self).row_normalization_stats(spec)
            }
        }

        impl<M: RowNormalizationKernel<F> + ?Sized, F: Scalar> RowNormalizationKernel<F>
            for $reference
        {
            fn row_normalized_matvec_into<
                X: VectorView<F> + ?Sized,
                Y: VectorViewMut<F> + ?Sized,
            >(
                &self,
                x: &X,
                centers: Option<&[F]>,
                scales: Option<&[F]>,
                out: &mut Y,
            ) -> Result<(), Self::Error> {
                (**self).row_normalized_matvec_into(x, centers, scales, out)
            }

            fn row_normalized_transpose_into<
                X: VectorView<F> + ?Sized,
                Y: VectorViewMut<F> + ?Sized,
            >(
                &self,
                x: &X,
                centers: Option<&[F]>,
                scales: Option<&[F]>,
                out: &mut Y,
            ) -> Result<(), Self::Error> {
                (**self).row_normalized_transpose_into(x, centers, scales, out)
            }
        }
    };
}
forward!(&M);
forward!(&mut M);
