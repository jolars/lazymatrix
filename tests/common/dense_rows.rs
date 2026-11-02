// External dense adapters can reuse the provided row kernels without backend dependencies.

macro_rules! implement {
    ($matrix:ty) => {
        impl<F: lazymatrix::Scalar> lazymatrix::RowStats<F> for $matrix {
            fn row_normalization_stats(
                &self,
                spec: lazymatrix::RowNormalization,
            ) -> Result<lazymatrix::RowNormalizationStats<F>, Self::Error> {
                Ok(self.dense_row_normalization_stats(spec))
            }
        }

        impl<F: lazymatrix::Scalar> lazymatrix::RowNormalizationKernel<F> for $matrix {
            fn row_normalized_matvec_into<
                X: lazymatrix::VectorView<F> + ?Sized,
                Y: lazymatrix::VectorViewMut<F> + ?Sized,
            >(
                &self,
                x: &X,
                centers: Option<&[F]>,
                scales: Option<&[F]>,
                out: &mut Y,
            ) -> Result<(), Self::Error> {
                self.dense_row_normalized_product_into(false, x, centers, scales, out)
            }

            fn row_normalized_transpose_into<
                X: lazymatrix::VectorView<F> + ?Sized,
                Y: lazymatrix::VectorViewMut<F> + ?Sized,
            >(
                &self,
                x: &X,
                centers: Option<&[F]>,
                scales: Option<&[F]>,
                out: &mut Y,
            ) -> Result<(), Self::Error> {
                self.dense_row_normalized_product_into(true, x, centers, scales, out)
            }
        }
    };
}
pub(crate) use implement;
