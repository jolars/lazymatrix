//! Row-normalized vector products over capable storage.

use crate::{
    LazyMatrix, MatTransposeVec, MatTransposeVecInto, MatVec, MatVecInto, RowNormalizationKernel,
    RowNormalizationParams, Scalar, VectorOwned, VectorView, VectorViewMut,
};

impl<M, V, F, S: AsRef<[F]>> MatVec<V> for LazyMatrix<M, RowNormalizationParams<F, S>>
where
    M: RowNormalizationKernel<F>,
    F: Scalar,
    V: VectorOwned<F, Owned = V> + VectorViewMut<F>,
{
    fn matvec(&self, x: &V) -> Result<V, Self::Error> {
        assert_eq!(x.len(), self.ncols(), "matvec: dimension mismatch");
        let mut out = V::owned_from_fn(self.nrows(), |_| F::zero());
        self.matvec_into(x, &mut out)?;
        Ok(out)
    }
}

impl<M, X, Y, F, S: AsRef<[F]>> MatVecInto<X, Y> for LazyMatrix<M, RowNormalizationParams<F, S>>
where
    M: RowNormalizationKernel<F>,
    F: Scalar,
    X: VectorView<F>,
    Y: VectorViewMut<F>,
{
    fn matvec_into(&self, x: &X, out: &mut Y) -> Result<(), Self::Error> {
        assert_eq!(x.len(), self.ncols(), "matvec_into: dimension mismatch");
        assert_eq!(
            out.len(),
            self.nrows(),
            "matvec_into: output dimension mismatch"
        );
        self.data.row_normalized_matvec_into(
            x,
            self.normalization().centers(),
            self.normalization().scales(),
            out,
        )
    }
}

impl<M, V, F, S: AsRef<[F]>> MatTransposeVec<V> for LazyMatrix<M, RowNormalizationParams<F, S>>
where
    M: RowNormalizationKernel<F>,
    F: Scalar,
    V: VectorOwned<F, Owned = V> + VectorViewMut<F>,
{
    fn mat_transpose_vec(&self, x: &V) -> Result<V, Self::Error> {
        assert_eq!(
            x.len(),
            self.nrows(),
            "mat_transpose_vec: dimension mismatch"
        );
        let mut out = V::owned_from_fn(self.ncols(), |_| F::zero());
        self.mat_transpose_vec_into(x, &mut out)?;
        Ok(out)
    }
}

impl<M, X, Y, F, S: AsRef<[F]>> MatTransposeVecInto<X, Y>
    for LazyMatrix<M, RowNormalizationParams<F, S>>
where
    M: RowNormalizationKernel<F>,
    F: Scalar,
    X: VectorView<F>,
    Y: VectorViewMut<F>,
{
    fn mat_transpose_vec_into(&self, x: &X, out: &mut Y) -> Result<(), Self::Error> {
        assert_eq!(
            x.len(),
            self.nrows(),
            "mat_transpose_vec_into: dimension mismatch"
        );
        assert_eq!(
            out.len(),
            self.ncols(),
            "mat_transpose_vec_into: output dimension mismatch"
        );
        self.data.row_normalized_transpose_into(
            x,
            self.normalization().centers(),
            self.normalization().scales(),
            out,
        )
    }
}
