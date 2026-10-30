//! Dense row-normalized vector products.

use crate::{
    LazyMatrix, MatTransposeVec, MatTransposeVecInto, MatVec, MatVecInto, MatrixErrorType,
    MatrixRead, RowNormalizationParams, Scalar, VectorOwned, VectorView, VectorViewMut,
};

impl<M: MatrixRead<F> + MatrixErrorType, F: Scalar> LazyMatrix<M, RowNormalizationParams<F>> {
    fn entry(&self, row: usize, column: usize) -> F {
        let mut value = self.data.get(row, column);
        if let Some(centers) = self.normalization().centers() {
            value = value - centers[row];
        }
        if let Some(scales) = self.normalization().scales() {
            value = value / scales[row];
        }
        value
    }
}

impl<M, V, F> MatVec<V> for LazyMatrix<M, RowNormalizationParams<F>>
where
    M: MatrixRead<F> + MatrixErrorType,
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

impl<M, X, Y, F> MatVecInto<X, Y> for LazyMatrix<M, RowNormalizationParams<F>>
where
    M: MatrixRead<F> + MatrixErrorType,
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
        for i in 0..self.nrows() {
            let value = (0..self.ncols()).map(|j| self.entry(i, j) * x.get(j)).sum();
            out.set(i, value);
        }
        Ok(())
    }
}

impl<M, V, F> MatTransposeVec<V> for LazyMatrix<M, RowNormalizationParams<F>>
where
    M: MatrixRead<F> + MatrixErrorType,
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

impl<M, X, Y, F> MatTransposeVecInto<X, Y> for LazyMatrix<M, RowNormalizationParams<F>>
where
    M: MatrixRead<F> + MatrixErrorType,
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
        for j in 0..self.ncols() {
            let value = (0..self.nrows()).map(|i| self.entry(i, j) * x.get(i)).sum();
            out.set(j, value);
        }
        Ok(())
    }
}
