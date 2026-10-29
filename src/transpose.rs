//! Borrowed transposes of linear operators.

use crate::{
    MatMatInto, MatTransposeMatInto, MatTransposeVec, MatTransposeVecInto,
    MatTransposeVecScaledInto, MatVec, MatVecInto, MatVecScaledInto, MatrixErrorType, MatrixRead,
    MatrixShape, MatrixWrite, Scalar,
};

/// A borrowed view of an operator's transpose.
///
/// Construction takes O(1) time, allocates nothing, and does not read the
/// operator. Products swap forward and transpose calls on the original
/// operator, retaining its fitted normalization, numerical behavior, error
/// type, and workspace requirements. No transposed or normalized matrix is
/// materialized. Allocating products still allocate their output vectors or
/// matrices, and reusable-output products may use the original operator's
/// internal scratch. Output may be partial after an error.
///
/// Each product capability is available independently when the original
/// operator supports the opposite product. This includes vector, fused vector,
/// and batched products. Transposing the view again represents the original
/// operator.
///
/// The view exposes operator products, not storage borrowing. It does not
/// implement [`crate::SparseColumns`] or [`crate::SparseRows`], even when the
/// original operator does. A storage-specific transpose may separately offer
/// such access when its borrowed slices satisfy the corresponding contracts.
///
/// ```
/// use lazymatrix::{MatVec, MatTransposeVec, Transpose};
///
/// fn apply_transpose<M, V>(operator: &M, input: &V) -> Result<V, M::Error>
/// where
///     M: MatTransposeVec<V>,
/// {
///     Transpose::new(operator).matvec(input)
/// }
/// ```
///
/// ```compile_fail
/// use lazymatrix::{SparseColumns, SparseRows, Transpose};
/// fn needs_rows<M: SparseRows<f64>>(_: &M) {}
/// fn transpose_columns<M: SparseColumns<f64>>(matrix: &M) {
///     needs_rows(&Transpose::new(matrix));
/// }
/// ```
///
/// ```compile_fail
/// use lazymatrix::{SparseColumns, SparseRows, Transpose};
/// fn needs_columns<M: SparseColumns<f64>>(_: &M) {}
/// fn transpose_rows<M: SparseRows<f64>>(matrix: &M) {
///     needs_columns(&Transpose::new(matrix));
/// }
/// ```
#[derive(Debug)]
pub struct Transpose<'a, M: ?Sized> {
    inner: &'a M,
}

impl<M: ?Sized> Copy for Transpose<'_, M> {}

impl<M: ?Sized> Clone for Transpose<'_, M> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, M: ?Sized> Transpose<'a, M> {
    /// Borrow an operator as its transpose without reading or copying it.
    pub const fn new(operator: &'a M) -> Self {
        Self { inner: operator }
    }

    /// Borrow the original operator, including any fitted normalization.
    pub const fn as_inner(&self) -> &'a M {
        self.inner
    }
}

impl<M: MatrixShape + ?Sized> MatrixShape for Transpose<'_, M> {
    fn nrows(&self) -> usize {
        self.inner.ncols()
    }

    fn ncols(&self) -> usize {
        self.inner.nrows()
    }
}

impl<M: MatrixErrorType + ?Sized> MatrixErrorType for Transpose<'_, M> {
    type Error = M::Error;
}

impl<M: MatTransposeVec<V> + ?Sized, V> MatVec<V> for Transpose<'_, M> {
    fn matvec(&self, x: &V) -> Result<V, Self::Error> {
        self.inner.mat_transpose_vec(x)
    }
}

impl<M: MatVec<V> + ?Sized, V> MatTransposeVec<V> for Transpose<'_, M> {
    fn mat_transpose_vec(&self, x: &V) -> Result<V, Self::Error> {
        self.inner.matvec(x)
    }
}

impl<M: MatTransposeVecInto<X, Y> + ?Sized, X, Y> MatVecInto<X, Y> for Transpose<'_, M> {
    fn matvec_into(&self, x: &X, out: &mut Y) -> Result<(), Self::Error> {
        self.inner.mat_transpose_vec_into(x, out)
    }
}

impl<M: MatVecInto<X, Y> + ?Sized, X, Y> MatTransposeVecInto<X, Y> for Transpose<'_, M> {
    fn mat_transpose_vec_into(&self, x: &X, out: &mut Y) -> Result<(), Self::Error> {
        self.inner.matvec_into(x, out)
    }
}

impl<M: MatTransposeVecScaledInto<X, Y, F> + ?Sized, X, Y, F> MatVecScaledInto<X, Y, F>
    for Transpose<'_, M>
{
    fn matvec_scaled_into(&self, alpha: F, x: &X, beta: F, out: &mut Y) -> Result<(), Self::Error> {
        self.inner
            .mat_transpose_vec_scaled_into(alpha, x, beta, out)
    }
}

impl<M: MatVecScaledInto<X, Y, F> + ?Sized, X, Y, F> MatTransposeVecScaledInto<X, Y, F>
    for Transpose<'_, M>
{
    fn mat_transpose_vec_scaled_into(
        &self,
        alpha: F,
        x: &X,
        beta: F,
        out: &mut Y,
    ) -> Result<(), Self::Error> {
        self.inner.matvec_scaled_into(alpha, x, beta, out)
    }
}

impl<M: MatTransposeMatInto<F> + ?Sized, F: Scalar> MatMatInto<F> for Transpose<'_, M> {
    fn matmat_into<R, O>(&self, rhs: &R, out: &mut O) -> Result<(), Self::Error>
    where
        R: MatrixRead<F> + ?Sized,
        O: MatrixRead<F> + MatrixWrite<F> + ?Sized,
    {
        self.inner.mat_transpose_mat_into(rhs, out)
    }
}

impl<M: MatMatInto<F> + ?Sized, F: Scalar> MatTransposeMatInto<F> for Transpose<'_, M> {
    fn mat_transpose_mat_into<R, O>(&self, rhs: &R, out: &mut O) -> Result<(), Self::Error>
    where
        R: MatrixRead<F> + ?Sized,
        O: MatrixRead<F> + MatrixWrite<F> + ?Sized,
    {
        self.inner.matmat_into(rhs, out)
    }
}
