use super::{MatrixShape, Scalar};

/// Borrowed access to rows stored contiguously in sparse form.
///
/// Implement this capability only when a backend can return the stored column
/// indices and values for one row as slices in O(1) time, without gathering,
/// copying, or allocating. Structurally absent entries are implicit zeros.
/// This capability describes raw storage, independently of normalization.
pub trait SparseRows<F: Scalar>: MatrixShape {
    /// Return the stored `(column index, raw value)` slices for row `i`.
    ///
    /// The two slices have equal length and corresponding entries in the
    /// backend's storage order. Explicitly stored zeros remain present, and
    /// each column index is less than `self.ncols()`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.nrows()`.
    fn sparse_row(&self, i: usize) -> (&[usize], &[F]);
}

impl<M, F> SparseRows<F> for &M
where
    M: SparseRows<F> + ?Sized,
    F: Scalar,
{
    fn sparse_row(&self, i: usize) -> (&[usize], &[F]) {
        (**self).sparse_row(i)
    }
}

impl<M, F> SparseRows<F> for &mut M
where
    M: SparseRows<F> + ?Sized,
    F: Scalar,
{
    fn sparse_row(&self, i: usize) -> (&[usize], &[F]) {
        (**self).sparse_row(i)
    }
}
