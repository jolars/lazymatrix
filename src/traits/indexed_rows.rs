use super::{MatrixErrorType, MatrixShape, Scalar};

/// Native traversal of a raw matrix restricted to indexed rows.
///
/// This is an operational capability, not borrowed access. Implementations may
/// allocate bounded workspace or return a storage error. In-memory adapters
/// visit dense entries directly, filter CSC columns, or visit selected CSR rows.
pub trait IndexedRowSource<F: Scalar>: MatrixShape + MatrixErrorType {
    /// Whether traversal visits every selected cell, including zeros.
    ///
    /// Dense adapters set this to preserve directly centered products. Sparse
    /// sources should retain the default even when every cell happens to be stored.
    const IS_DENSE: bool = false;

    /// Visit selected logical entries as `(local row, column, value)`.
    ///
    /// `rows` must contain strictly increasing, unique indices below `nrows`.
    /// Row `i` in the result corresponds to source row `rows[i]`. Omitted cells
    /// are zero. Visit each stored cell once, summing duplicate raw entries in
    /// storage order before calling `visit`. Empty selections perform no reads.
    /// Traversal order across cells is unspecified.
    ///
    /// # Errors
    /// Returns the source error; the visitor may already have received entries.
    fn for_each_indexed_entry(
        &self,
        rows: &[usize],
        visit: impl FnMut(usize, usize, F),
    ) -> Result<(), Self::Error>;
}

impl<F: Scalar, M: IndexedRowSource<F> + ?Sized> IndexedRowSource<F> for &M {
    const IS_DENSE: bool = M::IS_DENSE;

    fn for_each_indexed_entry(
        &self,
        rows: &[usize],
        visit: impl FnMut(usize, usize, F),
    ) -> Result<(), Self::Error> {
        (**self).for_each_indexed_entry(rows, visit)
    }
}

impl<F: Scalar, M: IndexedRowSource<F> + ?Sized> IndexedRowSource<F> for &mut M {
    const IS_DENSE: bool = M::IS_DENSE;

    fn for_each_indexed_entry(
        &self,
        rows: &[usize],
        visit: impl FnMut(usize, usize, F),
    ) -> Result<(), Self::Error> {
        (**self).for_each_indexed_entry(rows, visit)
    }
}
