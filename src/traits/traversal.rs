//! Explicit raw traversal for streaming and sparse storage.

use std::ops::Range;

use super::{MatrixErrorType, MatrixRead, MatrixShape, Scalar};

/// The cells and ordering exposed by [`MatrixTraversal`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraversalLayout {
    /// Every cell, including zeros and storage fill values, in any order.
    Full,
    /// Stored cells in increasing row order, then increasing column order.
    SparseRows,
    /// Stored cells in increasing column order, then increasing row order.
    SparseColumns,
}

/// Explicit traversal of raw matrix entries within a rectangle.
///
/// Unlike borrowing capabilities, traversal may gather, decode, or read data.
/// Full traversal visits every selected cell exactly once. Sparse traversal
/// omits implicit zeros, sums duplicate entries at each cell in their storage
/// order, and visits unique cells in the order declared by [`TraversalLayout`].
/// Explicitly stored zeros, including sums that cancel to zero, remain visits.
/// Visitor indices always refer to the original matrix, rather than the
/// selected rectangle. The backing matrix must remain unchanged during a scan
/// and between repeated scans used by fitting.
///
/// Implementations restrict outer storage scans and chunk reads to the
/// selection. Sparse coalescing may use workspace proportional to the selected
/// inner dimension; full storage must not materialize the selected rectangle.
/// Chunk and codec buffers may include a whole intersecting storage chunk.
pub trait MatrixTraversal<F: Scalar>: MatrixShape + MatrixErrorType {
    /// Describe which entries are visited and their native ordering.
    fn traversal_layout(&self) -> TraversalLayout;

    /// Visit raw entries in a selected rectangle.
    ///
    /// Empty rectangles perform no reads or callbacks.
    ///
    /// # Errors
    /// Returns the backend's error if traversal fails. Earlier callbacks may
    /// already have run; callers must discard incomplete statistics or honor
    /// the partial-output contract of the operation using this traversal.
    ///
    /// # Panics
    /// Panics for reversed or out-of-bounds ranges. Implementations must
    /// validate both ranges before reads or callbacks, even for empty ranges.
    fn for_each_entry(
        &self,
        rows: Range<usize>,
        columns: Range<usize>,
        visitor: impl FnMut(usize, usize, F),
    ) -> Result<(), Self::Error>;

    /// Traverse dense indexed storage without allocating or gathering entries.
    ///
    /// Custom dense implementations can delegate [`Self::for_each_entry`] to
    /// this helper and declare [`TraversalLayout::Full`].
    ///
    /// # Panics
    /// Panics for reversed or out-of-bounds ranges before reading entries.
    fn dense_for_each_entry(
        &self,
        rows: Range<usize>,
        columns: Range<usize>,
        visitor: impl FnMut(usize, usize, F),
    ) -> Result<(), Self::Error>
    where
        Self: MatrixRead<F>,
    {
        dense_for_each_entry(self, rows, columns, visitor)
    }
}

/// Traverse a rectangle of dense indexed storage in column order.
///
/// Entries include zeros and nonfinite values. No workspace is allocated;
/// empty rectangles perform no reads or callbacks.
///
/// # Panics
/// Panics for reversed or out-of-bounds ranges before reading entries.
pub fn dense_for_each_entry<M, F>(
    matrix: &M,
    rows: Range<usize>,
    columns: Range<usize>,
    mut visitor: impl FnMut(usize, usize, F),
) -> Result<(), M::Error>
where
    M: MatrixRead<F> + MatrixErrorType + ?Sized,
    F: Scalar,
{
    validate_traversal(matrix, &rows, &columns);
    if rows.is_empty() || columns.is_empty() {
        return Ok(());
    }
    for j in columns {
        for i in rows.clone() {
            visitor(i, j, matrix.get(i, j));
        }
    }
    Ok(())
}

pub(crate) fn validate_traversal<M: MatrixShape + ?Sized>(
    matrix: &M,
    rows: &Range<usize>,
    columns: &Range<usize>,
) {
    assert!(
        rows.start <= rows.end && rows.end <= matrix.nrows(),
        "traversal row range out of bounds"
    );
    assert!(
        columns.start <= columns.end && columns.end <= matrix.ncols(),
        "traversal column range out of bounds"
    );
}

macro_rules! forward {
    ($reference:ty) => {
        impl<M: MatrixTraversal<F> + ?Sized, F: Scalar> MatrixTraversal<F> for $reference {
            fn traversal_layout(&self) -> TraversalLayout {
                (**self).traversal_layout()
            }
            fn for_each_entry(
                &self,
                rows: Range<usize>,
                columns: Range<usize>,
                visitor: impl FnMut(usize, usize, F),
            ) -> Result<(), Self::Error> {
                (**self).for_each_entry(rows, columns, visitor)
            }
        }
    };
}
forward!(&M);
forward!(&mut M);
