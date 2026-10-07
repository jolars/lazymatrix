use std::ops::Range;

use super::{MatrixErrorType, MatrixShape, Scalar};

/// A packed, row-major block borrowing caller-owned read storage.
///
/// Dimensions describe the logical rectangle, without storage-chunk padding.
/// Entries are raw matrix values, including fill values and nonfinite values.
#[derive(Clone, Copy, Debug)]
pub struct DenseBlock<'a, F> {
    values: &'a [F],
    nrows: usize,
    ncols: usize,
}

impl<'a, F> DenseBlock<'a, F> {
    /// Borrow packed values with the given logical dimensions.
    ///
    /// # Panics
    /// Panics if the dimensions overflow or their product differs from the
    /// length of `values`.
    pub fn new(values: &'a [F], nrows: usize, ncols: usize) -> Self {
        let len = nrows.checked_mul(ncols).expect("block dimensions overflow");
        assert_eq!(values.len(), len, "block dimensions must match values");
        Self {
            values,
            nrows,
            ncols,
        }
    }

    /// Borrow the packed row-major values.
    pub fn values(&self) -> &'a [F] {
        self.values
    }

    /// Read an entry using indices relative to the block.
    ///
    /// # Panics
    /// Panics if either index is outside the logical dimensions.
    pub fn get(&self, row: usize, column: usize) -> F
    where
        F: Copy,
    {
        assert!(
            row < self.nrows && column < self.ncols,
            "block index out of bounds"
        );
        self.values[row * self.ncols + column]
    }
}

impl<F> MatrixShape for DenseBlock<'_, F> {
    fn nrows(&self) -> usize {
        self.nrows
    }
    fn ncols(&self) -> usize {
        self.ncols
    }
}

/// Experimental fallible access to raw rectangular matrix blocks.
///
/// Unlike borrowed column and row capabilities, this capability may perform
/// decoding and I/O. Callers choose rectangles and provide initialized buffer
/// storage. Implementations overwrite exactly the first `rows.len() *
/// columns.len()` elements and return a packed row-major view of that prefix.
/// The unused tail remains unchanged, including after an error. Empty
/// rectangles return empty views without storage reads.
///
/// A returned view borrows only the buffer, preventing its reuse while the view
/// remains live. Backends may require additional chunk and codec workspace;
/// the buffer length is not a bound on total working memory. Reading multiple
/// rectangles in the same storage chunk may decode that chunk repeatedly.
///
/// ```compile_fail
/// use lazymatrix::{ReadBlock, MatrixShape};
/// fn overlapping_reads<M: ReadBlock<f64>>(matrix: &M, buffer: &mut [f64]) {
///     let first = matrix.read_block(0..1, 0..1, buffer).unwrap();
///     let second = matrix.read_block(0..1, 0..1, buffer).unwrap();
///     assert_eq!(first.values(), second.values());
/// }
/// ```
pub trait ReadBlock<F: Scalar>: MatrixShape + MatrixErrorType {
    /// Read an in-bounds rectangle into caller-owned storage.
    ///
    /// # Errors
    /// Returns the backend's error on a read or decoding failure. No view is
    /// returned; the entire requested prefix is unspecified and must be
    /// discarded. A successful retry overwrites that prefix completely.
    ///
    /// # Panics
    /// Panics for reversed or out-of-bounds ranges, dimension overflow, or a
    /// buffer shorter than the requested number of elements. Implementations
    /// must validate these conditions before accessing storage.
    fn read_block<'buf>(
        &self,
        rows: Range<usize>,
        columns: Range<usize>,
        buffer: &'buf mut [F],
    ) -> Result<DenseBlock<'buf, F>, Self::Error>;
}

impl<M: ReadBlock<F> + ?Sized, F: Scalar> ReadBlock<F> for &M {
    fn read_block<'buf>(
        &self,
        rows: Range<usize>,
        columns: Range<usize>,
        buffer: &'buf mut [F],
    ) -> Result<DenseBlock<'buf, F>, Self::Error> {
        (**self).read_block(rows, columns, buffer)
    }
}

#[cfg(feature = "zarrs_all")]
pub(crate) fn validate_rectangle(
    shape: &impl MatrixShape,
    rows: &Range<usize>,
    columns: &Range<usize>,
    capacity: usize,
) -> usize {
    assert!(
        rows.start <= rows.end && rows.end <= shape.nrows(),
        "block row range out of bounds"
    );
    assert!(
        columns.start <= columns.end && columns.end <= shape.ncols(),
        "block column range out of bounds"
    );
    let len = rows
        .len()
        .checked_mul(columns.len())
        .expect("block dimensions overflow");
    assert!(capacity >= len, "block buffer too short");
    len
}
