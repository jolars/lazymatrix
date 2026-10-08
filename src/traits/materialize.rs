use super::{MatrixErrorType, MatrixShape, MatrixWrite, Scalar};

/// Allocate owned dense storage in an explicitly selected backend.
pub trait MatrixOwned<F: Scalar>: MatrixWrite<F> + Sized {
    /// Allocate a zero-filled matrix with shape `(nrows, ncols)`.
    ///
    /// Size overflow panics. Allocation failure follows the backend's behavior.
    fn zeros(nrows: usize, ncols: usize) -> Self;
}

/// Explicitly materialize logical normalized entries into dense storage.
///
/// Apply active subtraction followed by active division. Include implicit
/// zeros, and sum duplicate raw entries before normalization. Successful calls
/// overwrite the whole output without depending on its previous values.
/// Sparse implementations use one working row or column, not a full matrix.
pub trait MaterializeDense<F: Scalar>: MatrixShape + MatrixErrorType {
    /// Write the optionally normalized matrix to `out`.
    ///
    /// # Errors
    /// Returns the source's read or decoding error. Output may be partial.
    ///
    /// # Panics
    /// Validate output shape, parameter lengths, and nonzero scales before
    /// writing or reading storage. Invalid dimensions or parameters panic.
    fn materialize_normalized_into<O: MatrixWrite<F> + ?Sized>(
        &self,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut O,
    ) -> Result<(), Self::Error>;
}

/// Infallible, in-place normalization of writable dense storage.
///
/// This capability performs no I/O and does not change shape or storage layout.
/// Apply subtraction and division only for the active components, preserving
/// negative and nonfinite parameters. Do not recompute statistics.
pub trait DenseNormalize<F: Scalar>: MatrixWrite<F> {
    /// Normalize each entry in the existing storage.
    ///
    /// # Panics
    /// Validate parameter lengths and reject exact zero scales before writing.
    fn normalize_in_place(&mut self, centers: Option<&[F]>, scales: Option<&[F]>);
}

impl<M: MaterializeDense<F> + ?Sized, F: Scalar> MaterializeDense<F> for &M {
    fn materialize_normalized_into<O: MatrixWrite<F> + ?Sized>(
        &self,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut O,
    ) -> Result<(), Self::Error> {
        (**self).materialize_normalized_into(centers, scales, out)
    }
}

impl<M: MaterializeDense<F> + ?Sized, F: Scalar> MaterializeDense<F> for &mut M {
    fn materialize_normalized_into<O: MatrixWrite<F> + ?Sized>(
        &self,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut O,
    ) -> Result<(), Self::Error> {
        (**self).materialize_normalized_into(centers, scales, out)
    }
}

impl<M: MatrixWrite<F> + ?Sized, F: Scalar> MatrixWrite<F> for &mut M {
    fn set(&mut self, row: usize, column: usize, value: F) {
        (**self).set(row, column, value);
    }
}

impl<M: DenseNormalize<F> + ?Sized, F: Scalar> DenseNormalize<F> for &mut M {
    fn normalize_in_place(&mut self, centers: Option<&[F]>, scales: Option<&[F]>) {
        (**self).normalize_in_place(centers, scales);
    }
}
