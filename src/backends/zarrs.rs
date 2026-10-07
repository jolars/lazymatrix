//! Synchronous matrix operations on chunked Zarr arrays.

mod stats;

use std::fmt;
use std::marker::PhantomData;

use zarrs::array::ElementOwned;
use zarrs::array::codec::CodecOptions;
use zarrs::array::{Array, ArrayError};
use zarrs::array_subset::ArraySubset;
use zarrs::storage::ReadableStorageTraits;

use crate::{
    MatTransposeVec, MatTransposeVecInto, MatTransposeVecScaledInto, MatVec, MatVecInto,
    MatVecScaledInto, MatrixErrorType, MatrixShape, Scalar, VectorView, VectorViewMut,
};

/// Failure to interpret or read a Zarr matrix.
#[derive(Debug)]
#[non_exhaustive]
pub enum ZarrMatrixError {
    /// The array has a rank other than two.
    InvalidRank(usize),
    /// A dimension or decoded chunk cannot be represented in addressable memory.
    SizeOverflow,
    /// The element type is incompatible, or Zarr data could not be read or decoded.
    Array(Box<ArrayError>),
}

impl fmt::Display for ZarrMatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRank(rank) => {
                write!(f, "expected a two-dimensional Zarr array, got rank {rank}")
            }
            Self::SizeOverflow => {
                f.write_str("Zarr dimension or decoded chunk exceeds addressable memory")
            }
            Self::Array(error) => write!(f, "Zarr matrix: {error}"),
        }
    }
}

impl std::error::Error for ZarrMatrixError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Array(error) => Some(error.as_ref()),
            _ => None,
        }
    }
}

impl From<ArrayError> for ZarrMatrixError {
    fn from(error: ArrayError) -> Self {
        Self::Array(Box::new(error))
    }
}

/// A read-only, two-dimensional Zarr array presented as a matrix.
///
/// Supports `f32` and `f64` elements from synchronous readable stores. Products
/// and statistics read one chunk at a time in grid order, including fill values
/// for missing chunks. Allocating products use `Vec<F>`; reusable-output products
/// accept [`VectorView`] and [`VectorViewMut`]. No ndarray feature is required.
///
/// Working vectors and column statistics occupy O(nrows + ncols) memory. Chunk
/// buffers and codec workspaces depend on the largest decoded storage chunk
/// (the outer shard for a sharded array). Choose chunks that fit in RAM: this
/// adapter does not impose a byte budget, cache chunks, or prefetch data.
/// The `parallel` feature does not parallelize its scans.
///
/// The backing array must remain unchanged between normalization and products,
/// and throughout each operation. The adapter provides no snapshot isolation.
/// It deliberately provides no borrowed column or row capabilities, since data
/// must be loaded and decoded before it can be borrowed.
/// [`crate::ReadBlock`] instead loads caller-chosen rectangles into an explicit
/// reusable buffer. Reads decode intersecting chunks serially; chunk and codec
/// workspace is additional to the caller's buffer, including outer shards.
#[derive(Debug)]
pub struct ZarrMatrix<S: ?Sized, F = f64> {
    array: Array<S>,
    shape: [usize; 2],
    scalar: PhantomData<F>,
}

impl<S: ReadableStorageTraits + ?Sized + 'static, F: Scalar + ElementOwned> ZarrMatrix<S, F> {
    /// Wrap an opened array after checking its rank, shape, and scalar type.
    ///
    /// This reads no chunks. Missing chunks retain the array's configured fill
    /// value, which need not be zero.
    ///
    /// # Errors
    /// Returns an error for a rank other than two, dimensions that do not fit
    /// `usize`, or an element type different from `F`. Data is not converted.
    pub fn try_new(array: Array<S>) -> Result<Self, ZarrMatrixError> {
        if array.shape().len() != 2 {
            return Err(ZarrMatrixError::InvalidRank(array.shape().len()));
        }
        F::validate_data_type(array.data_type())?;
        let shape = [
            usize::try_from(array.shape()[0]).map_err(|_| ZarrMatrixError::SizeOverflow)?,
            usize::try_from(array.shape()[1]).map_err(|_| ZarrMatrixError::SizeOverflow)?,
        ];
        Ok(Self {
            array,
            shape,
            scalar: PhantomData,
        })
    }

    fn for_each_value(
        &self,
        mut visit: impl FnMut(usize, usize, F),
    ) -> Result<(), ZarrMatrixError> {
        if self.nrows() == 0 || self.ncols() == 0 {
            return Ok(());
        }
        let mut options = CodecOptions::default();
        options.set_concurrent_target(1);
        let grid = self.array.chunk_grid_shape();
        for i in 0..grid[0] {
            for j in 0..grid[1] {
                let indices = [i, j];
                let chunk_shape = self.array.chunk_shape(&indices)?;
                let rows = usize::try_from(chunk_shape[0].get())
                    .map_err(|_| ZarrMatrixError::SizeOverflow)?;
                let cols = usize::try_from(chunk_shape[1].get())
                    .map_err(|_| ZarrMatrixError::SizeOverflow)?;
                rows.checked_mul(cols)
                    .and_then(|n| n.checked_mul(std::mem::size_of::<F>()))
                    .filter(|&bytes| bytes <= isize::MAX as usize)
                    .ok_or(ZarrMatrixError::SizeOverflow)?;
                let origin = self.array.chunk_origin(&indices)?;
                let row_start =
                    usize::try_from(origin[0]).map_err(|_| ZarrMatrixError::SizeOverflow)?;
                let col_start =
                    usize::try_from(origin[1]).map_err(|_| ZarrMatrixError::SizeOverflow)?;
                let valid_rows = rows.min(self.nrows().saturating_sub(row_start));
                let valid_cols = cols.min(self.ncols().saturating_sub(col_start));
                let values = self
                    .array
                    .retrieve_chunk_elements_opt::<F>(&indices, &options)?;
                // Edge chunks contain padding outside the logical matrix.
                for row in 0..valid_rows {
                    for col in 0..valid_cols {
                        visit(row_start + row, col_start + col, values[row * cols + col]);
                    }
                }
            }
        }
        Ok(())
    }
}

impl<S: ?Sized, F> ZarrMatrix<S, F> {
    /// Borrow the original array without changing its metadata.
    pub fn as_inner(&self) -> &Array<S> {
        &self.array
    }

    /// Recover the original array without copying its data.
    pub fn into_inner(self) -> Array<S> {
        self.array
    }
}

impl<S: ?Sized, F> MatrixShape for ZarrMatrix<S, F> {
    fn nrows(&self) -> usize {
        self.shape[0]
    }
    fn ncols(&self) -> usize {
        self.shape[1]
    }
}

impl<S: ?Sized, F> MatrixErrorType for ZarrMatrix<S, F> {
    type Error = ZarrMatrixError;
}

impl<S, F> crate::ReadBlock<F> for ZarrMatrix<S, F>
where
    S: ReadableStorageTraits + ?Sized + 'static,
    F: Scalar + ElementOwned,
{
    fn read_block<'buf>(
        &self,
        rows: std::ops::Range<usize>,
        columns: std::ops::Range<usize>,
        buffer: &'buf mut [F],
    ) -> Result<crate::DenseBlock<'buf, F>, Self::Error> {
        let len = crate::traits::validate_rectangle(self, &rows, &columns, buffer.len());
        let output = &mut buffer[..len];
        if len == 0 {
            return Ok(crate::DenseBlock::new(output, rows.len(), columns.len()));
        }
        let subset = ArraySubset::new_with_start_shape(
            vec![rows.start as u64, columns.start as u64],
            vec![rows.len() as u64, columns.len() as u64],
        )
        .map_err(ArrayError::from)?;
        let chunks = self
            .array
            .chunks_in_array_subset(&subset)
            .map_err(ArrayError::from)?
            .ok_or_else(|| ArrayError::InvalidArraySubset(subset, self.array.shape().to_vec()))?;
        let mut options = CodecOptions::default();
        options.set_concurrent_target(1);
        for indices in chunks.indices() {
            let chunk_shape = self.array.chunk_shape(&indices)?;
            let chunk_rows =
                usize::try_from(chunk_shape[0].get()).map_err(|_| ZarrMatrixError::SizeOverflow)?;
            let chunk_cols =
                usize::try_from(chunk_shape[1].get()).map_err(|_| ZarrMatrixError::SizeOverflow)?;
            chunk_rows
                .checked_mul(chunk_cols)
                .and_then(|n| n.checked_mul(std::mem::size_of::<F>()))
                .filter(|&bytes| bytes <= isize::MAX as usize)
                .ok_or(ZarrMatrixError::SizeOverflow)?;
            let origin = self.array.chunk_origin(&indices)?;
            let row_start =
                usize::try_from(origin[0]).map_err(|_| ZarrMatrixError::SizeOverflow)?;
            let col_start =
                usize::try_from(origin[1]).map_err(|_| ZarrMatrixError::SizeOverflow)?;
            let row_end = row_start.saturating_add(chunk_rows).min(rows.end);
            let col_end = col_start.saturating_add(chunk_cols).min(columns.end);
            let values = self
                .array
                .retrieve_chunk_elements_opt::<F>(&indices, &options)?;
            for row in row_start.max(rows.start)..row_end {
                let count = col_end - col_start.max(columns.start);
                let source =
                    (row - row_start) * chunk_cols + columns.start.saturating_sub(col_start);
                let destination =
                    (row - rows.start) * columns.len() + col_start.saturating_sub(columns.start);
                output[destination..destination + count]
                    .copy_from_slice(&values[source..source + count]);
            }
        }
        Ok(crate::DenseBlock::new(output, rows.len(), columns.len()))
    }
}

impl<S, F, X, Y> MatVecInto<X, Y> for ZarrMatrix<S, F>
where
    S: ReadableStorageTraits + ?Sized + 'static,
    F: Scalar + ElementOwned,
    X: VectorView<F>,
    Y: VectorViewMut<F>,
{
    fn matvec_into(&self, x: &X, out: &mut Y) -> Result<(), Self::Error> {
        self.matvec_scaled_into(F::one(), x, F::zero(), out)
    }
}

impl<S, F, X, Y> MatVecScaledInto<X, Y, F> for ZarrMatrix<S, F>
where
    S: ReadableStorageTraits + ?Sized + 'static,
    F: Scalar + ElementOwned,
    X: VectorView<F>,
    Y: VectorViewMut<F>,
{
    fn matvec_scaled_into(&self, alpha: F, x: &X, beta: F, out: &mut Y) -> Result<(), Self::Error> {
        assert_eq!(
            x.len(),
            self.ncols(),
            "matvec_scaled_into: dimension mismatch"
        );
        assert_eq!(
            out.len(),
            self.nrows(),
            "matvec_scaled_into: output dimension mismatch"
        );
        crate::traits::scale_output(beta, out);
        if alpha == F::zero() {
            return Ok(());
        }
        self.for_each_value(|row, col, value| {
            out.set(row, out.get(row) + alpha * (value * x.get(col)));
        })
    }
}

impl<S, F, X, Y> MatTransposeVecInto<X, Y> for ZarrMatrix<S, F>
where
    S: ReadableStorageTraits + ?Sized + 'static,
    F: Scalar + ElementOwned,
    X: VectorView<F>,
    Y: VectorViewMut<F>,
{
    fn mat_transpose_vec_into(&self, x: &X, out: &mut Y) -> Result<(), Self::Error> {
        self.mat_transpose_vec_scaled_into(F::one(), x, F::zero(), out)
    }
}

impl<S, F, X, Y> MatTransposeVecScaledInto<X, Y, F> for ZarrMatrix<S, F>
where
    S: ReadableStorageTraits + ?Sized + 'static,
    F: Scalar + ElementOwned,
    X: VectorView<F>,
    Y: VectorViewMut<F>,
{
    fn mat_transpose_vec_scaled_into(
        &self,
        alpha: F,
        x: &X,
        beta: F,
        out: &mut Y,
    ) -> Result<(), Self::Error> {
        assert_eq!(
            x.len(),
            self.nrows(),
            "mat_transpose_vec_scaled_into: dimension mismatch"
        );
        assert_eq!(
            out.len(),
            self.ncols(),
            "mat_transpose_vec_scaled_into: output dimension mismatch"
        );
        crate::traits::scale_output(beta, out);
        if alpha == F::zero() {
            return Ok(());
        }
        self.for_each_value(|row, col, value| {
            out.set(col, out.get(col) + alpha * (value * x.get(row)));
        })
    }
}

impl<S: ReadableStorageTraits + ?Sized + 'static, F: Scalar + ElementOwned> MatVec<Vec<F>>
    for ZarrMatrix<S, F>
{
    fn matvec(&self, x: &Vec<F>) -> Result<Vec<F>, Self::Error> {
        assert_eq!(x.len(), self.ncols(), "matvec: dimension mismatch");
        let mut out = vec![F::zero(); self.nrows()];
        self.matvec_into(x, &mut out)?;
        Ok(out)
    }
}

impl<S: ReadableStorageTraits + ?Sized + 'static, F: Scalar + ElementOwned> MatTransposeVec<Vec<F>>
    for ZarrMatrix<S, F>
{
    fn mat_transpose_vec(&self, x: &Vec<F>) -> Result<Vec<F>, Self::Error> {
        assert_eq!(
            x.len(),
            self.nrows(),
            "mat_transpose_vec: dimension mismatch"
        );
        let mut out = vec![F::zero(); self.ncols()];
        self.mat_transpose_vec_into(x, &mut out)?;
        Ok(out)
    }
}
