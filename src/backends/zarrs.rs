//! Synchronous matrix operations on chunked Zarr arrays.

// Each release implements array access from the same compatibility source.
#![allow(clippy::duplicate_mod)]

mod stats;
mod submatrix;

#[cfg(feature = "zarrs_v0_22")]
mod v0_22;
#[cfg(feature = "zarrs_v0_23")]
mod v0_23;

use std::fmt;
use std::marker::PhantomData;
use std::ops::Range;

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
    Array(Box<dyn std::error::Error + Send + Sync>),
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

mod private {
    use super::ZarrMatrixError;
    use std::ops::Range;

    pub trait ArrayRead<F> {
        type ReadOptions;
        fn serial_options() -> Self::ReadOptions;
        fn shape(&self) -> &[u64];
        fn validate_scalar(&self) -> Result<(), ZarrMatrixError>;
        fn chunk_grid_shape(&self) -> &[u64];
        fn chunk_shape(&self, indices: &[u64]) -> Result<[u64; 2], ZarrMatrixError>;
        fn chunk_origin(&self, indices: &[u64]) -> Result<[u64; 2], ZarrMatrixError>;
        fn chunk_ranges(
            &self,
            rows: &Range<usize>,
            columns: &Range<usize>,
        ) -> Result<[Range<u64>; 2], ZarrMatrixError>;
        fn read_chunk(
            &self,
            indices: &[u64],
            options: &Self::ReadOptions,
        ) -> Result<Vec<F>, ZarrMatrixError>;
    }
}

/// A synchronous readable Zarr array supported by [`ZarrMatrix`].
///
/// Implemented for each enabled zarrs release's `Array<S>` with `f32` and `f64`
/// elements. This trait is sealed; downstream crates cannot implement it.
pub trait ZarrArray<F: Scalar>: private::ArrayRead<F> {}

/// A read-only, two-dimensional Zarr array presented as a matrix.
///
/// The array parameter `A` is the complete zarrs `Array<S>` type, rather than
/// its storage type. Both supported releases can coexist in the same build.
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
/// Median and quantile-range fitting use exact repeated scans with O(ncols)
/// selection state. Their scan counts depend on the values; fitting a median
/// and a quantile range together shares those scans.
///
/// The backing array must remain unchanged between normalization and products,
/// and throughout each operation. The adapter provides no snapshot isolation.
/// It deliberately provides no borrowed column or row capabilities, since data
/// must be loaded and decoded before it can be borrowed.
/// [`crate::ReadBlock`] instead loads caller-chosen rectangles into an explicit
/// reusable buffer. Reads decode intersecting chunks serially; chunk and codec
/// workspace is additional to the caller's buffer, including outer shards.
/// [`crate::LazyMatrix::submatrix`] provides normalized rectangular products,
/// weighted operations, and eager conversion through the same bounded reads.
/// Weighted Gram products use fixed panels and may decode chunks repeatedly.
#[derive(Debug)]
pub struct ZarrMatrix<A, F = f64> {
    array: A,
    shape: [usize; 2],
    scalar: PhantomData<F>,
}

impl<A: ZarrArray<F>, F: Scalar> ZarrMatrix<A, F> {
    /// Wrap an opened array after checking its rank, shape, and scalar type.
    ///
    /// This reads no chunks. Missing chunks retain the array's configured fill
    /// value, which need not be zero.
    ///
    /// # Errors
    /// Returns an error for a rank other than two, dimensions that do not fit
    /// `usize`, or an element type different from `F`. Data is not converted.
    pub fn try_new(array: A) -> Result<Self, ZarrMatrixError> {
        if array.shape().len() != 2 {
            return Err(ZarrMatrixError::InvalidRank(array.shape().len()));
        }
        array.validate_scalar()?;
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
        let options = A::serial_options();
        let grid = self.array.chunk_grid_shape();
        for i in 0..grid[0] {
            for j in 0..grid[1] {
                let indices = [i, j];
                let chunk_shape = self.array.chunk_shape(&indices)?;
                let rows =
                    usize::try_from(chunk_shape[0]).map_err(|_| ZarrMatrixError::SizeOverflow)?;
                let cols =
                    usize::try_from(chunk_shape[1]).map_err(|_| ZarrMatrixError::SizeOverflow)?;
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
                let values = self.array.read_chunk(&indices, &options)?;
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

    fn for_each_row_in(
        &self,
        rows: Range<usize>,
        columns: Range<usize>,
        mut visit: impl FnMut(usize, usize, &[F]),
    ) -> Result<(), ZarrMatrixError> {
        if rows.is_empty() || columns.is_empty() {
            return Ok(());
        }
        let [row_chunks, col_chunks] = self.array.chunk_ranges(&rows, &columns)?;
        let options = A::serial_options();
        for i in row_chunks {
            for j in col_chunks.clone() {
                let indices = [i, j];
                let chunk_shape = self.array.chunk_shape(&indices)?;
                let chunk_rows =
                    usize::try_from(chunk_shape[0]).map_err(|_| ZarrMatrixError::SizeOverflow)?;
                let chunk_cols =
                    usize::try_from(chunk_shape[1]).map_err(|_| ZarrMatrixError::SizeOverflow)?;
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
                let first_col = col_start.max(columns.start);
                let values = self.array.read_chunk(&indices, &options)?;
                for row in row_start.max(rows.start)..row_end {
                    let source = (row - row_start) * chunk_cols + first_col - col_start;
                    visit(
                        row,
                        first_col,
                        &values[source..source + col_end - first_col],
                    );
                }
            }
        }
        Ok(())
    }
}

impl<A, F> ZarrMatrix<A, F> {
    /// Borrow the original array without changing its metadata.
    pub fn as_inner(&self) -> &A {
        &self.array
    }

    /// Recover the original array without copying its data.
    pub fn into_inner(self) -> A {
        self.array
    }
}

impl<A, F> MatrixShape for ZarrMatrix<A, F> {
    fn nrows(&self) -> usize {
        self.shape[0]
    }
    fn ncols(&self) -> usize {
        self.shape[1]
    }
}

impl<A, F> MatrixErrorType for ZarrMatrix<A, F> {
    type Error = ZarrMatrixError;
}

macro_rules! batched_products {
    ($trait:ident, $product:ident, $transpose:expr) => {
        impl<A: ZarrArray<F>, F: Scalar> crate::$trait<F> for ZarrMatrix<A, F> {
            fn $product<R, O>(&self, rhs: &R, out: &mut O) -> Result<(), Self::Error>
            where
                R: crate::MatrixRead<F> + ?Sized,
                O: crate::MatrixRead<F> + crate::MatrixWrite<F> + ?Sized,
            {
                crate::matmat::validate(self, rhs, out, $transpose);
                crate::matmat::zero(out);
                if rhs.ncols() == 0 || out.nrows() == 0 {
                    return Ok(());
                }
                self.for_each_value(|row, col, value| {
                    let (destination, source) = if $transpose { (col, row) } else { (row, col) };
                    for k in 0..rhs.ncols() {
                        out.set(
                            destination,
                            k,
                            out.get(destination, k) + value * rhs.get(source, k),
                        );
                    }
                })
            }
        }
    };
}
batched_products!(MatMatInto, matmat_into, false);
batched_products!(MatTransposeMatInto, mat_transpose_mat_into, true);

impl<A, F> crate::ReadBlock<F> for ZarrMatrix<A, F>
where
    A: ZarrArray<F>,
    F: Scalar,
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
        self.for_each_row_in(rows.clone(), columns.clone(), |row, col, values| {
            let destination = (row - rows.start) * columns.len() + col - columns.start;
            output[destination..destination + values.len()].copy_from_slice(values);
        })?;
        Ok(crate::DenseBlock::new(output, rows.len(), columns.len()))
    }
}

impl<A, F, X, Y> MatVecInto<X, Y> for ZarrMatrix<A, F>
where
    A: ZarrArray<F>,
    F: Scalar,
    X: VectorView<F>,
    Y: VectorViewMut<F>,
{
    fn matvec_into(&self, x: &X, out: &mut Y) -> Result<(), Self::Error> {
        self.matvec_scaled_into(F::one(), x, F::zero(), out)
    }
}

impl<A, F, X, Y> MatVecScaledInto<X, Y, F> for ZarrMatrix<A, F>
where
    A: ZarrArray<F>,
    F: Scalar,
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

impl<A, F, X, Y> MatTransposeVecInto<X, Y> for ZarrMatrix<A, F>
where
    A: ZarrArray<F>,
    F: Scalar,
    X: VectorView<F>,
    Y: VectorViewMut<F>,
{
    fn mat_transpose_vec_into(&self, x: &X, out: &mut Y) -> Result<(), Self::Error> {
        self.mat_transpose_vec_scaled_into(F::one(), x, F::zero(), out)
    }
}

impl<A, F, X, Y> MatTransposeVecScaledInto<X, Y, F> for ZarrMatrix<A, F>
where
    A: ZarrArray<F>,
    F: Scalar,
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

impl<A: ZarrArray<F>, F: Scalar> MatVec<Vec<F>> for ZarrMatrix<A, F> {
    fn matvec(&self, x: &Vec<F>) -> Result<Vec<F>, Self::Error> {
        assert_eq!(x.len(), self.ncols(), "matvec: dimension mismatch");
        let mut out = vec![F::zero(); self.nrows()];
        self.matvec_into(x, &mut out)?;
        Ok(out)
    }
}

impl<A: ZarrArray<F>, F: Scalar> MatTransposeVec<Vec<F>> for ZarrMatrix<A, F> {
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

impl<A, F> crate::MaterializeDense<F> for ZarrMatrix<A, F>
where
    A: ZarrArray<F>,
    F: Scalar,
{
    fn materialize_normalized_into<O: crate::MatrixWrite<F> + ?Sized>(
        &self,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut O,
    ) -> Result<(), Self::Error> {
        crate::materialize::validate_output(self, out, centers, scales);
        self.for_each_value(|row, col, value| {
            out.set(
                row,
                col,
                crate::materialize::normalized(value, col, centers, scales),
            );
        })
    }
}
