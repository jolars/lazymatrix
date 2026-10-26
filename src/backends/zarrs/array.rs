use super::{ArraySubset, CodecOptions, read_chunk, validate_scalar, zarrs};
use crate::Scalar;
use crate::backends::zarrs::{ZarrArray, ZarrMatrixError, private};
use std::ops::Range;
use zarrs::array::{Array, ArrayError, ElementOwned};
use zarrs::storage::ReadableStorageTraits;

impl From<ArrayError> for ZarrMatrixError {
    fn from(error: ArrayError) -> Self {
        Self::Array(Box::new(error))
    }
}

impl<S: ReadableStorageTraits + ?Sized + 'static, F: Scalar + ElementOwned> ZarrArray<F>
    for Array<S>
{
}

impl<S: ReadableStorageTraits + ?Sized + 'static, F: Scalar + ElementOwned> private::ArrayRead<F>
    for Array<S>
{
    type ReadOptions = CodecOptions;

    fn serial_options() -> Self::ReadOptions {
        let mut options = CodecOptions::default();
        options.set_concurrent_target(1);
        options
    }

    fn shape(&self) -> &[u64] {
        self.shape()
    }

    fn validate_scalar(&self) -> Result<(), ZarrMatrixError> {
        validate_scalar::<S, F>(self)?;
        Ok(())
    }

    fn chunk_grid_shape(&self) -> &[u64] {
        self.chunk_grid_shape()
    }

    fn chunk_shape(&self, indices: &[u64]) -> Result<[u64; 2], ZarrMatrixError> {
        let shape = self.chunk_shape(indices)?;
        Ok([shape[0].get(), shape[1].get()])
    }

    fn chunk_origin(&self, indices: &[u64]) -> Result<[u64; 2], ZarrMatrixError> {
        let origin = self.chunk_origin(indices)?;
        Ok([origin[0], origin[1]])
    }

    fn chunk_ranges(
        &self,
        rows: &Range<usize>,
        columns: &Range<usize>,
    ) -> Result<[Range<u64>; 2], ZarrMatrixError> {
        let subset = ArraySubset::new_with_start_shape(
            vec![rows.start as u64, columns.start as u64],
            vec![rows.len() as u64, columns.len() as u64],
        )
        .map_err(ArrayError::from)?;
        let chunks = self
            .chunks_in_array_subset(&subset)
            .map_err(ArrayError::from)?
            .ok_or_else(|| ArrayError::InvalidArraySubset(subset, self.shape().to_vec()))?;
        let start = chunks.start();
        let shape = chunks.shape();
        Ok([start[0]..start[0] + shape[0], start[1]..start[1] + shape[1]])
    }

    fn read_chunk(
        &self,
        indices: &[u64],
        options: &Self::ReadOptions,
    ) -> Result<Vec<F>, ZarrMatrixError> {
        Ok(read_chunk(self, indices, options)?)
    }
}
