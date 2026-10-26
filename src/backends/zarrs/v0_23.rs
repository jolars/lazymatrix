use crate::zarrs_0_23 as zarrs;
use zarrs::array::ArraySubset;
use zarrs::array::codec::api::CodecOptions;
use zarrs::array::{Array, ArrayError, ElementOwned};
use zarrs::storage::ReadableStorageTraits;

#[path = "array.rs"]
mod array;

fn read_chunk<S: ReadableStorageTraits + ?Sized + 'static, F: ElementOwned>(
    array: &Array<S>,
    indices: &[u64],
    options: &CodecOptions,
) -> Result<Vec<F>, ArrayError> {
    array.retrieve_chunk_opt::<Vec<F>>(indices, options)
}

fn validate_scalar<S: ReadableStorageTraits + ?Sized + 'static, F: ElementOwned>(
    array: &Array<S>,
) -> Result<(), ArrayError> {
    F::validate_data_type(array.data_type()).map_err(ArrayError::from)
}
