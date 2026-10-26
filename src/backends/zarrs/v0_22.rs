use crate::zarrs_0_22 as zarrs;
use zarrs::array::codec::CodecOptions;
use zarrs::array::{Array, ArrayError, ElementOwned};
use zarrs::array_subset::ArraySubset;
use zarrs::storage::ReadableStorageTraits;

#[path = "array.rs"]
mod array;

fn read_chunk<S: ReadableStorageTraits + ?Sized + 'static, F: ElementOwned>(
    array: &Array<S>,
    indices: &[u64],
    options: &CodecOptions,
) -> Result<Vec<F>, ArrayError> {
    array.retrieve_chunk_elements_opt::<F>(indices, options)
}

fn validate_scalar<S: ReadableStorageTraits + ?Sized + 'static, F: ElementOwned>(
    array: &Array<S>,
) -> Result<(), ArrayError> {
    F::validate_data_type(array.data_type())
}
