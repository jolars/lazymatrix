#![allow(dead_code)]

pub(crate) use ::zarrs_0_22 as zarrs;
use zarrs::array::{Array, ArrayError, DataType, Element};
use zarrs::storage::{ReadableWritableStorageTraits, WritableStorageTraits};

pub(crate) fn float32() -> DataType {
    DataType::Float32
}
pub(crate) fn float64() -> DataType {
    DataType::Float64
}

pub(crate) fn sharding_codec(
    subchunks: [u64; 2],
    compressed: bool,
) -> std::sync::Arc<zarrs::array::codec::ShardingCodec> {
    use zarrs::array::codec::{GzipCodec, array_to_bytes::sharding::ShardingCodecBuilder};
    let mut builder = ShardingCodecBuilder::new(subchunks.as_slice().try_into().unwrap());
    if compressed {
        builder.bytes_to_bytes_codecs(vec![std::sync::Arc::new(GzipCodec::new(5).unwrap())]);
    }
    std::sync::Arc::new(builder.build())
}

pub(crate) fn store_chunk<S: WritableStorageTraits + ?Sized + 'static, F: Element>(
    array: &Array<S>,
    indices: &[u64],
    values: &[F],
) -> Result<(), ArrayError> {
    array.store_chunk_elements(indices, values)
}

pub(crate) fn store_subset<S: ReadableWritableStorageTraits + ?Sized + 'static, F: Element>(
    array: &Array<S>,
    values: &[F],
) -> Result<(), ArrayError> {
    array.store_array_subset_elements(&array.subset_all(), values)
}
