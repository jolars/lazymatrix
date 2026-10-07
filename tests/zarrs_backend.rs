#![cfg(feature = "zarrs_all")]
//! Chunked storage behavior and parity with the shared dense oracle.

#[path = "common/backend_aliases.rs"]
mod backend_aliases;
use backend_aliases::zarrs;
#[path = "common/runner.rs"]
mod common;

use lazymatrix::{ColumnStats, LazyMatrix, MatVec, Normalization, ZarrMatrix, ZarrMatrixError};
use std::sync::Arc;
use zarrs::array::{Array, ArrayBuilder, DataType};
use zarrs::storage::store::MemoryStore;

fn build(tm: &common::TestMatrix, chunk: [u64; 2]) -> ZarrMatrix<MemoryStore> {
    let array = ArrayBuilder::new(
        vec![tm.nrows as u64, tm.ncols as u64],
        chunk.to_vec(),
        DataType::Float64,
        0.0f64,
    )
    .build(Arc::new(MemoryStore::new()), "/matrix")
    .unwrap();
    if tm.nrows != 0 && tm.ncols != 0 {
        let values: Vec<_> = tm.dense.iter().flatten().copied().collect();
        array
            .store_array_subset_elements(&array.subset_all(), &values)
            .unwrap();
    }
    ZarrMatrix::try_new(array).unwrap()
}

#[test]
fn zarrs_backend_suite() {
    for chunks in [[3, 2], [1, 7], [16, 1], [32, 32]] {
        common::run_fused_suite(|tm| build(tm, chunks), |v| v.to_vec());
        common::run_backend_suite(|tm| build(tm, chunks), |v| v.to_vec(), |v| v.clone());
    }
}

#[test]
fn missing_chunks_use_their_fill_value() {
    use lazymatrix::{MatTransposeVecScaledInto, MatVecScaledInto};
    let array = ArrayBuilder::new(vec![3, 2], vec![2, 2], DataType::Float64, 5.0f64)
        .build(Arc::new(MemoryStore::new()), "/matrix")
        .unwrap();
    let matrix = ZarrMatrix::<_, f64>::try_new(array).unwrap();
    assert_eq!(matrix.col_means().unwrap(), [5.0, 5.0]);
    assert_eq!(matrix.matvec(&vec![2.0, 3.0]).unwrap(), [25.0; 3]);
    let lazy = LazyMatrix::new(&matrix, Normalization::default()).unwrap();
    assert_eq!(lazy.matvec(&vec![2.0, 3.0]).unwrap(), [25.0; 3]);
    let mut forward = vec![3.0; 3];
    matrix
        .matvec_scaled_into(2.0, &vec![2.0, 3.0], -1.0, &mut forward)
        .unwrap();
    assert_eq!(forward, [47.0; 3]);
    let mut transpose = vec![f64::NAN; 2];
    matrix
        .mat_transpose_vec_scaled_into(2.0, &vec![1.0; 3], 0.0, &mut transpose)
        .unwrap();
    assert_eq!(transpose, [30.0; 2]);
}

#[test]
fn fused_products_preserve_scan_counts_and_skip_reads_for_zero_alpha() {
    use lazymatrix::{MatTransposeVecScaledInto, MatVecScaledInto, WithIntercept};
    fn exercise<O>(op: &O, matrix: &ZarrMatrix<TrackedStore>, store: &TrackedStore)
    where
        O: MatVec<Vec<f64>>
            + MatTransposeVec<Vec<f64>>
            + MatVecScaledInto<Vec<f64>, Vec<f64>, f64>
            + MatTransposeVecScaledInto<Vec<f64>, Vec<f64>, f64>,
    {
        let x = vec![1.0; op.ncols()];
        let u = vec![1.0; op.nrows()];
        let expected_forward = op.matvec(&x).unwrap();
        let expected_transpose = op.mat_transpose_vec(&u).unwrap();
        store.reset();
        let mut forward = vec![3.0; op.nrows()];
        op.matvec_scaled_into(2.0, &x, -1.0, &mut forward).unwrap();
        assert_eq!(
            forward,
            expected_forward
                .iter()
                .map(|p| 2.0 * p - 3.0)
                .collect::<Vec<_>>()
        );
        assert_scans(store, matrix, 1);
        store.reset();
        let mut transpose = vec![3.0; op.ncols()];
        op.mat_transpose_vec_scaled_into(2.0, &u, -1.0, &mut transpose)
            .unwrap();
        assert_eq!(
            transpose,
            expected_transpose
                .iter()
                .map(|p| 2.0 * p - 3.0)
                .collect::<Vec<_>>()
        );
        assert_scans(store, matrix, 1);
        store.reset();
        store.fail_after.store(0, Ordering::Relaxed);
        op.matvec_scaled_into(-0.0, &vec![f64::NAN; op.ncols()], 0.0, &mut forward)
            .unwrap();
        op.mat_transpose_vec_scaled_into(0.0, &vec![f64::NAN; op.nrows()], 0.0, &mut transpose)
            .unwrap();
        assert_eq!(forward, vec![0.0; op.nrows()]);
        assert_eq!(transpose, vec![0.0; op.ncols()]);
        assert_scans(store, matrix, 0);
        store.reset();
    }
    for compressed in [false, true] {
        let (matrix, store) = tracked(compressed);
        exercise(&matrix, &matrix, &store);
        for center in [None, Some(vec![2.0; 3])] {
            for scale in [None, Some(vec![-4.0; 3])] {
                let lazy = LazyMatrix::from_parts(&matrix, center.clone(), scale);
                exercise(&lazy, &matrix, &store);
                exercise(&WithIntercept::new(&lazy), &matrix, &store);
            }
        }
    }
}

#[test]
fn fused_read_failures_do_not_apply_normalization_or_intercept_corrections() {
    use lazymatrix::{MatVecScaledInto, WithIntercept};
    let (matrix, store) = tracked(false);
    let lazy = LazyMatrix::from_parts(&matrix, Some(vec![2.0; 3]), Some(vec![4.0; 3]));
    let mut raw = vec![11.0; 5];
    let mut normalized = raw.clone();
    store.fail_after.store(2, Ordering::Relaxed);
    assert!(
        matrix
            .matvec_scaled_into(2.0, &vec![0.25; 3], 1.0, &mut raw)
            .is_err()
    );
    store.reset();
    store.fail_after.store(2, Ordering::Relaxed);
    assert!(
        WithIntercept::new(&lazy)
            .matvec_scaled_into(2.0, &vec![3.0, 1.0, 1.0, 1.0], 1.0, &mut normalized)
            .is_err()
    );
    assert_eq!(normalized, raw);
    assert_eq!(store.keys.lock().unwrap().len(), 3);
    store.reset();
    store.fail_after.store(2, Ordering::Relaxed);
    let mut out = vec![11.0; 4];
    let mut scratch = vec![0.0; 3];
    assert!(
        WithIntercept::new(&lazy)
            .mat_transpose_vec_scaled_with_workspace(
                2.0,
                &vec![1.0; 5],
                1.0,
                &mut out,
                &mut scratch
            )
            .is_err()
    );
    assert_eq!(out, [11.0; 4]);
    assert_eq!(store.keys.lock().unwrap().len(), 3);
}

#[test]
fn constructor_rejects_wrong_rank_and_scalar_type() {
    let store = Arc::new(MemoryStore::new());
    let rank_one = ArrayBuilder::new(vec![2], vec![2], DataType::Float64, 0.0f64)
        .build(store.clone(), "/vector")
        .unwrap();
    assert!(matches!(
        ZarrMatrix::<_, f64>::try_new(rank_one),
        Err(ZarrMatrixError::InvalidRank(1))
    ));
    let wrong_type: Array<_> = ArrayBuilder::new(vec![2, 2], vec![2, 2], DataType::Float32, 0.0f32)
        .build(store, "/matrix")
        .unwrap();
    assert!(matches!(
        ZarrMatrix::<_, f64>::try_new(wrong_type),
        Err(ZarrMatrixError::Array(_))
    ));
}

use lazymatrix::{Centering, MatTransposeVec, MatTransposeVecInto, MatVecInto, Scaling};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use zarrs::array::codec::GzipCodec;
use zarrs::storage::byte_range::ByteRangeIterator;
use zarrs::storage::{
    MaybeBytes, MaybeBytesIterator, ReadableStorageTraits, StorageError, StoreKey,
    WritableStorageTraits,
};

struct TrackedStore {
    inner: Arc<MemoryStore>,
    keys: Mutex<Vec<String>>,
    fail_after: AtomicUsize,
}

impl TrackedStore {
    fn reset(&self) {
        self.keys.lock().unwrap().clear();
        self.fail_after.store(usize::MAX, Ordering::Relaxed);
    }

    fn record(&self, key: &StoreKey) -> Result<(), StorageError> {
        let mut keys = self.keys.lock().unwrap();
        let index = keys.len();
        keys.push(key.to_string());
        if index >= self.fail_after.load(Ordering::Relaxed) {
            return Err(StorageError::Other("injected read failure".into()));
        }
        Ok(())
    }
}

impl ReadableStorageTraits for TrackedStore {
    fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        self.record(key)?;
        self.inner.get(key)
    }
    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        self.record(key)?;
        self.inner.get_partial_many(key, ranges)
    }
    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        self.inner.size_key(key)
    }
    fn supports_get_partial(&self) -> bool {
        self.inner.supports_get_partial()
    }
}

fn tracked(compressed: bool) -> (ZarrMatrix<TrackedStore>, Arc<TrackedStore>) {
    let inner = Arc::new(MemoryStore::new());
    let mut builder = ArrayBuilder::new(vec![5, 3], vec![2, 2], DataType::Float64, 0.0f64);
    if compressed {
        builder.bytes_to_bytes_codecs(vec![Arc::new(GzipCodec::new(5).unwrap())]);
    }
    let array = builder.build(inner.clone(), "/matrix").unwrap();
    array.store_metadata().unwrap();
    let values: Vec<f64> = (1..=15).map(f64::from).collect();
    array
        .store_array_subset_elements(&array.subset_all(), &values)
        .unwrap();
    let store = Arc::new(TrackedStore {
        inner,
        keys: Mutex::new(Vec::new()),
        fail_after: AtomicUsize::new(usize::MAX),
    });
    let array = Array::open(store.clone(), "/matrix").unwrap();
    store.reset();
    let matrix = ZarrMatrix::try_new(array).unwrap();
    assert!(store.keys.lock().unwrap().is_empty());
    (matrix, store)
}

fn assert_scans(store: &TrackedStore, matrix: &ZarrMatrix<TrackedStore>, scans: usize) {
    let expected: Vec<_> = (0..3)
        .flat_map(|row| (0..2).map(move |col| matrix.as_inner().chunk_key(&[row, col]).to_string()))
        .collect();
    assert_eq!(
        *store.keys.lock().unwrap(),
        (0..scans)
            .flat_map(|_| expected.iter().cloned())
            .collect::<Vec<_>>()
    );
}

#[test]
fn normalization_and_products_read_each_chunk_once_per_scan() {
    // For these unsharded bytes/gzip fixtures, one logical chunk retrieval is
    // also one store read. Other codecs may perform several range requests.
    for compressed in [false, true] {
        let (matrix, store) = tracked(compressed);
        for center in [Centering::None, Centering::Mean, Centering::Min] {
            for scale in [
                Scaling::None,
                Scaling::Sd,
                Scaling::Range,
                Scaling::L1,
                Scaling::L2,
                Scaling::MaxAbs,
            ] {
                store.reset();
                let lazy = LazyMatrix::new(&matrix, Normalization::new(center, scale)).unwrap();
                let scans = if center == Centering::None && scale == Scaling::None {
                    0
                } else if scale == Scaling::Sd
                    || (center != Centering::None
                        && matches!(scale, Scaling::L1 | Scaling::L2 | Scaling::MaxAbs))
                {
                    2
                } else {
                    1
                };
                assert_scans(&store, &matrix, scans);
                store.reset();
                let y = lazy.matvec(&vec![1.0; 3]).unwrap();
                assert_scans(&store, &matrix, 1);
                store.reset();
                lazy.mat_transpose_vec(&y).unwrap();
                assert_scans(&store, &matrix, 1);
            }
        }
    }
}

#[test]
fn read_errors_stop_the_scan_and_output_can_be_reused() {
    let (matrix, store) = tracked(false);
    let lazy = LazyMatrix::from_parts(&matrix, Some(vec![2.0; 3]), Some(vec![4.0; 3]));
    let mut forward = vec![f64::NAN; 5];
    let mut transpose = vec![f64::NAN; 3];
    store.fail_after.store(2, Ordering::Relaxed);
    let error = lazy.matvec_into(&vec![1.0; 3], &mut forward).unwrap_err();
    assert!(std::error::Error::source(&error).is_some());
    assert!(error.to_string().contains("injected read failure"));
    assert_eq!(store.keys.lock().unwrap().len(), 3);
    store.reset();
    lazy.matvec_into(&vec![1.0; 3], &mut forward).unwrap();
    assert_eq!(forward, [0.0, 2.25, 4.5, 6.75, 9.0]);
    store.reset();
    store.fail_after.store(2, Ordering::Relaxed);
    assert!(
        lazy.mat_transpose_vec_into(&vec![1.0; 5], &mut transpose)
            .is_err()
    );
    assert_eq!(store.keys.lock().unwrap().len(), 3);
    store.reset();
    lazy.mat_transpose_vec_into(&vec![1.0; 5], &mut transpose)
        .unwrap();
    assert_eq!(transpose, [6.25, 7.5, 8.75]);
    for fail_after in [0, 7] {
        store.reset();
        store.fail_after.store(fail_after, Ordering::Relaxed);
        assert!(
            LazyMatrix::new(&matrix, Normalization::new(Centering::Mean, Scaling::Sd)).is_err()
        );
        assert_eq!(store.keys.lock().unwrap().len(), fail_after + 1);
    }
}

#[test]
fn corrupt_compressed_chunks_return_decoding_errors() {
    let (matrix, store) = tracked(true);
    let key = matrix.as_inner().chunk_key(&[0, 0]);
    store.inner.set(&key, vec![1, 2, 3].into()).unwrap();
    assert!(matrix.matvec(&vec![1.0; 3]).is_err());
    assert_eq!(store.keys.lock().unwrap().len(), 1);
}

#[test]
fn f32_normalization_and_products_match_known_values() {
    let array = ArrayBuilder::new(vec![2, 1], vec![1, 1], DataType::Float32, 0.0_f32)
        .build(Arc::new(MemoryStore::new()), "/f32")
        .unwrap();
    array
        .store_array_subset_elements(&array.subset_all(), &[1.0_f32, 3.0])
        .unwrap();
    common::check_fused_f32(&ZarrMatrix::<_, f32>::try_new(array).unwrap(), |v| {
        v.to_vec()
    });
    let array = ArrayBuilder::new(vec![3, 2], vec![2, 1], DataType::Float32, 0.0f32)
        .build(Arc::new(MemoryStore::new()), "/matrix")
        .unwrap();
    array
        .store_array_subset_elements(&array.subset_all(), &[1.0f32, 2.0, 2.0, 4.0, 3.0, 6.0])
        .unwrap();
    let matrix = ZarrMatrix::<_, f32>::try_new(array).unwrap();
    let lazy = LazyMatrix::new(matrix, Normalization::new(Centering::Mean, Scaling::Sd)).unwrap();
    assert_eq!(lazy.centers().unwrap(), [2.0, 4.0]);
    let root = (1.5f32).sqrt();
    let forward = lazy.matvec(&vec![1.0, 2.0]).unwrap();
    for (got, expected) in forward.into_iter().zip([-3.0 * root, 0.0, 3.0 * root]) {
        approx::assert_abs_diff_eq!(got, expected, epsilon = 1e-6);
    }
    let transpose = lazy.mat_transpose_vec(&vec![1.0, -1.0, 2.0]).unwrap();
    for got in transpose {
        approx::assert_abs_diff_eq!(got, root, epsilon = 1e-6);
    }
}

#[test]
fn edge_padding_is_excluded_and_nan_fill_propagates() {
    use lazymatrix::{MatTransposeVecScaledInto, MatVecScaledInto};
    let array = ArrayBuilder::new(vec![1, 1], vec![2, 2], DataType::Float64, f64::NAN)
        .build(Arc::new(MemoryStore::new()), "/matrix")
        .unwrap();
    array
        .store_chunk_elements(&[0, 0], &[3.0, f64::NAN, f64::NAN, f64::NAN])
        .unwrap();
    let matrix = ZarrMatrix::<_, f64>::try_new(array).unwrap();
    assert_eq!(matrix.col_means().unwrap(), [3.0]);
    assert_eq!(matrix.matvec(&vec![2.0]).unwrap(), [6.0]);
    let mut out = vec![f64::NAN];
    matrix
        .matvec_scaled_into(2.0, &vec![2.0], 0.0, &mut out)
        .unwrap();
    assert_eq!(out, [12.0]);
    matrix
        .mat_transpose_vec_scaled_into(2.0, &vec![2.0], 0.0, &mut out)
        .unwrap();
    assert_eq!(out, [12.0]);
    matrix.as_inner().erase_chunk(&[0, 0]).unwrap();
    assert!(matrix.col_means().unwrap()[0].is_nan());
    assert!(matrix.matvec(&vec![0.0]).unwrap()[0].is_nan());
    matrix
        .matvec_scaled_into(1.0, &vec![0.0], 0.0, &mut out)
        .unwrap();
    assert!(out[0].is_nan());
}

#[test]
fn oversized_chunks_fail_before_reading_or_allocating_them() {
    let array = ArrayBuilder::new(vec![1, 1], vec![u64::MAX, 2], DataType::Float64, 0.0f64)
        .build(Arc::new(MemoryStore::new()), "/matrix")
        .unwrap();
    let matrix = ZarrMatrix::<_, f64>::try_new(array).unwrap();
    assert!(matches!(
        matrix.col_means(),
        Err(ZarrMatrixError::SizeOverflow)
    ));
}

#[cfg(feature = "ndarray_all")]
macro_rules! ndarray_suite {
    ($name:ident, $backend:ident) => {
        mod $name {
            use super::*;
            use crate::backend_aliases::$backend as ndarray;
            #[test]
            fn fused_normalization_matches_ndarray_statistics() {
                let (matrix, _) = tracked(false);
                let dense = ndarray::Array2::from_shape_fn((5, 3), |(i, j)| (3 * i + j + 1) as f64);
                for center in [Centering::None, Centering::Mean, Centering::Min] {
                    for scale in [
                        Scaling::None,
                        Scaling::Sd,
                        Scaling::Range,
                        Scaling::L1,
                        Scaling::L2,
                        Scaling::MaxAbs,
                    ] {
                        let spec = Normalization::new(center, scale);
                        let chunked = LazyMatrix::new(&matrix, spec).unwrap();
                        let resident = LazyMatrix::new(&dense, spec).unwrap();
                        assert_eq!(chunked.centers().is_some(), resident.centers().is_some());
                        assert_eq!(chunked.scales().is_some(), resident.scales().is_some());
                        common::assert_close(
                            chunked.centers().unwrap_or(&[]),
                            resident.centers().unwrap_or(&[]),
                            1e-12,
                        );
                        common::assert_close(
                            chunked.scales().unwrap_or(&[]),
                            resident.scales().unwrap_or(&[]),
                            1e-12,
                        );
                    }
                }
            }

            #[test]
            fn ndarray_vectors_and_strided_outputs_work_with_chunked_storage() {
                use ndarray::{Array1, array, s};

                let (matrix, _) = tracked(false);
                let input_storage = array![1.0, -99.0, 2.0, -99.0, 4.0];
                let input = input_storage.slice(s![..;2]);
                let mut output = Array1::from_elem(10, -99.0);
                matrix
                    .matvec_into(&input, &mut output.slice_mut(s![..;-2]))
                    .unwrap();
                assert_eq!(
                    output.slice(s![..;-2]),
                    array![17.0, 38.0, 59.0, 80.0, 101.0]
                );
                assert!(output.slice(s![..;2]).iter().all(|&x| x == -99.0));

                let lazy = LazyMatrix::from_parts(matrix, Some(vec![2.0, 3.0, 4.0]), Some(vec![1.0, 2.0, 4.0]));
                lazy.matvec_into(&input.to_owned(), &mut output.slice_mut(s![..;-2]))
                    .unwrap();
                assert_eq!(output.slice(s![..;-2]), array![-3.0, 6.0, 15.0, 24.0, 33.0]);
                assert!(output.slice(s![..;2]).iter().all(|&x| x == -99.0));

                let row_storage = array![1.0, -99.0, 2.0, -99.0, 3.0, -99.0, 4.0, -99.0, 5.0];
                let rows = row_storage.slice(s![..;2]);
                let mut transpose = Array1::from_elem(6, -99.0);
                lazy.mat_transpose_vec_into(&rows, &mut transpose.slice_mut(s![..;-2]))
                    .unwrap();
                assert_eq!(transpose.slice(s![..;-2]), array![105.0, 52.5, 26.25]);
                assert!(transpose.slice(s![..;2]).iter().all(|&x| x == -99.0));
            }
        }
    };
}

#[cfg(feature = "ndarray_v0_15")]
ndarray_suite!(ndarray_0_15, ndarray_0_15);

#[cfg(feature = "ndarray_v0_16")]
ndarray_suite!(ndarray_0_16, ndarray_0_16);

#[cfg(feature = "ndarray_v0_17")]
ndarray_suite!(ndarray_0_17, ndarray_0_17);

#[test]
fn intercept_adds_no_storage_reads_and_forwards_failures() {
    use lazymatrix::WithIntercept;
    let (matrix, store) = tracked(false);
    let lazy = LazyMatrix::from_parts(&matrix, Some(vec![2.0; 3]), Some(vec![4.0; 3]));
    let augmented = WithIntercept::new(&lazy);
    assert!(store.keys.lock().unwrap().is_empty());
    let mut forward = vec![f64::NAN; 5];
    augmented
        .matvec_into(&vec![2.0, 1.0, 1.0, 1.0], &mut forward)
        .unwrap();
    assert_eq!(forward, [2.0, 4.25, 6.5, 8.75, 11.0]);
    assert_scans(&store, &matrix, 1);
    store.reset();
    assert_eq!(
        augmented.mat_transpose_vec(&vec![1.0; 5]).unwrap(),
        [5.0, 6.25, 7.5, 8.75]
    );
    assert_scans(&store, &matrix, 1);
    store.reset();
    store.fail_after.store(2, Ordering::Relaxed);
    assert!(
        augmented
            .matvec_into(&vec![2.0, 1.0, 1.0, 1.0], &mut forward)
            .is_err()
    );
    assert_eq!(store.keys.lock().unwrap().len(), 3);
    store.reset();
    store.fail_after.store(2, Ordering::Relaxed);
    let mut transpose = vec![99.0; 4];
    assert!(
        augmented
            .mat_transpose_vec_into(&vec![1.0; 5], &mut transpose)
            .is_err()
    );
    assert_eq!(transpose[0], 99.0);
    assert_eq!(store.keys.lock().unwrap().len(), 3);
    store.reset();
    augmented
        .matvec_into(&vec![2.0, 1.0, 1.0, 1.0], &mut forward)
        .unwrap();
    assert_eq!(forward, [2.0, 4.25, 6.5, 8.75, 11.0]);
}
