//! Row fitting and products must retain the Zarr storage and scan boundaries.

use super::*;
use lazymatrix::{
    MatTransposeVecScaledInto, MatVecScaledInto, MatrixShape, RowNormalization,
    RowNormalizationKernel, WeightedColumnSumsInto, WeightedGramInto,
};

#[test]
fn row_normalization_matches_the_shared_oracle() {
    for chunks in [[3, 2], [1, 7], [16, 1], [32, 32]] {
        common::run_row_normalization_suite(|tm| build(tm, chunks), |v| v.to_vec());
        common::run_row_normalization_f32_suite(
            |raw| {
                let array = ArrayBuilder::new(
                    vec![raw.len() as u64, raw[0].len() as u64],
                    chunks.to_vec(),
                    float32(),
                    0.0_f32,
                )
                .build(Arc::new(MemoryStore::new()), "/row-f32")
                .unwrap();
                store_subset(&array, &raw.iter().flatten().copied().collect::<Vec<_>>()).unwrap();
                ZarrMatrix::<_, f32>::try_new(array).unwrap()
            },
            |v| v.to_vec(),
        );
    }
}

#[test]
fn row_fitting_and_products_stream_serial_scans() {
    for compressed in [false, true] {
        let (matrix, store) = tracked(compressed);
        for (spec, scans) in [
            (RowNormalization::default(), 0),
            (RowNormalization::standardize(), 2),
            (RowNormalization::min_max(), 2),
            (RowNormalization::new(Centering::None, Scaling::L2), 1),
            (RowNormalization::new(Centering::Mean, Scaling::L1), 2),
            (RowNormalization::new(Centering::Min, Scaling::MaxAbs), 2),
            (
                RowNormalization::new(Centering::Min, Scaling::Sd { df: 1 }),
                3,
            ),
        ] {
            store.reset();
            let lazy = LazyMatrix::new(&matrix, spec).unwrap();
            assert_scans(&store, &matrix, scans);
            store.reset();
            let frozen = lazy.normalization().view().apply_frozen(&matrix);
            assert_scans(&store, &matrix, 0);
            frozen.matvec(&vec![1.0; 3]).unwrap();
            assert_scans(&store, &matrix, 1);
            store.reset();
            frozen.mat_transpose_vec(&vec![1.0; 5]).unwrap();
            assert_scans(&store, &matrix, 1);
            store.reset();
            frozen
                .weighted_column_sums_into(&[1.; 5], &mut [f64::NAN; 3])
                .unwrap();
            assert_scans(&store, &matrix, 1);
            store.reset();
            frozen
                .weighted_gram_into(
                    &[1.; 5],
                    &mut common::GramOutput(vec![vec![f64::NAN; 3]; 3]),
                )
                .unwrap();
            assert_scans(&store, &matrix, 1);
            store.reset();
            frozen
                .matvec_scaled_into(-0.75, &vec![1.; 3], 0.25, &mut vec![1.; 5])
                .unwrap();
            assert_scans(&store, &matrix, 1);
            store.reset();
            frozen
                .mat_transpose_vec_scaled_into(-0.75, &vec![1.; 5], 0.25, &mut vec![1.; 3])
                .unwrap();
            assert_scans(&store, &matrix, 1);
            store.reset();
            frozen
                .matvec_scaled_into(0., &vec![f64::NAN; 3], 0., &mut vec![f64::NAN; 5])
                .unwrap();
            frozen
                .mat_transpose_vec_scaled_into(-0., &vec![f64::NAN; 5], -0., &mut vec![f64::NAN; 3])
                .unwrap();
            assert_scans(&store, &matrix, 0);
        }
        store.reset();
        let params = RowNormalization::robust().fit(&matrix).unwrap();
        assert_eq!(params.centers().unwrap(), [2., 5., 8., 11., 14.]);
        assert_eq!(params.scales().unwrap(), [1.; 5]);
        assert_scans(&store, &matrix, 2);
        for spec in [
            RowNormalization::new(Centering::Median, Scaling::None),
            RowNormalization::new(
                Centering::None,
                Scaling::QuantileRange(QuantileRange::default()),
            ),
        ] {
            store.reset();
            spec.fit(&matrix).unwrap();
            assert_scans(&store, &matrix, 2);
        }
    }
}

#[test]
fn row_scans_preserve_fill_and_exclude_nonfinite_padding() {
    for compressed in [false, true] {
        let mut builder = ArrayBuilder::new(vec![3, 3], vec![2, 2], float64(), 7.0_f64);
        if compressed {
            builder.bytes_to_bytes_codecs(vec![Arc::new(GzipCodec::new(5).unwrap())]);
        }
        let array = builder
            .build(Arc::new(MemoryStore::new()), "/row-fill")
            .unwrap();
        store_chunk(&array, &[0, 0], &[1., 3., 2., 4.]).unwrap();
        store_chunk(&array, &[1, 1], &[3., f64::NAN, f64::INFINITY, f64::NAN]).unwrap();
        let matrix = ZarrMatrix::<_, f64>::try_new(array).unwrap();
        let raw = [vec![1., 3., 7.], vec![2., 4., 7.], vec![7., 7., 3.]];
        common::batched::check_normalized_batches(&matrix, &raw);
        common::check_row_normalization(&matrix, &raw, |v| v.to_vec());
        common::check_row_submatrices(&matrix, &raw, |v| v.to_vec());
        let frozen = LazyMatrix::from_row_parts(&matrix, Some(vec![1., 2., 3.]), Some(vec![2.; 3]));
        common::check_row_normalized_products(
            &frozen,
            &[vec![0., 1., 3.], vec![0., 1., 2.5], vec![2., 2., 0.]],
            |v| v.to_vec(),
        );
        matrix.as_inner().erase_chunk(&[1, 1]).unwrap();
        assert_eq!(
            RowNormalization::min_max()
                .fit(&matrix)
                .unwrap()
                .scales()
                .unwrap()[2],
            1.
        );
    }
    let array = ArrayBuilder::new(vec![1, 1], vec![2, 2], float64(), f64::NAN)
        .build(Arc::new(MemoryStore::new()), "/row-nan-fill")
        .unwrap();
    store_chunk(&array, &[0, 0], &[3., f64::NAN, f64::NAN, f64::NAN]).unwrap();
    let matrix = ZarrMatrix::<_, f64>::try_new(array).unwrap();
    common::check_row_normalization(&matrix, &[vec![3.]], |v| v.to_vec());
    matrix.as_inner().erase_chunk(&[0, 0]).unwrap();
    common::check_row_normalization(&matrix, &[vec![f64::NAN]], |v| v.to_vec());
}

#[test]
fn row_scans_use_outer_shards_and_preserve_missing_subchunks() {
    for compressed in [false, true] {
        let inner = Arc::new(MemoryStore::new());
        let array = ArrayBuilder::new(vec![5, 5], vec![4, 4], float64(), 7.0_f64)
            .array_to_bytes_codec(compat::sharding_codec([2, 2], compressed))
            .build(inner.clone(), "/row-shards")
            .unwrap();
        let mut values = [7.; 16];
        values[0] = 1.;
        values[1] = 3.;
        store_chunk(&array, &[0, 0], &values).unwrap();
        let mut values = [f64::NAN; 16];
        values[0] = 3.;
        store_chunk(&array, &[1, 1], &values).unwrap();
        array.store_metadata().unwrap();
        let store = Arc::new(TrackedStore {
            inner,
            keys: Mutex::new(Vec::new()),
            fail_after: AtomicUsize::new(usize::MAX),
        });
        let array = Array::open(store.clone(), "/row-shards").unwrap();
        store.reset();
        let matrix = ZarrMatrix::<_, f64>::try_new(array).unwrap();
        let mut raw = vec![vec![7.; 5]; 5];
        raw[0][0] = 1.;
        raw[0][1] = 3.;
        raw[4][4] = 3.;
        common::batched::check_normalized_batches(&matrix, &raw);
        common::check_row_normalization(&matrix, &raw, |v| v.to_vec());
        common::check_row_submatrices(&matrix, &raw, |v| v.to_vec());
        store.reset();
        let lazy = LazyMatrix::new(&matrix, RowNormalization::standardize()).unwrap();
        let shard_keys: Vec<_> = (0..2)
            .flat_map(|i| (0..2).map(move |j| [i, j]))
            .map(|indices| matrix.as_inner().chunk_key(&indices).to_string())
            .collect();
        assert_eq!(
            *store.keys.lock().unwrap(),
            [shard_keys.clone(), shard_keys.clone()].concat()
        );
        store.reset();
        lazy.matvec(&vec![1.; 5]).unwrap();
        assert_eq!(*store.keys.lock().unwrap(), shard_keys);
        store.reset();
        lazy.mat_transpose_vec(&vec![1.; 5]).unwrap();
        assert_eq!(*store.keys.lock().unwrap(), shard_keys);
        let values = vec![1.; 5 * 33];
        let rhs = lazymatrix::DenseBlock::new(&values, 5, 33);
        store.reset();
        let _: common::MaterializeOutput = lazymatrix::MatMat::matmat(&lazy, &rhs).unwrap();
        assert_eq!(*store.keys.lock().unwrap(), shard_keys);
        store.reset();
        let _: common::MaterializeOutput =
            lazymatrix::MatTransposeMat::mat_transpose_mat(&lazy, &rhs).unwrap();
        assert_eq!(*store.keys.lock().unwrap(), shard_keys);
    }
}

#[test]
fn row_failures_stop_reads_and_allow_retry() {
    for compressed in [false, true] {
        let (matrix, store) = tracked(compressed);
        for spec in [RowNormalization::standardize(), RowNormalization::robust()] {
            for fail_after in [0, 2, 7] {
                store.reset();
                store.fail_after.store(fail_after, Ordering::Relaxed);
                let error = spec.fit(&matrix).unwrap_err();
                assert!(error.to_string().contains("injected read failure"));
                assert!(std::error::Error::source(&error).is_some());
                assert_eq!(store.keys.lock().unwrap().len(), fail_after + 1);
            }
            store.reset();
            spec.fit(&matrix).unwrap();
            assert_scans(&store, &matrix, 2);
        }
        let lazy = LazyMatrix::from_row_parts(&matrix, Some(vec![1.; 5]), Some(vec![2.; 5]));
        let mut gram = common::GramOutput(vec![vec![f64::NAN; 3]; 3]);
        let mut sums = vec![f64::NAN; 3];
        store.reset();
        store.fail_after.store(1, Ordering::Relaxed);
        let error = lazy.weighted_gram_into(&[1.; 5], &mut gram).unwrap_err();
        assert!(std::error::Error::source(&error).is_some());
        assert_eq!(store.keys.lock().unwrap().len(), 2);
        store.reset();
        lazy.weighted_gram_into(&[1.; 5], &mut gram).unwrap();
        assert_scans(&store, &matrix, 1);
        common::assert_gram(
            &gram,
            &[
                vec![0., 0.5, 1.],
                vec![1.5, 2., 2.5],
                vec![3., 3.5, 4.],
                vec![4.5, 5., 5.5],
                vec![6., 6.5, 7.],
            ],
            &[1.; 5],
        );
        store.reset();
        store.fail_after.store(1, Ordering::Relaxed);
        assert!(lazy.weighted_column_sums_into(&[1.; 5], &mut sums).is_err());
        assert_eq!(store.keys.lock().unwrap().len(), 2);
        store.reset();
        lazy.weighted_column_sums_into(&[1.; 5], &mut sums).unwrap();
        assert_eq!(sums, [15., 17.5, 20.]);
        assert_scans(&store, &matrix, 1);
        let mut forward = vec![f64::NAN; 5];
        store.reset();
        store.fail_after.store(1, Ordering::Relaxed);
        assert!(lazy.matvec_into(&vec![1.; 3], &mut forward).is_err());
        assert_eq!(store.keys.lock().unwrap().len(), 2);
        assert_eq!(forward, [0.5, 3.5, 0., 0., 0.]);
        store.reset();
        lazy.matvec_into(&vec![1.; 3], &mut forward).unwrap();
        assert_eq!(forward, [1.5, 6., 10.5, 15., 19.5]);
        let mut transpose = vec![f64::NAN; 3];
        store.reset();
        store.fail_after.store(1, Ordering::Relaxed);
        assert!(
            lazy.mat_transpose_vec_into(&vec![1.; 5], &mut transpose)
                .is_err()
        );
        assert_eq!(store.keys.lock().unwrap().len(), 2);
        assert_eq!(transpose, [1.5, 2.5, 0.]);
        store.reset();
        lazy.mat_transpose_vec_into(&vec![1.; 5], &mut transpose)
            .unwrap();
        assert_eq!(transpose, [15., 17.5, 20.]);
        let key = matrix.as_inner().chunk_key(&[0, 0]);
        store.inner.set(&key, vec![1, 2, 3].into()).unwrap();
        assert!(RowNormalization::standardize().fit(&matrix).is_err());
        assert!(lazy.matvec(&vec![1.; 3]).is_err());
    }
}

#[test]
fn row_kernels_validate_before_storage_or_output_access() {
    let (matrix, store) = tracked(false);
    for transpose in [false, true] {
        let (input, output) = if transpose { (5, 3) } else { (3, 5) };
        for (input, output, centers, scales) in [
            (input + 1, output, None, None),
            (input, output + 1, None, None),
            (input, output, Some(vec![0.; 4]), None),
            (input, output, None, Some(vec![1.; 4])),
            (input, output, None, Some(vec![0.; 5])),
            (input, output, None, Some(vec![-0.; 5])),
        ] {
            let mut out = vec![99.; output];
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if transpose {
                    matrix.row_normalized_transpose_into(
                        &vec![1.; input],
                        centers.as_deref(),
                        scales.as_deref(),
                        &mut out,
                    )
                } else {
                    matrix.row_normalized_matvec_into(
                        &vec![1.; input],
                        centers.as_deref(),
                        scales.as_deref(),
                        &mut out,
                    )
                }
            }));
            assert!(result.is_err());
            assert_eq!(out, vec![99.; output]);
            assert!(store.keys.lock().unwrap().is_empty());
        }
    }
    let array = ArrayBuilder::new(vec![1, 1], vec![u64::MAX, 2], float64(), 0.0_f64)
        .build(Arc::new(MemoryStore::new()), "/row-overflow")
        .unwrap();
    let matrix = ZarrMatrix::<_, f64>::try_new(array).unwrap();
    assert!(matches!(
        RowNormalization::standardize().fit(&matrix),
        Err(ZarrMatrixError::SizeOverflow)
    ));
    let lazy = LazyMatrix::from_row_parts(matrix, None, None);
    assert!(matches!(
        lazy.matvec(&vec![1.]),
        Err(ZarrMatrixError::SizeOverflow)
    ));
    assert!(matches!(
        lazy.mat_transpose_vec(&vec![1.]),
        Err(ZarrMatrixError::SizeOverflow)
    ));
}

#[test]
fn empty_rows_and_columns_skip_failing_storage() {
    let (_, store) = tracked(false);
    for shape in [vec![0, 3], vec![4, 0], vec![0, 0]] {
        let array = ArrayBuilder::new(shape, vec![2, 2], float64(), f64::NAN)
            .build(store.clone(), "/row-empty")
            .unwrap();
        let matrix = ZarrMatrix::<_, f64>::try_new(array).unwrap();
        store.reset();
        store.fail_after.store(0, Ordering::Relaxed);
        for spec in [
            RowNormalization::default(),
            RowNormalization::standardize(),
            RowNormalization::min_max(),
            RowNormalization::robust(),
        ] {
            let lazy = LazyMatrix::new(&matrix, spec).unwrap();
            assert_eq!(
                lazy.matvec(&vec![1.; matrix.ncols()]).unwrap(),
                vec![0.; matrix.nrows()]
            );
            assert_eq!(
                lazy.mat_transpose_vec(&vec![1.; matrix.nrows()]).unwrap(),
                vec![0.; matrix.ncols()]
            );
        }
        assert!(store.keys.lock().unwrap().is_empty());
    }
}

#[test]
fn row_and_column_chain_batches_scan_once_and_retry_after_errors() {
    use lazymatrix::{MatMat, MatMatInto, MatTransposeMat, MatTransposeMatInto, MatrixOwned};
    fn check<M: MatMatInto<f64> + MatTransposeMatInto<f64>>(
        op: &M,
        matrix: &ZarrMatrix<Array<TrackedStore>>,
        store: &TrackedStore,
    ) {
        for transpose in [false, true] {
            let (input, output) = if transpose {
                (op.nrows(), op.ncols())
            } else {
                (op.ncols(), op.nrows())
            };
            for k in [0, 1, 17, 33] {
                let values = vec![1.; input * k];
                let rhs = lazymatrix::DenseBlock::new(&values, input, k);
                store.reset();
                let expected: common::MaterializeOutput = if transpose {
                    op.mat_transpose_mat(&rhs).unwrap()
                } else {
                    op.matmat(&rhs).unwrap()
                };
                assert_scans(store, matrix, usize::from(k != 0));
                let mut out = common::MaterializeOutput::zeros(output, k);
                if k != 0 {
                    for fail_after in [0, 2] {
                        store.reset();
                        store.fail_after.store(fail_after, Ordering::Relaxed);
                        let result = if transpose {
                            op.mat_transpose_mat_into(&rhs, &mut out)
                        } else {
                            op.matmat_into(&rhs, &mut out)
                        };
                        assert!(result.is_err());
                        assert_eq!(store.keys.lock().unwrap().len(), fail_after + 1);
                        store.reset();
                        if transpose {
                            op.mat_transpose_mat_into(&rhs, &mut out).unwrap();
                        } else {
                            op.matmat_into(&rhs, &mut out).unwrap();
                        }
                        assert_scans(store, matrix, 1);
                        assert_eq!(out.values, expected.values);
                    }
                }
                store.reset();
                let mut wrong = common::MaterializeOutput::zeros(output + 1, k);
                wrong.values.fill(99.);
                assert!(
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        if transpose {
                            op.mat_transpose_mat_into(&rhs, &mut wrong)
                        } else {
                            op.matmat_into(&rhs, &mut wrong)
                        }
                    }))
                    .is_err()
                );
                assert!(wrong.values.iter().all(|&v| v == 99.));
                assert_scans(store, matrix, 0);
            }
        }
    }
    for compressed in [false, true] {
        let (matrix, store) = tracked(compressed);
        let row = RowNormalization::standardize().fit_apply(&matrix).unwrap();
        check(&row, &matrix, &store);
        let chain = ColumnNormalization::robust()
            .then(ColumnNormalization::standardize())
            .fit_apply(&matrix)
            .unwrap();
        check(&chain, &matrix, &store);
    }
}
