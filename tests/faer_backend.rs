#![cfg(feature = "faer_all")]
//! Verification of the faer dense and sparse backends against the dense oracle.

#[path = "common/backend_aliases.rs"]
mod backend_aliases;

#[path = "common/runner.rs"]
mod common;

macro_rules! backend_suite {
    ($name:ident, $backend:ident) => {
        mod $name {
            use crate::backend_aliases::$backend as faer;
            use crate::common;
            use common::TestMatrix;
            use faer::prelude::ReborrowMut;
            use faer::sparse::{SparseColMat, SparseRowMat, SymbolicSparseRowMat, Triplet};
            use faer::{Col, Mat};
            use lazymatrix::{
                Centering, ColumnNormalization, LazyMatrix, MatrixShape, Scaling, SparseRows,
            };

            fn build(tm: &TestMatrix) -> SparseColMat<usize, f64> {
                let triplets: Vec<Triplet<usize, usize, f64>> = tm
                    .triplets
                    .iter()
                    .map(|&(r, c, v)| Triplet::new(r, c, v))
                    .collect();
                SparseColMat::try_new_from_triplets(tm.nrows, tm.ncols, &triplets)
                    .expect("valid triplets")
            }

            fn to_col(v: &[f64]) -> Col<f64> {
                Col::from_fn(v.len(), |i| v[i])
            }

            fn check_csc_view<M>(matrix: &M, dense: &[Vec<f64>])
            where
                M: lazymatrix::ColumnStats<f64>
                    + lazymatrix::RawColumns<f64>
                    + lazymatrix::SparseColumns<f64>
                    + lazymatrix::MatVec<Col<f64>>
                    + lazymatrix::MatVecInto<Col<f64>>
                    + lazymatrix::MatTransposeVec<Col<f64>>
                    + lazymatrix::MatTransposeVecInto<Col<f64>>
                    + lazymatrix::MatVecScaledInto<Col<f64>, Col<f64>, f64>
                    + lazymatrix::MatTransposeVecScaledInto<Col<f64>, Col<f64>, f64>
                    + lazymatrix::MatMatInto<f64>
                    + lazymatrix::MatTransposeMatInto<f64>
                    + lazymatrix::WeightedGramKernel<f64>
                    + lazymatrix::WeightedColumnSumsKernel<f64>
                    + lazymatrix::MaterializeDense<f64>
                    + lazymatrix::SubmatrixKernel<f64>,
            {
                use lazymatrix::{RawColumn, WeightedColumnSumsInto, WeightedGramInto};
                let reference = Mat::from_fn(matrix.nrows(), matrix.ncols(), |i, j| dense[i][j]);
                common::check_column_stats_against(matrix, &reference);
                common::check_transpose_products(matrix, dense, &to_col, &from_col);
                common::check_fused_operator(matrix, dense, matrix.ncols(), &to_col);
                common::batched::check_batched(matrix, dense);
                let lazy = LazyMatrix::from_parts(
                    matrix,
                    Some(vec![0.75; matrix.ncols()]),
                    Some(vec![-2.0; matrix.ncols()]),
                );
                common::batched::check_batched(
                    &lazy,
                    &common::materialize(dense, lazy.centers(), lazy.scales()),
                );
                common::check_submatrix_weighted(matrix, dense);
                for j in 0..matrix.ncols() {
                    let raw = matrix.raw_column(j);
                    assert_eq!(raw.len(), matrix.nrows());
                    let mut column = vec![0.0; matrix.nrows()];
                    raw.for_each_stored(|i, value| column[i] += value);
                    for (i, &value) in column.iter().enumerate() {
                        approx::assert_abs_diff_eq!(value, dense[i][j], epsilon = 1e-12);
                    }
                }
                for center in [
                    Centering::None,
                    Centering::Mean,
                    Centering::Min,
                    Centering::Median,
                ] {
                    for scale in [
                        Scaling::None,
                        Scaling::Sd { df: 0 },
                        Scaling::Sd { df: 1 },
                        Scaling::L1,
                        Scaling::L2,
                        Scaling::MaxAbs,
                        Scaling::Range,
                        Scaling::QuantileRange(lazymatrix::QuantileRange::default()),
                    ] {
                        if matrix.nrows() <= 1
                            && (center != Centering::None || scale != Scaling::None)
                        {
                            continue;
                        }
                        let lazy = LazyMatrix::new(matrix, ColumnNormalization::new(center, scale))
                            .unwrap();
                        let normalized = common::materialize(dense, lazy.centers(), lazy.scales());
                        common::check_transpose_products(&lazy, &normalized, &to_col, &from_col);
                        common::check_fused_operator(&lazy, &normalized, matrix.ncols(), &to_col);
                        let eager = lazy.to_eager::<Mat<f64>>().unwrap();
                        for (i, row) in normalized.iter().enumerate() {
                            for (j, &value) in row.iter().enumerate() {
                                approx::assert_abs_diff_eq!(
                                    eager.data()[(i, j)],
                                    value,
                                    epsilon = 1e-12
                                );
                            }
                        }
                        let weights = common::random_vec(0xC5D, matrix.nrows());
                        let mut gram = common::GramOutput(vec![
                            vec![f64::NAN; matrix.ncols()];
                            matrix.ncols()
                        ]);
                        lazy.weighted_gram_into(&weights, &mut gram).unwrap();
                        common::assert_gram(&gram, &normalized, &weights);
                        let mut sums = vec![f64::NAN; matrix.ncols()];
                        lazy.weighted_column_sums_into(&weights, &mut sums).unwrap();
                        common::assert_close(
                            &sums,
                            &common::dense_matvec(
                                &common::dense_transpose(&normalized, matrix.ncols()),
                                &weights,
                            ),
                            1e-10,
                        );
                    }
                }
            }

            #[test]
            fn csc_borrowed_views_match_owned_and_dense() {
                for (n, p, density) in [
                    (9, 5, 0.4),
                    (3, 7, 0.6),
                    (4, 3, 0.0),
                    (0, 3, 0.0),
                    (5, 0, 0.0),
                    (0, 0, 0.0),
                ] {
                    let fixture = common::random_matrix(0xC5C, n, p, density);
                    let mut matrix = build(&fixture);
                    check_csc_view(&matrix, &fixture.dense);
                    check_csc_view(&matrix.as_ref(), &fixture.dense);
                    check_csc_view(&matrix.rb_mut(), &fixture.dense);
                    let csr = build_csr(&fixture);
                    check_csc_view(
                        &csr.as_ref().transpose(),
                        &common::dense_transpose(&fixture.dense, p),
                    );
                }
            }

            #[test]
            fn csc_views_preserve_offsets_capacity_and_duplicates() {
                use faer::sparse::{
                    SparseColMatMut, SparseColMatRef, SymbolicSparseColMat, SymbolicSparseColMatRef,
                };
                use lazymatrix::{RawColumns, SparseColumns};
                let mut rows = vec![99; 20];
                let mut values = vec![f64::NAN; 20];
                rows[1..5].copy_from_slice(&[2, 0, 2, 0]);
                values[1..5].copy_from_slice(&[1.0, 0.0, -2.0, 2.0]);
                rows[8..12].copy_from_slice(&[2, 0, 2, 1]);
                values[8..12].copy_from_slice(&[1.0, 2.0, 3.0, 0.0]);
                rows[14..17].copy_from_slice(&[0, 1, 2]);
                values[14..17].copy_from_slice(&[1.0, 2.0, 3.0]);
                let offsets = vec![1, 6, 8, 14, 18];
                let counts = vec![4, 0, 4, 3];
                let dense = vec![
                    vec![2.0, 0.0, 2.0, 1.0],
                    vec![0.0, 0.0, 0.0, 2.0],
                    vec![-1.0, 0.0, 4.0, 3.0],
                ];
                let symbolic =
                    SymbolicSparseColMat::new_unsorted_checked(3, 4, offsets, Some(counts), rows);
                let mut matrix = SparseColMat::new(symbolic, values);
                check_csc_view(&matrix, &dense);
                check_csc_view(&matrix.as_ref(), &dense);
                check_csc_view(&matrix.rb_mut(), &dense);
                let symbolic = SymbolicSparseColMatRef::new_unsorted_checked(
                    3,
                    2,
                    &matrix.col_ptr()[1..4],
                    Some(&matrix.col_nnz().unwrap()[1..3]),
                    matrix.row_idx(),
                );
                let view = SparseColMatRef::new(symbolic, matrix.val());
                let selected: Vec<Vec<_>> = dense.iter().map(|row| row[1..3].to_vec()).collect();
                check_csc_view(&view, &selected);
                assert_eq!(view.sparse_column(0), (&[][..], &[][..]));
                let (indices, values) = view.sparse_column(1);
                assert_eq!(indices, &[2, 0, 2, 1]);
                assert_eq!(indices.as_ptr(), matrix.row_idx()[8..].as_ptr());
                assert_eq!(values.as_ptr(), matrix.val()[8..].as_ptr());
                assert_eq!(view.raw_column(1).row_indices().as_ptr(), indices.as_ptr());
                assert_eq!(view.raw_column(1).values().as_ptr(), values.as_ptr());
                let mut storage = matrix.val().to_vec();
                let view = SparseColMatMut::new(symbolic, &mut storage);
                check_csc_view(&view, &selected);
                for (before, after) in matrix.val().iter().zip(storage) {
                    assert_eq!(before.to_bits(), after.to_bits());
                }
            }

            #[test]
            fn csc_view_statistics_cover_nonfinite_values_and_large_offsets() {
                use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
                for values in [
                    [5e11, 1e12 + 1.0, 5e11 + 2.0, 1e12 + 3.0],
                    [f64::INFINITY, 1.0, f64::NEG_INFINITY, 0.0],
                    [f64::NAN, 1.0, 2.0, 0.0],
                ] {
                    let symbolic = SymbolicSparseColMatRef::new_unsorted_checked(
                        3,
                        1,
                        &[0, 4],
                        None,
                        &[2, 0, 2, 1],
                    );
                    let view = SparseColMatRef::new(symbolic, &values);
                    let dense = Mat::from_fn(3, 1, |i, _| {
                        [values[1], values[3], values[0] + values[2]][i]
                    });
                    common::check_column_stats_against(&view, &dense);
                }
                let sparse = SparseColMat::try_new_from_triplets(
                    2,
                    1,
                    &[Triplet::new(0, 0, 1.0_f32), Triplet::new(1, 0, 3.0)],
                )
                .unwrap();
                common::check_column_stats_f32(&sparse.as_ref());
                common::check_fused_f32(&sparse.as_ref(), |values| {
                    Col::from_fn(values.len(), |i| values[i])
                });
            }

            #[test]
            fn batched_products() {
                use lazymatrix::{MatMat, MatrixRead};
                common::batched::run_batched_suite(build);
                common::batched::run_batched_suite(build_csr);
                common::batched::run_batched_suite(build_dense);
                let panels = common::random_matrix(74, 131, 129, 0.05);
                common::batched::check_batched(&build_dense(&panels), &panels.dense);
                common::batched::check_batched(&build(&panels), &panels.dense);
                let tm = common::random_matrix(73, 7, 3, 0.5);
                let mut matrix = build_csr(&tm);
                common::batched::check_batched(&matrix.as_ref(), &tm.dense);
                common::batched::check_batched(&matrix.rb_mut(), &tm.dense);
                let matrix = build(&tm);
                let lazy = LazyMatrix::from_parts(&matrix, Some(vec![1.; 3]), Some(vec![-2.; 3]));
                let eager = lazy.to_eager::<Mat<f64>>().unwrap();
                common::batched::check_batched(
                    &eager,
                    &common::materialize(&tm.dense, lazy.centers(), lazy.scales()),
                );
                let rhs = Mat::from_fn(3, 2, |i, j| (i + j) as f64);
                let native: Mat<f64> = lazy.matmat(&rhs.as_ref()).unwrap();
                let independent: common::MaterializeOutput = lazy.matmat(&rhs.as_ref()).unwrap();
                for i in 0..7 {
                    for j in 0..2 {
                        assert_eq!(native[(i, j)], independent.get(i, j));
                    }
                }
                let mut dense = Mat::from_fn(6, 6, |i, j| 1e12 + (i * 6 + j) as f64);
                let raw = dense.as_ref().submatrix(1, 1, 3, 2).transpose();
                let values: Vec<Vec<_>> = (0..2)
                    .map(|i| (0..3).map(|j| raw[(i, j)] - 1e12).collect())
                    .collect();
                let centered = LazyMatrix::with_centers(raw, vec![1e12; 3]);
                common::batched::check_batched(&centered, &values);
                let rhs = Mat::from_fn(3, 2, |i, j| (i + j) as f64);
                let expected: Mat<f64> = centered.matmat(&rhs.as_ref().reverse_cols()).unwrap();
                let mut storage = Mat::full(4, 4, 99.);
                lazymatrix::MatMatInto::matmat_into(
                    &centered,
                    &rhs.as_ref().reverse_cols(),
                    &mut storage.as_mut().submatrix_mut(1, 1, 2, 2),
                )
                .unwrap();
                for i in 0..4 {
                    for j in 0..4 {
                        assert_eq!(
                            storage[(i, j)],
                            if (1..3).contains(&i) && (1..3).contains(&j) {
                                expected[(i - 1, j - 1)]
                            } else {
                                99.
                            }
                        );
                    }
                }
                let values: Vec<Vec<_>> = (0..6)
                    .map(|i| (0..6).map(|j| dense[(i, j)]).collect())
                    .collect();
                common::batched::check_batched(&dense.as_mut(), &values);
                let matrix = Mat::from_fn(2, 1, |_, _| 1e38_f32);
                common::batched::check_batched_f32(&matrix, |v| Col::from_fn(v.len(), |i| v[i]));
                let nonfinite =
                    Mat::from_fn(2, 1, |i, _| if i == 0 { f32::NAN } else { f32::INFINITY });
                common::batched::check_batched_f32(&nonfinite, |v| Col::from_fn(v.len(), |i| v[i]));
                let sparse = SparseColMat::try_new_from_triplets(
                    2,
                    1,
                    &[
                        Triplet::new(0_usize, 0_usize, 1e38_f32),
                        Triplet::new(1, 0, 1e38_f32),
                    ],
                )
                .unwrap();
                common::batched::check_batched_f32(&sparse, |v| Col::from_fn(v.len(), |i| v[i]));
            }

            fn build_csr(tm: &TestMatrix) -> SparseRowMat<usize, f64> {
                let triplets: Vec<_> = tm
                    .triplets
                    .iter()
                    .map(|&(i, j, value)| Triplet::new(i, j, value))
                    .collect();
                SparseRowMat::try_new_from_triplets(tm.nrows, tm.ncols, &triplets).unwrap()
            }

            #[test]
            fn csr_backend_suite() {
                common::run_submatrix_suite(build_csr, to_col);
                common::run_backend_suite(build_csr, to_col, from_col);
                common::run_fused_suite(build_csr, to_col);
                common::run_materialization_suite(build_csr);
                common::run_eager_product_suite::<_, Mat<f64>, _>(build_csr, to_col, from_col);
                let tm = common::random_matrix(192, 7, 3, 0.4);
                let mut matrix = build_csr(&tm);
                common::check_fused_operator(&matrix.as_ref(), &tm.dense, 3, &to_col);
                common::check_fused_operator(&matrix.rb_mut(), &tm.dense, 3, &to_col);
                let mut matrix = SparseRowMat::try_new_from_triplets(
                    2,
                    1,
                    &[
                        Triplet::new(0_usize, 0_usize, 1.0_f32),
                        Triplet::new(1, 0, 3.0_f32),
                    ],
                )
                .unwrap();
                common::check_fused_f32(&matrix, |v| Col::from_fn(v.len(), |i| v[i]));
                common::check_column_stats_f32(&matrix);
                common::check_column_stats_f32(&matrix.as_ref());
                common::check_column_stats_f32(&matrix.rb_mut());
                common::batched::check_batched_f32(&matrix, |v| Col::from_fn(v.len(), |i| v[i]));
                common::batched::check_batched_f32(&matrix.as_ref(), |v| {
                    Col::from_fn(v.len(), |i| v[i])
                });
                common::check_fused_f32(&matrix.as_ref(), |v| Col::from_fn(v.len(), |i| v[i]));
                common::check_fused_f32(&matrix.rb_mut(), |v| Col::from_fn(v.len(), |i| v[i]));
            }

            #[test]
            fn csr_statistics_combine_duplicates_and_exclude_spare_capacity() {
                use lazymatrix::{ColumnStats, MatTransposeVec, MatVec};
                let symbolic = SymbolicSparseRowMat::new_unsorted_checked(
                    3,
                    2,
                    vec![0, 4, 5, 7],
                    Some(vec![3, 0, 1]),
                    vec![1, 0, 1, 99, 99, 0, 99],
                );
                let mut matrix =
                    SparseRowMat::new(symbolic, vec![2.0, 0.0, -3.0, 99.0, 99.0, 4.0, 99.0]);
                common::batched::check_batched(
                    &matrix,
                    &[vec![0., -1.], vec![0., 0.], vec![4., 0.]],
                );
                let check = |matrix: &dyn ColumnStats<f64, Error = std::convert::Infallible>| {
                    common::assert_close(
                        &matrix.col_means().unwrap(),
                        &[4.0 / 3.0, -1.0 / 3.0],
                        1e-10,
                    );
                    common::assert_close(&matrix.col_mins().unwrap(), &[0.0, -1.0], 1e-10);
                    common::assert_close(&matrix.col_ranges().unwrap(), &[4.0, 1.0], 1e-10);
                    common::assert_close(&matrix.col_medians().unwrap(), &[0.0, 0.0], 1e-10);
                    common::assert_close(
                        &matrix
                            .col_quantile_ranges(lazymatrix::QuantileRange::default())
                            .unwrap(),
                        &[2.0, 0.5],
                        1e-10,
                    );
                    common::assert_close(&matrix.col_l1().unwrap(), &[4.0, 1.0], 1e-10);
                    common::assert_close(&matrix.col_l2().unwrap(), &[4.0, 1.0], 1e-10);
                    common::assert_close(&matrix.col_maxabs().unwrap(), &[4.0, 1.0], 1e-10);
                    common::assert_close(
                        &matrix.col_sds(0).unwrap(),
                        &[(32.0_f64 / 9.0).sqrt(), (2.0_f64 / 9.0).sqrt()],
                        1e-10,
                    );
                    common::assert_close(
                        &matrix.col_sds(1).unwrap(),
                        &[(16.0_f64 / 3.0).sqrt(), (1.0_f64 / 3.0).sqrt()],
                        1e-10,
                    );
                    common::assert_close(
                        &matrix.col_l1_centered(&[1.0, 2.0]).unwrap(),
                        &[5.0, 7.0],
                        1e-10,
                    );
                    common::assert_close(
                        &matrix.col_l2_centered(&[1.0, 2.0]).unwrap(),
                        &[11.0_f64.sqrt(), 17.0_f64.sqrt()],
                        1e-10,
                    );
                    common::assert_close(
                        &matrix.col_maxabs_centered(&[1.0, 2.0]).unwrap(),
                        &[3.0, 3.0],
                        1e-10,
                    );
                };
                check(&matrix);
                check(&matrix.as_ref());
                check(&matrix.rb_mut());
                common::assert_close(
                    &from_col(&matrix.matvec(&to_col(&[2.0, 3.0])).unwrap()),
                    &[-3.0, 0.0, 8.0],
                    1e-10,
                );
                common::assert_close(
                    &from_col(&matrix.mat_transpose_vec(&to_col(&[2.0, 3.0, 4.0])).unwrap()),
                    &[16.0, -2.0],
                    1e-10,
                );
            }

            fn from_col(c: &Col<f64>) -> Vec<f64> {
                (0..c.nrows()).map(|i| c[i]).collect()
            }

            fn build_dense(tm: &TestMatrix) -> Mat<f64> {
                Mat::from_fn(tm.nrows, tm.ncols, |i, j| tm.dense[i][j])
            }

            #[test]
            fn eager_conversion_combines_duplicates_and_ignores_spare_capacity() {
                use lazymatrix::{ColumnStats, MatVec, MaterializeDense, MatrixOwned};
                let symbolic = faer::sparse::SymbolicSparseColMat::new_unsorted_checked(
                    3,
                    1,
                    vec![0, 4],
                    Some(vec![3]),
                    vec![2, 0, 2, 99],
                );
                let matrix = SparseColMat::new(symbolic, vec![0.25, -0.0, 0.75, 99.0]);
                common::assert_close(&matrix.col_medians().unwrap(), &[0.0], 0.0);
                common::assert_close(
                    &matrix
                        .col_quantile_ranges(lazymatrix::QuantileRange::default())
                        .unwrap(),
                    &[0.5],
                    0.0,
                );
                let raw = LazyMatrix::<_, f64>::from_parts(&matrix, None, None)
                    .to_eager::<Mat<f64>>()
                    .unwrap();
                assert_eq!(raw.data()[(0, 0)].to_bits(), (-0.0_f64).to_bits());
                let lazy = LazyMatrix::from_parts(&matrix, Some(vec![1.0]), Some(vec![-2.0]));
                let eager = lazy.to_eager::<Mat<f64>>().unwrap();
                assert_eq!(
                    eager.matvec(&Col::from_fn(1, |_| 1.0)).unwrap().as_ref()[0],
                    0.5
                );
                assert_eq!(eager.data()[(2, 0)], -0.0);
                let symbolic = SymbolicSparseRowMat::new_unsorted_checked(
                    1,
                    3,
                    vec![0, 4],
                    Some(vec![3]),
                    vec![2, 0, 2, 99],
                );
                let matrix = SparseRowMat::new(symbolic, vec![0.25_f32, -0.0, 0.75, 99.0]);
                common::check_materialize_f32(&matrix, &[-0.0, 0.0, 1.0]);
                common::check_materialize_f32(&matrix.as_ref(), &[-0.0, 0.0, 1.0]);
                let mut out = common::MaterializeOutput::zeros(1, 3);
                matrix
                    .materialize_normalized_into(None, None, &mut out)
                    .unwrap();
                assert_eq!(out.values[0].to_bits(), (-0.0_f32).to_bits());
            }

            #[test]
            fn eager_dense_conversion_supports_mutable_views_and_reuses_allocation() {
                use lazymatrix::MatVec;
                let dense = Mat::from_fn(2, 2, |i, j| (2 * i + j + 1) as f64);
                let pointer = dense.as_ref().col(0).as_ptr();
                let eager = LazyMatrix::with_centers(dense, vec![1.0, 2.0]).into_eager();
                assert_eq!(eager.data().as_ref().col(0).as_ptr(), pointer);
                let mut storage = Mat::full(4, 4, 99.0);
                {
                    let mut view = storage.as_mut().submatrix_mut(1, 1, 2, 2);
                    let lazy = LazyMatrix::with_centers(eager.data(), vec![1.0, 1.0]);
                    let borrowed = lazy.to_eager_into(&mut view).unwrap();
                    assert_eq!(borrowed.matvec(&Col::from_fn(2, |_| 1.0)).unwrap()[0], -2.0);
                    let inplace = LazyMatrix::with_scales(view, vec![2.0, 2.0]).into_eager();
                    assert_eq!(inplace.matvec(&Col::from_fn(2, |_| 1.0)).unwrap()[0], -1.0);
                }
                assert_eq!(storage[(0, 0)], 99.0);
                assert_eq!(storage[(3, 3)], 99.0);
                let dense = Mat::from_fn(2, 2, |i, j| (2 * i + j) as f32);
                common::check_materialize_f32(&dense, &[0.0, 1.0, 2.0, 3.0]);
            }

            #[test]
            fn fused_products_support_f32() {
                let matrix = Mat::from_fn(2, 1, |i, _| (2 * i + 1) as f32);
                common::check_column_stats_f32(&matrix);
                let to_v = |v: &[f32]| Col::from_fn(v.len(), |i| v[i]);
                common::check_fused_f32(&matrix, to_v);
                let sparse = SparseColMat::try_new_from_triplets(
                    2,
                    1,
                    &[
                        Triplet::new(0_usize, 0_usize, 1.0_f32),
                        Triplet::new(1, 0, 3.0_f32),
                    ],
                )
                .unwrap();
                common::check_fused_f32(&sparse, to_v);
                common::check_column_stats_f32(&sparse);
                common::run_dense_centered_f32_suite(
                    |dense| Mat::from_fn(dense.len(), dense[0].len(), |i, j| dense[i][j]),
                    to_v,
                );
            }

            #[test]
            fn centered_products_preserve_variation_through_dense_views() {
                let mut storage = Mat::from_fn(3, 7, |i, j| 1e12 + (i * 7 + j) as f64 / 4.0);
                let dense: Vec<Vec<_>> = (0..7)
                    .map(|i| (0..3).map(|j| storage[(j, i)]).collect())
                    .collect();
                for scales in [None, Some(vec![1.3, -2.7, 0.75])] {
                    common::check_dense_centered_products(
                        &storage.as_ref().transpose(),
                        &dense,
                        vec![1e12; 3],
                        scales.clone(),
                        &[0.3, -0.1, 0.7],
                        &[0.25; 7],
                        to_col,
                    );
                    common::check_dense_centered_products(
                        &storage.as_mut().transpose_mut(),
                        &dense,
                        vec![1e12; 3],
                        scales,
                        &[0.3, -0.1, 0.7],
                        &[0.25; 7],
                        to_col,
                    );
                }
            }

            #[test]
            fn faer_backend_suite() {
                common::run_submatrix_suite(build, to_col);
                common::run_submatrix_suite(build_dense, to_col);
                common::run_materialization_suite(build);
                common::run_materialization_suite(build_dense);
                common::run_eager_product_suite::<_, Mat<f64>, _>(build, to_col, from_col);
                common::run_eager_product_suite::<_, Mat<f64>, _>(build_dense, to_col, from_col);
                common::run_fused_suite(build, to_col);
                common::run_fused_suite(build_dense, to_col);
                common::run_dense_centered_product_suite(build_dense, to_col);
                common::run_gram_suite(build);
                common::run_gram_suite(build_dense);
                common::run_backend_suite(build, to_col, from_col);
                common::run_sparse_columns_suite(build);
                common::run_logical_columns_suite(build);
                common::run_backend_suite(build_dense, to_col, from_col);
                common::run_logical_columns_suite(build_dense);
            }

            #[test]
            fn dense_gram_views_preserve_strides_and_panel_edges() {
                use lazymatrix::{WeightedColumnSumsInto, WeightedGramInto, WithIntercept};
                let mut storage = Mat::from_fn(39, 302, |i, j| 1e12 + ((i * 3 + j) % 13) as f64);
                let design = storage
                    .as_ref()
                    .submatrix(1, 1, 37, 300)
                    .transpose()
                    .reverse_rows()
                    .reverse_cols();
                let dense: Vec<Vec<_>> = (0..300)
                    .map(|i| (0..37).map(|j| design[(i, j)]).collect())
                    .collect();
                let weight_storage =
                    Mat::from_fn(
                        2,
                        300,
                        |i, j| {
                            if i == 0 { 99.0 } else { (j % 7) as f64 - 3.0 }
                        },
                    );
                let weights = weight_storage.row(1).transpose().reverse_rows();
                let lazy =
                    LazyMatrix::from_parts(design, Some(vec![1e12; 37]), Some(vec![-2.0; 37]));
                let mut output = Mat::full(39, 39, -99.0);
                {
                    let mut view = output
                        .as_mut()
                        .submatrix_mut(1, 1, 37, 37)
                        .transpose_mut()
                        .reverse_rows_mut()
                        .reverse_cols_mut();
                    lazy.weighted_gram_into(&weights, &mut view).unwrap();
                    let actual = common::GramOutput(
                        (0..37)
                            .map(|i| (0..37).map(|j| view[(i, j)]).collect())
                            .collect(),
                    );
                    common::assert_gram(
                        &actual,
                        &common::materialize(&dense, lazy.centers(), lazy.scales()),
                        &(0..300).map(|i| weights[i]).collect::<Vec<_>>(),
                    );
                }
                for i in 0..39 {
                    for j in 0..39 {
                        if i == 0 || j == 0 || i == 38 || j == 38 {
                            assert_eq!(output[(i, j)], -99.0);
                        }
                    }
                }
                let mut sums = Mat::full(2, 37, 99.0);
                lazy.weighted_column_sums_into(
                    &weights,
                    &mut sums.row_mut(1).transpose_mut().reverse_rows_mut(),
                )
                .unwrap();
                let normalized = common::materialize(&dense, lazy.centers(), lazy.scales());
                for j in 0..37 {
                    let expected: f64 = (0..300).map(|i| normalized[i][j] * weights[i]).sum();
                    assert_eq!(sums[(1, 36 - j)], expected);
                    assert_eq!(sums[(0, j)], 99.0);
                }
                let mut expected = common::GramOutput(vec![vec![f64::NAN; 38]; 38]);
                WithIntercept::new(&lazy)
                    .weighted_gram_into(&weights, &mut expected)
                    .unwrap();
                let design = storage
                    .as_mut()
                    .submatrix_mut(1, 1, 37, 300)
                    .transpose_mut()
                    .reverse_rows_mut()
                    .reverse_cols_mut();
                let lazy =
                    LazyMatrix::from_parts(design, Some(vec![1e12; 37]), Some(vec![-2.0; 37]));
                let mut actual = common::GramOutput(vec![vec![f64::NAN; 38]; 38]);
                WithIntercept::new(&lazy)
                    .weighted_gram_into(&weights, &mut actual)
                    .unwrap();
                assert_eq!(actual.0, expected.0);
                let mut actual_sums = [f64::NAN; 37];
                lazy.weighted_column_sums_into(&weights, &mut actual_sums)
                    .unwrap();
                for j in 0..37 {
                    assert_eq!(actual_sums[j], sums[(1, 36 - j)]);
                }
                for i in 0..39 {
                    for j in 0..302 {
                        assert_eq!(storage[(i, j)], 1e12 + ((i * 3 + j) % 13) as f64);
                    }
                }
            }

            #[test]
            fn dense_weighted_products_support_f32_and_eager_views() {
                use lazymatrix::{WeightedColumnSumsInto, WeightedGramInto, WithIntercept};
                let dense = Mat::from_fn(2, 2, |i, j| [[1.0_f32, 0.0], [2.0, 3.0]][i][j]);
                let mut raw = Mat::full(2, 2, f32::NAN);
                dense
                    .weighted_gram_into(&[2.0_f32, -1.0], &mut raw)
                    .unwrap();
                assert_eq!(
                    raw,
                    Mat::from_fn(2, 2, |i, j| [[-2.0_f32, -6.0], [-6.0, -9.0]][i][j])
                );
                let lazy =
                    LazyMatrix::from_parts(dense, Some(vec![0.5, 1.0]), Some(vec![-2.0, 4.0]));
                let eager = lazy.to_eager::<Mat<f32>>().unwrap();
                let mut expected = Mat::full(3, 3, f32::NAN);
                WithIntercept::new(&lazy)
                    .weighted_gram_into(&[2.0_f32, -1.0], &mut expected)
                    .unwrap();
                let mut actual = Mat::full(3, 3, f32::NAN);
                WithIntercept::new(&eager)
                    .weighted_gram_into(&[2.0_f32, -1.0], &mut actual)
                    .unwrap();
                assert_eq!(actual, expected);
                let mut storage = Mat::full(4, 4, 99.0_f32);
                let mut view = storage.as_mut().submatrix_mut(1, 1, 2, 2);
                let eager = lazy.to_eager_into(&mut view).unwrap();
                WithIntercept::new(&eager)
                    .weighted_gram_into(&[2.0_f32, -1.0], &mut actual)
                    .unwrap();
                assert_eq!(actual, expected);
                let mut sums = [f32::NAN; 2];
                eager
                    .weighted_column_sums_into(&[2.0_f32, -1.0], &mut sums)
                    .unwrap();
                assert_eq!(sums, [expected[(0, 1)], expected[(0, 2)]]);
            }

            #[test]
            fn faer_weighted_norms_combine_unsorted_duplicates() {
                use lazymatrix::{MatTransposeVec, MatVec};
                let symbolic = faer::sparse::SymbolicSparseColMat::new_unsorted_checked(
                    3,
                    1,
                    vec![0, 4],
                    Some(vec![3]),
                    vec![2, 0, 2, 99],
                );
                let matrix = SparseColMat::new(symbolic, vec![0.25, 1.0, 0.75, 99.0]);
                let lazy = LazyMatrix::from_parts(&matrix, Some(vec![1.0]), Some(vec![-2.0]));
                let restricted = lazy.submatrix(0..3, 0..1);
                let column = restricted.sparse_column(0);
                assert_eq!(
                    column.stored_entries().collect::<Vec<_>>(),
                    vec![(2, 0.25), (0, 1.0), (2, 0.75)]
                );
                approx::assert_abs_diff_eq!(column.norm_squared(), 0.25);
                let eager = restricted.to_eager::<Mat<f64>>().unwrap();
                assert_eq!(eager.data()[(1, 0)], 0.5);
                assert_eq!(restricted.matvec(&vec![2.0]).unwrap(), vec![0.0, 1.0, 0.0]);
                assert_eq!(
                    restricted.mat_transpose_vec(&vec![1.0, 2.0, 3.0]).unwrap(),
                    vec![1.0]
                );
                let weights = [1e16, 1.0, 1e16];
                for column in [lazy.column(0), lazy.sparse_column(0)] {
                    approx::assert_abs_diff_eq!(column.weighted_norm_squared(&weights), 0.25);
                    approx::assert_abs_diff_eq!(
                        column.weighted_norm_squared_with_sum(&weights, weights.iter().sum()),
                        0.25
                    );
                }
            }

            #[test]
            fn faer_gram_uses_occupied_columns_and_combines_duplicates() {
                use lazymatrix::{SparseColumns, WeightedGramInto};
                let symbolic = faer::sparse::SymbolicSparseColMat::new_unsorted_checked(
                    3,
                    2,
                    vec![0, 4, 6],
                    Some(vec![3, 1]),
                    vec![2, 0, 2, 99, 1, 99],
                );
                let matrix = SparseColMat::new(symbolic, vec![1.0, 0.0, -2.0, 99.0, 4.0, 99.0]);
                assert_eq!(
                    matrix.sparse_column(0),
                    (&[2, 0, 2][..], &[1.0, 0.0, -2.0][..])
                );
                let lazy =
                    LazyMatrix::from_parts(&matrix, Some(vec![1.0, 2.0]), Some(vec![2.0, -1.0]));
                let mut out = Mat::full(2, 2, f64::NAN);
                lazy.weighted_gram_into(&[0.5, 2.0, -1.0], &mut out.as_mut())
                    .unwrap();
                let actual = common::GramOutput(
                    (0..2)
                        .map(|i| (0..2).map(|j| out[(i, j)]).collect())
                        .collect(),
                );
                let dense = vec![vec![0.0, 0.0], vec![0.0, 4.0], vec![-1.0, 0.0]];
                common::check_submatrix_weighted(&matrix, &dense);
                common::assert_gram(
                    &actual,
                    &common::materialize(&dense, lazy.centers(), lazy.scales()),
                    &[0.5, 2.0, -1.0],
                );
                let augmented = lazymatrix::WithIntercept::new(&lazy);
                for weights in [[0.5, 2.0, -1.0], [f64::INFINITY, 0.0, 1.0]] {
                    let mut gram = Mat::full(3, 3, f64::NAN);
                    augmented.weighted_gram_into(&weights, &mut gram).unwrap();
                    let actual = common::GramOutput(
                        (0..3)
                            .map(|i| (0..3).map(|j| gram[(i, j)]).collect())
                            .collect(),
                    );
                    let normalized = common::materialize(&dense, lazy.centers(), lazy.scales());
                    let normalized: Vec<Vec<_>> = normalized
                        .iter()
                        .map(|r| std::iter::once(1.0).chain(r.iter().copied()).collect())
                        .collect();
                    common::assert_gram(&actual, &normalized, &weights);
                }
                let mut owned = Mat::zeros(2, 2);
                lazy.weighted_gram_into(&[0.5, 2.0, -1.0], &mut owned)
                    .unwrap();
                assert_eq!(owned, out);
            }

            #[test]
            fn faer_sparse_rows_suite() {
                common::run_sparse_rows_suite(|tm| {
                    let triplets: Vec<_> = tm
                        .triplets
                        .iter()
                        .map(|&(i, j, value)| Triplet::new(i, j, value))
                        .collect();
                    SparseRowMat::try_new_from_triplets(tm.nrows, tm.ncols, &triplets).unwrap()
                });
            }

            #[test]
            fn faer_sparse_rows_borrow_only_occupied_storage() {
                let symbolic = SymbolicSparseRowMat::new_checked(
                    3,
                    4,
                    vec![0, 3, 5, 7],
                    Some(vec![2, 0, 1]),
                    vec![0, 2, 99, 99, 99, 1, 99],
                );
                let mut matrix =
                    SparseRowMat::new(symbolic, vec![1.0_f32, 0.0, 99.0, 99.0, 99.0, -2.0, 99.0]);
                let indices_ptr = matrix.col_idx().as_ptr();
                let values_ptr = matrix.val().as_ptr();
                let check = |rows: &dyn SparseRows<f32>| {
                    assert_eq!(rows.nrows(), 3);
                    assert_eq!(rows.ncols(), 4);
                    let (indices, values) = rows.sparse_row(0);
                    assert_eq!(indices, &[0, 2]);
                    assert_eq!(values, &[1.0, 0.0]);
                    assert_eq!(indices.as_ptr(), indices_ptr);
                    assert_eq!(values.as_ptr(), values_ptr);
                    assert_eq!(rows.sparse_row(1), (&[][..], &[][..]));
                    assert_eq!(rows.sparse_row(2), (&[1][..], &[-2.0][..]));
                    let lazy =
                        LazyMatrix::from_parts(rows, Some(vec![0.5; 4]), Some(vec![-2.0; 4]));
                    let row = lazy.row(0);
                    assert_eq!(row.column_indices().as_ptr(), indices_ptr);
                    assert_eq!(row.values().as_ptr(), values_ptr);
                    assert_eq!(row.values(), &[1.0, 0.0]);
                    assert_eq!(
                        row.stored_corrections().collect::<Vec<_>>(),
                        vec![(0, -0.5), (2, -0.0)]
                    );
                    assert!(lazy.row(1).values().is_empty());
                    assert_eq!(lazy.row(1).implicit_value(3), 0.25);
                };
                check(&matrix);
                check(&matrix.as_ref());
                check(&matrix.rb_mut());

                let csc = build(&common::random_matrix(76, 5, 3, 0.5));
                let transposed = csc.as_ref().transpose();
                let lazy = LazyMatrix::from_parts(transposed, None, None);
                assert_eq!(MatrixShape::nrows(&transposed), 3);
                assert_eq!(MatrixShape::ncols(&transposed), 5);
                for i in 0..3 {
                    let (indices, values) = transposed.sparse_row(i);
                    let range = csc.col_range(i);
                    assert_eq!(indices, &csc.row_idx()[range.clone()]);
                    assert_eq!(indices.as_ptr(), csc.row_idx()[range.clone()].as_ptr());
                    assert_eq!(values.as_ptr(), csc.val()[range].as_ptr());
                    assert_eq!(lazy.row(i).column_indices().as_ptr(), indices.as_ptr());
                    assert_eq!(lazy.row(i).values().as_ptr(), values.as_ptr());
                }
            }

            #[test]
            fn faer_sparse_rows_preserve_unsorted_entries() {
                let symbolic = SymbolicSparseRowMat::new_unsorted_checked(
                    1,
                    3,
                    vec![0, 3],
                    None,
                    vec![2, 0, 2],
                );
                let matrix = SparseRowMat::new(symbolic, vec![1.0, 0.0, -2.0]);
                let (columns, values) = matrix.sparse_row(0);
                assert_eq!(columns, &[2, 0, 2]);
                assert_eq!(values, &[1.0, 0.0, -2.0]);
                assert_eq!(columns.as_ptr(), matrix.col_idx().as_ptr());
                assert_eq!(values.as_ptr(), matrix.val().as_ptr());
                let lazy = LazyMatrix::from_parts(
                    matrix,
                    Some(vec![0.5, -1.0, 2.0]),
                    Some(vec![2.0, -4.0, 0.5]),
                );
                let row = lazy.row(0);
                assert_eq!(row.column_indices(), &[2, 0, 2]);
                assert_eq!(
                    row.stored_corrections().collect::<Vec<_>>(),
                    vec![(2, 2.0), (0, 0.0), (2, -4.0)]
                );
            }

            #[test]
            fn faer_strided_views_are_borrowed() {
                let design_storage = Mat::from_fn(2, 4, |i, j| (i * 4 + j + 1) as f64);
                let design = design_storage.as_ref().transpose();
                common::check_fused_operator(
                    &design,
                    &(0..4)
                        .map(|i| (0..2).map(|j| design[(i, j)]).collect())
                        .collect::<Vec<Vec<f64>>>(),
                    2,
                    &to_col,
                );
                let lazy = LazyMatrix::new(
                    design,
                    ColumnNormalization::new(Centering::Mean, Scaling::L2),
                )
                .unwrap();

                let vector_storage = Mat::from_fn(2, 4, |i, j| (i + j + 1) as f64);
                let vector = vector_storage.row(1).transpose();
                let column = lazy.column(0);
                let expected_dot = (0..4)
                    .map(|i| {
                        let raw = design_storage[(0, i)];
                        let center = 2.5;
                        let scale = 5.0_f64.sqrt();
                        (raw - center) / scale * vector[i]
                    })
                    .sum::<f64>();
                approx::assert_abs_diff_eq!(column.dot(&vector), expected_dot, epsilon = 1e-12);

                let mut destination_storage = Mat::zeros(2, 4);
                let mut destination = destination_storage.row_mut(1).transpose_mut();
                column.scaled_add_to(0.5, &mut destination);
                for i in 0..4 {
                    let expected = 0.5 * (design_storage[(0, i)] - 2.5) / 5.0_f64.sqrt();
                    approx::assert_abs_diff_eq!(destination[i], expected, epsilon = 1e-12);
                }
            }
        }
    };
}

#[cfg(feature = "faer_v0_22")]
backend_suite!(v0_22, faer_0_22);

#[cfg(feature = "faer_v0_23")]
backend_suite!(v0_23, faer_0_23);

#[cfg(feature = "faer_v0_24")]
backend_suite!(v0_24, faer_0_24);
