#![cfg(feature = "nalgebra_all")]
//! Verification of the nalgebra sparse backend against the dense oracle.

#[path = "common/backend_aliases.rs"]
mod backend_aliases;

#[path = "common/runner.rs"]
mod common;

macro_rules! backend_suite {
    ($name:ident, $backend:ident, $sparse:ident) => {
        mod $name {
            use crate::backend_aliases::$backend as nalgebra;
            use crate::backend_aliases::$sparse as nalgebra_sparse;
            use crate::common;
            use common::TestMatrix;
            use lazymatrix::{Centering, ColumnNormalization, LazyMatrix, Scaling, SparseRows};
            use nalgebra::{DMatrix, DMatrixView, DVector, DVectorView, DVectorViewMut, Dyn};
            use nalgebra_sparse::{CooMatrix, CscMatrix, CsrMatrix};

            fn build(tm: &TestMatrix) -> CscMatrix<f64> {
                let mut coo = CooMatrix::new(tm.nrows, tm.ncols);
                for &(r, c, v) in &tm.triplets {
                    coo.push(r, c, v);
                }
                CscMatrix::from(&coo)
            }

            fn to_dvec(v: &[f64]) -> DVector<f64> {
                DVector::from_column_slice(v)
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
                let matrix = build(&tm);
                let lazy =
                    LazyMatrix::from_column_parts(&matrix, Some(vec![1.; 3]), Some(vec![-2.; 3]));
                let eager = lazy.to_eager::<DMatrix<f64>>().unwrap();
                common::batched::check_batched(
                    &eager,
                    &common::materialize(
                        &tm.dense,
                        lazy.normalization().centers(),
                        lazy.normalization().scales(),
                    ),
                );
                let rhs = DMatrix::from_fn(3, 2, |i, j| (i + j) as f64);
                let view: DMatrixView<'_, f64> = rhs.as_view();
                let native: DMatrix<f64> = lazy.matmat(&view).unwrap();
                let independent: common::MaterializeOutput = lazy.matmat(&view).unwrap();
                for i in 0..7 {
                    for j in 0..2 {
                        assert_eq!(native[(i, j)], independent.get(i, j));
                    }
                }
                let dense = DMatrix::from_fn(6, 6, |i, j| 1e12 + (i * 6 + j) as f64);
                let raw = dense.view_with_steps((0, 0), (3, 2), (1, 2));
                let values: Vec<Vec<_>> = (0..3)
                    .map(|i| (0..2).map(|j| raw[(i, j)] - 1e12).collect())
                    .collect();
                let centered = LazyMatrix::from_column_parts(raw, Some(vec![1e12; 2]), None);
                common::batched::check_batched(&centered, &values);
                let rhs = DMatrix::from_fn(4, 4, |i, j| (i + j) as f64);
                let input = rhs.view_with_steps((0, 0), (2, 2), (1, 1));
                let expected: DMatrix<f64> = centered.matmat(&input).unwrap();
                let mut storage = DMatrix::from_element(6, 4, 99.);
                lazymatrix::MatMatInto::matmat_into(
                    &centered,
                    &input,
                    &mut storage.view_with_steps_mut((0, 0), (3, 2), (1, 1)),
                )
                .unwrap();
                for i in 0..6 {
                    for j in 0..4 {
                        assert_eq!(
                            storage[(i, j)],
                            if i % 2 == 0 && j % 2 == 0 {
                                expected[(i / 2, j / 2)]
                            } else {
                                99.
                            }
                        );
                    }
                }
                common::batched::check_batched_f32(
                    &DMatrix::from_element(2, 1, 1e38_f32),
                    DVector::from_column_slice,
                );
                let sparse =
                    CscMatrix::try_from_csc_data(2, 1, vec![0, 2], vec![0, 1], vec![1e38_f32; 2])
                        .unwrap();
                common::batched::check_batched_f32(&sparse, DVector::from_column_slice);
            }

            fn build_csr(tm: &TestMatrix) -> CsrMatrix<f64> {
                let mut coo = CooMatrix::new(tm.nrows, tm.ncols);
                for &(i, j, value) in &tm.triplets {
                    coo.push(i, j, value);
                }
                CsrMatrix::from(&coo)
            }

            #[test]
            fn csr_backend_suite() {
                common::run_submatrix_suite(build_csr, to_dvec);
                common::run_column_pipeline_suite(build_csr, to_dvec);
                common::run_backend_suite(build_csr, to_dvec, from_dvec);
                common::run_fused_suite(build_csr, to_dvec);
                common::run_materialization_suite(build_csr);
                common::run_eager_product_suite::<_, DMatrix<f64>, _>(
                    build_csr, to_dvec, from_dvec,
                );
                let matrix = CsrMatrix::try_from_csr_data(
                    2,
                    1,
                    vec![0, 1, 2],
                    vec![0, 0],
                    vec![1.0_f32, 3.0],
                )
                .unwrap();
                common::check_fused_f32(&matrix, DVector::from_column_slice);
                common::batched::check_batched_f32(&matrix, DVector::from_column_slice);
            }

            fn from_dvec(v: &DVector<f64>) -> Vec<f64> {
                v.as_slice().to_vec()
            }

            fn build_dense(tm: &TestMatrix) -> DMatrix<f64> {
                DMatrix::from_fn(tm.nrows, tm.ncols, |i, j| tm.dense[i][j])
            }

            #[test]
            fn eager_conversion_supports_csr_f32_and_mutable_dense_views() {
                use lazymatrix::MatVec;
                let matrix = CsrMatrix::try_from_csr_data(
                    2,
                    2,
                    vec![0, 1, 2],
                    vec![0, 1],
                    vec![2.0_f32, 4.0],
                )
                .unwrap();
                common::check_materialize_f32(&matrix, &[2.0, 0.0, 0.0, 4.0]);
                let dense = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
                let pointer = dense.as_ptr();
                let eager =
                    LazyMatrix::from_column_parts(dense, Some(vec![1.0, 2.0]), None).into_eager();
                assert_eq!(eager.data().as_ptr(), pointer);
                let mut storage = DMatrix::from_element(4, 4, 99.0);
                {
                    let mut view = storage.view_mut((1, 1), (2, 2));
                    let borrowed =
                        LazyMatrix::from_column_parts(eager.data(), Some(vec![1.0, 1.0]), None)
                            .to_eager_into(&mut view)
                            .unwrap();
                    assert_eq!(
                        borrowed.matvec(&DVector::from_element(2, 1.0)).unwrap()[0],
                        -2.0
                    );
                    let inplace = LazyMatrix::from_column_parts(view, None, Some(vec![2.0, 2.0]))
                        .into_eager();
                    assert_eq!(
                        inplace.matvec(&DVector::from_element(2, 1.0)).unwrap()[0],
                        -1.0
                    );
                }
                assert_eq!(storage[(0, 0)], 99.0);
                assert_eq!(storage[(3, 3)], 99.0);
                common::check_materialize_f32(
                    &DMatrix::from_row_slice(2, 2, &[0.0_f32, 1.0, 2.0, 3.0]),
                    &[0.0, 1.0, 2.0, 3.0],
                );
            }

            #[test]
            fn sparse_products_overwrite_nonfinite_destinations() {
                use lazymatrix::{MatTransposeVecInto, MatVecInto};
                let matrix = build(&common::random_matrix(191, 7, 3, 0.4));
                let x = DVector::from_element(3, 1.0);
                let u = DVector::from_element(7, 1.0);
                for old in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                    let mut forward = DVector::from_element(7, old);
                    let mut transpose = DVector::from_element(3, old);
                    matrix.matvec_into(&x, &mut forward).unwrap();
                    matrix.mat_transpose_vec_into(&u, &mut transpose).unwrap();
                    assert!(forward.iter().all(|x| x.is_finite()));
                    assert!(transpose.iter().all(|x| x.is_finite()));
                }
            }

            #[test]
            fn fused_products_support_f32() {
                let matrix = DMatrix::from_column_slice(2, 1, &[1.0_f32, 3.0]);
                common::check_column_stats_f32(&matrix);
                common::check_fused_f32(&matrix, DVector::from_column_slice);
                let sparse =
                    CscMatrix::try_from_csc_data(2, 1, vec![0, 2], vec![0, 1], vec![1.0_f32, 3.0])
                        .unwrap();
                common::check_fused_f32(&sparse, DVector::from_column_slice);
                common::check_column_stats_f32(&sparse);
                common::check_column_stats_f32(&CsrMatrix::from(&sparse));
                common::run_dense_centered_f32_suite(
                    |dense| DMatrix::from_fn(dense.len(), dense[0].len(), |i, j| dense[i][j]),
                    DVector::from_column_slice,
                );
            }

            #[test]
            fn centered_products_preserve_variation_through_dense_views() {
                let mut storage = DMatrix::from_fn(14, 6, |i, j| 1e12 + (i * 6 + j) as f64 / 4.0);
                let dense: Vec<Vec<_>> = (0..7)
                    .map(|i| (0..3).map(|j| storage[(2 * i, 2 * j)]).collect())
                    .collect();
                for scales in [None, Some(vec![1.3, -2.7, 0.75])] {
                    common::check_dense_centered_products(
                        &storage.view_with_steps((0, 0), (7, 3), (1, 1)),
                        &dense,
                        vec![1e12; 3],
                        scales.clone(),
                        &[0.3, -0.1, 0.7],
                        &[0.25; 7],
                        to_dvec,
                    );
                    common::check_dense_centered_products(
                        &storage.view_with_steps_mut((0, 0), (7, 3), (1, 1)),
                        &dense,
                        vec![1e12; 3],
                        scales,
                        &[0.3, -0.1, 0.7],
                        &[0.25; 7],
                        to_dvec,
                    );
                }
            }

            #[test]
            fn row_normalization() {
                common::run_dense_pipeline_suite(build_dense, to_dvec);
                use lazymatrix::{MatTransposeVec, MatTransposeVecInto, MatVec, MatVecInto};
                common::run_row_normalization_suite(build, to_dvec);
                common::run_row_normalization_suite(build_csr, to_dvec);
                common::run_row_normalization_suite(build_dense, to_dvec);
                let sparse = |raw: &[Vec<f32>]| {
                    let mut coo = CooMatrix::new(raw.len(), raw[0].len());
                    for (i, row) in raw.iter().enumerate() {
                        for (j, &value) in row.iter().enumerate() {
                            if value != 0.0 {
                                coo.push(i, j, value);
                            }
                        }
                    }
                    CscMatrix::from(&coo)
                };
                common::run_row_normalization_f32_suite(sparse, DVector::from_column_slice);
                common::run_row_normalization_f32_suite(
                    |raw| CsrMatrix::from(&sparse(raw)),
                    DVector::from_column_slice,
                );
                common::run_row_normalization_f32_suite(
                    |dense| DMatrix::from_fn(dense.len(), dense[0].len(), |i, j| dense[i][j]),
                    DVector::from_column_slice,
                );
                let mut storage = DMatrix::from_fn(14, 6, |i, j| 1e12 + (i * 6 + j) as f64 / 4.0);
                common::check_row_normalization_view(
                    &storage.view_with_steps((0, 0), (7, 3), (1, 1)),
                    to_dvec,
                );
                common::check_row_normalization_view(
                    &storage.view_with_steps_mut((0, 0), (7, 3), (1, 1)),
                    to_dvec,
                );
                common::check_normalized_column_view(&storage.view_with_steps(
                    (1, 1),
                    (6, 2),
                    (1, 1),
                ));
                common::check_normalized_column_view(&storage.view_with_steps_mut(
                    (1, 1),
                    (6, 2),
                    (1, 1),
                ));
                let normalized = lazymatrix::LazyMatrix::new(
                    storage.view_with_steps((0, 0), (3, 2), (1, 1)),
                    lazymatrix::RowNormalization::standardize(),
                )
                .unwrap();
                let coefficients = to_dvec(&[1.0, 99.0, -2.0, 99.0]);
                let expected = normalized.matvec(&to_dvec(&[1.0, -2.0])).unwrap();
                let mut out = DVector::from_element(6, 99.0);
                normalized
                    .matvec_into(
                        &coefficients.rows_with_step(0, 2, 1),
                        &mut out.rows_with_step_mut(0, 3, 1),
                    )
                    .unwrap();
                for i in 0..3 {
                    assert_eq!(out[2 * i], expected[i]);
                    assert_eq!(out[2 * i + 1], 99.0);
                }
                let observations = to_dvec(&[1.0, 99.0, -2.0, 99.0, 0.5, 99.0]);
                let expected = normalized
                    .mat_transpose_vec(&to_dvec(&[1.0, -2.0, 0.5]))
                    .unwrap();
                let mut out = DVector::from_element(4, 99.0);
                normalized
                    .mat_transpose_vec_into(
                        &observations.rows_with_step(0, 3, 1),
                        &mut out.rows_with_step_mut(0, 2, 1),
                    )
                    .unwrap();
                for i in 0..2 {
                    assert_eq!(out[2 * i], expected[i]);
                    assert_eq!(out[2 * i + 1], 99.0);
                }
            }

            #[test]
            fn nalgebra_backend_suite() {
                common::run_submatrix_suite(build, to_dvec);
                common::run_submatrix_suite(build_dense, to_dvec);
                common::run_materialization_suite(build);
                common::run_materialization_suite(build_dense);
                common::run_eager_product_suite::<_, DMatrix<f64>, _>(build, to_dvec, from_dvec);
                common::run_eager_product_suite::<_, DMatrix<f64>, _>(
                    build_dense,
                    to_dvec,
                    from_dvec,
                );
                common::run_fused_suite(build, to_dvec);
                common::run_fused_suite(build_dense, to_dvec);
                common::run_dense_centered_product_suite(build_dense, to_dvec);
                common::run_gram_suite(build);
                common::run_column_pipeline_suite(build, to_dvec);
                common::run_backend_suite(build, to_dvec, from_dvec);
                common::run_sparse_columns_suite(build);
                common::run_logical_columns_suite(build);
                common::run_column_pipeline_suite(build_dense, to_dvec);
                common::run_backend_suite(build_dense, to_dvec, from_dvec);
                common::run_logical_columns_suite(build_dense);
            }

            #[test]
            fn dense_column_reductions() {
                common::run_column_reductions_suite::<f32, _, _>(
                    |v| DMatrix::from_column_slice(v.len(), 1, v),
                    DVector::from_column_slice,
                );
                common::run_column_reductions_suite::<f64, _, _>(
                    |v| DMatrix::from_column_slice(v.len(), 1, v),
                    DVector::from_column_slice,
                );
            }

            #[test]
            fn vector_reductions() {
                common::run_vector_reductions_suite::<f32, _>(DVector::from_column_slice);
                common::run_vector_reductions_suite::<f64, _>(DVector::from_column_slice);
            }

            #[test]
            fn contiguous_column_and_vector_slices_preserve_view_offsets() {
                use lazymatrix::RawColumns;
                for len in [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65] {
                    let mut storage =
                        DMatrix::from_fn(len + 7, 4, |i, j| (i + j * 100) as f64 / 8.0);
                    let expected: Vec<_> = (3..3 + len).map(|i| storage[(i, 1)]).collect();
                    let ptr = &storage[(3, 1)] as *const f64;
                    let view = storage.view_with_steps((3, 1), (len, 2), (0, 1));
                    let column = view.raw_column(0);
                    common::check_contiguous_column(&column, &expected, ptr);
                    common::check_contiguous_vector(&column, &expected, ptr);

                    let range = storage.column(1);
                    let range = range.rows(3, len);
                    common::check_contiguous_vector(&range, &expected, ptr);
                    let view = storage.view_mut((3, 1), (len, 2));
                    common::check_contiguous_column(&view.raw_column(0), &expected, ptr);
                    let mut range = storage.column_mut(1);
                    let vector = range.rows_mut(3, len);
                    common::check_contiguous_column(&vector, &expected, ptr);
                    common::check_contiguous_vector(&vector, &expected, ptr);

                    let mut owned = DVector::from_column_slice(&expected);
                    let ptr = owned.as_ptr();
                    common::check_contiguous_column(&owned, &expected, ptr);
                    common::check_contiguous_vector(&owned, &expected, ptr);
                    let view: DVectorView<'_, f64> = owned.as_view();
                    common::check_contiguous_vector(&view, &expected, ptr);
                    let view: DVectorViewMut<'_, f64> = owned.as_view_mut();
                    common::check_contiguous_vector(&view, &expected, ptr);
                }
                let fixed = nalgebra::SVector::<f64, 3>::new(1.0, 2.0, 3.0);
                common::check_contiguous_column(&fixed, &[1.0, 2.0, 3.0], fixed.as_ptr());
                common::check_contiguous_vector(&fixed, &[1.0, 2.0, 3.0], fixed.as_ptr());
            }

            #[test]
            fn column_reductions_preserve_strided_logical_order() {
                use lazymatrix::{RawColumn, RawColumns, VectorView};
                let mut storage = DMatrix::from_fn(70, 4, |i, j| (i + j * 3) as f64 / 8.0);
                let contiguous = storage.view_with_steps((3, 1), (33, 1), (0, 1));
                let strided = storage.view_with_steps((3, 1), (33, 1), (1, 0));
                assert!(contiguous.raw_column(0).dense_values().is_some());
                assert!(strided.raw_column(0).dense_values().is_none());
                let vector_storage = DVector::from_fn(70, |i, _| i as f64 / 16.0);
                let vector = vector_storage.rows_with_step(2, 33, 0);
                let strided_vector = vector_storage.rows_with_step(2, 33, 1);
                assert!(vector.contiguous_values().is_some());
                assert!(strided_vector.contiguous_values().is_none());
                for matrix in [contiguous, strided] {
                    let raw: Vec<_> = (0..33).map(|i| matrix[(i, 0)]).collect();
                    for input in [vector, strided_vector] {
                        let values: Vec<_> = (0..33).map(|i| input[i]).collect();
                        common::check_column_reduction_view(&matrix, &input, &raw, &values);
                    }
                }
                let mut vector_storage = vector_storage;
                let raw: Vec<_> = (0..33).map(|i| storage[(i + 3, 1)]).collect();
                let values: Vec<_> = (0..33).map(|i| vector_storage[i + 2]).collect();
                let input = vector_storage.rows_mut(2, 33);
                let matrix = storage.view_mut((3, 1), (33, 1));
                common::check_column_reduction_view(&matrix, &input, &raw, &values);
                let matrix = storage.view_with_steps_mut((3, 1), (33, 1), (1, 0));
                assert!(matrix.raw_column(0).dense_values().is_none());
                let raw: Vec<_> = (0..33).map(|i| matrix[(i, 0)]).collect();
                let input = vector_storage.rows_with_step_mut(2, 33, 1);
                assert!(input.contiguous_values().is_none());
                let values: Vec<_> = (0..33).map(|i| input[i]).collect();
                common::check_column_reduction_view(&matrix, &input, &raw, &values);
            }

            #[test]
            fn nalgebra_gram_writes_owned_and_borrowed_f32_outputs() {
                use lazymatrix::WeightedGramInto;
                let matrix = CscMatrix::try_from_csc_data(
                    2,
                    2,
                    vec![0, 2, 3],
                    vec![0, 1, 1],
                    vec![1.0_f32, 2.0, 3.0],
                )
                .unwrap();
                let mut out = DMatrix::from_element(2, 2, f32::NAN);
                matrix
                    .weighted_gram_into(&[2.0_f32, -1.0], &mut out)
                    .unwrap();
                assert_eq!(
                    out,
                    DMatrix::from_row_slice(2, 2, &[-2.0, -6.0, -6.0, -9.0])
                );
                let mut storage = DMatrix::from_element(4, 4, -99.0);
                matrix
                    .weighted_gram_into(&[2.0_f32, -1.0], &mut storage.view_mut((1, 1), (2, 2)))
                    .unwrap();
                assert_eq!(storage.view((1, 1), (2, 2)), out);
                assert_eq!(storage[(0, 0)], -99.0);
            }

            #[test]
            fn nalgebra_sparse_rows_suite() {
                common::run_sparse_rows_suite(|tm| {
                    let mut coo = CooMatrix::new(tm.nrows, tm.ncols);
                    for &(i, j, value) in &tm.triplets {
                        coo.push(i, j, value);
                    }
                    CsrMatrix::from(&coo)
                });
            }

            #[test]
            fn nalgebra_sparse_rows_borrow_original_storage() {
                let matrix = CsrMatrix::try_from_csr_data(
                    3,
                    4,
                    vec![0, 1, 3, 3],
                    vec![2, 0, 3],
                    vec![1.0_f32, 0.0, -2.0],
                )
                .unwrap();
                let (columns, values) = matrix.sparse_row(1);
                assert_eq!(columns, &[0, 3]);
                assert_eq!(values, &[0.0, -2.0]);
                assert_eq!(columns.as_ptr(), matrix.col_indices()[1..].as_ptr());
                assert_eq!(values.as_ptr(), matrix.values()[1..].as_ptr());
                let lazy =
                    LazyMatrix::from_column_parts(&matrix, Some(vec![0.5; 4]), Some(vec![-2.0; 4]));
                let row = lazy.row(1);
                assert_eq!(row.column_indices().as_ptr(), columns.as_ptr());
                assert_eq!(row.values().as_ptr(), values.as_ptr());
                assert_eq!(
                    row.stored_corrections().collect::<Vec<_>>(),
                    vec![(0, -0.0), (3, 1.0)]
                );
            }

            #[test]
            fn nalgebra_strided_views_are_borrowed() {
                let design_storage = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0];
                let design = DMatrixView::<_, Dyn, Dyn>::from_slice_with_strides(
                    &design_storage,
                    4,
                    2,
                    2,
                    1,
                );
                common::check_fused_operator(
                    &design,
                    &(0..4)
                        .map(|i| (0..2).map(|j| design[(i, j)]).collect())
                        .collect::<Vec<Vec<f64>>>(),
                    2,
                    &to_dvec,
                );
                let lazy = LazyMatrix::new(
                    design,
                    ColumnNormalization::new(Centering::Mean, Scaling::L2),
                )
                .unwrap();

                let vector_storage = [1.0, -99.0, 2.0, -99.0, 3.0, -99.0, 4.0];
                let vector =
                    DVectorView::<_, Dyn, Dyn>::from_slice_with_strides(&vector_storage, 4, 2, 1);
                let column = lazy.column(0);
                let expected_dot = (0..4)
                    .map(|i| {
                        let raw = (i + 1) as f64;
                        (raw - 2.5) / 5.0_f64.sqrt() * vector[i]
                    })
                    .sum::<f64>();
                approx::assert_abs_diff_eq!(column.dot(&vector), expected_dot, epsilon = 1e-12);

                let mut destination_storage = [0.0; 7];
                let mut destination = DVectorViewMut::<_, Dyn, Dyn>::from_slice_with_strides_mut(
                    &mut destination_storage,
                    4,
                    2,
                    1,
                );
                column.scaled_add_to(0.5, &mut destination);
                for i in 0..4 {
                    let expected = 0.5 * ((i + 1) as f64 - 2.5) / 5.0_f64.sqrt();
                    approx::assert_abs_diff_eq!(destination[i], expected, epsilon = 1e-12);
                }
            }
        }
    };
}

#[cfg(feature = "nalgebra_v0_32")]
backend_suite!(v0_32, nalgebra_0_32, nalgebra_sparse_0_9);

#[cfg(feature = "nalgebra_v0_33")]
backend_suite!(v0_33, nalgebra_0_33, nalgebra_sparse_0_10);

#[cfg(feature = "nalgebra_v0_34")]
backend_suite!(v0_34, nalgebra_0_34, nalgebra_sparse_0_11);

#[cfg(feature = "nalgebra_v0_35")]
backend_suite!(v0_35, nalgebra_0_35, nalgebra_sparse_0_12);
