#![cfg(feature = "faer_all")]
//! Verification of the faer sparse backend against the dense oracle.

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
                Centering, LazyMatrix, MatrixShape, Normalization, Scaling, SparseRows,
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

            fn from_col(c: &Col<f64>) -> Vec<f64> {
                (0..c.nrows()).map(|i| c[i]).collect()
            }

            fn build_dense(tm: &TestMatrix) -> Mat<f64> {
                Mat::from_fn(tm.nrows, tm.ncols, |i, j| tm.dense[i][j])
            }

            #[test]
            fn eager_conversion_combines_duplicates_and_ignores_spare_capacity() {
                use lazymatrix::{MatVec, MaterializeDense, MatrixOwned};
                let symbolic = faer::sparse::SymbolicSparseColMat::new_unsorted_checked(
                    3,
                    1,
                    vec![0, 4],
                    Some(vec![3]),
                    vec![2, 0, 2, 99],
                );
                let matrix = SparseColMat::new(symbolic, vec![0.25, -0.0, 0.75, 99.0]);
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
            }

            #[test]
            fn faer_backend_suite() {
                common::run_materialization_suite(build);
                common::run_materialization_suite(build_dense);
                common::run_eager_product_suite::<_, Mat<f64>, _>(build, to_col, from_col);
                common::run_eager_product_suite::<_, Mat<f64>, _>(build_dense, to_col, from_col);
                common::run_fused_suite(build, to_col);
                common::run_fused_suite(build_dense, to_col);
                common::run_gram_suite(build);
                common::run_backend_suite(build, to_col, from_col);
                common::run_sparse_columns_suite(build);
                common::run_logical_columns_suite(build);
                common::run_backend_suite(build_dense, to_col, from_col);
                common::run_logical_columns_suite(build_dense);
            }

            #[test]
            fn faer_weighted_norms_combine_unsorted_duplicates() {
                let symbolic = faer::sparse::SymbolicSparseColMat::new_unsorted_checked(
                    3,
                    1,
                    vec![0, 4],
                    Some(vec![3]),
                    vec![2, 0, 2, 99],
                );
                let matrix = SparseColMat::new(symbolic, vec![0.25, 1.0, 0.75, 99.0]);
                let lazy = LazyMatrix::from_parts(&matrix, Some(vec![1.0]), Some(vec![-2.0]));
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
                let lazy =
                    LazyMatrix::new(design, Normalization::new(Centering::Mean, Scaling::L2))
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
