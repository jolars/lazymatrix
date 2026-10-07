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
            use lazymatrix::{Centering, LazyMatrix, Normalization, Scaling, SparseRows};
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

            fn from_dvec(v: &DVector<f64>) -> Vec<f64> {
                v.as_slice().to_vec()
            }

            fn build_dense(tm: &TestMatrix) -> DMatrix<f64> {
                DMatrix::from_fn(tm.nrows, tm.ncols, |i, j| tm.dense[i][j])
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
            fn nalgebra_backend_suite() {
                common::run_gram_suite(build);
                common::run_backend_suite(build, to_dvec, from_dvec);
                common::run_sparse_columns_suite(build);
                common::run_logical_columns_suite(build);
                common::run_backend_suite(build_dense, to_dvec, from_dvec);
                common::run_logical_columns_suite(build_dense);
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
                let lazy = LazyMatrix::from_parts(&matrix, Some(vec![0.5; 4]), Some(vec![-2.0; 4]));
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
                let lazy =
                    LazyMatrix::new(design, Normalization::new(Centering::Mean, Scaling::L2))
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
