#![cfg(feature = "sprs_all")]
//! Verification of sprs storage and borrowed views against the dense oracle.

#[path = "common/backend_aliases.rs"]
mod backend_aliases;
use backend_aliases::sprs;

#[path = "common/runner.rs"]
mod common;

use common::{TestMatrix, assert_close};
use lazymatrix::{
    Centering, ColumnStats, LazyMatrix, MatTransposeVec, MatTransposeVecInto, MatVec, MatVecInto,
    Normalization, Scaling, SparseColumns, SparseRows, SprsCsc, SprsCsr,
};
use sprs::{CsMat, CsMatI, TriMat};

fn build(tm: &TestMatrix) -> CsMat<f64> {
    let mut triplets = TriMat::new((tm.nrows, tm.ncols));
    for &(row, col, value) in &tm.triplets {
        triplets.add_triplet(row, col, value);
    }
    triplets.to_csc()
}

#[test]
fn sprs_backend_suite() {
    common::run_fused_suite(build, |v| v.to_vec());
    common::run_fused_suite(|tm| build(tm).to_csr(), |v| v.to_vec());
    common::run_fused_suite(|tm| SprsCsc::try_new(build(tm)).unwrap(), |v| v.to_vec());
    common::run_fused_suite(
        |tm| SprsCsr::try_new(build(tm).to_csr()).unwrap(),
        |v| v.to_vec(),
    );
    common::run_gram_suite(|tm| SprsCsc::try_new(build(tm)).unwrap());
    common::run_backend_suite(build, |v| v.to_vec(), Clone::clone);
    common::run_backend_suite(|tm| build(tm).to_csr(), |v| v.to_vec(), Clone::clone);
    common::run_backend_suite(
        |tm| SprsCsr::try_new(build(tm).to_csr()).unwrap(),
        |v| v.to_vec(),
        Clone::clone,
    );
    common::run_sparse_rows_suite(|tm| SprsCsr::try_new(build(tm).to_csr()).unwrap());
    common::run_backend_suite(
        |tm| SprsCsc::try_new(build(tm)).unwrap(),
        |v| v.to_vec(),
        Clone::clone,
    );
    common::run_sparse_columns_suite(|tm| SprsCsc::try_new(build(tm)).unwrap());
    common::run_logical_columns_suite(|tm| SprsCsc::try_new(build(tm)).unwrap());
    common::run_logical_columns_suite(|tm| {
        let mut triplets = sprs::TriMatI::<_, u32>::new((tm.nrows, tm.ncols));
        for &(row, col, value) in &tm.triplets {
            triplets.add_triplet(row, col, value);
        }
        SprsCsc::try_new(triplets.to_csc::<u64>()).unwrap()
    });
}

#[test]
fn sprs_weighted_norms_preserve_f32_implicit_contributions() {
    let matrix = CsMat::new_csc((3, 1), vec![0, 2], vec![0, 2], vec![1.0_f32, 1.0]);
    let matrix = SprsCsc::try_new(matrix.view()).unwrap();
    let lazy = LazyMatrix::with_centers(matrix, vec![1.0_f32]);
    let weights = [1e8_f32, 1.0, 1e8];
    approx::assert_abs_diff_eq!(lazy.column(0).weighted_norm_squared(&weights), 1.0);
    approx::assert_abs_diff_eq!(lazy.sparse_column(0).weighted_norm_squared(&weights), 1.0);
    approx::assert_abs_diff_eq!(
        lazy.column(0)
            .weighted_norm_squared_with_sum(&weights, weights.iter().sum()),
        1.0
    );
    approx::assert_abs_diff_eq!(
        lazy.sparse_column(0)
            .weighted_norm_squared_with_sum(&weights, weights.iter().sum()),
        1.0
    );
}

#[test]
fn sprs_gram_accepts_sliced_columns_and_explicit_zeros() {
    use lazymatrix::WeightedGramInto;
    let matrix = CsMat::new_csc(
        (3, 3),
        vec![0, 1, 3, 4],
        vec![0, 0, 2, 1],
        vec![99.0, 0.0, 2.0, -1.0],
    );
    let lazy = LazyMatrix::from_parts(
        SprsCsc::try_new(matrix.slice_outer(1..3)).unwrap(),
        Some(vec![1.0, 2.0]),
        None,
    );
    let mut out = common::GramOutput(vec![vec![f64::NAN; 2]; 2]);
    lazy.weighted_gram_into(&[1.0, 2.0, 3.0], &mut out).unwrap();
    let dense = vec![vec![0.0, 0.0], vec![0.0, -1.0], vec![2.0, 0.0]];
    common::assert_gram(
        &out,
        &common::materialize(&dense, lazy.centers(), None),
        &[1.0, 2.0, 3.0],
    );
}

#[test]
fn sprs_sparse_centered_gram_keeps_implicit_rows_with_large_weights() {
    use lazymatrix::WeightedGramInto;

    let n = 1_000;
    let p = 32;
    let matrix = CsMat::new_csc((n, p), (0..=p).collect(), vec![0; p], vec![1.0_f64; p]);
    let lazy = LazyMatrix::with_centers(SprsCsc::try_new(matrix).unwrap(), vec![1.0; p]);
    let mut weights = vec![1.0_f64; n];
    weights[0] = 1e16;
    let mut out = common::GramOutput(vec![vec![f64::NAN; p]; p]);
    lazy.weighted_gram_into(&weights, &mut out).unwrap();
    for row in &out.0 {
        assert!(row.iter().all(|&value| value == 999.0));
    }
}

#[test]
fn sprs_centered_gram_matches_oracle_for_sparse_pairs() {
    use lazymatrix::WeightedGramInto;

    let (n, p) = (1_000, 32);
    let mut dense = vec![vec![0.0; p]; n];
    let mut triplets = TriMat::new((n, p));
    for j in 0..p {
        for i in (j % 17..n).step_by(97) {
            let value = (i % 7) as f64 - 2.0;
            dense[i][j] = value;
            triplets.add_triplet(i, j, value);
        }
    }
    let centers: Vec<_> = (0..p).map(|j| 0.1 + j as f64 / 100.0).collect();
    let weights: Vec<_> = (0..n).map(|i| 0.5 + (i % 11) as f64 / 11.0).collect();
    let matrix = SprsCsc::try_new(triplets.to_csc::<usize>()).unwrap();
    let lazy = LazyMatrix::with_centers(matrix, centers);
    let mut out = common::GramOutput(vec![vec![f64::NAN; p]; p]);
    lazy.weighted_gram_into(&weights, &mut out).unwrap();
    common::assert_gram(
        &out,
        &common::materialize(&dense, lazy.centers(), None),
        &weights,
    );
}

#[test]
fn sprs_csr_wrapper_checks_orientation_and_borrows_sliced_rows() {
    let matrix = CsMat::new(
        (3, 4),
        vec![0, 1, 3, 4],
        vec![0, 1, 3, 2],
        vec![5.0, 0.0, 2.0, -1.0],
    );
    let data_ptr = matrix.data().as_ptr();
    let indices_ptr = matrix.indices().as_ptr();
    let wrapped = SprsCsr::try_new(matrix).unwrap();
    assert_eq!(wrapped.as_inner().data().as_ptr(), data_ptr);
    let mut matrix = wrapped.into_inner();
    assert_eq!(matrix.indices().as_ptr(), indices_ptr);
    let wrapped = SprsCsr::try_new(matrix.slice_outer(1..3)).unwrap();
    let (columns, values) = wrapped.sparse_row(0);
    assert_eq!(columns, &[1, 3]);
    assert_eq!(values, &[0.0, 2.0]);
    assert_eq!(columns.as_ptr(), matrix.indices()[1..].as_ptr());
    assert_eq!(values.as_ptr(), matrix.data()[1..].as_ptr());
    assert_eq!(wrapped.as_inner().rows(), 2);
    assert_eq!(wrapped.sparse_row(1), (&[2][..], &[-1.0][..]));
    let lazy = LazyMatrix::from_parts(&wrapped, Some(vec![0.5; 4]), Some(vec![-2.0; 4]));
    let row = lazy.row(0);
    assert_eq!(row.column_indices().as_ptr(), columns.as_ptr());
    assert_eq!(row.values().as_ptr(), values.as_ptr());
    assert_eq!(
        row.stored_corrections().collect::<Vec<_>>(),
        vec![(1, -0.0), (3, -1.0)]
    );
    assert_eq!(lazy.row(1).values(), &[-1.0]);

    let wrapped = SprsCsr::try_new(matrix.view_mut()).unwrap();
    assert_eq!(wrapped.sparse_row(0).1.as_ptr(), data_ptr);
    let lazy = LazyMatrix::from_parts(&wrapped, None, None);
    assert_eq!(lazy.row(0).values().as_ptr(), data_ptr);
    let mut recovered = wrapped.into_inner();
    recovered.data_mut()[0] = 7.0;
    assert_eq!(matrix.data()[0], 7.0);

    let csc = matrix.to_csc();
    let data_ptr = csc.data().as_ptr();
    let rejected = SprsCsr::try_new(csc).unwrap_err();
    assert!(rejected.is_csc());
    assert_eq!(rejected.data().as_ptr(), data_ptr);
    let wrapped = SprsCsr::try_new(rejected.transpose_view()).unwrap();
    for i in 0..rejected.cols() {
        let range = rejected.indptr().outer_inds_sz(i);
        let (columns, values) = wrapped.sparse_row(i);
        assert_eq!(columns, &rejected.indices()[range.clone()]);
        assert_eq!(values.as_ptr(), rejected.data()[range].as_ptr());
    }
}

#[test]
fn sprs_csr_wrapper_supports_alternate_index_and_pointer_widths() {
    let matrix =
        CsMatI::<f32, usize, u64>::new((2, 3), vec![0, 1, 3], vec![1, 0, 2], vec![2.0, 0.0, -1.0]);
    let wrapped = SprsCsr::try_new(matrix.view()).unwrap();
    assert_eq!(wrapped.sparse_row(1), (&[0, 2][..], &[0.0, -1.0][..]));
    assert_eq!(
        wrapped.sparse_row(1).0.as_ptr(),
        matrix.indices()[1..].as_ptr()
    );
    assert_eq!(
        wrapped.sparse_row(1).1.as_ptr(),
        matrix.data()[1..].as_ptr()
    );

    let narrow =
        CsMatI::<f32, u32, u64>::new((2, 3), vec![0, 1, 3], vec![1, 0, 2], vec![2.0, 0.0, -1.0]);
    let wrapped = SprsCsr::try_new(narrow).unwrap();
    assert_eq!(
        wrapped.matvec(&vec![1.0, 2.0, 3.0]).unwrap(),
        vec![4.0, -3.0]
    );
    assert_eq!(wrapped.col_means().unwrap(), vec![0.0, 1.0, -0.5]);
}

#[test]
fn sprs_csc_wrapper_checks_orientation_without_copying() {
    let matrix = CsMat::new_csc(
        (4, 3),
        vec![0, 1, 3, 4],
        vec![0, 1, 3, 2],
        vec![5.0, 0.0, 2.0, -1.0],
    );
    let data_ptr = matrix.data().as_ptr();
    let wrapped = SprsCsc::try_new(matrix).unwrap();
    assert_eq!(wrapped.as_inner().data().as_ptr(), data_ptr);
    let matrix = wrapped.into_inner();
    let view = matrix.slice_outer(1..3);
    let wrapped = SprsCsc::try_new(view).unwrap();
    let (rows, values) = wrapped.sparse_column(0);
    assert_eq!(rows, &[1, 3]);
    assert_eq!(values, &[0.0, 2.0]);
    assert_eq!(rows.as_ptr(), matrix.indices()[1..].as_ptr());
    assert_eq!(values.as_ptr(), matrix.data()[1..].as_ptr());
    let lazy = LazyMatrix::new(wrapped, Normalization::new(Centering::Mean, Scaling::L2)).unwrap();
    assert_eq!(lazy.sparse_column(0).values().len(), 2);
    assert_close(
        &lazy.matvec(&vec![1.0, 0.0]).unwrap(),
        &[
            -0.5 / 3.0_f64.sqrt(),
            -0.5 / 3.0_f64.sqrt(),
            -0.5 / 3.0_f64.sqrt(),
            1.5 / 3.0_f64.sqrt(),
        ],
        1e-12,
    );
    let csr = matrix.to_csr();
    let data_ptr = csr.data().as_ptr();
    let rejected = SprsCsc::try_new(csr).unwrap_err();
    assert!(rejected.is_csr());
    assert_eq!(rejected.data().as_ptr(), data_ptr);
}

#[test]
fn sprs_views_and_alternate_indices_support_products_and_statistics() {
    let mut csc =
        CsMatI::<f32, u32, u64>::new_csc((3, 2), vec![0, 2, 3], vec![0, 2, 1], vec![1.0, 3.0, 2.0]);
    let csr = csc.to_csr();
    for matrix in [csc.view(), csr.view()] {
        assert_eq!(
            matrix.matvec(&vec![2.0, -1.0]).unwrap(),
            vec![2.0, -2.0, 6.0]
        );
        assert_eq!(matrix.col_means().unwrap(), vec![4.0 / 3.0, 2.0 / 3.0]);
        assert_eq!(
            matrix.mat_transpose_vec(&vec![1.0, 2.0, 3.0]).unwrap(),
            vec![10.0, 4.0]
        );
    }
    let lazy = LazyMatrix::new(
        csc.view_mut(),
        Normalization::new(Centering::Mean, Scaling::Sd),
    )
    .unwrap();
    assert!(
        lazy.matvec(&vec![1.0, 2.0])
            .unwrap()
            .iter()
            .all(|v| v.is_finite())
    );
}

#[test]
fn sprs_sliced_and_transposed_views_use_local_dimensions() {
    let csc = build(&common::random_matrix(57, 8, 6, 0.5));
    let csr = csc.to_csr();
    for matrix in [
        csc.slice_outer(1..5),
        csr.slice_outer(2..7),
        csc.transpose_view(),
        csr.transpose_view(),
    ] {
        let dense: Vec<Vec<_>> = (0..matrix.rows())
            .map(|i| {
                (0..matrix.cols())
                    .map(|j| matrix.get(i, j).copied().unwrap_or(0.0))
                    .collect()
            })
            .collect();
        let v = common::random_vec(58, matrix.cols());
        let u = common::random_vec(59, matrix.rows());
        for center in [Centering::None, Centering::Mean, Centering::Min] {
            for scale in [
                Scaling::None,
                Scaling::Sd,
                Scaling::L1,
                Scaling::L2,
                Scaling::MaxAbs,
                Scaling::Range,
            ] {
                let spec = Normalization::new(center, scale);
                let lazy = LazyMatrix::new(matrix, spec).unwrap();
                let owned = LazyMatrix::new(matrix.to_owned(), spec).unwrap();
                assert_eq!(lazy.centers(), owned.centers());
                assert_eq!(lazy.scales(), owned.scales());
                let normalized = common::materialize(&dense, lazy.centers(), lazy.scales());
                assert_close(
                    &lazy.matvec(&v).unwrap(),
                    &common::dense_matvec(&normalized, &v),
                    1e-10,
                );
                assert_close(
                    &lazy.mat_transpose_vec(&u).unwrap(),
                    &common::dense_tmatvec(&normalized, &u),
                    1e-10,
                );
                if matrix.is_csc() {
                    let columns = LazyMatrix::new(SprsCsc::try_new(matrix).unwrap(), spec).unwrap();
                    for j in 0..matrix.cols() {
                        let column = columns.column(j);
                        let range = matrix.indptr().outer_inds_sz(j);
                        assert_eq!(column.raw().data().as_ptr(), matrix.data()[range].as_ptr());
                        approx::assert_abs_diff_eq!(
                            column.dot(&u),
                            common::dense_tmatvec(&normalized, &u)[j],
                            epsilon = 1e-10
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn sprs_f32_normalization_and_mutable_csc_view() {
    let mut csc = CsMat::new_csc(
        (2, 2),
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![1.0_f32, 3.0, 3.0, 3.0],
    );
    let csr = csc.to_csr();
    for matrix in [csc.view(), csr.view()] {
        let lazy =
            LazyMatrix::new(matrix, Normalization::new(Centering::Mean, Scaling::Sd)).unwrap();
        assert_eq!(lazy.centers(), Some([2.0, 3.0].as_slice()));
        assert_eq!(lazy.scales(), Some([1.0, 1.0].as_slice()));
        assert_eq!(lazy.matvec(&vec![2.0, 1.0]).unwrap(), vec![-2.0, 2.0]);
        assert_eq!(
            lazy.mat_transpose_vec(&vec![1.0, 2.0]).unwrap(),
            vec![1.0, 0.0]
        );
    }
    let wrapped = SprsCsc::try_new(csc.view_mut()).unwrap();
    let lazy = LazyMatrix::new(wrapped, Normalization::new(Centering::Mean, Scaling::Sd)).unwrap();
    assert_eq!(lazy.column(0).dot(&[1.0, 2.0]), 1.0);
}

#[test]
fn sprs_infinite_statistics_preserve_ieee_values() {
    let csc = CsMat::new_csc(
        (2, 2),
        vec![0, 1, 2],
        vec![0, 1],
        vec![f64::INFINITY, f64::NEG_INFINITY],
    );
    let csr = csc.to_csr();
    for matrix in [csc.view(), csr.view()] {
        assert_eq!(
            matrix.col_means().unwrap(),
            vec![f64::INFINITY, f64::NEG_INFINITY]
        );
        assert!(matrix.col_sds().unwrap().iter().all(|v| v.is_nan()));
        assert_eq!(matrix.col_mins().unwrap(), vec![0.0, f64::NEG_INFINITY]);
        assert_eq!(matrix.col_ranges().unwrap(), vec![f64::INFINITY; 2]);
        assert_eq!(matrix.col_maxabs().unwrap(), vec![f64::INFINITY; 2]);
        assert_eq!(
            matrix.col_l1_centered(&[1.0, -1.0]).unwrap(),
            vec![f64::INFINITY; 2]
        );
        assert_eq!(
            matrix.col_l2_centered(&[1.0, -1.0]).unwrap(),
            vec![f64::INFINITY; 2]
        );
        assert_eq!(
            matrix.col_maxabs_centered(&[1.0, -1.0]).unwrap(),
            vec![f64::INFINITY; 2]
        );
    }
}

#[test]
fn sprs_reusable_products_validate_dimensions_and_overwrite_nan() {
    let csc = CsMat::new_csc((3, 2), vec![0, 1, 2], vec![0, 2], vec![2.0, -1.0]);
    for matrix in [csc.view(), csc.transpose_view()] {
        let mut out = vec![f64::NAN; matrix.rows()];
        matrix
            .matvec_into(&vec![1.0; matrix.cols()], &mut out)
            .unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(
            std::panic::catch_unwind(|| matrix.matvec(&vec![1.0; matrix.cols() + 1]).unwrap())
                .is_err()
        );
        assert!(
            std::panic::catch_unwind(|| matrix
                .mat_transpose_vec(&vec![1.0; matrix.rows() + 1])
                .unwrap())
            .is_err()
        );
        assert!(
            std::panic::catch_unwind(|| matrix
                .matvec_into(&vec![1.0; matrix.cols()], &mut vec![0.0; matrix.rows() + 1])
                .unwrap())
            .is_err()
        );
        assert!(
            std::panic::catch_unwind(|| matrix
                .mat_transpose_vec_into(
                    &vec![1.0; matrix.rows()],
                    &mut vec![0.0; matrix.cols() + 1]
                )
                .unwrap())
            .is_err()
        );
    }
}

#[cfg(feature = "ndarray_all")]
macro_rules! ndarray_suite {
    ($name:ident, $backend:ident) => {
        mod $name {
            use super::*;
            use crate::backend_aliases::$backend as ndarray;
            #[test]
            fn sprs_ndarray_vectors_and_strided_destinations() {
                use ndarray::{Array1, array, s};
                common::run_backend_suite(build, |v| Array1::from_vec(v.to_vec()), |v| v.to_vec());
                common::run_backend_suite(
                    |tm| build(tm).to_csr(),
                    |v| Array1::from_vec(v.to_vec()),
                    |v| v.to_vec(),
                );
                common::run_backend_suite(
                    |tm| SprsCsr::try_new(build(tm).to_csr()).unwrap(),
                    |v| Array1::from_vec(v.to_vec()),
                    |v| v.to_vec(),
                );
                let matrix = CsMat::new_csc((3, 2), vec![0, 1, 2], vec![0, 2], vec![2.0, -1.0]);
                let storage = array![1.0, -99.0, 2.0];
                let mut out = Array1::from_elem(6, f64::NAN);
                matrix
                    .matvec_into(&storage.slice(s![..;-2]), &mut out.slice_mut(s![..;-2]))
                    .unwrap();
                assert_eq!(out.slice(s![..;-2]).to_vec(), vec![4.0, 0.0, -1.0]);
                assert!(out.slice(s![..;2]).iter().all(|v| v.is_nan()));

                let csr = SprsCsr::try_new(matrix.to_csr()).unwrap();
                out.fill(f64::NAN);
                csr.matvec_into(&storage.slice(s![..;-2]), &mut out.slice_mut(s![..;-2]))
                    .unwrap();
                assert_eq!(out.slice(s![..;-2]).to_vec(), vec![4.0, 0.0, -1.0]);
                assert!(out.slice(s![..;2]).iter().all(|v| v.is_nan()));

                let lazy = LazyMatrix::new(
                    SprsCsc::try_new(matrix.view()).unwrap(),
                    Normalization::new(Centering::Mean, Scaling::Sd),
                )
                .unwrap();
                let input = array![1.0, 3.0, 2.0];
                let mut transpose_out = array![f64::NAN, -99.0, f64::NAN, -99.0];
                lazy.mat_transpose_vec_into(
                    &input.slice(s![..;-1]),
                    &mut transpose_out.slice_mut(s![..;2]),
                )
                .unwrap();
                assert_close(
                    &transpose_out.slice(s![..;2]).to_vec(),
                    &lazy.mat_transpose_vec(&vec![2.0, 3.0, 1.0]).unwrap(),
                    1e-12,
                );
                assert_eq!(transpose_out.slice(s![1..;2]).to_vec(), vec![-99.0; 2]);
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
