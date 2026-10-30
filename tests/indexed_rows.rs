//! Indexed raw row selection across every enabled in-memory backend release.
#[path = "common/runner.rs"]
mod common;

#[cfg(feature = "ndarray_all")]
macro_rules! ndarray_suite {
    ($name:ident, $backend:ident) => { mod $name {
        use $backend as ndarray;
        #[test]
        fn indexed_rows() {
            for column_major in [false, true] {
                use ndarray::ShapeBuilder;
                common::run_indexed_rows_suite(|tm| {
                    let shape = (tm.nrows, tm.ncols).set_f(column_major);
                    ndarray::Array2::from_shape_fn(shape, |(i, j)| tm.dense[i][j])
                }, |v| ndarray::Array1::from_vec(v.to_vec()));
            }
            let x = ndarray::Array2::from_shape_fn((7, 6), |(i, j)| 1e12 + (i * 6 + j) as f64);
            let view = x.slice(ndarray::s![..;-2, ..;2]);
            let rows = lazymatrix::IndexedRows::new(view, &[0, 2]);
            let lazy = lazymatrix::LazyMatrix::from_normalization(rows, lazymatrix::ColumnNormalizationParams::from_parts(3, Some(vec![1e12; 3]), None));
            let expected = vec![114.0, 42.0];
            use lazymatrix::{MatVec, MatVecInto, MatTransposeVecInto};
            assert_eq!(lazy.matvec(&vec![1.0; 3]).unwrap(), expected);
            let mut output = ndarray::Array1::from_elem(4, 99.0);
            lazy.matvec_into(&vec![1.0; 3], &mut output.slice_mut(ndarray::s![..;2])).unwrap();
            assert_eq!(output.to_vec(), vec![114.0, 99.0, 42.0, 99.0]);
            let mut transposed = ndarray::Array1::from_elem(6, 99.0);
            lazy.mat_transpose_vec_into(&vec![1.0; 2], &mut transposed.slice_mut(ndarray::s![..;2])).unwrap();
            assert_eq!(transposed.to_vec(), vec![48.0, 99.0, 52.0, 99.0, 56.0, 99.0]);
        }
        use crate::common;
    }};
}
#[cfg(feature = "ndarray_v0_15")]
ndarray_suite!(ndarray_15, ndarray_0_15);
#[cfg(feature = "ndarray_v0_16")]
ndarray_suite!(ndarray_16, ndarray_0_16);
#[cfg(feature = "ndarray_v0_17")]
ndarray_suite!(ndarray_17, ndarray);

#[cfg(feature = "faer_all")]
macro_rules! faer_suite {
    ($name:ident, $backend:ident) => {
        mod $name {
            use crate::common;
            use $backend as faer;
            #[test]
            fn duplicates_offsets_and_borrowed_views() {
                use faer::prelude::ReborrowMut;
                use faer::sparse::{
                    SparseColMat, SparseColMatRef, SparseRowMat, SymbolicSparseColMatRef,
                    SymbolicSparseRowMat,
                };
                use lazymatrix::{IndexedRows, MatVec, RawColumn, RawColumns, SparseRows};
                let indices = [99, 2, 0, 2, 0, 99, 99, 1, 2, 99];
                let values = [
                    f64::NAN,
                    1.0,
                    0.0,
                    -2.0,
                    2.0,
                    f64::NAN,
                    f64::NAN,
                    2.0,
                    3.0,
                    f64::NAN,
                ];
                let symbolic = SymbolicSparseColMatRef::new_unsorted_checked(
                    3,
                    2,
                    &[1, 7, 10],
                    Some(&[4, 2]),
                    &indices,
                );
                let view = SparseColMatRef::new(symbolic, &values);
                let mut matrix = SparseColMat::new(symbolic.to_owned().unwrap(), values.to_vec());
                fn check<M: lazymatrix::IndexedRowSource<f64> + RawColumns<f64>>(matrix: M) {
                    let selection = IndexedRows::new(matrix, &[0, 2]);
                    let reference =
                        faer::Mat::from_fn(2, 2, |i, j| [[2.0, 0.0], [-1.0, 3.0]][i][j]);
                    common::check_column_stats_against(&selection, &reference);
                    assert_eq!(selection.matvec(&vec![1.0, 1.0]).unwrap(), vec![2.0, 2.0]);
                    let column = selection.raw_column(0);
                    assert_eq!(column.stored_len(), 4);
                    let mut entries = Vec::new();
                    column.for_each_stored(|i, v| entries.push((i, v)));
                    assert_eq!(entries, vec![(1, 1.0), (0, 0.0), (1, -2.0), (0, 2.0)]);
                }
                check(view);
                check(matrix.as_ref());
                check(matrix.rb_mut());
                check(matrix);
                let symbolic = SymbolicSparseRowMat::new_unsorted_checked(
                    3,
                    2,
                    vec![0, 4, 5, 7],
                    Some(vec![3, 0, 1]),
                    vec![1, 0, 1, 99, 99, 0, 99],
                );
                let mut csr =
                    SparseRowMat::new(symbolic, vec![2.0, 0.0, -3.0, 99.0, 99.0, 4.0, 99.0]);
                fn check_row<M: lazymatrix::IndexedRowSource<f64> + SparseRows<f64>>(matrix: M) {
                    let selected = IndexedRows::new(matrix, &[0, 2]);
                    let reference =
                        faer::Mat::from_fn(2, 2, |i, j| [[0.0, -1.0], [4.0, 0.0]][i][j]);
                    common::check_column_stats_against(&selected, &reference);
                    assert_eq!(
                        selected.sparse_row(0),
                        (&[1, 0, 1][..], &[2.0, 0.0, -3.0][..])
                    );
                    assert_eq!(selected.matvec(&vec![1.0, 1.0]).unwrap(), vec![-1.0, 4.0]);
                }
                check_row(csr.as_ref());
                check_row(csr.rb_mut());
                check_row(csr);
            }
            #[test]
            fn indexed_rows() {
                common::run_indexed_rows_suite(
                    |tm| faer::Mat::from_fn(tm.nrows, tm.ncols, |i, j| tm.dense[i][j]),
                    |v| faer::Col::from_fn(v.len(), |i| v[i]),
                );
                let sparse = |tm: &common::TestMatrix| {
                    let triplets: Vec<_> = tm
                        .triplets
                        .iter()
                        .map(|&(i, j, v)| faer::sparse::Triplet::new(i, j, v))
                        .collect();
                    faer::sparse::SparseColMat::<usize, f64>::try_new_from_triplets(
                        tm.nrows, tm.ncols, &triplets,
                    )
                    .unwrap()
                };
                common::run_indexed_rows_suite(sparse, |v| faer::Col::from_fn(v.len(), |i| v[i]));
                common::run_indexed_rows_suite(
                    |tm| sparse(tm).to_row_major().unwrap(),
                    |v| faer::Col::from_fn(v.len(), |i| v[i]),
                );
            }
        }
    };
}
#[cfg(feature = "faer_v0_22")]
faer_suite!(faer_22, faer_0_22);
#[cfg(feature = "faer_v0_23")]
faer_suite!(faer_23, faer_0_23);
#[cfg(feature = "faer_v0_24")]
faer_suite!(faer_24, faer);

#[cfg(feature = "nalgebra_all")]
macro_rules! nalgebra_suite {
    ($name:ident, $backend:ident, $sparse:ident) => {
        mod $name {
            use crate::common;
            use $backend as nalgebra;
            use $sparse as sparse;
            #[test]
            fn indexed_rows() {
                common::run_indexed_rows_suite(
                    |tm| nalgebra::DMatrix::from_fn(tm.nrows, tm.ncols, |i, j| tm.dense[i][j]),
                    |v| nalgebra::DVector::from_column_slice(v),
                );
                let csr = |tm: &common::TestMatrix| {
                    let mut coo = sparse::CooMatrix::new(tm.nrows, tm.ncols);
                    for &(i, j, v) in &tm.triplets {
                        coo.push(i, j, v);
                    }
                    sparse::CsrMatrix::from(&coo)
                };
                common::run_indexed_rows_suite(csr, |v| nalgebra::DVector::from_column_slice(v));
                common::run_indexed_rows_suite(
                    |tm| sparse::CscMatrix::from(&csr(tm)),
                    |v| nalgebra::DVector::from_column_slice(v),
                );
            }
        }
    };
}
#[cfg(feature = "nalgebra_v0_32")]
nalgebra_suite!(nalgebra_32, nalgebra_0_32, nalgebra_sparse_0_9);
#[cfg(feature = "nalgebra_v0_33")]
nalgebra_suite!(nalgebra_33, nalgebra_0_33, nalgebra_sparse_0_10);
#[cfg(feature = "nalgebra_v0_34")]
nalgebra_suite!(nalgebra_34, nalgebra_0_34, nalgebra_sparse_0_11);
#[cfg(feature = "nalgebra_v0_35")]
nalgebra_suite!(nalgebra_35, nalgebra, nalgebra_sparse);

#[cfg(feature = "sprs_all")]
#[test]
fn sprs_indexed_rows() {
    let csc = |tm: &common::TestMatrix| {
        let mut coo = sprs::TriMat::new((tm.nrows, tm.ncols));
        for &(i, j, v) in &tm.triplets {
            coo.add_triplet(i, j, v);
        }
        coo.to_csc::<usize>()
    };
    common::run_indexed_rows_suite(csc, |v| v.to_vec());
    common::run_indexed_rows_suite(|tm| csc(tm).to_csr(), |v| v.to_vec());
    common::run_indexed_rows_suite(
        |tm| lazymatrix::SprsCsc::try_new(csc(tm)).unwrap(),
        |v| v.to_vec(),
    );
    common::run_indexed_rows_suite(
        |tm| lazymatrix::SprsCsr::try_new(csc(tm).to_csr()).unwrap(),
        |v| v.to_vec(),
    );
}

#[test]
fn dependency_light_sources_share_scans_and_forward_errors() {
    use lazymatrix::{
        ColumnNormalization, ColumnStats, IndexedRowSource, IndexedRows, LazyMatrix, MatVecInto,
        MatrixErrorType, MatrixShape,
    };
    use std::cell::Cell;
    struct Source {
        scans: Cell<usize>,
        fail: Cell<bool>,
    }
    impl MatrixShape for Source {
        fn nrows(&self) -> usize {
            4
        }
        fn ncols(&self) -> usize {
            1
        }
    }
    impl MatrixErrorType for Source {
        type Error = std::io::Error;
    }
    impl IndexedRowSource<f64> for Source {
        fn for_each_indexed_entry(
            &self,
            rows: &[usize],
            mut visit: impl FnMut(usize, usize, f64),
        ) -> Result<(), Self::Error> {
            self.scans.set(self.scans.get() + 1);
            for (i, &row) in rows.iter().enumerate() {
                visit(i, 0, row as f64 + 1.0);
                if self.fail.get() {
                    return Err(std::io::Error::other("injected read failure"));
                }
            }
            Ok(())
        }
    }
    let source = Source {
        scans: Cell::new(0),
        fail: Cell::new(false),
    };
    let selected = IndexedRows::new(&source, &[0, 2]);
    let (centers, scales) = selected
        .normalization_stats(ColumnNormalization::standardize())
        .unwrap();
    assert_eq!(source.scans.get(), 2);
    assert_eq!(centers, Some(vec![2.0]));
    assert_eq!(scales, Some(vec![1.0]));
    let borrowed = &selected;
    borrowed
        .normalization_stats(ColumnNormalization::standardize())
        .unwrap();
    assert_eq!(source.scans.get(), 4);
    selected
        .normalization_stats(ColumnNormalization::default())
        .unwrap();
    assert_eq!(source.scans.get(), 4);
    let before = source.scans.get();
    selected
        .normalization_stats(ColumnNormalization::robust())
        .unwrap();
    let combined = source.scans.get() - before;
    let before = source.scans.get();
    selected.col_medians().unwrap();
    selected
        .col_quantile_ranges(lazymatrix::QuantileRange::default())
        .unwrap();
    assert!(combined < source.scans.get() - before);
    source.fail.set(true);
    assert!(selected.col_means().is_err());
    assert!(selected.col_sds(0).is_err());
    assert!(selected.col_medians().is_err());
    assert!(
        selected
            .normalization_stats(ColumnNormalization::robust())
            .is_err()
    );
    assert!(LazyMatrix::new(&selected, ColumnNormalization::standardize()).is_err());
    let lazy = LazyMatrix::from_normalization(
        &selected,
        lazymatrix::ColumnNormalizationParams::from_parts(1, Some(vec![10.0]), None),
    );
    let mut out = vec![99.0; 2];
    assert!(lazy.matvec_into(&vec![1.0], &mut out).is_err());
    assert_eq!(out[0], 1.0);
    let before = source.scans.get();
    let empty = IndexedRows::new(&source, &[]);
    empty
        .normalization_stats(ColumnNormalization::robust())
        .unwrap();
    empty
        .matvec_into(&vec![1.0], &mut Vec::<f64>::new())
        .unwrap();
    assert_eq!(source.scans.get(), before);
}

#[cfg(feature = "sprs_all")]
#[test]
fn borrowed_columns_and_rows_preserve_local_mapping() {
    use lazymatrix::{IndexedRows, MatVec, RawColumn, RawColumns, SparseRows, SprsCsc, SprsCsr};
    let matrix = sprs::CsMat::new_csc(
        (4, 2),
        vec![0, 3, 5],
        vec![0, 1, 3, 1, 2],
        vec![99.0, 2.0, 0.0, 4.0, 88.0],
    );
    let source = SprsCsc::try_new(matrix.view()).unwrap();
    let selection = IndexedRows::new(source, &[1, 3]);
    let column = selection.raw_column(0);
    assert_eq!((column.len(), column.stored_len()), (2, 2));
    let mut entries = Vec::new();
    column.for_each_stored(|i, v| entries.push((i, v)));
    assert_eq!(entries, vec![(0, 2.0), (1, 0.0)]);
    assert_eq!(selection.matvec(&vec![1.0, 1.0]).unwrap(), vec![6.0, 0.0]);
    let csr = matrix.to_csr();
    let source = SprsCsr::try_new(csr.view()).unwrap();
    let rows = IndexedRows::<_, f64>::new(source, &[1, 3]);
    assert_eq!(rows.sparse_row(0), source.sparse_row(1));
    assert!(std::ptr::eq(
        rows.sparse_row(0).0.as_ptr(),
        source.sparse_row(1).0.as_ptr()
    ));
    assert!(std::ptr::eq(
        rows.sparse_row(0).1.as_ptr(),
        source.sparse_row(1).1.as_ptr()
    ));
}
