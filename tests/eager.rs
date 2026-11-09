use std::{cell::Cell, convert::Infallible, ops::Range};

#[path = "common/runner.rs"]
mod common;

#[cfg(feature = "ndarray_all")]
#[path = "common/backend_aliases.rs"]
mod backend_aliases;

use lazymatrix::{
    ColumnNormalizationParams, DenseNormalize, EagerMatrix, FittedRowNormalization, LazyMatrix,
    MatVec, MaterializeDense, MatrixErrorType, MatrixOwned, MatrixRead, MatrixShape,
    MatrixTraversal, MatrixWrite, NormalizationChainParams, RowNormalization,
    RowNormalizationParams, TraversalLayout,
};

#[derive(Clone, Debug)]
struct Dense {
    rows: usize,
    columns: usize,
    values: Vec<f64>,
}

impl MatrixShape for Dense {
    fn nrows(&self) -> usize {
        self.rows
    }
    fn ncols(&self) -> usize {
        self.columns
    }
}

impl MatrixErrorType for Dense {
    type Error = Infallible;
}

impl MatrixWrite<f64> for Dense {
    fn set(&mut self, row: usize, column: usize, value: f64) {
        self.values[row * self.columns + column] = value;
    }
}

impl MatrixRead<f64> for Dense {
    fn get(&self, row: usize, column: usize) -> f64 {
        self.values[row * self.columns + column]
    }
}

impl MatrixTraversal<f64> for Dense {
    fn traversal_layout(&self) -> TraversalLayout {
        TraversalLayout::Full
    }
    fn for_each_entry(
        &self,
        rows: Range<usize>,
        columns: Range<usize>,
        visit: impl FnMut(usize, usize, f64),
    ) -> Result<(), Self::Error> {
        self.dense_for_each_entry(rows, columns, visit)
    }
}

impl MatrixOwned<f64> for Dense {
    fn zeros(rows: usize, columns: usize) -> Self {
        Self {
            rows,
            columns,
            values: vec![0.0; rows * columns],
        }
    }
}

impl MaterializeDense<f64> for Dense {
    fn materialize_normalized_into<O: MatrixWrite<f64> + ?Sized>(
        &self,
        centers: Option<&[f64]>,
        scales: Option<&[f64]>,
        out: &mut O,
    ) -> Result<(), Infallible> {
        for row in 0..self.rows {
            for col in 0..self.columns {
                let mut value = self.values[row * self.columns + col];
                if let Some(c) = centers {
                    value -= c[col];
                }
                if let Some(s) = scales {
                    value /= s[col];
                }
                out.set(row, col, value);
            }
        }
        Ok(())
    }
}

impl DenseNormalize<f64> for Dense {
    fn normalize_in_place(&mut self, centers: Option<&[f64]>, scales: Option<&[f64]>) {
        for row in self.values.chunks_mut(self.columns) {
            for (col, value) in row.iter_mut().enumerate() {
                if let Some(c) = centers {
                    *value -= c[col];
                }
                if let Some(s) = scales {
                    *value /= s[col];
                }
            }
        }
    }
}

impl MatVec<Vec<f64>> for Dense {
    fn matvec(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(self
            .values
            .chunks(self.columns)
            .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
            .collect())
    }
}

fn source() -> Dense {
    Dense {
        rows: 2,
        columns: 2,
        values: vec![2.0, 4.0, 6.0, 8.0],
    }
}

#[test]
fn row_and_chain_conversion_matches_the_shared_oracle() {
    common::run_normalized_materialization_suite(|tm| Dense {
        rows: tm.nrows,
        columns: tm.ncols,
        values: tm.dense.iter().flatten().copied().collect(),
    });
}

#[test]
fn row_and_chain_conversion_owns_metadata_and_preserves_in_place_borrows() {
    let rows = RowNormalizationParams::from_parts(2, Some(vec![2., 6.]), Some(vec![-2., 4.]));
    let cols = ColumnNormalizationParams::from_parts(2, Some(vec![1., -1.]), Some(vec![2., -3.]));
    let mut raw = source();
    let pointer = raw.values.as_ptr();
    let rp = rows.centers().unwrap().as_ptr();
    let cp = cols.scales().unwrap().as_ptr();
    let chain = NormalizationChainParams::from_parts(
        FittedRowNormalization::from_parts(RowNormalization::standardize(), rows.view()),
        cols.view(),
    );
    let eager = chain.view().apply_frozen(&mut raw).into_eager();
    assert_eq!(eager.data().values.as_ptr(), pointer);
    assert_eq!(
        eager
            .normalization()
            .first()
            .parameters()
            .centers()
            .unwrap()
            .as_ptr(),
        rp
    );
    assert_eq!(eager.normalization().next().scales().unwrap().as_ptr(), cp);
    assert_eq!(eager.matvec(&vec![1., 2.]).unwrap(), [-0.5, -1.5]);
    let mut destination = Dense::zeros(2, 2);
    let (copied, copied_into) = {
        let owned_chain = NormalizationChainParams::from_parts(
            FittedRowNormalization::from_parts(RowNormalization::standardize(), rows.clone()),
            cols.clone(),
        );
        let op = owned_chain.view().apply_frozen(source());
        let copied = op.to_eager::<Dense>().unwrap();
        assert_ne!(
            copied
                .normalization()
                .first()
                .parameters()
                .centers()
                .unwrap()
                .as_ptr(),
            rp
        );
        assert_ne!(copied.normalization().next().scales().unwrap().as_ptr(), cp);
        assert_eq!(
            copied.normalization().first().recipe(),
            RowNormalization::standardize()
        );
        (copied, op.to_eager_into(&mut destination).unwrap())
    };
    assert_eq!(copied.data().values, eager.data().values);
    assert_eq!(copied_into.data().values, eager.data().values);
    let (raw, params) = eager.into_parts();
    assert_eq!(raw.values.as_ptr(), pointer);
    assert_eq!(params.first().parameters().centers().unwrap().as_ptr(), rp);
    let row_pointer = raw.values.as_ptr();
    let row = rows.view().apply_frozen(raw).into_eager();
    assert_eq!(row.data().values.as_ptr(), row_pointer);
    assert_eq!(row.normalization().centers().unwrap().as_ptr(), rp);
}

struct Tracked {
    data: Dense,
    scans: Cell<usize>,
    fail: bool,
    layout: TraversalLayout,
}
impl MatrixShape for Tracked {
    fn nrows(&self) -> usize {
        self.data.rows
    }
    fn ncols(&self) -> usize {
        self.data.columns
    }
}
impl MatrixErrorType for Tracked {
    type Error = std::io::Error;
}
impl MatrixTraversal<f64> for Tracked {
    fn traversal_layout(&self) -> TraversalLayout {
        self.layout
    }
    fn for_each_entry(
        &self,
        rows: Range<usize>,
        columns: Range<usize>,
        mut visit: impl FnMut(usize, usize, f64),
    ) -> Result<(), Self::Error> {
        self.scans.set(self.scans.get() + 1);
        assert_eq!(rows, 0..self.nrows());
        assert_eq!(columns, 0..self.ncols());
        for i in rows {
            for j in columns.clone() {
                visit(i, j, self.data.get(i, j));
                if self.fail {
                    return Err(std::io::Error::other("injected read failure"));
                }
            }
        }
        Ok(())
    }
}

#[test]
fn conversion_scans_once_and_propagates_read_errors() {
    for layout in [TraversalLayout::Full, TraversalLayout::SparseRows] {
        for fail in [false, true] {
            let source = Tracked {
                data: source(),
                scans: Cell::new(0),
                fail,
                layout,
            };
            let rows = RowNormalizationParams::from_parts(2, Some(vec![1.; 2]), Some(vec![-2.; 2]));
            let cols =
                ColumnNormalizationParams::from_parts(2, Some(vec![3.; 2]), Some(vec![4.; 2]));
            let chain = NormalizationChainParams::from_parts(cols.view(), cols.view());
            let mut out = Dense::zeros(2, 2);
            macro_rules! check {
                ($op:expr) => {{
                    source.scans.set(0);
                    let result = $op.to_eager_into(&mut out);
                    assert_eq!(
                        result.err().map(|err| err.to_string()),
                        fail.then(|| "injected read failure".to_string())
                    );
                    assert_eq!(source.scans.get(), 1);
                }};
            }
            check!(rows.view().apply_frozen(&source));
            check!(chain.view().apply_frozen(&source));
            let mixed = NormalizationChainParams::from_parts(
                FittedRowNormalization::from_parts(RowNormalization::standardize(), rows.view()),
                chain,
            );
            check!(mixed.apply_frozen(&source));
        }
    }
}

#[test]
fn row_and_chain_shapes_are_checked_before_reads_or_writes() {
    let source = Tracked {
        data: source(),
        scans: Cell::new(0),
        fail: true,
        layout: TraversalLayout::Full,
    };
    let rows = RowNormalizationParams::<f64>::from_parts(2, None, None);
    let cols = ColumnNormalizationParams::<f64>::from_parts(2, None, None);
    let mut out = Dense::zeros(1, 2);
    out.values.fill(99.);
    macro_rules! check {
        ($op:expr) => {{
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _ = $op.to_eager_into(&mut out);
                }))
                .is_err()
            );
            assert_eq!(source.scans.get(), 0);
            assert_eq!(out.values, [99.; 2]);
        }};
    }
    check!(rows.view().apply_frozen(&source));
    check!(NormalizationChainParams::from_parts(cols.view(), cols.view()).apply_frozen(&source));
}

#[test]
fn materialization_retains_parameters_without_applying_them_twice() {
    fn forward<M: MatVec<Vec<f64>>, P>(
        matrix: &EagerMatrix<M, P>,
        coefficients: &Vec<f64>,
    ) -> Result<Vec<f64>, M::Error> {
        matrix.matvec(coefficients)
    }

    let source = source();
    let lazy = LazyMatrix::from_column_parts(&source, Some(vec![2.0, 4.0]), Some(vec![2.0, -2.0]));
    let eager: EagerMatrix<Dense> = lazy.to_eager().unwrap();
    assert_eq!(eager.data().values, [0.0, -0.0, 2.0, -2.0]);
    assert_eq!(
        eager.normalization().centers(),
        lazy.normalization().centers()
    );
    assert_eq!(
        eager.normalization().scales(),
        lazy.normalization().scales()
    );
    assert_eq!(forward(&eager, &vec![1.0, 2.0]).unwrap(), [0.0, -2.0]);
    assert_eq!(source.values, [2.0, 4.0, 6.0, 8.0]);
    let prediction = LazyMatrix::from_parts(&source, eager.normalization().clone());
    assert_eq!(prediction.matvec(&vec![1.0, 2.0]).unwrap(), [0.0, -2.0]);
}

#[cfg(feature = "ndarray_all")]
macro_rules! eager_submatrix_test {
    ($feature:literal, $name:ident, $backend:ident) => {
        #[cfg(feature = $feature)]
        #[test]
        fn $name() {
            use backend_aliases::$backend::{Array2, array};
            use lazymatrix::{MatTransposeVec, Submatrix};

            let raw = array![[1., 2., 3.], [3., 6., 5.], [5., 10., 7.]];
            let lazy = LazyMatrix::from_column_parts(
                raw,
                Some(vec![1., 2., 3.]),
                Some(vec![2., -2., 0.5]),
            );
            let eager = lazy.to_eager::<Array2<f64>>().unwrap();
            let selected = Submatrix::new(&eager, 1..3, 1..3);
            assert!(std::ptr::eq(selected.as_inner(), &eager));
            assert_eq!(
                selected.matvec(&array![2., -1.]).unwrap(),
                array![-8., -16.]
            );
            assert_eq!(
                selected.mat_transpose_vec(&array![0.5, 1.]).unwrap(),
                array![-5., 10.],
            );
            assert_eq!(
                eager.normalization().centers(),
                lazy.normalization().centers(),
            );
            assert_eq!(
                eager.normalization().scales(),
                lazy.normalization().scales()
            );
        }
    };
}

#[cfg(feature = "ndarray_all")]
eager_submatrix_test!("ndarray_v0_15", eager_submatrix_v0_15, ndarray_0_15);
#[cfg(feature = "ndarray_all")]
eager_submatrix_test!("ndarray_v0_16", eager_submatrix_v0_16, ndarray_0_16);
#[cfg(feature = "ndarray_all")]
eager_submatrix_test!("ndarray_v0_17", eager_submatrix_v0_17, ndarray_0_17);

#[test]
fn consuming_conversion_reuses_dense_storage() {
    let source = source();
    let pointer = source.values.as_ptr();
    let centers = vec![2.0, 4.0];
    let center_ptr = centers.as_ptr();
    let eager = LazyMatrix::from_column_parts(source, Some(centers), None).into_eager();
    assert_eq!(eager.data().values.as_ptr(), pointer);
    assert_eq!(eager.data().values, [0.0, 0.0, 4.0, 4.0]);
    let (data, params) = eager.into_parts();
    assert_eq!(data.values.as_ptr(), pointer);
    assert_eq!(params.centers().unwrap().as_ptr(), center_ptr);
    assert_eq!(params.centers().unwrap(), &[2.0, 4.0]);
}

#[test]
fn consuming_conversion_retains_borrowed_parameter_storage() {
    let params =
        ColumnNormalizationParams::from_parts(2, Some(vec![2.0, 4.0]), Some(vec![2.0, -2.0]));
    let source = source();
    let data_ptr = source.values.as_ptr();
    let center_ptr = params.centers().unwrap().as_ptr();
    let eager = params.view().apply(source).into_eager();
    assert_eq!(eager.data().values.as_ptr(), data_ptr);
    assert_eq!(
        eager.normalization().centers().unwrap().as_ptr(),
        center_ptr
    );
    assert_eq!(eager.matvec(&vec![1.0, 2.0]).unwrap(), [0.0, -2.0]);
    let (_, view) = eager.into_parts();
    let (ncols, centers, scales) = view.into_parts();
    assert_eq!(ncols, 2);
    assert_eq!(centers.unwrap().as_ptr(), center_ptr);
    assert_eq!(scales.unwrap().as_ptr(), params.scales().unwrap().as_ptr());
}

#[test]
fn materialization_owns_metadata_independently_of_borrowed_parameters() {
    let copied = {
        let params =
            ColumnNormalizationParams::from_parts(2, Some(vec![2.0, 4.0]), Some(vec![2.0, -2.0]));
        let lazy = params.view().apply(source());
        let eager: EagerMatrix<Dense> = lazy.to_eager().unwrap();
        assert_ne!(
            eager.normalization().centers().unwrap().as_ptr(),
            params.centers().unwrap().as_ptr()
        );
        eager
    };
    assert_eq!(copied.matvec(&vec![1.0, 2.0]).unwrap(), [0.0, -2.0]);
    let mut destination = Dense::zeros(2, 2);
    let copied_into = {
        let params =
            ColumnNormalizationParams::from_parts(2, Some(vec![2.0, 4.0]), Some(vec![2.0, -2.0]));
        params
            .view()
            .apply(source())
            .to_eager_into(&mut destination)
            .unwrap()
    };
    assert_eq!(copied_into.matvec(&vec![1.0, 2.0]).unwrap(), [0.0, -2.0]);
}

#[test]
fn reusable_output_provides_a_working_borrowed_operator() {
    let lazy = LazyMatrix::from_column_parts(source(), None, Some(vec![2.0, -2.0]));
    let mut destination = Dense::zeros(2, 2);
    let eager = lazy.to_eager_into(&mut destination).unwrap();
    assert_eq!(eager.matvec(&vec![1.0, 2.0]).unwrap(), [-3.0, -5.0]);
    assert_eq!(eager.normalization().scales(), Some(&[2.0, -2.0][..]));
}

#[test]
fn output_dimensions_are_checked_before_writing() {
    let lazy = LazyMatrix::<_>::from_column_parts(source(), None, None);
    let mut destination = Dense::zeros(1, 2);
    destination.values.fill(99.0);
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = lazy.to_eager_into(&mut destination);
    }));
    assert!(panic.is_err());
    assert_eq!(destination.values, [99.0, 99.0]);
}

#[test]
fn fitted_parameters_check_shape_even_when_both_axes_are_inactive() {
    let params = ColumnNormalizationParams::<f64>::from_parts(3, None, None);
    assert!(std::panic::catch_unwind(|| { LazyMatrix::from_parts(source(), params) }).is_err());
}

#[test]
fn reusable_output_wrapper_outlives_its_source() {
    let mut destination = Dense::zeros(2, 2);
    let eager = {
        let lazy = LazyMatrix::from_column_parts(source(), Some(vec![2.0, 4.0]), None);
        lazy.to_eager_into(&mut destination).unwrap()
    };
    assert_eq!(eager.matvec(&vec![1.0, 1.0]).unwrap(), [0.0, 8.0]);
}

#[test]
#[should_panic(expected = "dense allocation size overflow")]
fn allocating_conversion_checks_size_before_allocating_or_reading() {
    let source = Dense {
        rows: usize::MAX,
        columns: 2,
        values: vec![],
    };
    let _ = LazyMatrix::<_>::from_column_parts(source, None, None).to_eager::<Dense>();
}
