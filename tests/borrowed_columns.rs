use lazymatrix::{
    ColumnNormalizationParams, Columns, FittedRowNormalization, LazyMatrix, MatrixShape,
    NormalizationChainParams, RawColumn, RawColumns, RowNormalization, RowNormalizationParams,
    Scalar, SparseColumns,
};
use std::cell::Cell;

#[path = "common/runner.rs"]
mod common;

struct Csc<'a, F> {
    nrows: usize,
    offsets: &'a [usize],
    rows: &'a [usize],
    values: &'a [F],
    scans: Cell<usize>,
}

impl<F> MatrixShape for Csc<'_, F> {
    fn nrows(&self) -> usize {
        self.nrows
    }
    fn ncols(&self) -> usize {
        self.offsets.len() - 1
    }
}

impl<F: Scalar> SparseColumns<F> for Csc<'_, F> {
    fn sparse_column(&self, j: usize) -> (&[usize], &[F]) {
        assert!(j < self.ncols());
        let range = self.offsets[j]..self.offsets[j + 1];
        (&self.rows[range.clone()], &self.values[range])
    }
}

impl<F: Scalar> RawColumns<F> for Csc<'_, F> {
    type Column<'a>
        = Column<'a, F>
    where
        Self: 'a;
    fn raw_column(&self, j: usize) -> Self::Column<'_> {
        let (rows, values) = self.sparse_column(j);
        Column {
            len: self.nrows,
            rows,
            values,
            scans: &self.scans,
        }
    }
}

#[derive(Clone, Copy)]
struct Column<'a, F> {
    len: usize,
    rows: &'a [usize],
    values: &'a [F],
    scans: &'a Cell<usize>,
}

impl<F: Scalar> RawColumn<F> for Column<'_, F> {
    fn len(&self) -> usize {
        self.len
    }
    fn stored_len(&self) -> usize {
        self.values.len()
    }
    fn for_each_stored(&self, mut visit: impl FnMut(usize, F)) {
        self.scans.set(self.scans.get() + 1);
        for (&row, &value) in self.rows.iter().zip(self.values) {
            visit(row, value);
        }
    }
}

#[test]
fn views_borrow_only_storage_and_saved_parameters() {
    let rows = [99, 2, 0, 2, 1, 2, 99];
    let values = [99., 1e16, 0., -1e16, -0., 1., 99.];
    let matrix = Csc {
        nrows: 4,
        offsets: &[1, 6, 6],
        rows: &rows,
        values: &values,
        scans: Cell::new(0),
    };
    let row_params = RowNormalizationParams::from_parts(
        4,
        Some(vec![2., -3., 4., 1.]),
        Some(vec![-0.5, 4., 2., -2.]),
    );
    let column_params =
        ColumnNormalizationParams::from_parts(2, Some(vec![1., -2.]), Some(vec![2., -4.]));
    let params = NormalizationChainParams::from_parts(
        FittedRowNormalization::from_parts(RowNormalization::standardize(), row_params.view()),
        column_params.view(),
    );
    let op = params.view().apply_frozen(&matrix);
    let column = op.column(0);
    assert_eq!(matrix.scans.get(), 0);
    assert_eq!(column.raw().values.as_ptr(), values[1..].as_ptr());
    assert!(std::ptr::eq(column.normalization(), op.normalization()));
    assert_eq!(
        column
            .normalization()
            .first()
            .parameters()
            .centers()
            .unwrap()
            .as_ptr(),
        row_params.centers().unwrap().as_ptr()
    );
    assert_eq!(
        column.normalization().next().scales().unwrap().as_ptr(),
        column_params.scales().unwrap().as_ptr()
    );
    let expected = [1.5, -0.125, -1.25, -0.25];
    common::check_normalized_column(&column, &expected);
    common::check_normalized_column(&Columns::column(&op, 0), &expected);
    let column_op = LazyMatrix::from_column_parts(&matrix, Some(vec![1., 0.]), Some(vec![2., 1.]));
    for (i, expected) in [-0.5, -0.5, 0., -0.5].into_iter().enumerate() {
        assert_eq!(column_op.column(0).get(i), expected);
    }

    let erased: &dyn SparseColumns<f64> = &matrix;
    let sparse_op = params.apply_frozen(erased);
    let sparse = sparse_op.sparse_column(0);
    assert_eq!(sparse.row_indices(), &[2, 0, 2, 1, 2]);
    assert_eq!(sparse.row_indices().as_ptr(), rows[1..].as_ptr());
    assert_eq!(sparse.values().as_ptr(), values[1..].as_ptr());
    assert_eq!(sparse.values()[3].to_bits(), (-0.0_f64).to_bits());
    common::check_normalized_column(&sparse, &expected);
    assert_eq!(
        sparse
            .stored_corrections()
            .map(|(i, _)| i)
            .collect::<Vec<_>>(),
        vec![2, 0, 2, 1, 2]
    );
    let empty = sparse_op.sparse_column(1);
    assert!(!empty.is_empty());
    assert!(empty.values().is_empty());
    for i in 0..4 {
        let background =
            ((0. - row_params.centers().unwrap()[i]) / row_params.scales().unwrap()[i] + 2.) / -4.;
        assert_eq!(empty.get(i), background);
        assert_eq!(empty.implicit_value(i), background);
    }

    fn copyable<T: Copy>(_: T) {}
    let owned = NormalizationChainParams::from_parts(
        FittedRowNormalization::from_parts(RowNormalization::standardize(), row_params),
        column_params,
    )
    .apply_frozen(&matrix);
    copyable(owned.column(0));
    copyable(owned.sparse_column(0));
    common::check_normalized_column(&owned.column(0), &expected);
}

#[test]
fn logical_entries_preserve_nonfinite_values_and_signed_zeros() {
    let matrix = Csc {
        nrows: 4,
        offsets: &[0, 4, 4],
        rows: &[0, 1, 2, 3],
        values: &[f32::MAX, f32::INFINITY, f32::NAN, -0.],
        scans: Cell::new(0),
    };
    let identity = LazyMatrix::from_row_parts(&matrix, None, None);
    assert_eq!(identity.column(0).get(3).to_bits(), (-0.0_f32).to_bits());
    let negative = LazyMatrix::from_row_parts(&matrix, None, Some(vec![-2.; 4]));
    assert_eq!(negative.column(0).get(3).to_bits(), 0.0_f32.to_bits());
    let op = LazyMatrix::from_row_parts(
        &matrix,
        Some(vec![f32::MAX, 0., 0., -0.]),
        Some(vec![0.5, 2., 1., -2.]),
    );
    let column = op.sparse_column(0);
    assert_eq!(column.get(0), 0.);
    assert_eq!(column.get(1), f32::INFINITY);
    assert!(column.get(2).is_nan());
    assert_eq!(column.get(3).to_bits(), (-0.0_f32).to_bits());
    assert_eq!(column.implicit_value(0), f32::NEG_INFINITY);
    assert_eq!(column.stored_corrections().next().unwrap().1, f32::INFINITY);
    assert!(column.sum().is_nan());
    assert!(column.norm_squared().is_nan());
    assert!(column.dot(&[1.; 4][..]).is_nan());
    let mut out = vec![0.; 4];
    column.scaled_add_to(0., &mut out);
    assert_eq!(out[0], 0.);
    assert!(out[1].is_nan());
    assert!(out[2].is_nan());

    let chain = NormalizationChainParams::from_parts(
        FittedRowNormalization::from_parts(
            RowNormalization::standardize(),
            RowNormalizationParams::from_parts(4, None, Some(vec![2.; 4])),
        ),
        ColumnNormalizationParams::from_parts(2, Some(vec![f32::INFINITY, f32::NAN]), None),
    )
    .apply_frozen(&matrix);
    assert!(chain.column(0).get(1).is_nan());
    assert!(chain.column(1).implicit_value(0).is_nan());
    assert!(
        chain
            .sparse_column(0)
            .stored_corrections()
            .nth(1)
            .unwrap()
            .1
            .is_infinite()
    );
}

#[test]
fn coalescing_preserves_stored_signed_zero_and_sparse_scan_bounds() {
    for indices in [&[0, 1, 2][..], &[2, 0, 2, 1][..], &[2, 0, 1][..]] {
        let values: Vec<f64> = indices
            .iter()
            .map(|&i| if i == 1 { -0.0 } else { 0.0 })
            .collect();
        let matrix = Csc {
            nrows: 4,
            offsets: &[0, indices.len()],
            rows: indices,
            values: &values,
            scans: Cell::new(0),
        };
        let op = LazyMatrix::from_row_parts(&matrix, None, None);
        let column = op.column(0);
        assert_eq!(column.get(1).to_bits(), (-0.0_f64).to_bits());
        assert_eq!(column.get(3).to_bits(), 0.0_f64.to_bits());
        matrix.scans.set(0);
        let mut out = vec![-0.0; 4];
        column.scaled_add_to(1., &mut out);
        assert_eq!(out[1].to_bits(), (-0.0_f64).to_bits());
        assert_eq!(out[3].to_bits(), 0.0_f64.to_bits());
        assert!(matrix.scans.get() <= 2);
    }
}

#[test]
fn column_operations_keep_stage_order_and_normalize_before_squaring() {
    let matrix = Csc {
        nrows: 1,
        offsets: &[0, 1],
        rows: &[0],
        values: &[f32::MAX],
        scans: Cell::new(0),
    };
    let params = NormalizationChainParams::from_parts(
        FittedRowNormalization::from_parts(
            RowNormalization::standardize(),
            RowNormalizationParams::from_parts(1, None, Some(vec![1e-30])),
        ),
        ColumnNormalizationParams::from_parts(1, None, Some(vec![1e30])),
    );
    let op = params.apply_frozen(&matrix);
    assert_eq!(op.column(0).get(0), f32::INFINITY);
    assert_eq!(op.column(0).sum(), f32::INFINITY);
    assert_eq!(
        op.column(0).dot_with_sum(&vec![1.], f32::NAN),
        f32::INFINITY
    );
    assert_eq!(
        op.sparse_column(0).stored_corrections().next().unwrap().1,
        f32::INFINITY
    );
    let mut out = vec![0.];
    op.column(0).scaled_add_to(1., &mut out);
    assert_eq!(out[0], f32::INFINITY);

    let matrix = Csc {
        nrows: 2,
        offsets: &[0, 1],
        rows: &[0],
        values: &[1e200],
        scans: Cell::new(0),
    };
    let op = LazyMatrix::from_row_parts(&matrix, None, Some(vec![1e200; 2]));
    assert_eq!(op.column(0).norm_squared(), 1.);
    assert_eq!(op.column(0).weighted_norm_squared(&vec![1., 1.]), 1.);
}

#[test]
fn weighted_reductions_preserve_small_implicit_contributions() {
    let matrix = Csc {
        nrows: 2,
        offsets: &[0, 1],
        rows: &[0],
        values: &[1.],
        scans: Cell::new(0),
    };
    let params = RowNormalizationParams::from_parts(2, Some(vec![1., -2.]), None);
    let op = params.apply_frozen(&matrix);
    let column = op.column(0);
    assert_eq!(column.weighted_norm_squared(&vec![1e20, 1.]), 4.);
    assert_eq!(
        column.weighted_norm_squared_with_sum(&vec![1e20, 1.], f64::NAN),
        4.
    );
    assert_eq!(
        column.weighted_dot_with_sum(&vec![1., 1.], &vec![1e20, -1.], f64::NAN),
        -2.
    );
    assert!(
        column
            .weighted_norm_squared(&vec![f64::INFINITY, 1.])
            .is_nan()
    );
}

#[test]
fn logical_column_bounds_and_dimensions_panic_even_for_identity_parameters() {
    for nrows in [0, 2] {
        let matrix = Csc {
            nrows,
            offsets: &[0, 0],
            rows: &[],
            values: &[] as &[f64],
            scans: Cell::new(0),
        };
        let op = LazyMatrix::from_row_parts(&matrix, None, None);
        let column = op.column(0);
        assert_eq!(column.len(), nrows);
        assert_eq!(column.is_empty(), nrows == 0);
        common::check_normalized_column(&column, &vec![0.; nrows]);
        let panic = |f: &dyn Fn()| {
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err())
        };
        for row in [nrows, usize::MAX] {
            panic(&|| {
                column.get(row);
            });
            panic(&|| {
                column.normalize_value(row, 0.);
            });
            panic(&|| {
                column.implicit_value(row);
            });
        }
        for j in [1, usize::MAX] {
            panic(&|| {
                op.column(j);
            });
            panic(&|| {
                op.sparse_column(j);
            });
        }
        let wrong = vec![0.; nrows + 1];
        let right = vec![0.; nrows];
        panic(&|| {
            column.dot(&wrong);
        });
        panic(&|| {
            column.dot_with_sum(&wrong, 0.);
        });
        panic(&|| {
            column.weighted_dot(&right, &wrong);
        });
        panic(&|| {
            column.weighted_dot(&wrong, &right);
        });
        panic(&|| {
            column.weighted_dot_with_sum(&wrong, &right, 0.);
        });
        panic(&|| {
            column.weighted_norm_squared(&wrong);
        });
        panic(&|| {
            column.weighted_norm_squared_with_sum(&wrong, 0.);
        });
        panic(&|| {
            column.scaled_add_to(0., &mut wrong.clone());
        });
        let columns = LazyMatrix::from_column_parts(&matrix, None, None);
        panic(&|| {
            columns.column(0).get(nrows);
        });
    }
}
