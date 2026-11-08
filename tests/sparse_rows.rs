use lazymatrix::{LazyMatrix, MatrixShape, Scalar, SparseRows};

struct Csr<'a, F> {
    ncols: usize,
    offsets: &'a [usize],
    columns: &'a [usize],
    values: &'a [F],
}

impl<F> MatrixShape for Csr<'_, F> {
    fn nrows(&self) -> usize {
        self.offsets.len() - 1
    }

    fn ncols(&self) -> usize {
        self.ncols
    }
}

impl<F: Scalar> SparseRows<F> for Csr<'_, F> {
    fn sparse_row(&self, i: usize) -> (&[usize], &[F]) {
        assert!(i < self.nrows());
        let range = self.offsets[i]..self.offsets[i + 1];
        (&self.columns[range.clone()], &self.values[range])
    }
}

#[test]
fn sparse_rows_supports_borrowed_trait_objects_without_backends() {
    let columns = [2, 0, 1];
    let values = [f32::NAN, f32::INFINITY, -0.0];
    let matrix = Csr {
        ncols: 3,
        offsets: &[0, 1, 3],
        columns: &columns,
        values: &values,
    };
    let erased: &dyn SparseRows<f32> = &matrix;
    let borrowed = &erased;
    let (indices, raw) = <&dyn SparseRows<f32> as SparseRows<f32>>::sparse_row(borrowed, 1);
    assert_eq!(borrowed.nrows(), 2);
    assert_eq!(borrowed.ncols(), 3);
    assert_eq!(indices, &[0, 1]);
    assert_eq!(indices.as_ptr(), columns[1..].as_ptr());
    assert_eq!(raw.as_ptr(), values[1..].as_ptr());
    assert_eq!(raw[0], f32::INFINITY);
    assert_eq!(raw[1].to_bits(), (-0.0_f32).to_bits());
    assert!(borrowed.sparse_row(0).1[0].is_nan());
}

#[test]
fn lazy_rows_borrow_storage_and_nonfinite_parameters_from_trait_objects() {
    let columns = [2, 0, 1];
    let values = [f32::NAN, f32::INFINITY, -0.0];
    let matrix = Csr {
        ncols: 3,
        offsets: &[0, 1, 3],
        columns: &columns,
        values: &values,
    };
    let erased: &dyn SparseRows<f32> = &matrix;
    let lazy = LazyMatrix::from_column_parts(
        erased,
        Some(vec![f32::NEG_INFINITY, -0.0, f32::NAN]),
        Some(vec![f32::INFINITY, -2.0, f32::NAN]),
    );
    let row = lazy.row(1);
    assert_eq!(row.len(), 3);
    assert!(!row.is_empty());
    assert_eq!(row.column_indices(), &[0, 1]);
    assert_eq!(row.column_indices().as_ptr(), columns[1..].as_ptr());
    assert_eq!(row.values().as_ptr(), values[1..].as_ptr());
    assert_eq!(row.values()[1].to_bits(), (-0.0_f32).to_bits());
    assert_eq!(
        row.centers().unwrap().as_ptr(),
        lazy.normalization().centers().unwrap().as_ptr()
    );
    assert_eq!(
        row.scales().unwrap().as_ptr(),
        lazy.normalization().scales().unwrap().as_ptr()
    );
    assert_eq!(row.center(1).to_bits(), (-0.0_f32).to_bits());
    assert_eq!(row.scale(1), -2.0);
    assert!(row.implicit_value(0).is_nan());
    assert!(row.implicit_value(2).is_nan());
    let corrections: Vec<_> = row.stored_corrections().collect();
    assert_eq!(corrections[0].0, 0);
    assert!(corrections[0].1.is_nan());
    assert_eq!(corrections[1], (1, 0.0));
    assert!(lazy.row(0).values()[0].is_nan());
    assert!(lazy.row(0).stored_corrections().next().unwrap().1.is_nan());
}

#[test]
fn lazy_rows_preserve_unsorted_duplicates_and_affine_structure() {
    let matrix = Csr {
        ncols: 3,
        offsets: &[0, 3, 3],
        columns: &[2, 0, 2],
        values: &[1.0, 0.0, -2.0],
    };
    let lazy = LazyMatrix::from_column_parts(
        matrix,
        Some(vec![0.5, -1.0, 2.0]),
        Some(vec![2.0, -4.0, 0.5]),
    );
    let row = lazy.row(0);
    assert_eq!(row.column_indices(), &[2, 0, 2]);
    assert_eq!(row.values(), &[1.0, 0.0, -2.0]);
    assert_eq!(
        row.stored_corrections().collect::<Vec<_>>(),
        vec![(2, 2.0), (0, 0.0), (2, -4.0)]
    );
    let mut logical: Vec<_> = (0..row.len()).map(|j| row.implicit_value(j)).collect();
    for (j, correction) in row.stored_corrections() {
        logical[j] += correction;
    }
    assert_eq!(logical, vec![-0.25, -0.25, -6.0]);
    let empty = lazy.row(1);
    assert!(!empty.is_empty());
    assert!(empty.values().is_empty());
    assert_eq!(
        (0..empty.len())
            .map(|j| empty.implicit_value(j))
            .collect::<Vec<_>>(),
        vec![-0.25, -0.25, -4.0]
    );
}

#[test]
fn lazy_row_helpers_check_column_bounds_without_normalization() {
    let matrix = Csr {
        ncols: 2,
        offsets: &[0, 0],
        columns: &[],
        values: &[] as &[f64],
    };
    let lazy = LazyMatrix::from_column_parts(matrix, None, None);
    let row = lazy.row(0);
    assert_eq!(row.centers(), None);
    assert_eq!(row.scales(), None);
    assert_eq!(row.center(1), 0.0);
    assert_eq!(row.scale(1), 1.0);
    for j in [row.len(), usize::MAX] {
        assert!(std::panic::catch_unwind(|| row.center(j)).is_err());
        assert!(std::panic::catch_unwind(|| row.scale(j)).is_err());
        assert!(std::panic::catch_unwind(|| row.implicit_value(j)).is_err());
    }
}

#[test]
fn lazy_rows_handle_zero_dimensions() {
    for nrows in [0, 2] {
        let offsets = vec![0; nrows + 1];
        let matrix = Csr {
            ncols: 0,
            offsets: &offsets,
            columns: &[],
            values: &[] as &[f64],
        };
        let lazy = LazyMatrix::from_column_parts(matrix, Some(vec![]), Some(vec![]));
        for i in 0..nrows {
            let row = lazy.row(i);
            assert_eq!(row.len(), 0);
            assert!(row.is_empty());
            assert!(row.column_indices().is_empty());
            assert!(row.values().is_empty());
            assert_eq!(row.centers(), Some(&[][..]));
            assert_eq!(row.scales(), Some(&[][..]));
            assert!(row.stored_corrections().next().is_none());
            assert!(std::panic::catch_unwind(|| row.implicit_value(0)).is_err());
        }
        for i in [nrows, usize::MAX] {
            assert!(std::panic::catch_unwind(|| lazy.row(i)).is_err());
        }
    }
}

#[test]
fn affine_row_parts_can_overflow_when_direct_normalization_is_finite() {
    let matrix = Csr {
        ncols: 1,
        offsets: &[0, 1],
        columns: &[0],
        values: &[f32::MAX],
    };
    let lazy = LazyMatrix::from_column_parts(matrix, Some(vec![f32::MAX]), Some(vec![0.5]));
    let row = lazy.row(0);
    let direct = (row.values()[0] - row.center(0)) / row.scale(0);
    assert_eq!(direct, 0.0);
    assert_eq!(row.implicit_value(0), f32::NEG_INFINITY);
    let (_, correction) = row.stored_corrections().next().unwrap();
    assert_eq!(correction, f32::INFINITY);
    assert!((row.implicit_value(0) + correction).is_nan());
}

#[test]
fn row_and_chain_reads_coalesce_before_applying_saved_stages() {
    use lazymatrix::{
        ColumnNormalizationParams, FittedRowNormalization, NormalizationChainParams,
        RowNormalization, RowNormalizationParams,
    };

    let matrix = Csr {
        ncols: 4,
        offsets: &[0, 5, 5],
        columns: &[2, 0, 2, 1, 2],
        values: &[1e16, 0., -1e16, -0., 1.],
    };
    let erased: &dyn SparseRows<f64> = &matrix;
    let rows = RowNormalizationParams::from_parts(2, Some(vec![2., -3.]), Some(vec![-0.5, 4.]));
    let columns = ColumnNormalizationParams::from_parts(
        4,
        Some(vec![1., -2., 3., 4.]),
        Some(vec![2., -4., 0.5, 8.]),
    );
    let params = NormalizationChainParams::from_parts(
        FittedRowNormalization::from_parts(RowNormalization::standardize(), rows.view()),
        columns.view(),
    );
    let op = params.view().apply_frozen(erased);
    let row = op.row(0);
    // Copying a row view does not require owned parameters to implement Copy.
    fn copyable<T: Copy>(_: T) {}
    copyable(row);
    assert_eq!(row.column_indices(), &[2, 0, 2, 1, 2]);
    assert_eq!(row.values().as_ptr(), matrix.values.as_ptr());
    assert!(std::ptr::eq(row.normalization(), op.normalization()));
    assert_eq!(
        row.stored_corrections().map(|(j, _)| j).collect::<Vec<_>>(),
        matrix.columns
    );
    for i in 0..2 {
        let row = op.row(i);
        for j in 0..4 {
            let raw = if i == 0 && j == 2 { 1. } else { 0. };
            let expected = ((raw - rows.centers().unwrap()[i]) / rows.scales().unwrap()[i]
                - columns.centers().unwrap()[j])
                / columns.scales().unwrap()[j];
            assert_eq!(row.get(j), expected);
            assert_eq!(row.normalize_value(j, raw), expected);
            let background = ((0. - rows.centers().unwrap()[i]) / rows.scales().unwrap()[i]
                - columns.centers().unwrap()[j])
                / columns.scales().unwrap()[j];
            assert_eq!(row.implicit_value(j), background);
        }
    }
    let expected = op.row(0).get(2);
    let owned = NormalizationChainParams::from_parts(
        FittedRowNormalization::from_parts(RowNormalization::standardize(), rows),
        columns,
    )
    .apply_frozen(erased);
    copyable(owned.row(0));
    assert_eq!(owned.row(0).get(2), expected);
}

#[test]
fn direct_row_reads_preserve_nonfinite_and_signed_zero_semantics() {
    use lazymatrix::{
        ColumnNormalizationParams, FittedRowNormalization, NormalizationChainParams,
        RowNormalization, RowNormalizationParams,
    };

    let matrix = Csr {
        ncols: 4,
        offsets: &[0, 4],
        columns: &[0, 1, 2, 3],
        values: &[f32::MAX, f32::INFINITY, f32::NAN, -0.],
    };
    let op = LazyMatrix::from_row_parts(&matrix, Some(vec![f32::MAX]), Some(vec![0.5]));
    let row = op.row(0);
    assert_eq!(row.get(0), 0.);
    assert_eq!(row.get(1), f32::INFINITY);
    assert!(row.get(2).is_nan());
    assert_eq!(row.implicit_value(0), f32::NEG_INFINITY);
    assert_eq!(row.stored_corrections().next().unwrap().1, f32::INFINITY);

    let params = NormalizationChainParams::from_parts(
        FittedRowNormalization::from_parts(
            RowNormalization::standardize(),
            RowNormalizationParams::from_parts(1, None, Some(vec![2.])),
        ),
        ColumnNormalizationParams::from_parts(
            4,
            Some(vec![0., f32::INFINITY, f32::NAN, -0.]),
            Some(vec![f32::INFINITY, 1., -2., -2.]),
        ),
    );
    let chain = params.apply_frozen(&matrix);
    let row = chain.row(0);
    assert_eq!(row.get(0), 0.);
    assert!(row.get(1).is_nan());
    assert!(row.get(2).is_nan());
    assert_eq!(row.get(3).to_bits(), (-0.0_f32).to_bits());
    assert_eq!(row.stored_corrections().nth(1).unwrap().1, f32::INFINITY);

    let identity = LazyMatrix::from_row_parts(&matrix, None, None);
    assert_eq!(identity.row(0).get(3).to_bits(), (-0.0_f32).to_bits());
    let negative = LazyMatrix::from_row_parts(&matrix, None, Some(vec![-2.]));
    assert_eq!(negative.row(0).get(3).to_bits(), 0.0_f32.to_bits());
}

#[test]
fn row_views_preserve_stage_arithmetic_with_extreme_parameters() {
    use lazymatrix::{
        ColumnNormalizationParams, FittedRowNormalization, NormalizationChainParams,
        RowNormalization, RowNormalizationParams,
    };

    let matrix = Csr {
        ncols: 1,
        offsets: &[0, 1],
        columns: &[0],
        values: &[f32::MAX],
    };
    let params = NormalizationChainParams::from_parts(
        FittedRowNormalization::from_parts(
            RowNormalization::standardize(),
            RowNormalizationParams::from_parts(1, None, Some(vec![1e-30])),
        ),
        ColumnNormalizationParams::from_parts(1, None, Some(vec![1e30])),
    );
    let op = params.apply_frozen(&matrix);
    // Combining reciprocal scales would erase the overflow in the first stage.
    assert_eq!(op.row(0).get(0), f32::INFINITY);
    assert_eq!(
        op.row(0).stored_corrections().next().unwrap().1,
        f32::INFINITY
    );
    assert_eq!(op.row(0).implicit_value(0), 0.);

    let op = LazyMatrix::from_row_parts(
        &matrix,
        Some(vec![f32::INFINITY]),
        Some(vec![f32::NEG_INFINITY]),
    );
    assert!(op.row(0).get(0).is_nan());
    assert!(op.row(0).implicit_value(0).is_nan());
    assert_eq!(
        op.row(0).stored_corrections().next().unwrap().1.to_bits(),
        (-0.0_f32).to_bits()
    );
}
