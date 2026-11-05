use lazymatrix::{LazyMatrix, MatrixShape, RawColumn, RawColumns};

struct ReductionOnly;

impl MatrixShape for ReductionOnly {
    fn nrows(&self) -> usize {
        3
    }
    fn ncols(&self) -> usize {
        1
    }
}

impl RawColumns<f64> for ReductionOnly {
    type Column<'a> = Self;

    fn raw_column(&self, j: usize) -> Self::Column<'_> {
        assert_eq!(j, 0);
        Self
    }
}

impl RawColumn<f64> for ReductionOnly {
    fn len(&self) -> usize {
        3
    }
    fn stored_len(&self) -> usize {
        3
    }
    fn for_each_stored(&self, _visit: impl FnMut(usize, f64)) {
        panic!("the specialized reduction should avoid traversal");
    }
    fn centered_sum_squares(&self, center: f64) -> Option<f64> {
        assert_eq!(center, 4.0);
        Some(26.0)
    }
}

#[test]
fn logical_norm_scales_the_specialized_centered_reduction() {
    let matrix = LazyMatrix::from_column_parts(ReductionOnly, Some(vec![4.0]), Some(vec![-2.0]));
    assert!(matrix.column(0).raw().dense_values().is_none());
    assert_eq!(matrix.column(0).norm_squared(), 6.5);
}
