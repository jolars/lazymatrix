//! Logical restrictions work for external operators without backend features.

use std::{cell::RefCell, convert::Infallible, ops::Range};

use lazymatrix::{
    ColumnNormalization, LazyMatrix, MatTransposeVec, MatVec, MatVecScaledInto, MatrixErrorType,
    MatrixRead, MatrixShape, MatrixTraversal, RectangularProducts, RowNormalization, RowStats,
    Submatrix, Transpose, TraversalLayout, VectorView, VectorViewMut,
};

struct Dense {
    values: Vec<Vec<f64>>,
    reads: RefCell<Vec<(usize, usize)>>,
}

impl Dense {
    fn new(values: Vec<Vec<f64>>) -> Self {
        Self {
            values,
            reads: RefCell::new(Vec::new()),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn product<X: VectorView<f64> + ?Sized, Y: VectorViewMut<f64> + ?Sized>(
        &self,
        rows: Range<usize>,
        columns: Range<usize>,
        transpose: bool,
        alpha: f64,
        x: &X,
        beta: f64,
        out: &mut Y,
    ) {
        assert!(rows.start <= rows.end && rows.end <= self.nrows());
        assert!(columns.start <= columns.end && columns.end <= self.ncols());
        let (input, output) = if transpose {
            (rows.len(), columns.len())
        } else {
            (columns.len(), rows.len())
        };
        assert_eq!(x.len(), input);
        assert_eq!(out.len(), output);
        for o in 0..output {
            let product = if alpha == 0.0 {
                0.0
            } else {
                (0..input)
                    .map(|k| {
                        let (i, j) = if transpose { (k, o) } else { (o, k) };
                        self.get(rows.start + i, columns.start + j) * x.get(k)
                    })
                    .sum::<f64>()
                    * alpha
            };
            out.set(
                o,
                if beta == 0.0 {
                    product
                } else {
                    product + beta * out.get(o)
                },
            );
        }
    }
}

impl MatrixShape for Dense {
    fn nrows(&self) -> usize {
        self.values.len()
    }
    fn ncols(&self) -> usize {
        self.values.first().map_or(0, Vec::len)
    }
}

impl MatrixErrorType for Dense {
    type Error = Infallible;
}

impl MatrixRead<f64> for Dense {
    fn get(&self, row: usize, column: usize) -> f64 {
        self.reads.borrow_mut().push((row, column));
        self.values[row][column]
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

impl RowStats<f64> for Dense {
    fn row_normalization_stats(
        &self,
        spec: RowNormalization,
    ) -> Result<lazymatrix::RowNormalizationStats<f64>, Self::Error> {
        Ok(self.dense_row_normalization_stats(spec))
    }
}

impl RectangularProducts for Dense {
    type Scalar = f64;
    fn rectangle_matvec_into<X, Y>(
        &self,
        rows: Range<usize>,
        columns: Range<usize>,
        alpha: f64,
        x: &X,
        beta: f64,
        out: &mut Y,
    ) -> Result<(), Self::Error>
    where
        X: VectorView<f64> + ?Sized,
        Y: VectorViewMut<f64> + ?Sized,
    {
        self.product(rows, columns, false, alpha, x, beta, out);
        Ok(())
    }
    fn rectangle_transpose_into<X, Y>(
        &self,
        rows: Range<usize>,
        columns: Range<usize>,
        alpha: f64,
        x: &X,
        beta: f64,
        out: &mut Y,
    ) -> Result<(), Self::Error>
    where
        X: VectorView<f64> + ?Sized,
        Y: VectorViewMut<f64> + ?Sized,
    {
        self.product(rows, columns, true, alpha, x, beta, out);
        Ok(())
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn external_parent_restrictions_flatten_and_transpose() {
    let parent = Dense::new(vec![vec![1., 2., 3.], vec![4., 5., 6.], vec![7., 8., 9.]]);
    let outer = Submatrix::new(&parent, 1..3, 0..3);
    let selected = outer.submatrix(0..2, 1..3);
    assert!(std::ptr::eq(selected.as_inner(), &parent));
    assert_eq!(selected.row_range(), 1..3);
    assert_eq!(selected.column_range(), 1..3);
    assert!(parent.reads.borrow().is_empty());
    assert_eq!(selected.matvec(&vec![2., -1.]).unwrap(), vec![4., 7.]);
    assert_eq!(*parent.reads.borrow(), vec![(1, 1), (1, 2), (2, 1), (2, 2)]);
    let transposed = Transpose::new(&parent);
    let transpose_selection = Submatrix::new(&transposed, 1..3, 1..3);
    assert_eq!(
        transpose_selection.matvec(&vec![1., -2.]).unwrap(),
        selected.mat_transpose_vec(&vec![1., -2.]).unwrap()
    );
    let wrapped = Submatrix::new(&outer, 0..2, 1..3);
    assert_eq!(wrapped.matvec(&vec![2., -1.]).unwrap(), vec![4., 7.]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn frozen_rows_retain_full_column_fit_and_share_parent() {
    let raw = Dense::new(vec![vec![0., 2., 4., 6.], vec![10., 14., 18., 22.]]);
    let normalized = LazyMatrix::new(&raw, RowNormalization::standardize()).unwrap();
    raw.reads.borrow_mut().clear();
    let selected = normalized.submatrix(0..2, 1..3);
    assert!(std::ptr::eq(selected.as_inner(), &normalized));
    let result = selected.matvec(&vec![1., 2.]).unwrap();
    for value in result {
        approx::assert_relative_eq!(value, 1. / 5_f64.sqrt(), epsilon = 1e-12);
    }
    assert!(
        raw.reads
            .borrow()
            .iter()
            .all(|&(i, j)| i < 2 && (1..3).contains(&j))
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn selection_excludes_nonfinite_cells_and_zero_alpha_skips_reads() {
    let raw = Dense::new(vec![
        vec![f64::NAN; 4],
        vec![f64::INFINITY, 3., 7., f64::NAN],
        vec![f64::NEG_INFINITY, 9., 17., f64::NAN],
    ]);
    let normalized =
        LazyMatrix::from_row_parts(&raw, Some(vec![0., 1., 5.]), Some(vec![1., 2., 4.]));
    let selected = normalized.submatrix(1..3, 1..3);
    assert_eq!(selected.matvec(&vec![2., -1.]).unwrap(), vec![-1., -1.]);
    assert_eq!(
        selected.mat_transpose_vec(&vec![1., -2.]).unwrap(),
        vec![-1., -3.]
    );
    raw.reads.borrow_mut().clear();
    let mut out = vec![f64::NAN; 2];
    selected
        .matvec_scaled_into(0., &vec![f64::NAN; 2], 0., &mut out)
        .unwrap();
    assert_eq!(out, vec![0.; 2]);
    assert!(raw.reads.borrow().is_empty());
    #[cfg(not(target_arch = "wasm32"))]
    {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            selected
                .matvec_scaled_into(0., &vec![1.], 0., &mut out)
                .unwrap();
        }));
        assert!(result.is_err());
        assert!(raw.reads.borrow().is_empty());
    }
}

struct IndexedOnly(Dense);
impl MatrixShape for IndexedOnly {
    fn nrows(&self) -> usize {
        self.0.nrows()
    }
    fn ncols(&self) -> usize {
        self.0.ncols()
    }
}
impl MatrixErrorType for IndexedOnly {
    type Error = Infallible;
}
impl MatrixRead<f64> for IndexedOnly {
    fn get(&self, row: usize, column: usize) -> f64 {
        self.0.get(row, column)
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn mixed_rectangles_need_only_dense_indexed_access() {
    let raw = IndexedOnly(Dense::new(vec![
        vec![1., 3., 8.],
        vec![-2., 5., 3.],
        vec![6., 4., 0.],
    ]));
    let fitted = RowNormalization::standardize()
        .then(ColumnNormalization::standardize())
        .fit_apply(&raw)
        .unwrap();
    let selected = fitted.submatrix(1..3, 1..3);
    let row = fitted.normalization().first().parameters();
    let column = fitted.normalization().next();
    let x = vec![2., -0.5];
    let expected: Vec<_> = (1..3)
        .map(|i| {
            (1..3)
                .map(|j| {
                    let row_value =
                        (raw.0.values[i][j] - row.centers().unwrap()[i]) / row.scales().unwrap()[i];
                    let logical =
                        (row_value - column.centers().unwrap()[j]) / column.scales().unwrap()[j];
                    logical * x[j - 1]
                })
                .sum::<f64>()
        })
        .collect();
    raw.0.reads.borrow_mut().clear();
    let result = selected.matvec(&x).unwrap();
    assert_eq!(*raw.0.reads.borrow(), vec![(1, 1), (1, 2), (2, 1), (2, 2)]);
    for (actual, expected) in result.iter().zip(expected) {
        approx::assert_relative_eq!(*actual, expected, epsilon = 1e-12);
    }
    let u = vec![1.5, -2.];
    let transposed = selected.mat_transpose_vec(&u).unwrap();
    let left: f64 = u.iter().zip(result).map(|(a, b)| a * b).sum();
    let right: f64 = x.iter().zip(transposed).map(|(a, b)| a * b).sum();
    approx::assert_relative_eq!(left, right, epsilon = 1e-12);
}
