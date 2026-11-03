//! Public native traversal supports downstream backends without optional features.

use lazymatrix::{
    Centering, ColumnNormalization, ColumnNormalizationStats, ColumnStats, MatTransposeVec,
    MatTransposeVecScaledInto, MatVec, MatVecScaledInto, MatrixErrorType, MatrixRead, MatrixShape,
    MatrixTraversal, QuantileRange, Scaling, TraversalLayout, dense_for_each_entry,
};
use std::{cell::Cell, convert::Infallible, ops::Range};

struct Dense {
    values: [[f64; 3]; 4],
    reads: Cell<usize>,
}
impl MatrixShape for Dense {
    fn nrows(&self) -> usize {
        4
    }
    fn ncols(&self) -> usize {
        3
    }
}
impl MatrixErrorType for Dense {
    type Error = Infallible;
}
impl MatrixRead<f64> for Dense {
    fn get(&self, i: usize, j: usize) -> f64 {
        self.reads.set(self.reads.get() + 1);
        self.values[i][j]
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
        visitor: impl FnMut(usize, usize, f64),
    ) -> Result<(), Infallible> {
        self.dense_for_each_entry(rows, columns, visitor)
    }
}

struct Sparse {
    dense: Dense,
    csc: bool,
}
impl MatrixShape for Sparse {
    fn nrows(&self) -> usize {
        self.dense.nrows()
    }
    fn ncols(&self) -> usize {
        self.dense.ncols()
    }
}
impl MatrixErrorType for Sparse {
    type Error = Infallible;
}
impl MatrixTraversal<f64> for Sparse {
    fn traversal_layout(&self) -> TraversalLayout {
        if self.csc {
            TraversalLayout::SparseColumns
        } else {
            TraversalLayout::SparseRows
        }
    }
    fn for_each_entry(
        &self,
        rows: Range<usize>,
        columns: Range<usize>,
        mut visitor: impl FnMut(usize, usize, f64),
    ) -> Result<(), Infallible> {
        assert!(rows.start <= rows.end && rows.end <= self.nrows());
        assert!(columns.start <= columns.end && columns.end <= self.ncols());
        let (outer, inner) = if self.csc {
            (columns, rows)
        } else {
            (rows, columns)
        };
        for o in outer {
            for k in inner.clone() {
                let (i, j) = if self.csc { (k, o) } else { (o, k) };
                let value = self.dense.get(i, j);
                if value != 0. {
                    visitor(i, j, value);
                }
            }
        }
        Ok(())
    }
}

macro_rules! unused_statistics {
    ($($name:ident $(($arg:ident: $ty:ty))?),* $(,)?) => {
        $(fn $name(&self $(, $arg: $ty)?) -> Result<Vec<f64>, Infallible> {
            $(let _ = $arg;)?
            panic!("initial fitting must use the combined statistics hook");
        })*
    };
}
macro_rules! column_statistics {
    ($ty:ty, $values:expr) => {
        impl ColumnStats<f64> for $ty {
            fn normalization_stats(
                &self, spec: ColumnNormalization,
            ) -> Result<ColumnNormalizationStats<f64>, Infallible> {
                assert_eq!(spec, ColumnNormalization::standardize());
                let values: &[[f64; 3]; 4] = $values(self);
                let means: Vec<_> = (0..3).map(|j| values.iter().map(|r| r[j]).sum::<f64>() / 4.).collect();
                let scales = (0..3).map(|j| {
                    (values.iter().map(|r| (r[j] - means[j]).powi(2)).sum::<f64>() / 3.).sqrt()
                }).collect();
                Ok((Some(means), Some(scales)))
            }
            unused_statistics!(
                col_medians, col_quantile_ranges(range: QuantileRange), col_means, col_sds(df: usize),
                col_mins, col_ranges, col_maxabs, col_l1, col_l2,
                col_l1_centered(centers: &[f64]), col_l2_centered(centers: &[f64]),
                col_maxabs_centered(centers: &[f64]),
            );
        }
    };
}
fn dense_values(matrix: &Dense) -> &[[f64; 3]; 4] {
    &matrix.values
}
fn sparse_values(matrix: &Sparse) -> &[[f64; 3]; 4] {
    &matrix.dense.values
}
column_statistics!(Dense, dense_values);
column_statistics!(Sparse, sparse_values);

fn data() -> Dense {
    Dense {
        values: [[1., 0., 3.], [0., 5., -2.], [4., 0., 0.], [-3., 2., 0.]],
        reads: Cell::new(0),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dense_traversal_uses_parent_coordinates_and_borrowed_forwarding() {
    let mut matrix = data();
    let mut visited = Vec::new();
    (&&mut matrix)
        .for_each_entry(1..3, 1..3, |i, j, v| visited.push((i, j, v)))
        .unwrap();
    assert_eq!(
        visited,
        vec![(1, 1, 5.), (2, 1, 0.), (1, 2, -2.), (2, 2, 0.)]
    );
    assert_eq!(matrix.reads.get(), 4);
    dense_for_each_entry(&matrix, 0..0, 0..3, |_, _, _| panic!("empty traversal")).unwrap();
    matrix
        .for_each_entry(0..4, 2..2, |_, _, _| panic!("empty traversal"))
        .unwrap();
    assert_eq!(matrix.reads.get(), 4);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn invalid_dense_ranges_panic_before_reads_even_when_empty() {
    let matrix = data();
    for (rows, columns) in [
        (Range { start: 3, end: 2 }, 0..0),
        (0..0, 4..4),
        (5..5, 0..0),
    ] {
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                matrix
                    .for_each_entry(rows, columns, |_, _, _| panic!("invalid callback"))
                    .unwrap();
            }))
            .is_err()
        );
    }
    assert_eq!(matrix.reads.get(), 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn column_chains_fit_and_multiply_custom_dense_and_sparse_sources() {
    let recipe = ColumnNormalization::standardize()
        .then(ColumnNormalization::new(Centering::Mean, Scaling::L1));
    let dense = recipe.fit_apply(data()).unwrap();
    let x = vec![0.25, -2., 3.];
    let u = vec![-1., 0.5, 0.25, 2.];
    let expected = dense.matvec(&x).unwrap();
    let transpose = dense.mat_transpose_vec(&u).unwrap();
    for csc in [true, false] {
        let sparse = recipe.fit_apply(Sparse { dense: data(), csc }).unwrap();
        let actual = sparse.matvec(&x).unwrap();
        let actual_t = sparse.mat_transpose_vec(&u).unwrap();
        for (actual, expected) in actual.iter().zip(&expected) {
            approx::assert_relative_eq!(actual, expected, epsilon = 1e-12);
        }
        for (actual, expected) in actual_t.iter().zip(&transpose) {
            approx::assert_relative_eq!(actual, expected, epsilon = 1e-12);
        }
        let left: f64 = actual.iter().zip(&u).map(|(a, b)| a * b).sum();
        let right: f64 = actual_t.iter().zip(&x).map(|(a, b)| a * b).sum();
        approx::assert_relative_eq!(left, right, epsilon = 1e-12);
        for (left, right) in [
            (
                dense.normalization().first(),
                sparse.normalization().first(),
            ),
            (dense.normalization().next(), sparse.normalization().next()),
        ] {
            for (left, right) in [
                (left.centers().unwrap(), right.centers().unwrap()),
                (left.scales().unwrap(), right.scales().unwrap()),
            ] {
                for (a, b) in left.iter().zip(right) {
                    approx::assert_relative_eq!(a, b, epsilon = 1e-12);
                }
            }
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn empty_chain_products_preserve_nonfinite_alpha_without_reads() {
    fn check<M: MatrixTraversal<f64> + ColumnStats<f64>>(source: &M, reads: &Cell<usize>) {
        let operator = ColumnNormalization::standardize()
            .then(ColumnNormalization::new(Centering::Mean, Scaling::L1))
            .fit_apply(source)
            .unwrap();
        let before = reads.get();
        for alpha in [f64::INFINITY, f64::NAN] {
            let mut out = vec![123.; 2];
            operator
                .submatrix(1..3, 1..1)
                .matvec_scaled_into(alpha, &Vec::<f64>::new(), 0., &mut out)
                .unwrap();
            assert!(out.iter().all(|value| value.is_nan()));
            operator
                .submatrix(1..1, 1..3)
                .mat_transpose_vec_scaled_into(alpha, &Vec::<f64>::new(), 0., &mut out)
                .unwrap();
            assert!(out.iter().all(|value| value.is_nan()));
        }
        assert_eq!(reads.get(), before);
    }
    let dense = data();
    check(&dense, &dense.reads);
    for csc in [true, false] {
        let sparse = Sparse { dense: data(), csc };
        check(&sparse, &sparse.dense.reads);
    }
}
