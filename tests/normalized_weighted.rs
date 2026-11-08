//! Weighted row and chain operations work without optional backend dependencies.

use std::{cell::RefCell, ops::Range};

use lazymatrix::{
    ColumnNormalizationParams, LazyMatrix, MatrixErrorType, MatrixShape, MatrixTraversal,
    NormalizationChainParams, RowNormalizationParams, TraversalLayout, WeightedColumnSumsInto,
    WeightedGramInto, WithIntercept,
};

#[path = "common/runner.rs"]
mod common;

struct Source {
    raw: Vec<Vec<f64>>,
    columns: usize,
    layout: TraversalLayout,
    scans: RefCell<Vec<(Range<usize>, Range<usize>)>>,
    fail: bool,
}

impl Source {
    fn new(rows: usize, columns: usize, layout: TraversalLayout) -> Self {
        Self {
            raw: common::random_matrix(457, rows, columns, 0.3).dense,
            columns,
            layout,
            scans: RefCell::new(Vec::new()),
            fail: false,
        }
    }
}

impl MatrixShape for Source {
    fn nrows(&self) -> usize {
        self.raw.len()
    }
    fn ncols(&self) -> usize {
        self.columns
    }
}

impl MatrixErrorType for Source {
    type Error = std::io::Error;
}

impl MatrixTraversal<f64> for Source {
    fn traversal_layout(&self) -> TraversalLayout {
        self.layout
    }

    fn for_each_entry(
        &self,
        rows: Range<usize>,
        columns: Range<usize>,
        mut visit: impl FnMut(usize, usize, f64),
    ) -> Result<(), Self::Error> {
        assert!(!rows.is_empty() && !columns.is_empty());
        self.scans
            .borrow_mut()
            .push((rows.clone(), columns.clone()));
        let csc = self.layout != TraversalLayout::SparseRows;
        let (outer, inner) = if csc {
            (columns, rows)
        } else {
            (rows, columns)
        };
        for o in outer {
            for k in inner.clone() {
                let (i, j) = if csc { (k, o) } else { (o, k) };
                if self.layout == TraversalLayout::Full || self.raw[i][j] != 0. {
                    visit(i, j, self.raw[i][j]);
                    if self.fail {
                        return Err(std::io::Error::other("weighted scan failed"));
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn row_and_column_chain_panels_preserve_boundaries_and_stage_order() {
    for layout in [
        TraversalLayout::Full,
        TraversalLayout::SparseRows,
        TraversalLayout::SparseColumns,
    ] {
        let source = Source::new(35, 37, layout);
        let centers: Vec<_> = (0..35).map(|i| i as f64 / 4.).collect();
        let scales: Vec<_> = (0..35).map(|i| if i % 2 == 0 { -2. } else { 4. }).collect();
        let params =
            RowNormalizationParams::from_parts(35, Some(centers.clone()), Some(scales.clone()));
        let op = params.view().apply_frozen(&source);
        let expected = common::materialize_rows(&source.raw, Some(&centers), Some(&scales));
        common::check_normalized_weighted(&op, &expected);
        let intercept: Vec<_> = expected
            .iter()
            .map(|row| std::iter::once(1.).chain(row.iter().copied()).collect())
            .collect();
        let mut augmented = common::GramOutput(vec![vec![f64::NAN; 38]; 38]);
        WithIntercept::new(&op)
            .weighted_gram_into(&[1.; 35], &mut augmented)
            .unwrap();
        common::assert_gram(&augmented, &intercept, &[1.; 35]);

        let first = ColumnNormalizationParams::from_parts(
            37,
            Some((0..37).map(|j| j as f64 / 4.).collect()),
            Some(vec![-2.; 37]),
        );
        let next =
            ColumnNormalizationParams::from_parts(37, Some(vec![0.75; 37]), Some(vec![1.5; 37]));
        let expected: Vec<Vec<_>> = source
            .raw
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(j, &v)| ((v - j as f64 / 4.) / -2. - 0.75) / 1.5)
                    .collect()
            })
            .collect();
        let params = NormalizationChainParams::from_parts(first, next);
        let op = params.view().apply_frozen(&source);
        common::check_normalized_weighted(&op, &expected);
        let mut out = common::GramOutput(vec![vec![f64::NAN; 37]; 37]);
        source.scans.borrow_mut().clear();
        op.weighted_gram_into(&[1.; 35], &mut out).unwrap();
        let scans = source.scans.borrow();
        assert!(scans.len() > 1);
        assert!(scans.iter().all(|(r, c)| r.len() <= 32 && c.len() <= 32));
        assert!(scans.iter().any(|(r, c)| r.end == 35 && c.end == 37));
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn normalized_weighted_operations_preserve_ieee_parameters_and_large_offsets() {
    for layout in [
        TraversalLayout::Full,
        TraversalLayout::SparseRows,
        TraversalLayout::SparseColumns,
    ] {
        let mut source = Source::new(3, 3, layout);
        source.raw = vec![vec![1e16, 1e16 + 2., 1e16 + 4.]; 3];
        let op = LazyMatrix::from_row_parts(&source, Some(vec![1e16; 3]), Some(vec![-2.; 3]));
        common::check_normalized_weighted(&op, &vec![vec![0., -1., -2.]; 3]);

        source.raw = vec![vec![0.; 3]; 3];
        for centers in [None, Some(vec![0.75, f64::INFINITY, f64::NAN])] {
            for scales in [None, Some(vec![-2., f64::INFINITY, f64::NAN])] {
                let op = LazyMatrix::from_row_parts(&source, centers.clone(), scales.clone());
                let expected =
                    common::materialize_rows(&source.raw, centers.as_deref(), scales.as_deref());
                common::check_normalized_weighted(&op, &expected);
            }
        }
        let params = NormalizationChainParams::from_parts(
            ColumnNormalizationParams::from_parts(3, Some(vec![1e16; 3]), None),
            ColumnNormalizationParams::from_parts(3, Some(vec![0.5; 3]), Some(vec![-2.; 3])),
        );
        source.raw = vec![vec![1e16, 1e16 + 2., 1e16 + 4.]; 3];
        common::check_normalized_weighted(
            &params.apply_frozen(&source),
            &vec![vec![0.25, -0.75, -1.75]; 3],
        );
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn empty_products_overwrite_outputs_without_storage_reads() {
    for (rows, columns) in [(0, 3), (4, 0), (0, 0)] {
        for layout in [
            TraversalLayout::Full,
            TraversalLayout::SparseRows,
            TraversalLayout::SparseColumns,
        ] {
            let source = Source::new(rows, columns, layout);
            let op = LazyMatrix::from_row_parts(&source, Some(vec![f64::NAN; rows]), None);
            common::check_normalized_weighted(&op, &source.raw);
            let params = NormalizationChainParams::from_parts(
                ColumnNormalizationParams::from_parts(columns, Some(vec![f64::NAN; columns]), None),
                ColumnNormalizationParams::<f64>::from_parts(columns, None, None),
            );
            common::check_normalized_weighted(&params.apply_frozen(&source), &source.raw);
            assert!(source.scans.borrow().is_empty());
        }
    }
}

fn check_errors<M>(op: &M, source: &Source)
where
    M: WeightedGramInto<f64, Error = std::io::Error> + WeightedColumnSumsInto<f64>,
{
    let mut out = common::GramOutput(vec![vec![19.; op.ncols()]; op.ncols()]);
    let mut sums = vec![19.; op.ncols()];
    let weights = vec![1.; op.nrows()];
    assert_eq!(
        op.weighted_gram_into(&weights, &mut out)
            .unwrap_err()
            .to_string(),
        "weighted scan failed"
    );
    assert_eq!(
        op.weighted_column_sums_into(&weights, &mut sums)
            .unwrap_err()
            .to_string(),
        "weighted scan failed"
    );
    assert!(out.0.iter().flatten().all(|&v| v == 19.));
    if source.layout != TraversalLayout::Full {
        assert!(sums.iter().all(|&v| v == 19.));
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::panic::{AssertUnwindSafe, catch_unwind};
        source.scans.borrow_mut().clear();
        for length in [op.nrows() - 1, op.nrows() + 1] {
            let bad_weights = vec![1.; length];
            assert!(
                catch_unwind(AssertUnwindSafe(
                    || op.weighted_gram_into(&bad_weights, &mut out)
                ))
                .is_err()
            );
            assert!(
                catch_unwind(AssertUnwindSafe(
                    || op.weighted_column_sums_into(&bad_weights, &mut sums)
                ))
                .is_err()
            );
        }
        let mut wrong = common::GramOutput(vec![vec![19.; op.ncols() + 1]; op.ncols()]);
        assert!(
            catch_unwind(AssertUnwindSafe(
                || op.weighted_gram_into(&weights, &mut wrong)
            ))
            .is_err()
        );
        let mut wrong = vec![19.; op.ncols() + 1];
        assert!(
            catch_unwind(AssertUnwindSafe(
                || op.weighted_column_sums_into(&weights, &mut wrong)
            ))
            .is_err()
        );
        assert!(source.scans.borrow().is_empty());
        assert!(out.0.iter().flatten().all(|&v| v == 19.));
        assert!(wrong.iter().all(|&v| v == 19.));
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn operational_errors_propagate_and_invalid_dimensions_skip_reads() {
    for layout in [
        TraversalLayout::Full,
        TraversalLayout::SparseRows,
        TraversalLayout::SparseColumns,
    ] {
        let mut source = Source::new(3, 3, layout);
        source.raw[0][0] = 7.;
        source.fail = true;
        let op = LazyMatrix::from_row_parts(&source, Some(vec![f64::NAN; 3]), None);
        check_errors(&op, &source);
        let params = NormalizationChainParams::from_parts(
            ColumnNormalizationParams::from_parts(3, Some(vec![f64::NAN; 3]), None),
            ColumnNormalizationParams::<f64>::from_parts(3, None, None),
        );
        check_errors(&params.apply_frozen(&source), &source);
    }
}
