//! Native raw traversal for column-only pipelines.

#[cfg(any(feature = "faer_all", feature = "nalgebra_all", feature = "sprs_all"))]
use std::ops::Range;

#[cfg(any(feature = "faer_all", feature = "nalgebra_all", feature = "sprs_all"))]
use crate::Scalar;

pub(crate) use crate::MatrixTraversal as Source;
pub(crate) use crate::TraversalLayout as Layout;

#[cfg(feature = "faer_all")]
macro_rules! dense_source {
    ($matrix:ty) => {
        impl<F: crate::Scalar> crate::MatrixTraversal<F> for $matrix
        where
            Self: crate::ColumnStats<F>,
        {
            fn traversal_layout(&self) -> crate::TraversalLayout {
                crate::TraversalLayout::Full
            }
            fn for_each_entry(
                &self,
                rows: std::ops::Range<usize>,
                columns: std::ops::Range<usize>,
                visit: impl FnMut(usize, usize, F),
            ) -> Result<(), Self::Error> {
                crate::dense_for_each_entry(self, rows, columns, visit)
            }
        }
    };
}
#[cfg(feature = "faer_all")]
pub(crate) use dense_source;

#[cfg(any(feature = "faer_all", feature = "nalgebra_all", feature = "sprs_all"))]
// Full scans need to eliminate selection filters inside their entry callbacks.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sparse<F: Scalar>(
    nrows: usize,
    ncols: usize,
    rows: Range<usize>,
    columns: Range<usize>,
    csc: bool,
    mut outer: impl FnMut(usize, &mut dyn FnMut(usize, F)),
    mut visit: impl FnMut(usize, usize, F),
) {
    assert!(
        rows.start <= rows.end && rows.end <= nrows,
        "traversal row range out of bounds"
    );
    assert!(
        columns.start <= columns.end && columns.end <= ncols,
        "traversal column range out of bounds"
    );
    if rows.is_empty() || columns.is_empty() {
        return;
    }
    let (outer_range, inner_range) = if csc {
        (columns, rows)
    } else {
        (rows, columns)
    };
    let mut values = vec![F::zero(); inner_range.len()];
    let mut seen = vec![false; inner_range.len()];
    let mut touched = Vec::new();
    for o in outer_range {
        outer(o, &mut |i, value| {
            if !inner_range.contains(&i) {
                return;
            }
            let local = i - inner_range.start;
            if seen[local] {
                values[local] = values[local] + value;
            } else {
                values[local] = value;
                seen[local] = true;
                touched.push(local);
            }
        });
        touched.sort_unstable();
        for local in touched.drain(..) {
            let i = inner_range.start + local;
            let (row, column) = if csc { (i, o) } else { (o, i) };
            visit(row, column, values[local]);
            seen[local] = false;
        }
    }
}

#[cfg(any(feature = "faer_all", feature = "nalgebra_all"))]
macro_rules! sparse_source {
    ($matrix:ty, $csc:expr, $borrow:path) => {
        impl<F: crate::Scalar> crate::MatrixTraversal<F> for $matrix {
            fn traversal_layout(&self) -> crate::TraversalLayout {
                if $csc {
                    crate::TraversalLayout::SparseColumns
                } else {
                    crate::TraversalLayout::SparseRows
                }
            }
            fn for_each_entry(
                &self,
                rows: std::ops::Range<usize>,
                columns: std::ops::Range<usize>,
                visit: impl FnMut(usize, usize, F),
            ) -> Result<(), Self::Error> {
                crate::normalization::pipeline::sparse(
                    crate::MatrixShape::nrows(self),
                    crate::MatrixShape::ncols(self),
                    rows,
                    columns,
                    $csc,
                    |outer, f| {
                        let (indices, values) = $borrow(self, outer);
                        for (&i, &value) in indices.iter().zip(values) {
                            f(i, value);
                        }
                    },
                    visit,
                );
                Ok(())
            }
        }
    };
}
#[cfg(any(feature = "faer_all", feature = "nalgebra_all"))]
pub(crate) use sparse_source;

#[cfg(all(
    test,
    any(feature = "faer_all", feature = "nalgebra_all", feature = "sprs_all")
))]
mod tests {
    use super::sparse;

    #[test]
    fn selected_sparse_traversal_coalesces_before_sorted_visitation() {
        for csc in [true, false] {
            let rows = 1..3;
            let columns = 2..4;
            let (outer_range, inner_range) = if csc {
                (columns.clone(), rows.clone())
            } else {
                (rows.clone(), columns.clone())
            };
            let mut scanned = Vec::new();
            let mut visited = Vec::new();
            sparse(
                4,
                5,
                rows,
                columns,
                csc,
                |outer, emit| {
                    scanned.push(outer);
                    emit(0, f64::NAN);
                    emit(inner_range.start + 1, 1e16);
                    emit(inner_range.start, 5.);
                    emit(inner_range.start + 1, -1e16);
                    emit(inner_range.start + 1, 1.);
                    emit(inner_range.start, -5.);
                },
                |i, j, value| visited.push((i, j, value)),
            );
            assert_eq!(scanned, outer_range.clone().collect::<Vec<_>>());
            let expected: Vec<_> = outer_range
                .flat_map(|outer| {
                    inner_range.clone().enumerate().map(move |(offset, inner)| {
                        let (i, j) = if csc { (inner, outer) } else { (outer, inner) };
                        (i, j, offset as f64)
                    })
                })
                .collect();
            assert_eq!(visited, expected);
        }
    }

    #[test]
    fn empty_sparse_rectangles_do_not_scan_storage() {
        for csc in [true, false] {
            for (rows, columns) in [(1..1, 0..5), (0..4, 2..2)] {
                sparse::<f64>(
                    4,
                    5,
                    rows,
                    columns,
                    csc,
                    |_, _| panic!("empty traversal reads outer storage"),
                    |_, _, _| panic!("empty traversal emits entries"),
                );
            }
        }
    }
}
