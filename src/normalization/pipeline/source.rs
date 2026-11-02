//! Native raw traversal for column-only pipelines.

use crate::{ColumnStats, MatrixErrorType, MatrixRead, MatrixShape, Scalar};

#[derive(Clone, Copy)]
pub(crate) enum Layout {
    Full,
    #[cfg(any(feature = "faer_all", feature = "nalgebra_all", feature = "sprs_all"))]
    SparseRows,
    #[cfg(any(feature = "faer_all", feature = "nalgebra_all", feature = "sprs_all"))]
    SparseColumns,
}

pub(crate) trait Source<F: Scalar>: MatrixShape + MatrixErrorType + ColumnStats<F> {
    fn layout(&self) -> Layout {
        Layout::Full
    }

    // Sparse traversal coalesces duplicates in storage order, then visits
    // unique cells in sorted native outer/inner order.
    fn visit(&self, visit: impl FnMut(usize, usize, F)) -> Result<(), Self::Error>;
}

macro_rules! forward {
    ($reference:ty) => {
        impl<M: Source<F> + ?Sized, F: Scalar> Source<F> for $reference {
            fn layout(&self) -> Layout {
                (**self).layout()
            }
            fn visit(&self, visit: impl FnMut(usize, usize, F)) -> Result<(), Self::Error> {
                (**self).visit(visit)
            }
        }
    };
}
forward!(&M);
forward!(&mut M);

pub(crate) fn dense<M: MatrixRead<F> + ?Sized, F: Scalar>(
    data: &M,
    mut visit: impl FnMut(usize, usize, F),
) {
    for j in 0..data.ncols() {
        for i in 0..data.nrows() {
            visit(i, j, data.get(i, j));
        }
    }
}

#[cfg(feature = "faer_all")]
macro_rules! dense_source {
    ($matrix:ty) => {
        impl<F: crate::Scalar> crate::normalization::pipeline::Source<F> for $matrix
        where
            Self: crate::ColumnStats<F>,
        {
            fn visit(&self, visit: impl FnMut(usize, usize, F)) -> Result<(), Self::Error> {
                crate::normalization::pipeline::source::dense(self, visit);
                Ok(())
            }
        }
    };
}
#[cfg(feature = "faer_all")]
pub(crate) use dense_source;

#[cfg(any(feature = "faer_all", feature = "nalgebra_all", feature = "sprs_all"))]
pub(crate) fn sparse<F: Scalar>(
    nrows: usize,
    ncols: usize,
    csc: bool,
    mut outer: impl FnMut(usize, &mut dyn FnMut(usize, F)),
    mut visit: impl FnMut(usize, usize, F),
) {
    if nrows == 0 || ncols == 0 {
        return;
    }
    let (nouter, ninner) = if csc { (ncols, nrows) } else { (nrows, ncols) };
    let mut values = vec![F::zero(); ninner];
    let mut seen = vec![false; ninner];
    let mut touched = Vec::new();
    for o in 0..nouter {
        outer(o, &mut |i, value| {
            if seen[i] {
                values[i] = values[i] + value;
            } else {
                values[i] = value;
                seen[i] = true;
                touched.push(i);
            }
        });
        touched.sort_unstable();
        for i in touched.drain(..) {
            let (row, column) = if csc { (i, o) } else { (o, i) };
            visit(row, column, values[i]);
            seen[i] = false;
        }
    }
}

#[cfg(any(feature = "faer_all", feature = "nalgebra_all"))]
macro_rules! sparse_source {
    ($matrix:ty, $csc:expr, $borrow:path) => {
        impl<F: crate::Scalar> crate::normalization::pipeline::Source<F> for $matrix
        where
            Self: crate::ColumnStats<F>,
        {
            fn layout(&self) -> crate::normalization::pipeline::source::Layout {
                if $csc {
                    crate::normalization::pipeline::source::Layout::SparseColumns
                } else {
                    crate::normalization::pipeline::source::Layout::SparseRows
                }
            }
            fn visit(&self, visit: impl FnMut(usize, usize, F)) -> Result<(), Self::Error> {
                crate::normalization::pipeline::sparse(
                    crate::MatrixShape::nrows(self),
                    crate::MatrixShape::ncols(self),
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
