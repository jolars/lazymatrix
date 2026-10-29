//! Native traversal helpers shared across backend releases.

#[cfg(any(feature = "faer_all", feature = "nalgebra_all", feature = "sprs_all"))]
use crate::Scalar;

/// Keep scratch proportional to the selected inner dimension, including duplicates.
#[cfg(any(feature = "faer_all", feature = "nalgebra_all", feature = "sprs_all"))]
struct Accumulator<F> {
    values: Vec<F>,
    seen: Vec<bool>,
    touched: Vec<usize>,
}
#[cfg(any(feature = "faer_all", feature = "nalgebra_all", feature = "sprs_all"))]
impl<F: Scalar> Accumulator<F> {
    fn new(len: usize) -> Self {
        Self {
            values: vec![F::zero(); len],
            seen: vec![false; len],
            touched: Vec::new(),
        }
    }
    fn add(&mut self, i: usize, value: F) {
        if self.seen[i] {
            self.values[i] = self.values[i] + value;
        } else {
            self.values[i] = value;
            self.seen[i] = true;
            self.touched.push(i);
        }
    }
    fn drain(&mut self, mut visit: impl FnMut(usize, F)) {
        for i in self.touched.drain(..) {
            visit(i, self.values[i]);
            self.seen[i] = false;
        }
    }
}

#[cfg(any(feature = "faer_all", feature = "nalgebra_all", feature = "sprs_all"))]
pub(crate) fn sparse<F: Scalar>(
    rows: &[usize],
    ncols: usize,
    csc: bool,
    mut outer: impl FnMut(usize, &mut dyn FnMut(usize, F)),
    mut visit: impl FnMut(usize, usize, F),
) {
    if rows.is_empty() || ncols == 0 {
        return;
    }
    let mut sums = Accumulator::new(if csc { rows.len() } else { ncols });
    if csc {
        for j in 0..ncols {
            outer(j, &mut |i, value| {
                if let Ok(local) = rows.binary_search(&i) {
                    sums.add(local, value);
                }
            });
            sums.drain(|i, value| visit(i, j, value));
        }
    } else {
        for (i, &source) in rows.iter().enumerate() {
            outer(source, &mut |j, value| sums.add(j, value));
            sums.drain(|j, value| visit(i, j, value));
        }
    }
}

#[cfg(feature = "faer_all")]
macro_rules! dense_source {
    ($matrix:ty) => {
        impl<F: crate::Scalar> crate::IndexedRowSource<F> for $matrix {
            const IS_DENSE: bool = true;
            fn for_each_indexed_entry(
                &self,
                rows: &[usize],
                mut visit: impl FnMut(usize, usize, F),
            ) -> Result<(), Self::Error> {
                for j in 0..crate::MatrixShape::ncols(self) {
                    for (i, &row) in rows.iter().enumerate() {
                        visit(i, j, crate::MatrixRead::get(self, row, j));
                    }
                }
                Ok(())
            }
        }
    };
}
#[cfg(feature = "faer_all")]
pub(crate) use dense_source;

#[cfg(any(feature = "faer_all", feature = "nalgebra_all"))]
macro_rules! sparse_source {
    ($matrix:ty, $csc:expr, $borrow:path) => {
        impl<F: crate::Scalar> crate::IndexedRowSource<F> for $matrix {
            fn for_each_indexed_entry(
                &self,
                rows: &[usize],
                visit: impl FnMut(usize, usize, F),
            ) -> Result<(), Self::Error> {
                crate::indexed_rows_kernels::sparse(
                    rows,
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
