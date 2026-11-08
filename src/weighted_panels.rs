//! Fixed-size logical panels shared by weighted kernels.

use std::ops::Range;

use crate::{MatrixShape, MatrixWrite, Scalar, VectorView, VectorViewMut};

const TILE: usize = 32;

pub(crate) struct Panel<F> {
    values: [F; TILE * TILE],
    rows: usize,
    columns: usize,
}

impl<F: Scalar> Panel<F> {
    fn new() -> Self {
        Self {
            values: [F::zero(); TILE * TILE],
            rows: 0,
            columns: 0,
        }
    }

    fn read<E>(
        &mut self,
        rows: Range<usize>,
        columns: Range<usize>,
        load: &mut impl FnMut(Range<usize>, Range<usize>, &mut Self) -> Result<(), E>,
    ) -> Result<(), E> {
        self.rows = rows.len();
        self.columns = columns.len();
        load(rows, columns, self)
    }

    pub(crate) fn get(&self, i: usize, j: usize) -> F {
        self.values[i * TILE + j]
    }
}

impl<F> MatrixShape for Panel<F> {
    fn nrows(&self) -> usize {
        self.rows
    }
    fn ncols(&self) -> usize {
        self.columns
    }
}

impl<F: Scalar> MatrixWrite<F> for Panel<F> {
    fn set(&mut self, i: usize, j: usize, value: F) {
        self.values[i * TILE + j] = value;
    }
}

pub(crate) fn sums<F, W, O, E>(
    shape: (usize, usize),
    weights: &W,
    out: &mut O,
    mut read: impl FnMut(Range<usize>, Range<usize>, &mut Panel<F>) -> Result<(), E>,
) -> Result<(), E>
where
    F: Scalar,
    W: VectorView<F> + ?Sized,
    O: VectorViewMut<F> + ?Sized,
{
    let (nrows, ncols) = shape;
    crate::weighted_sums::validate(nrows, ncols, weights, None, None, out);
    let mut panel = Panel::new();
    for j in (0..ncols).step_by(TILE) {
        let end = (ncols - j).min(TILE) + j;
        let mut sums = [F::zero(); TILE];
        for i in (0..nrows).step_by(TILE) {
            let nr = (nrows - i).min(TILE);
            panel.read(i..i + nr, j..end, &mut read)?;
            for r in 0..nr {
                let weight = weights.get(i + r);
                for (a, sum) in sums[..end - j].iter_mut().enumerate() {
                    *sum = *sum + panel.get(r, a) * weight;
                }
            }
        }
        for (a, &value) in sums[..end - j].iter().enumerate() {
            out.set(j + a, value);
        }
    }
    Ok(())
}

pub(crate) fn gram<F, W, O, E>(
    shape: (usize, usize),
    weights: &W,
    out: &mut O,
    mut read: impl FnMut(Range<usize>, Range<usize>, &mut Panel<F>) -> Result<(), E>,
) -> Result<(), E>
where
    F: Scalar,
    W: VectorView<F> + ?Sized,
    O: MatrixWrite<F> + ?Sized,
{
    let (nrows, ncols) = shape;
    crate::gram::validate(nrows, ncols, weights, None, None, out);
    let mut left = Panel::new();
    let mut right = Panel::new();
    for j in (0..ncols).step_by(TILE) {
        let j_end = (ncols - j).min(TILE) + j;
        for k in (j..ncols).step_by(TILE) {
            let k_end = (ncols - k).min(TILE) + k;
            let mut block = [F::zero(); TILE * TILE];
            for i in (0..nrows).step_by(TILE) {
                let nr = (nrows - i).min(TILE);
                left.read(i..i + nr, j..j_end, &mut read)?;
                if j != k {
                    right.read(i..i + nr, k..k_end, &mut read)?;
                }
                let rhs = if j == k { &left } else { &right };
                for r in 0..nr {
                    let weight = weights.get(i + r);
                    for a in 0..j_end - j {
                        let weighted = left.get(r, a) * weight;
                        let first = if j == k { a } else { 0 };
                        for b in first..k_end - k {
                            let sum = &mut block[a * TILE + b];
                            *sum = *sum + weighted * rhs.get(r, b);
                        }
                    }
                }
            }
            for a in 0..j_end - j {
                for b in 0..k_end - k {
                    if j + a <= k + b {
                        let value = block[a * TILE + b];
                        out.set(j + a, k + b, value);
                        if j + a != k + b {
                            out.set(k + b, j + a, value);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}
