//! Bounded normalized panels for rectangular weighted operations.
#![allow(clippy::too_many_arguments)]

use std::ops::Range;

use crate::{MatrixShape, MatrixWrite, Scalar, SubmatrixKernel, VectorView, VectorViewMut};

const TILE: usize = 32;

struct Panel<F> {
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

    fn read<M: SubmatrixKernel<F> + ?Sized>(
        &mut self,
        matrix: &M,
        rows: Range<usize>,
        columns: Range<usize>,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
    ) -> Result<(), M::Error> {
        self.rows = rows.len();
        self.columns = columns.len();
        matrix.materialize_submatrix_into(rows, columns, centers, scales, self)
    }

    fn get(&self, i: usize, j: usize) -> F {
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

pub(crate) fn sums<F, M, W, O>(
    matrix: &M,
    rows: Range<usize>,
    columns: Range<usize>,
    centers: Option<&[F]>,
    scales: Option<&[F]>,
    weights: &W,
    out: &mut O,
) -> Result<(), M::Error>
where
    F: Scalar,
    M: SubmatrixKernel<F> + ?Sized,
    W: VectorView<F> + ?Sized,
    O: VectorViewMut<F> + ?Sized,
{
    crate::submatrix::validate_ranges(matrix, &rows, &columns);
    crate::weighted_sums::validate(rows.len(), columns.len(), weights, centers, scales, out);
    let mut panel = Panel::new();
    for j in (0..columns.len()).step_by(TILE) {
        let end = (columns.len() - j).min(TILE) + j;
        let mut sums = [F::zero(); TILE];
        for i in (0..rows.len()).step_by(TILE) {
            let nr = (rows.len() - i).min(TILE);
            panel.read(
                matrix,
                rows.start + i..rows.start + i + nr,
                columns.start + j..columns.start + end,
                centers.map(|c| &c[j..end]),
                scales.map(|s| &s[j..end]),
            )?;
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

pub(crate) fn gram<F, M, W, O>(
    matrix: &M,
    rows: Range<usize>,
    columns: Range<usize>,
    centers: Option<&[F]>,
    scales: Option<&[F]>,
    weights: &W,
    out: &mut O,
) -> Result<(), M::Error>
where
    F: Scalar,
    M: SubmatrixKernel<F> + ?Sized,
    W: VectorView<F> + ?Sized,
    O: MatrixWrite<F> + ?Sized,
{
    crate::submatrix::validate_ranges(matrix, &rows, &columns);
    crate::gram::validate(rows.len(), columns.len(), weights, centers, scales, out);
    let mut left = Panel::new();
    let mut right = Panel::new();
    for j in (0..columns.len()).step_by(TILE) {
        let j_end = (columns.len() - j).min(TILE) + j;
        for k in (j..columns.len()).step_by(TILE) {
            let k_end = (columns.len() - k).min(TILE) + k;
            let mut block = [F::zero(); TILE * TILE];
            for i in (0..rows.len()).step_by(TILE) {
                let nr = (rows.len() - i).min(TILE);
                let selected_rows = rows.start + i..rows.start + i + nr;
                left.read(
                    matrix,
                    selected_rows.clone(),
                    columns.start + j..columns.start + j_end,
                    centers.map(|c| &c[j..j_end]),
                    scales.map(|s| &s[j..j_end]),
                )?;
                if j != k {
                    right.read(
                        matrix,
                        selected_rows,
                        columns.start + k..columns.start + k_end,
                        centers.map(|c| &c[k..k_end]),
                        scales.map(|s| &s[k..k_end]),
                    )?;
                }
                let rhs = if j == k { &left } else { &right };
                for r in 0..nr {
                    let weight = weights.get(i + r);
                    for a in 0..j_end - j {
                        let weighted = left.get(r, a) * weight;
                        for b in 0..k_end - k {
                            if j + a <= k + b {
                                let sum = &mut block[a * TILE + b];
                                *sum = *sum + weighted * rhs.get(r, b);
                            }
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
