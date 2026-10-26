//! Stable weighted sums for normalized columns.

use crate::{Scalar, VectorView, VectorViewMut};

pub(crate) fn validate<F, W, O>(
    nrows: usize,
    ncols: usize,
    weights: &W,
    centers: Option<&[F]>,
    scales: Option<&[F]>,
    out: &O,
) where
    F: Scalar,
    W: VectorView<F> + ?Sized,
    O: VectorViewMut<F> + ?Sized,
{
    assert_eq!(
        weights.len(),
        nrows,
        "weighted_column_sums_into: weights length mismatch"
    );
    assert_eq!(
        out.len(),
        ncols,
        "weighted_column_sums_into: output length mismatch"
    );
    crate::gram::validate_normalization(ncols, centers, scales);
}

#[cfg(any(feature = "faer_all", feature = "ndarray_all"))]
pub(crate) fn dense_sums<F, W, O>(
    shape: (usize, usize),
    at: impl Fn(usize, usize) -> F,
    weights: &W,
    centers: Option<&[F]>,
    scales: Option<&[F]>,
    out: &mut O,
) where
    F: Scalar,
    W: VectorView<F> + ?Sized,
    O: VectorViewMut<F> + ?Sized,
{
    let (nrows, ncols) = shape;
    validate(nrows, ncols, weights, centers, scales, out);
    for j in 0..ncols {
        out.set(
            j,
            (0..nrows)
                .map(|i| crate::gram::normalize(at(i, j), j, centers, scales) * weights.get(i))
                .sum(),
        );
    }
}

#[cfg(any(feature = "faer_all", feature = "nalgebra_all", feature = "sprs_all"))]
pub(crate) fn sparse_sums<F, M, W, O>(
    matrix: &M,
    weights: &W,
    centers: Option<&[F]>,
    scales: Option<&[F]>,
    out: &mut O,
) where
    F: Scalar,
    M: crate::SparseColumns<F> + ?Sized,
    W: VectorView<F> + ?Sized,
    O: VectorViewMut<F> + ?Sized,
{
    use crate::gram::{WeightSums, normalize};

    let (n, p) = (matrix.nrows(), matrix.ncols());
    validate(n, p, weights, centers, scales, out);
    if n == 0 || p == 0 {
        for j in 0..p {
            out.set(j, F::zero());
        }
        return;
    }
    let finite_weights = (0..n).all(|i| weights.get(i).is_finite());
    let centered = (0..p).any(|j| normalize(F::zero(), j, centers, scales) != F::zero());
    let ranges = (centered && finite_weights).then(|| WeightSums::new(weights));
    let finite_ranges = ranges.as_ref().is_none_or(WeightSums::is_finite);
    let mut scratch = None;
    for j in 0..p {
        let (rows, values) = matrix.sparse_column(j);
        let background = normalize(F::zero(), j, centers, scales);
        let mut fast = finite_weights
            && finite_ranges
            && background.is_finite()
            && rows.windows(2).all(|pair| pair[0] < pair[1]);
        let mut sum = F::zero();
        if fast {
            let mut next = 0;
            for (&row, &value) in rows.iter().zip(values) {
                if next < row && background != F::zero() {
                    sum = sum + background * ranges.as_ref().unwrap().sum(next, row);
                }
                sum = sum + normalize(value, j, centers, scales) * weights.get(row);
                next = row + 1;
            }
            if next < n && background != F::zero() {
                sum = sum + background * ranges.as_ref().unwrap().sum(next, n);
            }
            fast = sum.is_finite();
        }
        if !fast {
            // Combine duplicate raw entries before normalization, and evaluate
            // implicit zeros individually to preserve nonfinite arithmetic.
            let column = scratch.get_or_insert_with(|| vec![F::zero(); n]);
            column.fill(F::zero());
            for (&row, &value) in rows.iter().zip(values) {
                column[row] = column[row] + value;
            }
            sum = (0..n)
                .map(|i| normalize(column[i], j, centers, scales) * weights.get(i))
                .sum();
        }
        out.set(j, sum);
    }
}
