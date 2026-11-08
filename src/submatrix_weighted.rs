//! Bounded normalized panels for rectangular weighted operations.
#![allow(clippy::too_many_arguments)]

use std::ops::Range;

use crate::{MatrixWrite, Scalar, SubmatrixKernel, VectorView, VectorViewMut};

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
    crate::weighted_panels::sums((rows.len(), columns.len()), weights, out, |r, c, panel| {
        matrix.materialize_submatrix_into(
            rows.start + r.start..rows.start + r.end,
            columns.start + c.start..columns.start + c.end,
            centers.map(|v| &v[c.clone()]),
            scales.map(|v| &v[c]),
            panel,
        )
    })
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
    crate::weighted_panels::gram((rows.len(), columns.len()), weights, out, |r, c, panel| {
        matrix.materialize_submatrix_into(
            rows.start + r.start..rows.start + r.end,
            columns.start + c.start..columns.start + c.end,
            centers.map(|v| &v[c.clone()]),
            scales.map(|v| &v[c]),
            panel,
        )
    })
}
