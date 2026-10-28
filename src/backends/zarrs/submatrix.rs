use std::ops::Range;

use super::{ZarrArray, ZarrMatrix};
use crate::{MatrixWrite, Scalar, SubmatrixKernel, VectorView, VectorViewMut};

#[allow(clippy::too_many_arguments)]
impl<A: ZarrArray<F>, F: Scalar> ZarrMatrix<A, F> {
    fn rectangular_product<X: VectorView<F> + ?Sized, Y: VectorViewMut<F> + ?Sized>(
        &self,
        rows: Range<usize>,
        columns: Range<usize>,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        transpose: bool,
        alpha: F,
        x: &X,
        beta: F,
        out: &mut Y,
    ) -> Result<(), super::ZarrMatrixError> {
        crate::submatrix::validate_ranges(self, &rows, &columns);
        crate::gram::validate_normalization(columns.len(), centers, scales);
        let (input, output) = if transpose {
            (rows.len(), columns.len())
        } else {
            (columns.len(), rows.len())
        };
        assert_eq!(x.len(), input, "submatrix input dimension mismatch");
        assert_eq!(out.len(), output, "submatrix output dimension mismatch");
        if alpha == F::zero() {
            crate::traits::scale_output(beta, out);
            return Ok(());
        }
        let mut products = vec![F::zero(); output];
        self.for_each_row_in(rows.clone(), columns.clone(), |row, first_col, values| {
            let i = row - rows.start;
            for (offset, &value) in values.iter().enumerate() {
                let j = first_col + offset - columns.start;
                if transpose {
                    products[j] = products[j] + value * x.get(i);
                } else {
                    let coefficient = scales.map_or_else(|| x.get(j), |s| x.get(j) / s[j]);
                    products[i] = products[i] + value * coefficient;
                }
            }
        })?;
        // Apply fitted corrections only after every storage read succeeds.
        // Empty inner dimensions represent empty sums even with nonfinite metadata.
        if input != 0 {
            if transpose {
                let sum = centers.map(|_| x.sum());
                for (j, value) in products.iter_mut().enumerate() {
                    if let Some(centers) = centers {
                        *value = *value - centers[j] * sum.unwrap();
                    }
                    if let Some(scales) = scales {
                        *value = *value / scales[j];
                    }
                }
            } else if let Some(centers) = centers {
                let correction: F = (0..columns.len())
                    .map(|j| {
                        let coefficient = scales.map_or_else(|| x.get(j), |s| x.get(j) / s[j]);
                        centers[j] * coefficient
                    })
                    .sum();
                for value in &mut products {
                    *value = *value - correction;
                }
            }
        }
        for (i, value) in products.into_iter().enumerate() {
            let product = alpha * value;
            out.set(
                i,
                if beta == F::zero() {
                    product
                } else {
                    product + beta * out.get(i)
                },
            );
        }
        Ok(())
    }
}

impl<A: ZarrArray<F>, F: Scalar> SubmatrixKernel<F> for ZarrMatrix<A, F> {
    fn submatrix_matvec_into<X: VectorView<F> + ?Sized, Y: VectorViewMut<F> + ?Sized>(
        &self,
        rows: Range<usize>,
        columns: Range<usize>,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        alpha: F,
        x: &X,
        beta: F,
        out: &mut Y,
    ) -> Result<(), Self::Error> {
        self.rectangular_product(rows, columns, centers, scales, false, alpha, x, beta, out)
    }

    fn submatrix_transpose_into<X: VectorView<F> + ?Sized, Y: VectorViewMut<F> + ?Sized>(
        &self,
        rows: Range<usize>,
        columns: Range<usize>,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        alpha: F,
        x: &X,
        beta: F,
        out: &mut Y,
    ) -> Result<(), Self::Error> {
        self.rectangular_product(rows, columns, centers, scales, true, alpha, x, beta, out)
    }

    fn materialize_submatrix_into<O: MatrixWrite<F> + ?Sized>(
        &self,
        rows: Range<usize>,
        columns: Range<usize>,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut O,
    ) -> Result<(), Self::Error> {
        crate::submatrix::validate_ranges(self, &rows, &columns);
        assert_eq!(
            (out.nrows(), out.ncols()),
            (rows.len(), columns.len()),
            "materialization output shape must match input"
        );
        crate::gram::validate_normalization(columns.len(), centers, scales);
        self.for_each_row_in(rows.clone(), columns.clone(), |row, first_col, values| {
            for (offset, &value) in values.iter().enumerate() {
                let j = first_col + offset - columns.start;
                out.set(
                    row - rows.start,
                    j,
                    crate::materialize::normalized(value, j, centers, scales),
                );
            }
        })
    }

    fn submatrix_weighted_column_sums_into<W, O>(
        &self,
        rows: Range<usize>,
        columns: Range<usize>,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        weights: &W,
        out: &mut O,
    ) -> Result<(), Self::Error>
    where
        W: VectorView<F> + ?Sized,
        O: VectorViewMut<F> + ?Sized,
    {
        crate::submatrix::validate_ranges(self, &rows, &columns);
        crate::weighted_sums::validate(rows.len(), columns.len(), weights, centers, scales, out);
        for j in 0..columns.len() {
            out.set(j, F::zero());
        }
        self.for_each_row_in(rows.clone(), columns.clone(), |row, first_col, values| {
            let weight = weights.get(row - rows.start);
            for (offset, &value) in values.iter().enumerate() {
                let j = first_col + offset - columns.start;
                let value = crate::materialize::normalized(value, j, centers, scales);
                out.set(j, out.get(j) + value * weight);
            }
        })
    }
}
