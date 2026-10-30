//! Shared batched normalization and borrowed wrapper adapters.

use crate::{
    LazyMatrix, MatMatInto, MatTransposeMatInto, MatrixRead, MatrixShape, MatrixWrite, Scalar,
    WithIntercept,
};

pub(crate) fn validate_rhs<M: MatrixShape + ?Sized, R: MatrixShape + ?Sized>(
    matrix: &M,
    rhs: &R,
    transpose: bool,
) {
    assert_eq!(
        rhs.nrows(),
        if transpose {
            matrix.nrows()
        } else {
            matrix.ncols()
        },
        "batched product: input dimension mismatch"
    );
}

pub(crate) fn validate<M, R, O>(matrix: &M, rhs: &R, out: &O, transpose: bool)
where
    M: MatrixShape + ?Sized,
    R: MatrixShape + ?Sized,
    O: MatrixShape + ?Sized,
{
    validate_rhs(matrix, rhs, transpose);
    assert_eq!(
        (out.nrows(), out.ncols()),
        (
            if transpose {
                matrix.ncols()
            } else {
                matrix.nrows()
            },
            rhs.ncols()
        ),
        "batched product: output dimension mismatch"
    );
}

pub(crate) fn zero<F: Scalar, O: MatrixWrite<F> + ?Sized>(out: &mut O) {
    for i in 0..out.nrows() {
        for k in 0..out.ncols() {
            out.set(i, k, F::zero());
        }
    }
}

#[cfg(any(
    feature = "faer_all",
    feature = "nalgebra_all",
    feature = "ndarray_all"
))]
pub(crate) mod dense {
    use super::*;

    /// Independent accumulators retain the vector kernels' reduction order.
    pub(crate) fn product<F, M, R, O>(
        matrix: &M,
        rhs: &R,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut O,
        transpose: bool,
    ) where
        F: Scalar,
        M: MatrixRead<F> + ?Sized,
        R: MatrixRead<F> + ?Sized,
        O: MatrixRead<F> + MatrixWrite<F> + ?Sized,
    {
        validate(matrix, rhs, out, transpose);
        crate::gram::validate_normalization(matrix.ncols(), centers, scales);
        const ROWS: usize = 16;
        const RESPONSES: usize = 16;
        let reduction = rhs.nrows();
        for start in (0..out.nrows()).step_by(ROWS) {
            let rows = ROWS.min(out.nrows() - start);
            for response in (0..rhs.ncols()).step_by(RESPONSES) {
                let responses = RESPONSES.min(rhs.ncols() - response);
                let mut sums = [std::iter::empty::<F>().sum::<F>(); ROWS * RESPONSES];
                for j in 0..reduction {
                    for i in 0..rows {
                        let (row, column) = if transpose {
                            (j, start + i)
                        } else {
                            (start + i, j)
                        };
                        let value = crate::materialize::normalized(
                            matrix.get(row, column),
                            column,
                            centers,
                            scales,
                        );
                        for k in 0..responses {
                            let sum = &mut sums[i * RESPONSES + k];
                            *sum = *sum + value * rhs.get(j, response + k);
                        }
                    }
                }
                for i in 0..rows {
                    for k in 0..responses {
                        out.set(start + i, response + k, sums[i * RESPONSES + k]);
                    }
                }
            }
        }
    }

    /// Bound native scratch independently of the design and response counts.
    pub(crate) fn native<F, M, R, O, B>(
        matrix: &M,
        rhs: &R,
        out: &mut O,
        transpose: bool,
        mut multiply: impl FnMut(&B, &B, &mut B, (usize, usize, usize), bool),
    ) where
        F: Scalar,
        M: MatrixRead<F> + ?Sized,
        R: MatrixRead<F> + ?Sized,
        O: MatrixRead<F> + MatrixWrite<F> + ?Sized,
        B: crate::MatrixOwned<F> + MatrixRead<F>,
    {
        validate(matrix, rhs, out, transpose);
        if rhs.ncols() == 0 || out.nrows() == 0 {
            return;
        }
        if rhs.nrows() == 0 {
            zero(out);
            return;
        }
        let mut maximum = F::zero();
        let mut finite_matrix = true;
        for j in 0..matrix.ncols() {
            for i in 0..matrix.nrows() {
                let value = matrix.get(i, j);
                maximum = maximum.max(value.abs());
                finite_matrix &= value.is_finite();
            }
        }
        let mut coefficient = F::zero();
        let finite_rhs = (0..rhs.nrows()).all(|i| {
            (0..rhs.ncols()).all(|k| {
                let value = rhs.get(i, k);
                coefficient = coefficient.max(value.abs());
                value.is_finite()
            })
        });
        if !finite_rhs
            || !finite_matrix
            || !(maximum * coefficient * F::from_usize(rhs.nrows()).unwrap()).is_finite()
        {
            product(matrix, rhs, None, None, out, transpose);
            return;
        }
        const TILE: usize = 128;
        const RESPONSES: usize = 16;
        let mut left = B::zeros(out.nrows().min(TILE), rhs.nrows().min(TILE));
        let mut right = B::zeros(rhs.nrows().min(TILE), rhs.ncols().min(RESPONSES));
        let mut block = B::zeros(out.nrows().min(TILE), rhs.ncols().min(RESPONSES));
        for start in (0..out.nrows()).step_by(TILE) {
            let rows = TILE.min(out.nrows() - start);
            for response in (0..rhs.ncols()).step_by(RESPONSES) {
                let responses = RESPONSES.min(rhs.ncols() - response);
                for inner in (0..rhs.nrows()).step_by(TILE) {
                    let depth = TILE.min(rhs.nrows() - inner);
                    for i in 0..rows {
                        for j in 0..depth {
                            left.set(
                                i,
                                j,
                                if transpose {
                                    matrix.get(inner + j, start + i)
                                } else {
                                    matrix.get(start + i, inner + j)
                                },
                            );
                        }
                    }
                    for j in 0..depth {
                        for k in 0..responses {
                            right.set(j, k, rhs.get(inner + j, response + k));
                        }
                    }
                    multiply(
                        &left,
                        &right,
                        &mut block,
                        (rows, depth, responses),
                        inner == 0,
                    );
                }
                for i in 0..rows {
                    for k in 0..responses {
                        out.set(start + i, response + k, block.get(i, k));
                    }
                }
            }
        }
    }
}

#[cfg(any(feature = "faer_all", feature = "nalgebra_all"))]
pub(crate) fn sparse_panels<F, R, O, B>(
    rhs: &R,
    out: &mut O,
    mut multiply: impl FnMut(&B, &mut B, usize),
) where
    F: Scalar,
    R: MatrixRead<F> + ?Sized,
    O: MatrixRead<F> + MatrixWrite<F> + ?Sized,
    B: crate::MatrixOwned<F> + MatrixRead<F>,
{
    if rhs.ncols() == 0 || out.nrows() == 0 {
        return;
    }
    if rhs.nrows() == 0 {
        zero(out);
        return;
    }
    const RESPONSES: usize = 16;
    let mut input = B::zeros(rhs.nrows(), rhs.ncols().min(RESPONSES));
    let mut output = B::zeros(out.nrows(), rhs.ncols().min(RESPONSES));
    for start in (0..rhs.ncols()).step_by(RESPONSES) {
        let width = RESPONSES.min(rhs.ncols() - start);
        for i in 0..rhs.nrows() {
            for k in 0..width {
                input.set(i, k, rhs.get(i, start + k));
            }
        }
        // Some native kernels evaluate beta * old_output even when beta is zero.
        zero(&mut output);
        multiply(&input, &mut output, width);
        for i in 0..out.nrows() {
            for k in 0..width {
                out.set(i, start + k, output.get(i, k));
            }
        }
    }
}

struct ScaledRows<'a, R: ?Sized, F> {
    rhs: &'a R,
    scales: &'a [F],
}
impl<R: MatrixShape + ?Sized, F> MatrixShape for ScaledRows<'_, R, F> {
    fn nrows(&self) -> usize {
        self.rhs.nrows()
    }
    fn ncols(&self) -> usize {
        self.rhs.ncols()
    }
}
impl<R: MatrixRead<F> + ?Sized, F: Scalar> MatrixRead<F> for ScaledRows<'_, R, F> {
    fn get(&self, row: usize, column: usize) -> F {
        self.rhs.get(row, column) / self.scales[row]
    }
}

pub(crate) fn normalized<M, F, R, O>(
    matrix: &M,
    rhs: &R,
    centers: Option<&[F]>,
    scales: Option<&[F]>,
    out: &mut O,
) -> Result<(), M::Error>
where
    M: MatMatInto<F> + ?Sized,
    F: Scalar,
    R: MatrixRead<F> + ?Sized,
    O: MatrixRead<F> + MatrixWrite<F> + ?Sized,
{
    validate(matrix, rhs, out, false);
    crate::gram::validate_normalization(matrix.ncols(), centers, scales);
    if rhs.ncols() == 0 || out.nrows() == 0 {
        return Ok(());
    }
    if matrix.ncols() == 0 {
        zero(out);
        return Ok(());
    }
    if let Some(scales) = scales {
        forward(matrix, &ScaledRows { rhs, scales }, centers, out)
    } else {
        forward(matrix, rhs, centers, out)
    }
}

fn forward<M, F, R, O>(
    matrix: &M,
    rhs: &R,
    centers: Option<&[F]>,
    out: &mut O,
) -> Result<(), M::Error>
where
    M: MatMatInto<F> + ?Sized,
    F: Scalar,
    R: MatrixRead<F> + ?Sized,
    O: MatrixRead<F> + MatrixWrite<F> + ?Sized,
{
    matrix.matmat_into(rhs, out)?;
    if let Some(centers) = centers {
        for k in 0..rhs.ncols() {
            let correction: F = centers
                .iter()
                .enumerate()
                .map(|(j, &c)| rhs.get(j, k) * c)
                .sum();
            for i in 0..out.nrows() {
                out.set(i, k, out.get(i, k) - correction);
            }
        }
    }
    Ok(())
}

pub(crate) fn transpose_normalized<M, F, R, O>(
    matrix: &M,
    rhs: &R,
    centers: Option<&[F]>,
    scales: Option<&[F]>,
    out: &mut O,
) -> Result<(), M::Error>
where
    M: MatTransposeMatInto<F> + ?Sized,
    F: Scalar,
    R: MatrixRead<F> + ?Sized,
    O: MatrixRead<F> + MatrixWrite<F> + ?Sized,
{
    validate(matrix, rhs, out, true);
    crate::gram::validate_normalization(matrix.ncols(), centers, scales);
    if rhs.ncols() == 0 || out.nrows() == 0 {
        return Ok(());
    }
    matrix.mat_transpose_mat_into(rhs, out)?;
    if centers.is_none() && scales.is_none() {
        return Ok(());
    }
    for k in 0..rhs.ncols() {
        let sum = centers.map(|_| (0..rhs.nrows()).map(|i| rhs.get(i, k)).sum::<F>());
        for j in 0..out.nrows() {
            let mut value = out.get(j, k);
            if let Some(centers) = centers {
                value = value - centers[j] * sum.unwrap();
            }
            if let Some(scales) = scales {
                value = value / scales[j];
            }
            out.set(j, k, value);
        }
    }
    Ok(())
}

impl<M: MatMatInto<F>, F: Scalar> MatMatInto<F>
    for LazyMatrix<M, crate::ColumnNormalizationParams<F>>
{
    fn matmat_into<R, O>(&self, rhs: &R, out: &mut O) -> Result<(), Self::Error>
    where
        R: MatrixRead<F> + ?Sized,
        O: MatrixRead<F> + MatrixWrite<F> + ?Sized,
    {
        self.data().matmat_normalized_into(
            rhs,
            self.normalization().centers(),
            self.normalization().scales(),
            out,
        )
    }
}
impl<M: MatTransposeMatInto<F>, F: Scalar> MatTransposeMatInto<F>
    for LazyMatrix<M, crate::ColumnNormalizationParams<F>>
{
    fn mat_transpose_mat_into<R, O>(&self, rhs: &R, out: &mut O) -> Result<(), Self::Error>
    where
        R: MatrixRead<F> + ?Sized,
        O: MatrixRead<F> + MatrixWrite<F> + ?Sized,
    {
        self.data().mat_transpose_mat_normalized_into(
            rhs,
            self.normalization().centers(),
            self.normalization().scales(),
            out,
        )
    }
}

struct RowOffset<M> {
    data: M,
    offset: usize,
    rows: usize,
}
impl<M: MatrixShape> MatrixShape for RowOffset<M> {
    fn nrows(&self) -> usize {
        self.rows
    }
    fn ncols(&self) -> usize {
        self.data.ncols()
    }
}
impl<M: MatrixRead<F>, F: Scalar> MatrixRead<F> for RowOffset<M> {
    fn get(&self, row: usize, column: usize) -> F {
        self.data.get(row + self.offset, column)
    }
}
impl<M: MatrixWrite<F>, F: Scalar> MatrixWrite<F> for RowOffset<M> {
    fn set(&mut self, row: usize, column: usize, value: F) {
        self.data.set(row + self.offset, column, value);
    }
}

impl<M: MatMatInto<F>, F: Scalar> MatMatInto<F> for WithIntercept<M, F> {
    fn matmat_into<R, O>(&self, rhs: &R, out: &mut O) -> Result<(), Self::Error>
    where
        R: MatrixRead<F> + ?Sized,
        O: MatrixRead<F> + MatrixWrite<F> + ?Sized,
    {
        validate(self, rhs, out, false);
        if rhs.ncols() == 0 || out.nrows() == 0 {
            return Ok(());
        }
        let predictors = RowOffset {
            data: rhs,
            offset: 1,
            rows: self.as_inner().ncols(),
        };
        if predictors.rows == 0 {
            zero(out);
        } else {
            self.as_inner().matmat_into(&predictors, out)?;
        }
        for k in 0..out.ncols() {
            let intercept = rhs.get(0, k);
            for i in 0..out.nrows() {
                out.set(i, k, out.get(i, k) + intercept);
            }
        }
        Ok(())
    }
}
impl<M: MatTransposeMatInto<F>, F: Scalar> MatTransposeMatInto<F> for WithIntercept<M, F> {
    fn mat_transpose_mat_into<R, O>(&self, rhs: &R, out: &mut O) -> Result<(), Self::Error>
    where
        R: MatrixRead<F> + ?Sized,
        O: MatrixRead<F> + MatrixWrite<F> + ?Sized,
    {
        validate(self, rhs, out, true);
        if rhs.ncols() == 0 {
            return Ok(());
        }
        let mut predictors = RowOffset {
            data: &mut *out,
            offset: 1,
            rows: self.as_inner().ncols(),
        };
        if predictors.rows != 0 {
            self.as_inner()
                .mat_transpose_mat_into(rhs, &mut predictors)?;
        }
        for k in 0..out.ncols() {
            out.set(0, k, (0..rhs.nrows()).map(|i| rhs.get(i, k)).sum());
        }
        Ok(())
    }
}
