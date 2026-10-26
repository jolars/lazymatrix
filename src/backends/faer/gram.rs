use super::{faer, faer_traits};
use crate::gram::{dense_gram_direct, dense_is_finite, normalize, validate};
use crate::{
    MatrixShape, MatrixWrite, Scalar, VectorView, VectorViewMut, WeightedColumnSumsInto,
    WeightedColumnSumsKernel, WeightedGramInto, WeightedGramKernel,
};
use faer::linalg::matmul::matmul;
use faer::sparse::SparseColMat;
use faer::{Accum, Mat, MatMut, MatRef, Par};

impl<F> MatrixShape for MatMut<'_, F> {
    fn nrows(&self) -> usize {
        MatMut::nrows(self)
    }
    fn ncols(&self) -> usize {
        MatMut::ncols(self)
    }
}
impl<F: Scalar> MatrixWrite<F> for Mat<F> {
    fn set(&mut self, row: usize, column: usize, value: F) {
        self[(row, column)] = value;
    }
}
impl<F: Scalar> MatrixWrite<F> for MatMut<'_, F> {
    fn set(&mut self, row: usize, column: usize, value: F) {
        self[(row, column)] = value;
    }
}

fn dense_gram<F, W, O>(
    matrix: MatRef<'_, F>,
    weights: &W,
    centers: Option<&[F]>,
    scales: Option<&[F]>,
    out: &mut O,
) where
    F: Scalar + faer_traits::ComplexField,
    W: VectorView<F> + ?Sized,
    O: MatrixWrite<F> + ?Sized,
{
    let (n, p) = (matrix.nrows(), matrix.ncols());
    validate(n, p, weights, centers, scales, out);
    if p == 0 {
        return;
    }
    if n == 0 || !dense_is_finite((n, p), |i, j| matrix[(i, j)], weights, centers, scales) {
        dense_gram_direct((n, p), |i, j| matrix[(i, j)], weights, centers, scales, out);
        return;
    }

    // Bounded panels allow native GEMM without retaining a normalized design.
    const ROWS: usize = 256;
    const COLS: usize = 32;
    let mut left = Mat::<F>::zeros(n.min(ROWS), p.min(COLS));
    let mut right = Mat::<F>::zeros(n.min(ROWS), p.min(COLS));
    let mut block = Mat::<F>::zeros(p.min(COLS), p.min(COLS));
    for j in (0..p).step_by(COLS) {
        let nj = (p - j).min(COLS);
        for k in (j..p).step_by(COLS) {
            let nk = (p - k).min(COLS);
            for row in (0..n).step_by(ROWS) {
                let nr = (n - row).min(ROWS);
                for a in 0..nj {
                    for i in 0..nr {
                        left[(i, a)] = normalize(matrix[(row + i, j + a)], j + a, centers, scales)
                            * weights.get(row + i);
                    }
                }
                for b in 0..nk {
                    for i in 0..nr {
                        right[(i, b)] = normalize(matrix[(row + i, k + b)], k + b, centers, scales);
                    }
                }
                matmul(
                    block.as_mut().submatrix_mut(0, 0, nj, nk),
                    if row == 0 { Accum::Replace } else { Accum::Add },
                    left.as_ref().submatrix(0, 0, nr, nj).transpose(),
                    right.as_ref().submatrix(0, 0, nr, nk),
                    F::one(),
                    Par::Seq,
                );
            }
            for a in 0..nj {
                for b in 0..nk {
                    if j + a <= k + b {
                        let value = block[(a, b)];
                        out.set(j + a, k + b, value);
                        if j + a != k + b {
                            out.set(k + b, j + a, value);
                        }
                    }
                }
            }
        }
    }
}

macro_rules! impl_dense_weighted_products {
    ($matrix:ty) => {
        impl<F: Scalar + faer_traits::ComplexField> WeightedGramInto<F> for $matrix {
            fn weighted_gram_into<W, O>(&self, weights: &W, out: &mut O) -> Result<(), Self::Error>
            where
                W: VectorView<F> + ?Sized,
                O: MatrixWrite<F> + ?Sized,
            {
                self.weighted_gram_normalized_into(weights, None, None, out)
            }
        }

        impl<F: Scalar + faer_traits::ComplexField> WeightedGramKernel<F> for $matrix {
            fn weighted_gram_normalized_into<W, O>(
                &self,
                weights: &W,
                centers: Option<&[F]>,
                scales: Option<&[F]>,
                out: &mut O,
            ) -> Result<(), Self::Error>
            where
                W: VectorView<F> + ?Sized,
                O: MatrixWrite<F> + ?Sized,
            {
                dense_gram(self.as_ref(), weights, centers, scales, out);
                Ok(())
            }
        }

        impl<F: Scalar> WeightedColumnSumsInto<F> for $matrix {
            fn weighted_column_sums_into<W, O>(
                &self,
                weights: &W,
                out: &mut O,
            ) -> Result<(), Self::Error>
            where
                W: VectorView<F> + ?Sized,
                O: VectorViewMut<F> + ?Sized,
            {
                self.weighted_column_sums_normalized_into(weights, None, None, out)
            }
        }

        impl<F: Scalar> WeightedColumnSumsKernel<F> for $matrix {
            fn weighted_column_sums_normalized_into<W, O>(
                &self,
                weights: &W,
                centers: Option<&[F]>,
                scales: Option<&[F]>,
                out: &mut O,
            ) -> Result<(), Self::Error>
            where
                W: VectorView<F> + ?Sized,
                O: VectorViewMut<F> + ?Sized,
            {
                crate::weighted_sums::dense_sums(
                    (self.nrows(), self.ncols()),
                    |i, j| self[(i, j)],
                    weights,
                    centers,
                    scales,
                    out,
                );
                Ok(())
            }
        }
    };
}

impl_dense_weighted_products!(Mat<F>);
impl_dense_weighted_products!(MatRef<'_, F>);
impl_dense_weighted_products!(MatMut<'_, F>);

impl<F> WeightedGramInto<F> for SparseColMat<usize, F>
where
    F: Scalar,
{
    fn weighted_gram_into<W, O>(&self, weights: &W, out: &mut O) -> Result<(), Self::Error>
    where
        W: VectorView<F> + ?Sized,
        O: MatrixWrite<F> + ?Sized,
    {
        self.weighted_gram_normalized_into(weights, None, None, out)
    }
}

impl<F> WeightedGramKernel<F> for SparseColMat<usize, F>
where
    F: Scalar,
{
    fn weighted_gram_normalized_into<W, O>(
        &self,
        weights: &W,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut O,
    ) -> Result<(), Self::Error>
    where
        W: VectorView<F> + ?Sized,
        O: MatrixWrite<F> + ?Sized,
    {
        crate::gram::sparse_gram(self, weights, centers, scales, out);
        Ok(())
    }
}

impl<F> WeightedColumnSumsInto<F> for SparseColMat<usize, F>
where
    F: Scalar,
{
    fn weighted_column_sums_into<W, O>(&self, weights: &W, out: &mut O) -> Result<(), Self::Error>
    where
        W: VectorView<F> + ?Sized,
        O: VectorViewMut<F> + ?Sized,
    {
        self.weighted_column_sums_normalized_into(weights, None, None, out)
    }
}

impl<F> WeightedColumnSumsKernel<F> for SparseColMat<usize, F>
where
    F: Scalar,
{
    fn weighted_column_sums_normalized_into<W, O>(
        &self,
        weights: &W,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut O,
    ) -> Result<(), Self::Error>
    where
        W: VectorView<F> + ?Sized,
        O: VectorViewMut<F> + ?Sized,
    {
        crate::weighted_sums::sparse_sums(self, weights, centers, scales, out);
        Ok(())
    }
}
