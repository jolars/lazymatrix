use super::ndarray;

use crate::gram::{normalize, validate};
use crate::{MatrixWrite, Scalar, VectorView, WeightedGramInto, WeightedGramKernel};
use ndarray::linalg::general_mat_mul;
use ndarray::{Array2, ArrayBase, Data, DataMut, Ix2, s};

impl<F, S> MatrixWrite<F> for ArrayBase<S, Ix2>
where
    F: Scalar + 'static,
    S: DataMut<Elem = F>,
{
    fn set(&mut self, row: usize, column: usize, value: F) {
        self[(row, column)] = value;
    }
}

impl<F, S> WeightedGramInto<F> for ArrayBase<S, Ix2>
where
    F: Scalar + 'static,
    S: Data<Elem = F>,
{
    fn weighted_gram_into<W, O>(&self, weights: &W, out: &mut O) -> Result<(), Self::Error>
    where
        W: VectorView<F> + ?Sized,
        O: MatrixWrite<F> + ?Sized,
    {
        self.weighted_gram_normalized_into(weights, None, None, out)
    }
}

impl<F, S> WeightedGramKernel<F> for ArrayBase<S, Ix2>
where
    F: Scalar + 'static,
    S: Data<Elem = F>,
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
        validate(self.nrows(), self.ncols(), weights, centers, scales, out);
        let (n, p) = (self.nrows(), self.ncols());
        if p == 0 {
            return Ok(());
        }
        if n == 0 {
            for j in 0..p {
                for k in 0..p {
                    out.set(j, k, F::zero());
                }
            }
            return Ok(());
        }
        let finite = (0..n).all(|i| weights.get(i).is_finite())
            && centers.is_none_or(|c| c.iter().all(|v| v.is_finite()))
            && scales.is_none_or(|s| s.iter().all(|v| v.is_finite()))
            && (0..p).all(|j| {
                (0..n).all(|i| {
                    let x = normalize(self[(i, j)], j, centers, scales);
                    x.is_finite() && (x * weights.get(i)).is_finite()
                })
            });
        if !finite {
            for j in 0..p {
                for k in j..p {
                    let sum = (0..n)
                        .map(|i| {
                            (normalize(self[(i, j)], j, centers, scales) * weights.get(i))
                                * normalize(self[(i, k)], k, centers, scales)
                        })
                        .sum();
                    out.set(j, k, sum);
                    if j != k {
                        out.set(k, j, sum);
                    }
                }
            }
            return Ok(());
        }
        // Bounded panels allow GEMM without retaining a normalized design.
        const ROWS: usize = 256;
        const COLS: usize = 32;
        let mut left = Array2::<F>::zeros((n.min(ROWS), p.min(COLS)));
        let mut right = left.clone();
        let mut block = Array2::<F>::zeros((p.min(COLS), p.min(COLS)));
        for j in (0..p).step_by(COLS) {
            let nj = (p - j).min(COLS);
            for k in (j..p).step_by(COLS) {
                let nk = (p - k).min(COLS);
                block.fill(F::zero());
                for row in (0..n).step_by(ROWS) {
                    let nr = (n - row).min(ROWS);
                    for i in 0..nr {
                        for a in 0..nj {
                            left[(i, a)] =
                                normalize(self[(row + i, j + a)], j + a, centers, scales)
                                    * weights.get(row + i);
                        }
                        for b in 0..nk {
                            right[(i, b)] =
                                normalize(self[(row + i, k + b)], k + b, centers, scales);
                        }
                    }
                    general_mat_mul(
                        F::one(),
                        &left.slice(s![..nr, ..nj]).t(),
                        &right.slice(s![..nr, ..nk]),
                        F::one(),
                        &mut block.slice_mut(s![..nj, ..nk]),
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
        Ok(())
    }
}

impl<F, S> crate::WeightedColumnSumsInto<F> for ArrayBase<S, Ix2>
where
    F: Scalar + 'static,
    S: Data<Elem = F>,
{
    fn weighted_column_sums_into<W, O>(&self, weights: &W, out: &mut O) -> Result<(), Self::Error>
    where
        W: VectorView<F> + ?Sized,
        O: crate::VectorViewMut<F> + ?Sized,
    {
        crate::WeightedColumnSumsKernel::weighted_column_sums_normalized_into(
            self, weights, None, None, out,
        )
    }
}

impl<F, S> crate::WeightedColumnSumsKernel<F> for ArrayBase<S, Ix2>
where
    F: Scalar + 'static,
    S: Data<Elem = F>,
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
        O: crate::VectorViewMut<F> + ?Sized,
    {
        crate::weighted_sums::validate(self.nrows(), self.ncols(), weights, centers, scales, out);
        for j in 0..self.ncols() {
            out.set(
                j,
                (0..self.nrows())
                    .map(|i| normalize(self[(i, j)], j, centers, scales) * weights.get(i))
                    .sum(),
            );
        }
        Ok(())
    }
}
