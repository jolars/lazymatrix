//! Matrix operations and column statistics for two-dimensional ndarray arrays.

use super::ndarray;

use ndarray::linalg::general_mat_vec_mul;
use ndarray::{Array1, ArrayBase, ArrayView1, Data, DataMut, Ix1, Ix2};

use crate::backends::support::{
    MaybeSend, MaybeSync, collect_columns, max_or_nan, min_or_nan, range_or_nan,
};
use crate::traits::{
    ColumnStats, MatTransposeVec, MatTransposeVecInto, MatTransposeVecScaledInto, MatVec,
    MatVecInto, MatVecScaledInto, MatrixShape, RawColumns, Scalar,
};

impl<S: Data> MatrixShape for ArrayBase<S, Ix2> {
    fn nrows(&self) -> usize {
        self.shape()[0]
    }

    fn ncols(&self) -> usize {
        self.shape()[1]
    }
}

impl<F, S> RawColumns<F> for ArrayBase<S, Ix2>
where
    F: Scalar + 'static,
    S: Data<Elem = F>,
{
    type Column<'a>
        = ArrayView1<'a, F>
    where
        Self: 'a;

    fn raw_column(&self, j: usize) -> Self::Column<'_> {
        assert!(j < self.ncols(), "column index out of bounds");
        self.column(j)
    }
}

impl<F, S> MatVec<Array1<F>> for ArrayBase<S, Ix2>
where
    F: Scalar + 'static,
    S: Data<Elem = F>,
{
    fn matvec(&self, x: &Array1<F>) -> Result<Array1<F>, Self::Error> {
        let mut out = Array1::zeros(self.nrows());
        self.matvec_into(x, &mut out)?;
        Ok(out)
    }
}

impl<F, S> MatTransposeVec<Array1<F>> for ArrayBase<S, Ix2>
where
    F: Scalar + 'static,
    S: Data<Elem = F>,
{
    fn mat_transpose_vec(&self, x: &Array1<F>) -> Result<Array1<F>, Self::Error> {
        let mut out = Array1::zeros(self.ncols());
        self.mat_transpose_vec_into(x, &mut out)?;
        Ok(out)
    }
}

impl<F, S, X, Y> MatVecInto<ArrayBase<X, Ix1>, ArrayBase<Y, Ix1>> for ArrayBase<S, Ix2>
where
    F: Scalar + 'static,
    S: Data<Elem = F>,
    X: Data<Elem = F>,
    Y: DataMut<Elem = F>,
{
    fn matvec_into(
        &self,
        x: &ArrayBase<X, Ix1>,
        out: &mut ArrayBase<Y, Ix1>,
    ) -> Result<(), Self::Error> {
        self.matvec_scaled_into(F::one(), x, F::zero(), out)
    }
}

impl<F, S, X, Y> MatVecScaledInto<ArrayBase<X, Ix1>, ArrayBase<Y, Ix1>, F> for ArrayBase<S, Ix2>
where
    F: Scalar + 'static,
    S: Data<Elem = F>,
    X: Data<Elem = F>,
    Y: DataMut<Elem = F>,
{
    fn matvec_scaled_into(
        &self,
        alpha: F,
        x: &ArrayBase<X, Ix1>,
        beta: F,
        out: &mut ArrayBase<Y, Ix1>,
    ) -> Result<(), Self::Error> {
        assert_eq!(
            self.ncols(),
            x.len(),
            "matvec_scaled_into: dimension mismatch"
        );
        assert_eq!(
            self.nrows(),
            out.len(),
            "matvec_scaled_into: output dimension mismatch"
        );
        if alpha == F::zero() {
            if beta == F::zero() {
                out.fill(F::zero());
            } else if beta != F::one() {
                out.mapv_inplace(|value| beta * value);
            }
            return Ok(());
        }
        general_mat_vec_mul(alpha, self, x, beta, out);
        Ok(())
    }
}

impl<F, S, X, Y> MatTransposeVecInto<ArrayBase<X, Ix1>, ArrayBase<Y, Ix1>> for ArrayBase<S, Ix2>
where
    F: Scalar + 'static,
    S: Data<Elem = F>,
    X: Data<Elem = F>,
    Y: DataMut<Elem = F>,
{
    fn mat_transpose_vec_into(
        &self,
        x: &ArrayBase<X, Ix1>,
        out: &mut ArrayBase<Y, Ix1>,
    ) -> Result<(), Self::Error> {
        self.mat_transpose_vec_scaled_into(F::one(), x, F::zero(), out)
    }
}

impl<F, S, X, Y> MatTransposeVecScaledInto<ArrayBase<X, Ix1>, ArrayBase<Y, Ix1>, F>
    for ArrayBase<S, Ix2>
where
    F: Scalar + 'static,
    S: Data<Elem = F>,
    X: Data<Elem = F>,
    Y: DataMut<Elem = F>,
{
    fn mat_transpose_vec_scaled_into(
        &self,
        alpha: F,
        x: &ArrayBase<X, Ix1>,
        beta: F,
        out: &mut ArrayBase<Y, Ix1>,
    ) -> Result<(), Self::Error> {
        assert_eq!(
            self.nrows(),
            x.len(),
            "mat_transpose_vec_scaled_into: dimension mismatch"
        );
        assert_eq!(
            self.ncols(),
            out.len(),
            "mat_transpose_vec_scaled_into: output dimension mismatch"
        );
        if alpha == F::zero() {
            if beta == F::zero() {
                out.fill(F::zero());
            } else if beta != F::one() {
                out.mapv_inplace(|value| beta * value);
            }
            return Ok(());
        }
        general_mat_vec_mul(alpha, &self.t(), x, beta, out);
        Ok(())
    }
}

impl<F, S> ColumnStats<F> for ArrayBase<S, Ix2>
where
    F: Scalar + MaybeSend + MaybeSync,
    S: Data<Elem = F> + MaybeSync,
{
    fn col_means(&self) -> Result<Vec<F>, Self::Error> {
        let n = F::from_usize(self.nrows()).unwrap();
        Ok(collect_columns(self.ncols(), |j| {
            self.column(j).iter().copied().sum::<F>() / n
        }))
    }

    fn col_sds(&self) -> Result<Vec<F>, Self::Error> {
        let centers = self.col_means()?;
        let n = F::from_usize(self.nrows()).unwrap();
        Ok(collect_columns(self.ncols(), |j| {
            (self
                .column(j)
                .iter()
                .map(|&value| {
                    let deviation = value - centers[j];
                    deviation * deviation
                })
                .sum::<F>()
                / n)
                .sqrt()
        }))
    }

    fn col_mins(&self) -> Result<Vec<F>, Self::Error> {
        Ok(collect_columns(self.ncols(), |j| {
            min_or_nan(self.column(j).iter().copied())
        }))
    }

    fn col_ranges(&self) -> Result<Vec<F>, Self::Error> {
        Ok(collect_columns(self.ncols(), |j| {
            range_or_nan(self.column(j).iter().copied())
        }))
    }

    fn col_maxabs(&self) -> Result<Vec<F>, Self::Error> {
        Ok(collect_columns(self.ncols(), |j| {
            max_or_nan(self.column(j).iter().map(|x| x.abs()))
        }))
    }

    fn col_l1(&self) -> Result<Vec<F>, Self::Error> {
        Ok(collect_columns(self.ncols(), |j| {
            self.column(j).iter().map(|x| x.abs()).sum()
        }))
    }

    fn col_l2(&self) -> Result<Vec<F>, Self::Error> {
        Ok(collect_columns(self.ncols(), |j| {
            self.column(j).iter().map(|&x| x * x).sum::<F>().sqrt()
        }))
    }

    fn col_l2_centered(&self, centers: &[F]) -> Result<Vec<F>, Self::Error> {
        assert_eq!(
            centers.len(),
            self.ncols(),
            "col_l2_centered: length mismatch"
        );
        Ok(collect_columns(self.ncols(), |j| {
            self.column(j)
                .iter()
                .map(|&value| {
                    let value = value - centers[j];
                    value * value
                })
                .sum::<F>()
                .sqrt()
        }))
    }

    fn col_l1_centered(&self, centers: &[F]) -> Result<Vec<F>, Self::Error> {
        assert_eq!(
            centers.len(),
            self.ncols(),
            "col_l1_centered: length mismatch"
        );
        Ok(collect_columns(self.ncols(), |j| {
            self.column(j)
                .iter()
                .map(|&value| (value - centers[j]).abs())
                .sum()
        }))
    }

    fn col_maxabs_centered(&self, centers: &[F]) -> Result<Vec<F>, Self::Error> {
        assert_eq!(
            centers.len(),
            self.ncols(),
            "col_maxabs_centered: length mismatch"
        );
        Ok(collect_columns(self.ncols(), |j| {
            max_or_nan(
                self.column(j)
                    .iter()
                    .map(|&value| (value - centers[j]).abs()),
            )
        }))
    }
}

impl<S: Data> crate::MatrixErrorType for ArrayBase<S, Ix2> {
    type Error = std::convert::Infallible;
}

impl<F: Scalar + 'static> crate::MatrixOwned<F> for ndarray::Array2<F> {
    fn zeros(nrows: usize, ncols: usize) -> Self {
        crate::materialize::validate_allocation::<F>(nrows, ncols);
        Self::from_elem((nrows, ncols), F::zero())
    }
}

impl<F, S> crate::MaterializeDense<F> for ArrayBase<S, Ix2>
where
    F: Scalar,
    S: Data<Elem = F>,
{
    fn materialize_normalized_into<O: crate::MatrixWrite<F> + ?Sized>(
        &self,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut O,
    ) -> Result<(), Self::Error> {
        crate::materialize::validate_output(self, out, centers, scales);
        for j in 0..self.ncols() {
            for i in 0..self.nrows() {
                out.set(
                    i,
                    j,
                    crate::materialize::normalized(self[(i, j)], j, centers, scales),
                );
            }
        }
        Ok(())
    }
}

impl<F, S> crate::DenseNormalize<F> for ArrayBase<S, Ix2>
where
    F: Scalar + 'static,
    S: DataMut<Elem = F>,
{
    fn normalize_in_place(&mut self, centers: Option<&[F]>, scales: Option<&[F]>) {
        crate::gram::validate_normalization(self.ncols(), centers, scales);
        for j in 0..self.ncols() {
            for i in 0..self.nrows() {
                self[(i, j)] = crate::materialize::normalized(self[(i, j)], j, centers, scales);
            }
        }
    }
}
