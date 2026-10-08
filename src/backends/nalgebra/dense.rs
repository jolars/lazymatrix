//! Dense nalgebra backend over owned matrices and immutable matrix views.

use super::nalgebra;

use nalgebra::base::storage::{RawStorage, RawStorageMut};
use nalgebra::{DVector, Dim, Matrix, MatrixView, U1};

use crate::backends::support::{
    MaybeSend, MaybeSync, collect_columns, max_or_nan, min_or_nan, range_or_nan,
};
use crate::traits::{
    ColumnStats, MatTransposeVec, MatTransposeVecInto, MatTransposeVecScaledInto, MatVec,
    MatVecInto, MatVecScaledInto, MatrixShape, RawColumn, RawColumns, Scalar, VectorView,
    VectorViewMut,
};

impl<F, R, S> VectorView<F> for Matrix<F, R, U1, S>
where
    F: Scalar + nalgebra::Scalar,
    R: Dim,
    S: RawStorage<F, R, U1>,
{
    fn len(&self) -> usize {
        self.nrows()
    }

    fn get(&self, index: usize) -> F {
        self[index]
    }
}

impl<F, R, S> VectorViewMut<F> for Matrix<F, R, U1, S>
where
    F: Scalar + nalgebra::Scalar,
    R: Dim,
    S: RawStorageMut<F, R, U1>,
{
    fn set(&mut self, index: usize, value: F) {
        self[index] = value;
    }
}

impl<F, R, S> RawColumn<F> for Matrix<F, R, U1, S>
where
    F: Scalar + nalgebra::Scalar,
    R: Dim,
    S: RawStorage<F, R, U1>,
{
    fn len(&self) -> usize {
        self.nrows()
    }

    fn stored_len(&self) -> usize {
        self.nrows()
    }

    fn for_each_stored(&self, mut f: impl FnMut(usize, F)) {
        for row in 0..self.nrows() {
            f(row, self[row]);
        }
    }

    fn affine_add_to<V>(&self, raw_multiplier: F, offset: F, destination: &mut V)
    where
        V: VectorViewMut<F> + ?Sized,
    {
        assert_eq!(
            destination.len(),
            self.nrows(),
            "destination length must equal column length"
        );
        for row in 0..self.nrows() {
            destination.set(
                row,
                destination.get(row) + raw_multiplier * self[row] + offset,
            );
        }
    }
}

impl<F, R, C, S> MatrixShape for Matrix<F, R, C, S>
where
    R: Dim,
    C: Dim,
    S: RawStorage<F, R, C>,
{
    fn nrows(&self) -> usize {
        Matrix::nrows(self)
    }

    fn ncols(&self) -> usize {
        Matrix::ncols(self)
    }
}

impl<F, R, C, S> RawColumns<F> for Matrix<F, R, C, S>
where
    F: Scalar + nalgebra::Scalar,
    R: Dim,
    C: Dim,
    S: RawStorage<F, R, C>,
{
    type Column<'a>
        = MatrixView<'a, F, R, U1, S::RStride, S::CStride>
    where
        Self: 'a;

    fn raw_column(&self, j: usize) -> Self::Column<'_> {
        assert!(j < self.ncols(), "column index out of bounds");
        self.column(j)
    }
}

impl<F, R, C, S> MatVec<DVector<F>> for Matrix<F, R, C, S>
where
    F: Scalar + nalgebra::Scalar,
    R: Dim,
    C: Dim,
    S: RawStorage<F, R, C>,
{
    fn matvec(&self, x: &DVector<F>) -> Result<DVector<F>, Self::Error> {
        let mut out = DVector::zeros(self.nrows());
        self.matvec_into(x, &mut out)?;
        Ok(out)
    }
}

impl<F, R, C, S> MatTransposeVec<DVector<F>> for Matrix<F, R, C, S>
where
    F: Scalar + nalgebra::Scalar,
    R: Dim,
    C: Dim,
    S: RawStorage<F, R, C>,
{
    fn mat_transpose_vec(&self, x: &DVector<F>) -> Result<DVector<F>, Self::Error> {
        let mut out = DVector::zeros(self.ncols());
        self.mat_transpose_vec_into(x, &mut out)?;
        Ok(out)
    }
}

impl<F, R, C, S> MatVecInto<DVector<F>> for Matrix<F, R, C, S>
where
    F: Scalar + nalgebra::Scalar,
    R: Dim,
    C: Dim,
    S: RawStorage<F, R, C>,
{
    fn matvec_into(&self, x: &DVector<F>, out: &mut DVector<F>) -> Result<(), Self::Error> {
        self.matvec_scaled_into(F::one(), x, F::zero(), out)
    }
}

impl<F, R, C, S> MatVecScaledInto<DVector<F>, DVector<F>, F> for Matrix<F, R, C, S>
where
    F: Scalar + nalgebra::Scalar,
    R: Dim,
    C: Dim,
    S: RawStorage<F, R, C>,
{
    fn matvec_scaled_into(
        &self,
        alpha: F,
        x: &DVector<F>,
        beta: F,
        out: &mut DVector<F>,
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
            crate::traits::scale_output(beta, out);
            return Ok(());
        }
        for i in 0..out.len() {
            let product = alpha * (0..x.len()).map(|j| self[(i, j)] * x[j]).sum::<F>();
            out[i] = if beta == F::zero() {
                product
            } else {
                product + beta * out[i]
            };
        }
        Ok(())
    }
}

impl<F, R, C, S> MatTransposeVecInto<DVector<F>> for Matrix<F, R, C, S>
where
    F: Scalar + nalgebra::Scalar,
    R: Dim,
    C: Dim,
    S: RawStorage<F, R, C>,
{
    fn mat_transpose_vec_into(
        &self,
        x: &DVector<F>,
        out: &mut DVector<F>,
    ) -> Result<(), Self::Error> {
        self.mat_transpose_vec_scaled_into(F::one(), x, F::zero(), out)
    }
}

impl<F, R, C, S> MatTransposeVecScaledInto<DVector<F>, DVector<F>, F> for Matrix<F, R, C, S>
where
    F: Scalar + nalgebra::Scalar,
    R: Dim,
    C: Dim,
    S: RawStorage<F, R, C>,
{
    fn mat_transpose_vec_scaled_into(
        &self,
        alpha: F,
        x: &DVector<F>,
        beta: F,
        out: &mut DVector<F>,
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
            crate::traits::scale_output(beta, out);
            return Ok(());
        }
        for i in 0..out.len() {
            let product = alpha * (0..x.len()).map(|j| self[(j, i)] * x[j]).sum::<F>();
            out[i] = if beta == F::zero() {
                product
            } else {
                product + beta * out[i]
            };
        }
        Ok(())
    }
}

impl<F, R, C, S> ColumnStats<F> for Matrix<F, R, C, S>
where
    F: Scalar + nalgebra::Scalar + MaybeSend + MaybeSync,
    R: Dim,
    C: Dim,
    S: RawStorage<F, R, C> + MaybeSync,
{
    fn col_means(&self) -> Result<Vec<F>, Self::Error> {
        let n = F::from_usize(self.nrows()).unwrap();
        Ok(collect_columns(self.ncols(), |j| {
            (0..self.nrows()).map(|i| self[(i, j)]).sum::<F>() / n
        }))
    }

    fn col_sds(&self) -> Result<Vec<F>, Self::Error> {
        let centers = self.col_means()?;
        let n = F::from_usize(self.nrows()).unwrap();
        Ok(collect_columns(self.ncols(), |j| {
            ((0..self.nrows())
                .map(|i| {
                    let deviation = self[(i, j)] - centers[j];
                    deviation * deviation
                })
                .sum::<F>()
                / n)
                .sqrt()
        }))
    }

    fn col_mins(&self) -> Result<Vec<F>, Self::Error> {
        Ok(collect_columns(self.ncols(), |j| {
            min_or_nan((0..self.nrows()).map(|i| self[(i, j)]))
        }))
    }

    fn col_ranges(&self) -> Result<Vec<F>, Self::Error> {
        Ok(collect_columns(self.ncols(), |j| {
            range_or_nan((0..self.nrows()).map(|i| self[(i, j)]))
        }))
    }

    fn col_maxabs(&self) -> Result<Vec<F>, Self::Error> {
        Ok(collect_columns(self.ncols(), |j| {
            max_or_nan((0..self.nrows()).map(|i| self[(i, j)].abs()))
        }))
    }

    fn col_l1(&self) -> Result<Vec<F>, Self::Error> {
        Ok(collect_columns(self.ncols(), |j| {
            (0..self.nrows()).map(|i| self[(i, j)].abs()).sum()
        }))
    }

    fn col_l2(&self) -> Result<Vec<F>, Self::Error> {
        Ok(collect_columns(self.ncols(), |j| {
            (0..self.nrows())
                .map(|i| self[(i, j)] * self[(i, j)])
                .sum::<F>()
                .sqrt()
        }))
    }

    fn col_l2_centered(&self, centers: &[F]) -> Result<Vec<F>, Self::Error> {
        assert_eq!(
            centers.len(),
            self.ncols(),
            "col_l2_centered: length mismatch"
        );
        Ok(collect_columns(self.ncols(), |j| {
            (0..self.nrows())
                .map(|i| {
                    let value = self[(i, j)] - centers[j];
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
            (0..self.nrows())
                .map(|i| (self[(i, j)] - centers[j]).abs())
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
            max_or_nan((0..self.nrows()).map(|i| (self[(i, j)] - centers[j]).abs()))
        }))
    }
}

impl<F, R, C, S> crate::MatrixErrorType for Matrix<F, R, C, S>
where
    R: Dim,
    C: Dim,
    S: RawStorage<F, R, C>,
{
    type Error = std::convert::Infallible;
}

impl<F: Scalar + nalgebra::Scalar> crate::MatrixOwned<F> for nalgebra::DMatrix<F> {
    fn zeros(nrows: usize, ncols: usize) -> Self {
        crate::materialize::validate_allocation::<F>(nrows, ncols);
        Self::from_element(nrows, ncols, F::zero())
    }
}

impl<F, R, C, S> crate::MaterializeDense<F> for Matrix<F, R, C, S>
where
    F: Scalar + nalgebra::Scalar,
    R: Dim,
    C: Dim,
    S: RawStorage<F, R, C>,
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

impl<F, R, C, S> crate::DenseNormalize<F> for Matrix<F, R, C, S>
where
    F: Scalar + nalgebra::Scalar,
    R: Dim,
    C: Dim,
    S: RawStorageMut<F, R, C>,
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
