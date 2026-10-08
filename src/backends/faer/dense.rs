//! Dense faer backend, including arbitrary-stride immutable matrix views.

use super::faer;

use faer::{Col, ColMut, ColRef, Mat, MatMut, MatRef};

use crate::backends::support::{
    MaybeSend, MaybeSync, collect_columns, max_or_nan, min_or_nan, range_or_nan,
};
use crate::traits::{
    ColumnStats, MatTransposeVec, MatTransposeVecInto, MatTransposeVecScaledInto, MatVec,
    MatVecInto, MatVecScaledInto, MatrixShape, RawColumn, RawColumns, Scalar, VectorView,
    VectorViewMut,
};

impl<F: Scalar> VectorView<F> for Col<F> {
    fn len(&self) -> usize {
        self.nrows()
    }

    fn get(&self, index: usize) -> F {
        self[index]
    }
}

impl<F: Scalar> VectorViewMut<F> for Col<F> {
    fn set(&mut self, index: usize, value: F) {
        self[index] = value;
    }
}

impl<F: Scalar> VectorView<F> for ColRef<'_, F> {
    fn len(&self) -> usize {
        self.nrows()
    }

    fn get(&self, index: usize) -> F {
        self[index]
    }
}

impl<F: Scalar> VectorView<F> for ColMut<'_, F> {
    fn len(&self) -> usize {
        self.nrows()
    }

    fn get(&self, index: usize) -> F {
        self[index]
    }
}

impl<F: Scalar> VectorViewMut<F> for ColMut<'_, F> {
    fn set(&mut self, index: usize, value: F) {
        self[index] = value;
    }
}

impl<F: Scalar> RawColumn<F> for ColRef<'_, F> {
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

impl<F> MatrixShape for Mat<F> {
    fn nrows(&self) -> usize {
        Mat::nrows(self)
    }

    fn ncols(&self) -> usize {
        Mat::ncols(self)
    }
}

impl<F> MatrixShape for MatRef<'_, F> {
    fn nrows(&self) -> usize {
        MatRef::nrows(self)
    }

    fn ncols(&self) -> usize {
        MatRef::ncols(self)
    }
}

impl<F: Scalar> RawColumns<F> for Mat<F> {
    type Column<'a>
        = ColRef<'a, F>
    where
        Self: 'a;

    fn raw_column(&self, j: usize) -> Self::Column<'_> {
        assert!(j < self.ncols(), "column index out of bounds");
        self.col(j)
    }
}

impl<F: Scalar> RawColumns<F> for MatRef<'_, F> {
    type Column<'a>
        = ColRef<'a, F>
    where
        Self: 'a;

    fn raw_column(&self, j: usize) -> Self::Column<'_> {
        assert!(j < self.ncols(), "column index out of bounds");
        (*self).col(j)
    }
}

fn matvec_scaled_into<F: Scalar>(
    nrows: usize,
    ncols: usize,
    at: impl Fn(usize, usize) -> F,
    alpha: F,
    x: &Col<F>,
    beta: F,
    out: &mut Col<F>,
) {
    assert_eq!(ncols, x.nrows(), "matvec_scaled_into: dimension mismatch");
    assert_eq!(
        nrows,
        out.nrows(),
        "matvec_scaled_into: output dimension mismatch"
    );
    if alpha == F::zero() {
        crate::traits::scale_output(beta, out);
        return;
    }
    for i in 0..nrows {
        let product = alpha * (0..ncols).map(|j| at(i, j) * x[j]).sum::<F>();
        out[i] = if beta == F::zero() {
            product
        } else {
            product + beta * out[i]
        };
    }
}

macro_rules! impl_dense_ops {
    ($matrix:ty) => {
        impl<F: Scalar> MatVec<Col<F>> for $matrix {
            fn matvec(&self, x: &Col<F>) -> Result<Col<F>, Self::Error> {
                let mut out = Col::from_fn(self.nrows(), |_| F::zero());
                self.matvec_into(x, &mut out)?;
                Ok(out)
            }
        }

        impl<F: Scalar> MatTransposeVec<Col<F>> for $matrix {
            fn mat_transpose_vec(&self, x: &Col<F>) -> Result<Col<F>, Self::Error> {
                let mut out = Col::from_fn(self.ncols(), |_| F::zero());
                self.mat_transpose_vec_into(x, &mut out)?;
                Ok(out)
            }
        }

        impl<F: Scalar> MatVecScaledInto<Col<F>, Col<F>, F> for $matrix {
            fn matvec_scaled_into(
                &self,
                alpha: F,
                x: &Col<F>,
                beta: F,
                out: &mut Col<F>,
            ) -> Result<(), Self::Error> {
                matvec_scaled_into(
                    self.nrows(),
                    self.ncols(),
                    |i, j| self[(i, j)],
                    alpha,
                    x,
                    beta,
                    out,
                );
                Ok(())
            }
        }
        impl<F: Scalar> MatTransposeVecScaledInto<Col<F>, Col<F>, F> for $matrix {
            fn mat_transpose_vec_scaled_into(
                &self,
                alpha: F,
                x: &Col<F>,
                beta: F,
                out: &mut Col<F>,
            ) -> Result<(), Self::Error> {
                matvec_scaled_into(
                    self.ncols(),
                    self.nrows(),
                    |i, j| self[(j, i)],
                    alpha,
                    x,
                    beta,
                    out,
                );
                Ok(())
            }
        }

        impl<F: Scalar> MatVecInto<Col<F>> for $matrix {
            fn matvec_into(&self, x: &Col<F>, out: &mut Col<F>) -> Result<(), Self::Error> {
                self.matvec_scaled_into(F::one(), x, F::zero(), out)
            }
        }

        impl<F: Scalar> MatTransposeVecInto<Col<F>> for $matrix {
            fn mat_transpose_vec_into(
                &self,
                x: &Col<F>,
                out: &mut Col<F>,
            ) -> Result<(), Self::Error> {
                self.mat_transpose_vec_scaled_into(F::one(), x, F::zero(), out)
            }
        }
    };
}

impl_dense_ops!(Mat<F>);
impl_dense_ops!(MatRef<'_, F>);
impl_dense_ops!(MatMut<'_, F>);

fn means<F>(
    nrows: usize,
    ncols: usize,
    at: impl Fn(usize, usize) -> F + MaybeSend + MaybeSync,
) -> Vec<F>
where
    F: Scalar + MaybeSend + MaybeSync,
{
    let n = F::from_usize(nrows).unwrap();
    collect_columns(ncols, |j| (0..nrows).map(|i| at(i, j)).sum::<F>() / n)
}

fn sds<F>(
    nrows: usize,
    ncols: usize,
    at: impl Fn(usize, usize) -> F + MaybeSend + MaybeSync,
) -> Vec<F>
where
    F: Scalar + MaybeSend + MaybeSync,
{
    let centers = means(nrows, ncols, &at);
    let n = F::from_usize(nrows).unwrap();
    collect_columns(ncols, |j| {
        ((0..nrows)
            .map(|i| {
                let deviation = at(i, j) - centers[j];
                deviation * deviation
            })
            .sum::<F>()
            / n)
            .sqrt()
    })
}

macro_rules! impl_dense_stats {
    ($matrix:ty) => {
        impl<F> ColumnStats<F> for $matrix
        where
            F: Scalar + MaybeSend + MaybeSync,
        {
            fn col_means(&self) -> Result<Vec<F>, Self::Error> {
                Ok(means(self.nrows(), self.ncols(), |i, j| self[(i, j)]))
            }

            fn col_sds(&self) -> Result<Vec<F>, Self::Error> {
                Ok(sds(self.nrows(), self.ncols(), |i, j| self[(i, j)]))
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
    };
}

impl_dense_stats!(Mat<F>);
impl_dense_stats!(MatRef<'_, F>);
impl_dense_stats!(MatMut<'_, F>);

impl<F> crate::MatrixErrorType for Mat<F> {
    type Error = std::convert::Infallible;
}

impl<F> crate::MatrixErrorType for MatRef<'_, F> {
    type Error = std::convert::Infallible;
}

impl<F> crate::MatrixErrorType for MatMut<'_, F> {
    type Error = std::convert::Infallible;
}
impl<F: Scalar> RawColumns<F> for MatMut<'_, F> {
    type Column<'a>
        = ColRef<'a, F>
    where
        Self: 'a;
    fn raw_column(&self, j: usize) -> Self::Column<'_> {
        self.as_ref().col(j)
    }
}
impl<F: Scalar> crate::MatrixOwned<F> for Mat<F> {
    fn zeros(nrows: usize, ncols: usize) -> Self {
        crate::materialize::validate_allocation::<F>(nrows, ncols);
        Mat::from_fn(nrows, ncols, |_, _| F::zero())
    }
}

impl<F: Scalar> crate::MaterializeDense<F> for Mat<F> {
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

impl<F: Scalar> crate::MaterializeDense<F> for MatRef<'_, F> {
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

impl<F: Scalar> crate::MaterializeDense<F> for MatMut<'_, F> {
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

impl<F: Scalar> crate::DenseNormalize<F> for Mat<F> {
    fn normalize_in_place(&mut self, centers: Option<&[F]>, scales: Option<&[F]>) {
        crate::gram::validate_normalization(self.ncols(), centers, scales);
        for j in 0..self.ncols() {
            for i in 0..self.nrows() {
                self[(i, j)] = crate::materialize::normalized(self[(i, j)], j, centers, scales);
            }
        }
    }
}

impl<F: Scalar> crate::DenseNormalize<F> for MatMut<'_, F> {
    fn normalize_in_place(&mut self, centers: Option<&[F]>, scales: Option<&[F]>) {
        crate::gram::validate_normalization(self.ncols(), centers, scales);
        for j in 0..self.ncols() {
            for i in 0..self.nrows() {
                self[(i, j)] = crate::materialize::normalized(self[(i, j)], j, centers, scales);
            }
        }
    }
}
