//! Sparse operators for sprs CSC and CSR matrices, with checked borrowing.

mod csc;
mod csr;
mod gram;
mod stats;

pub use csc::SprsCsc;
pub use csr::SprsCsr;

use std::ops::Deref;

use sprs::{CsMatBase, SpIndex};

use crate::traits::{
    MatTransposeVec, MatTransposeVecInto, MatTransposeVecScaledInto, MatVec, MatVecInto,
    MatVecScaledInto, MatrixShape, Scalar, VectorView, VectorViewMut,
};

impl<F, I, IP, IS, DS, Iptr> MatrixShape for CsMatBase<F, I, IP, IS, DS, Iptr>
where
    I: SpIndex,
    Iptr: SpIndex,
    IP: Deref<Target = [Iptr]>,
    IS: Deref<Target = [I]>,
    DS: Deref<Target = [F]>,
{
    fn nrows(&self) -> usize {
        self.rows()
    }

    fn ncols(&self) -> usize {
        self.cols()
    }
}

impl<F, I, IP, IS, DS, Iptr, X, Y> MatVecInto<X, Y> for CsMatBase<F, I, IP, IS, DS, Iptr>
where
    F: Scalar,
    I: SpIndex,
    Iptr: SpIndex,
    IP: Deref<Target = [Iptr]>,
    IS: Deref<Target = [I]>,
    DS: Deref<Target = [F]>,
    X: VectorView<F>,
    Y: VectorViewMut<F>,
{
    fn matvec_into(&self, x: &X, out: &mut Y) -> Result<(), Self::Error> {
        self.matvec_scaled_into(F::one(), x, F::zero(), out)
    }
}

impl<F, I, IP, IS, DS, Iptr, X, Y> MatVecScaledInto<X, Y, F> for CsMatBase<F, I, IP, IS, DS, Iptr>
where
    F: Scalar,
    I: SpIndex,
    Iptr: SpIndex,
    IP: Deref<Target = [Iptr]>,
    IS: Deref<Target = [I]>,
    DS: Deref<Target = [F]>,
    X: VectorView<F>,
    Y: VectorViewMut<F>,
{
    fn matvec_scaled_into(&self, alpha: F, x: &X, beta: F, out: &mut Y) -> Result<(), Self::Error> {
        assert_eq!(
            self.cols(),
            x.len(),
            "matvec_scaled_into: dimension mismatch"
        );
        assert_eq!(
            self.rows(),
            out.len(),
            "matvec_scaled_into: output dimension mismatch"
        );
        if alpha == F::zero() {
            for row in 0..out.len() {
                out.set(
                    row,
                    if beta == F::zero() {
                        F::zero()
                    } else {
                        beta * out.get(row)
                    },
                );
            }
            return Ok(());
        }
        if self.is_csr() {
            for (row, values) in self.outer_iterator().enumerate() {
                let product: F = values.iter().map(|(col, &v)| v * x.get(col)).sum();
                out.set(
                    row,
                    if beta == F::zero() {
                        alpha * product
                    } else {
                        alpha * product + beta * out.get(row)
                    },
                );
            }
        } else {
            for row in 0..out.len() {
                out.set(
                    row,
                    if beta == F::zero() {
                        F::zero()
                    } else {
                        beta * out.get(row)
                    },
                );
            }
            for (col, values) in self.outer_iterator().enumerate() {
                let coefficient = x.get(col);
                for (row, &value) in values.iter() {
                    out.set(row, out.get(row) + alpha * (value * coefficient));
                }
            }
        }
        Ok(())
    }
}

impl<F, I, IP, IS, DS, Iptr, X, Y> MatTransposeVecInto<X, Y> for CsMatBase<F, I, IP, IS, DS, Iptr>
where
    F: Scalar,
    I: SpIndex,
    Iptr: SpIndex,
    IP: Deref<Target = [Iptr]>,
    IS: Deref<Target = [I]>,
    DS: Deref<Target = [F]>,
    X: VectorView<F>,
    Y: VectorViewMut<F>,
{
    fn mat_transpose_vec_into(&self, x: &X, out: &mut Y) -> Result<(), Self::Error> {
        self.mat_transpose_vec_scaled_into(F::one(), x, F::zero(), out)
    }
}

impl<F, I, IP, IS, DS, Iptr, X, Y> MatTransposeVecScaledInto<X, Y, F>
    for CsMatBase<F, I, IP, IS, DS, Iptr>
where
    F: Scalar,
    I: SpIndex,
    Iptr: SpIndex,
    IP: Deref<Target = [Iptr]>,
    IS: Deref<Target = [I]>,
    DS: Deref<Target = [F]>,
    X: VectorView<F>,
    Y: VectorViewMut<F>,
{
    fn mat_transpose_vec_scaled_into(
        &self,
        alpha: F,
        x: &X,
        beta: F,
        out: &mut Y,
    ) -> Result<(), Self::Error> {
        assert_eq!(
            self.rows(),
            x.len(),
            "mat_transpose_vec_scaled_into: dimension mismatch"
        );
        assert_eq!(
            self.cols(),
            out.len(),
            "mat_transpose_vec_scaled_into: output dimension mismatch"
        );
        self.transpose_view()
            .matvec_scaled_into(alpha, x, beta, out)?;
        Ok(())
    }
}

macro_rules! allocating_products {
    ($vector:ty, $zeros:expr) => {
        impl<F, I, IP, IS, DS, Iptr> MatVec<$vector> for CsMatBase<F, I, IP, IS, DS, Iptr>
        where
            F: Scalar,
            I: SpIndex,
            Iptr: SpIndex,
            IP: Deref<Target = [Iptr]>,
            IS: Deref<Target = [I]>,
            DS: Deref<Target = [F]>,
        {
            fn matvec(&self, x: &$vector) -> Result<$vector, Self::Error> {
                let mut out = $zeros(self.rows());
                self.matvec_into(x, &mut out)?;
                Ok(out)
            }
        }

        impl<F, I, IP, IS, DS, Iptr> MatTransposeVec<$vector> for CsMatBase<F, I, IP, IS, DS, Iptr>
        where
            F: Scalar,
            I: SpIndex,
            Iptr: SpIndex,
            IP: Deref<Target = [Iptr]>,
            IS: Deref<Target = [I]>,
            DS: Deref<Target = [F]>,
        {
            fn mat_transpose_vec(&self, x: &$vector) -> Result<$vector, Self::Error> {
                let mut out = $zeros(self.cols());
                self.mat_transpose_vec_into(x, &mut out)?;
                Ok(out)
            }
        }
    };
}

allocating_products!(Vec<F>, |n| vec![F::zero(); n]);
#[cfg(feature = "ndarray_v0_15")]
allocating_products!(
    crate::ndarray_0_15::Array1<F>,
    crate::ndarray_0_15::Array1::zeros
);
#[cfg(feature = "ndarray_v0_16")]
allocating_products!(
    crate::ndarray_0_16::Array1<F>,
    crate::ndarray_0_16::Array1::zeros
);
#[cfg(feature = "ndarray_v0_17")]
allocating_products!(
    crate::ndarray_0_17::Array1<F>,
    crate::ndarray_0_17::Array1::zeros
);

impl<F, I, IP, IS, DS, Iptr> crate::MatrixErrorType for CsMatBase<F, I, IP, IS, DS, Iptr>
where
    I: SpIndex,
    Iptr: SpIndex,
    IP: Deref<Target = [Iptr]>,
    IS: Deref<Target = [I]>,
    DS: Deref<Target = [F]>,
{
    type Error = std::convert::Infallible;
}
