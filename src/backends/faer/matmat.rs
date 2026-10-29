use super::{faer, faer_traits};
use crate::{MatMatInto, MatTransposeMatInto, MatrixRead, MatrixWrite, Scalar};
use faer::prelude::Reborrow;
use faer::sparse::{
    SparseColMat, SparseColMatMut, SparseColMatRef, SparseRowMat, SparseRowMatMut, SparseRowMatRef,
};
use faer::{Accum, Mat, MatMut, MatRef, Par};

fn parallelism() -> Par {
    #[cfg(feature = "parallel")]
    {
        Par::rayon(0)
    }
    #[cfg(not(feature = "parallel"))]
    {
        Par::Seq
    }
}

macro_rules! dense {
    ($matrix:ty) => {
        impl<F: Scalar> MatrixRead<F> for $matrix {
            fn get(&self, row: usize, column: usize) -> F {
                self[(row, column)]
            }
        }
        impl<F: Scalar + faer_traits::ComplexField> MatMatInto<F> for $matrix {
            fn matmat_into<R, O>(&self, rhs: &R, out: &mut O) -> Result<(), Self::Error>
            where
                R: MatrixRead<F> + ?Sized,
                O: MatrixRead<F> + MatrixWrite<F> + ?Sized,
            {
                native(self, rhs, out, false);
                Ok(())
            }
            fn matmat_normalized_into<R, O>(
                &self,
                rhs: &R,
                centers: Option<&[F]>,
                scales: Option<&[F]>,
                out: &mut O,
            ) -> Result<(), Self::Error>
            where
                R: MatrixRead<F> + ?Sized,
                O: MatrixRead<F> + MatrixWrite<F> + ?Sized,
            {
                if centers.is_some() {
                    crate::matmat::dense::product(self, rhs, centers, scales, out, false);
                    Ok(())
                } else {
                    crate::matmat::normalized(self, rhs, centers, scales, out)
                }
            }
        }
        impl<F: Scalar + faer_traits::ComplexField> MatTransposeMatInto<F> for $matrix {
            fn mat_transpose_mat_into<R, O>(&self, rhs: &R, out: &mut O) -> Result<(), Self::Error>
            where
                R: MatrixRead<F> + ?Sized,
                O: MatrixRead<F> + MatrixWrite<F> + ?Sized,
            {
                native(self, rhs, out, true);
                Ok(())
            }
            fn mat_transpose_mat_normalized_into<R, O>(
                &self,
                rhs: &R,
                centers: Option<&[F]>,
                scales: Option<&[F]>,
                out: &mut O,
            ) -> Result<(), Self::Error>
            where
                R: MatrixRead<F> + ?Sized,
                O: MatrixRead<F> + MatrixWrite<F> + ?Sized,
            {
                if centers.is_some() {
                    crate::matmat::dense::product(self, rhs, centers, scales, out, true);
                    Ok(())
                } else {
                    crate::matmat::transpose_normalized(self, rhs, centers, scales, out)
                }
            }
        }
    };
}
dense!(Mat<F>);
dense!(MatRef<'_, F>);
dense!(MatMut<'_, F>);

fn native<F, M, R, O>(matrix: &M, rhs: &R, out: &mut O, transpose: bool)
where
    F: Scalar + faer_traits::ComplexField,
    M: MatrixRead<F> + ?Sized,
    R: MatrixRead<F> + ?Sized,
    O: MatrixRead<F> + MatrixWrite<F> + ?Sized,
{
    crate::matmat::dense::native(
        matrix,
        rhs,
        out,
        transpose,
        |left: &Mat<F>, right: &Mat<F>, output: &mut Mat<F>, (rows, depth, responses), first| {
            faer::linalg::matmul::matmul(
                output.as_mut().submatrix_mut(0, 0, rows, responses),
                if first { Accum::Replace } else { Accum::Add },
                left.as_ref().submatrix(0, 0, rows, depth),
                right.as_ref().submatrix(0, 0, depth, responses),
                F::one(),
                parallelism(),
            );
        },
    );
}

macro_rules! csc {
    ($matrix:ty) => {
        impl<F: Scalar + faer_traits::ComplexField> MatMatInto<F> for $matrix {
            fn matmat_into<R, O>(&self, rhs: &R, out: &mut O) -> Result<(), Self::Error>
            where
                R: MatrixRead<F> + ?Sized,
                O: MatrixRead<F> + MatrixWrite<F> + ?Sized,
            {
                crate::matmat::validate(self, rhs, out, false);
                crate::matmat::sparse_panels(
                    rhs,
                    out,
                    |input: &Mat<F>, output: &mut Mat<F>, width| {
                        faer::sparse::linalg::matmul::sparse_dense_matmul(
                            output.as_mut().subcols_mut(0, width),
                            Accum::Replace,
                            self.rb(),
                            input.as_ref().subcols(0, width),
                            F::one(),
                            parallelism(),
                        );
                    },
                );
                Ok(())
            }
        }
        impl<F: Scalar + faer_traits::ComplexField> MatTransposeMatInto<F> for $matrix {
            fn mat_transpose_mat_into<R, O>(&self, rhs: &R, out: &mut O) -> Result<(), Self::Error>
            where
                R: MatrixRead<F> + ?Sized,
                O: MatrixRead<F> + MatrixWrite<F> + ?Sized,
            {
                crate::matmat::validate(self, rhs, out, true);
                crate::matmat::sparse_panels(
                    rhs,
                    out,
                    |input: &Mat<F>, output: &mut Mat<F>, width| {
                        super::transpose::multiply_transpose_mat(
                            self.rb(),
                            input.as_ref().subcols(0, width),
                            output.as_mut().subcols_mut(0, width),
                            parallelism(),
                        );
                    },
                );
                Ok(())
            }
        }
    };
}
csc!(SparseColMat<usize, F>);
csc!(SparseColMatRef<'_, usize, F>);
csc!(SparseColMatMut<'_, usize, F>);

macro_rules! csr {
    ($matrix:ty) => {
        impl<F: Scalar + faer_traits::ComplexField> MatMatInto<F> for $matrix {
            fn matmat_into<R, O>(&self, rhs: &R, out: &mut O) -> Result<(), Self::Error>
            where
                R: MatrixRead<F> + ?Sized,
                O: MatrixRead<F> + MatrixWrite<F> + ?Sized,
            {
                crate::matmat::validate(self, rhs, out, false);
                crate::matmat::sparse_panels(
                    rhs,
                    out,
                    |input: &Mat<F>, output: &mut Mat<F>, width| {
                        super::transpose::multiply_csr_mat(
                            self.rb(),
                            input.as_ref().subcols(0, width),
                            output.as_mut().subcols_mut(0, width),
                            parallelism(),
                        );
                    },
                );
                Ok(())
            }
        }
        impl<F: Scalar + faer_traits::ComplexField> MatTransposeMatInto<F> for $matrix {
            fn mat_transpose_mat_into<R, O>(&self, rhs: &R, out: &mut O) -> Result<(), Self::Error>
            where
                R: MatrixRead<F> + ?Sized,
                O: MatrixRead<F> + MatrixWrite<F> + ?Sized,
            {
                crate::matmat::validate(self, rhs, out, true);
                crate::matmat::sparse_panels(
                    rhs,
                    out,
                    |input: &Mat<F>, output: &mut Mat<F>, width| {
                        faer::sparse::linalg::matmul::sparse_dense_matmul(
                            output.as_mut().subcols_mut(0, width),
                            Accum::Replace,
                            self.rb().transpose(),
                            input.as_ref().subcols(0, width),
                            F::one(),
                            parallelism(),
                        );
                    },
                );
                Ok(())
            }
        }
    };
}
csr!(SparseRowMat<usize, F>);
csr!(SparseRowMatRef<'_, usize, F>);
csr!(SparseRowMatMut<'_, usize, F>);
