//! Products, column statistics, and borrowed raw rows for faer CSR matrices and views.
//!
//! CSC storage does not provide contiguous rows:
//!
//! ```compile_fail
//! # #[cfg(all(feature = "faer_v0_22", not(any(feature = "faer_v0_23", feature = "faer_v0_24"))))]
//! # use faer_0_22 as faer;
//! # #[cfg(all(feature = "faer_v0_23", not(feature = "faer_v0_24")))]
//! # use faer_0_23 as faer;
//! use lazymatrix::SparseRows;
//! fn needs_rows<M: SparseRows<f64>>() {}
//! needs_rows::<faer::sparse::SparseColMat<usize, f64>>();
//! ```
//!
//! Other index widths cannot provide borrowed `usize` slices:
//!
//! ```compile_fail
//! # #[cfg(all(feature = "faer_v0_22", not(any(feature = "faer_v0_23", feature = "faer_v0_24"))))]
//! # use faer_0_22 as faer;
//! # #[cfg(all(feature = "faer_v0_23", not(feature = "faer_v0_24")))]
//! # use faer_0_23 as faer;
//! use lazymatrix::SparseRows;
//! fn needs_rows<M: SparseRows<f64>>() {}
//! needs_rows::<faer::sparse::SparseRowMat<u32, f64>>();
//! ```

use super::faer;

use faer::prelude::Reborrow;
use faer::sparse::{SparseRowMat, SparseRowMatMut, SparseRowMatRef};

use crate::{MatrixShape, Scalar, SparseRows};

impl<F> MatrixShape for SparseRowMat<usize, F> {
    fn nrows(&self) -> usize {
        self.symbolic().nrows()
    }

    fn ncols(&self) -> usize {
        self.symbolic().ncols()
    }
}

impl<F> MatrixShape for SparseRowMatRef<'_, usize, F> {
    fn nrows(&self) -> usize {
        self.symbolic().nrows()
    }

    fn ncols(&self) -> usize {
        self.symbolic().ncols()
    }
}

impl<F> MatrixShape for SparseRowMatMut<'_, usize, F> {
    fn nrows(&self) -> usize {
        self.symbolic().nrows()
    }

    fn ncols(&self) -> usize {
        self.symbolic().ncols()
    }
}

impl<F: Scalar> SparseRows<F> for SparseRowMatRef<'_, usize, F> {
    fn sparse_row(&self, i: usize) -> (&[usize], &[F]) {
        assert!(i < self.nrows(), "row index out of bounds");
        // A row can reserve more capacity than it actually stores.
        let range = self.row_range(i);
        (&self.col_idx()[range.clone()], &self.val()[range])
    }
}

impl<F: Scalar> SparseRows<F> for SparseRowMat<usize, F> {
    fn sparse_row(&self, i: usize) -> (&[usize], &[F]) {
        assert!(i < self.nrows(), "row index out of bounds");
        let range = self.row_range(i);
        (&self.col_idx()[range.clone()], &self.val()[range])
    }
}

impl<F: Scalar> SparseRows<F> for SparseRowMatMut<'_, usize, F> {
    fn sparse_row(&self, i: usize) -> (&[usize], &[F]) {
        assert!(i < self.nrows(), "row index out of bounds");
        let view = self.rb();
        let range = view.row_range(i);
        (&view.col_idx()[range.clone()], &view.val()[range])
    }
}

impl<F> crate::MatrixErrorType for SparseRowMat<usize, F> {
    type Error = std::convert::Infallible;
}

impl<F: Scalar> crate::MaterializeDense<F> for SparseRowMat<usize, F> {
    fn materialize_normalized_into<O: crate::MatrixWrite<F> + ?Sized>(
        &self,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut O,
    ) -> Result<(), Self::Error> {
        crate::materialize::sparse_outer(self, false, centers, scales, out, |outer, visit| {
            let (indices, values) = self.sparse_row(outer);
            for (&index, &value) in indices.iter().zip(values) {
                visit(index, value);
            }
        });
        Ok(())
    }
}

impl<F> crate::MatrixErrorType for SparseRowMatRef<'_, usize, F> {
    type Error = std::convert::Infallible;
}

impl<F: Scalar> crate::MaterializeDense<F> for SparseRowMatRef<'_, usize, F> {
    fn materialize_normalized_into<O: crate::MatrixWrite<F> + ?Sized>(
        &self,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut O,
    ) -> Result<(), Self::Error> {
        crate::materialize::sparse_outer(self, false, centers, scales, out, |outer, visit| {
            let (indices, values) = self.sparse_row(outer);
            for (&index, &value) in indices.iter().zip(values) {
                visit(index, value);
            }
        });
        Ok(())
    }
}

impl<F> crate::MatrixErrorType for SparseRowMatMut<'_, usize, F> {
    type Error = std::convert::Infallible;
}

impl<F: Scalar> crate::MaterializeDense<F> for SparseRowMatMut<'_, usize, F> {
    fn materialize_normalized_into<O: crate::MatrixWrite<F> + ?Sized>(
        &self,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut O,
    ) -> Result<(), Self::Error> {
        crate::materialize::sparse_outer(self, false, centers, scales, out, |outer, visit| {
            let (indices, values) = self.sparse_row(outer);
            for (&index, &value) in indices.iter().zip(values) {
                visit(index, value);
            }
        });
        Ok(())
    }
}

#[cfg(feature = "parallel")]
fn parallelism() -> faer::Par {
    faer::Par::rayon(0)
}
#[cfg(not(feature = "parallel"))]
fn parallelism() -> faer::Par {
    faer::Par::Seq
}

macro_rules! impl_products {
    ($matrix:ty) => {
        impl<F: Scalar + super::faer_traits::ComplexField> crate::MatVec<faer::Col<F>> for $matrix {
            fn matvec(&self, x: &faer::Col<F>) -> Result<faer::Col<F>, Self::Error> {
                let mut out = faer::Col::zeros(self.nrows());
                crate::MatVecInto::matvec_into(self, x, &mut out)?;
                Ok(out)
            }
        }
        impl<F: Scalar + super::faer_traits::ComplexField> crate::MatTransposeVec<faer::Col<F>>
            for $matrix
        {
            fn mat_transpose_vec(&self, x: &faer::Col<F>) -> Result<faer::Col<F>, Self::Error> {
                let mut out = faer::Col::zeros(self.ncols());
                crate::MatTransposeVecInto::mat_transpose_vec_into(self, x, &mut out)?;
                Ok(out)
            }
        }
        impl<F: Scalar + super::faer_traits::ComplexField> crate::MatVecInto<faer::Col<F>>
            for $matrix
        {
            fn matvec_into(
                &self,
                x: &faer::Col<F>,
                out: &mut faer::Col<F>,
            ) -> Result<(), Self::Error> {
                crate::MatVecScaledInto::matvec_scaled_into(self, F::one(), x, F::zero(), out)
            }
        }
        impl<F: Scalar + super::faer_traits::ComplexField> crate::MatTransposeVecInto<faer::Col<F>>
            for $matrix
        {
            fn mat_transpose_vec_into(
                &self,
                x: &faer::Col<F>,
                out: &mut faer::Col<F>,
            ) -> Result<(), Self::Error> {
                crate::MatTransposeVecScaledInto::mat_transpose_vec_scaled_into(
                    self,
                    F::one(),
                    x,
                    F::zero(),
                    out,
                )
            }
        }
        impl<F: Scalar + super::faer_traits::ComplexField>
            crate::MatVecScaledInto<faer::Col<F>, faer::Col<F>, F> for $matrix
        {
            fn matvec_scaled_into(
                &self,
                alpha: F,
                x: &faer::Col<F>,
                beta: F,
                out: &mut faer::Col<F>,
            ) -> Result<(), Self::Error> {
                assert_eq!(
                    x.nrows(),
                    self.ncols(),
                    "matvec_scaled_into: dimension mismatch"
                );
                assert_eq!(
                    out.nrows(),
                    self.nrows(),
                    "matvec_scaled_into: output dimension mismatch"
                );
                if alpha == F::zero() {
                    crate::traits::scale_output(beta, out);
                    return Ok(());
                }
                let accumulation = if beta == F::zero() {
                    faer::Accum::Replace
                } else {
                    crate::traits::scale_output(beta, out);
                    faer::Accum::Add
                };
                super::transpose::multiply_csr(
                    self.rb(),
                    x,
                    out,
                    alpha,
                    accumulation,
                    parallelism(),
                );
                Ok(())
            }
        }
        impl<F: Scalar + super::faer_traits::ComplexField>
            crate::MatTransposeVecScaledInto<faer::Col<F>, faer::Col<F>, F> for $matrix
        {
            fn mat_transpose_vec_scaled_into(
                &self,
                alpha: F,
                x: &faer::Col<F>,
                beta: F,
                out: &mut faer::Col<F>,
            ) -> Result<(), Self::Error> {
                assert_eq!(
                    x.nrows(),
                    self.nrows(),
                    "mat_transpose_vec_scaled_into: dimension mismatch"
                );
                assert_eq!(
                    out.nrows(),
                    self.ncols(),
                    "mat_transpose_vec_scaled_into: output dimension mismatch"
                );
                if alpha == F::zero() {
                    crate::traits::scale_output(beta, out);
                    return Ok(());
                }
                let accumulation = if beta == F::zero() {
                    faer::Accum::Replace
                } else {
                    crate::traits::scale_output(beta, out);
                    faer::Accum::Add
                };
                faer::sparse::linalg::matmul::sparse_dense_matmul(
                    out.as_mat_mut(),
                    accumulation,
                    self.rb().transpose(),
                    x.as_mat(),
                    alpha,
                    parallelism(),
                );
                Ok(())
            }
        }
    };
}

impl_products!(SparseRowMat<usize, F>);
impl_products!(SparseRowMatRef<'_, usize, F>);
impl_products!(SparseRowMatMut<'_, usize, F>);
crate::csr_stats::impl_column_stats!(SparseRowMat<usize, F>);
crate::csr_stats::impl_column_stats!(SparseRowMatRef<'_, usize, F>);
crate::csr_stats::impl_column_stats!(SparseRowMatMut<'_, usize, F>);
