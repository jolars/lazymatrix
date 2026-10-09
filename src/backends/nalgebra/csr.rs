//! Products, column statistics, and borrowed raw rows for nalgebra-sparse CSR matrices.
//!
//! CSC storage does not provide contiguous rows:
//!
//! ```compile_fail
//! # #[cfg(all(feature = "nalgebra_v0_32", not(any(feature = "nalgebra_v0_33", feature = "nalgebra_v0_34", feature = "nalgebra_v0_35"))))]
//! # use nalgebra_sparse_0_9 as nalgebra_sparse;
//! # #[cfg(all(feature = "nalgebra_v0_33", not(any(feature = "nalgebra_v0_34", feature = "nalgebra_v0_35"))))]
//! # use nalgebra_sparse_0_10 as nalgebra_sparse;
//! # #[cfg(all(feature = "nalgebra_v0_34", not(feature = "nalgebra_v0_35")))]
//! # use nalgebra_sparse_0_11 as nalgebra_sparse;
//! use lazymatrix::SparseRows;
//! fn needs_rows<M: SparseRows<f64>>() {}
//! needs_rows::<nalgebra_sparse::CscMatrix<f64>>();
//! ```

use super::{ClosedAddAssign, ClosedMulAssign, nalgebra, nalgebra_sparse};
use crate::{
    MatTransposeVec, MatTransposeVecInto, MatTransposeVecScaledInto, MatVec, MatVecInto,
    MatVecScaledInto,
};
use nalgebra::DVector;
use nalgebra_sparse::ops::{Op, serial::spmm_csr_dense};

use nalgebra_sparse::CsrMatrix;

use crate::{MatrixShape, Scalar, SparseRows};

impl<F> MatrixShape for CsrMatrix<F> {
    fn nrows(&self) -> usize {
        CsrMatrix::nrows(self)
    }

    fn ncols(&self) -> usize {
        CsrMatrix::ncols(self)
    }
}

impl<F: Scalar> SparseRows<F> for CsrMatrix<F> {
    fn sparse_row(&self, i: usize) -> (&[usize], &[F]) {
        assert!(i < self.nrows(), "row index out of bounds");
        let (offsets, columns, values) = self.csr_data();
        let range = offsets[i]..offsets[i + 1];
        (&columns[range.clone()], &values[range])
    }
}

impl<F> crate::MatrixErrorType for CsrMatrix<F> {
    type Error = std::convert::Infallible;
}

impl<F: Scalar> crate::MaterializeDense<F> for CsrMatrix<F> {
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

impl<F> MatVec<DVector<F>> for CsrMatrix<F>
where
    F: Scalar + nalgebra::Scalar + ClosedAddAssign + ClosedMulAssign,
{
    fn matvec(&self, x: &DVector<F>) -> Result<DVector<F>, Self::Error> {
        let mut out = DVector::zeros(self.nrows());
        self.matvec_into(x, &mut out)?;
        Ok(out)
    }
}

impl<F> MatTransposeVec<DVector<F>> for CsrMatrix<F>
where
    F: Scalar + nalgebra::Scalar + ClosedAddAssign + ClosedMulAssign,
{
    fn mat_transpose_vec(&self, x: &DVector<F>) -> Result<DVector<F>, Self::Error> {
        let mut out = DVector::zeros(self.ncols());
        self.mat_transpose_vec_into(x, &mut out)?;
        Ok(out)
    }
}

impl<F> MatVecInto<DVector<F>> for CsrMatrix<F>
where
    F: Scalar + nalgebra::Scalar + ClosedAddAssign + ClosedMulAssign,
{
    fn matvec_into(&self, x: &DVector<F>, out: &mut DVector<F>) -> Result<(), Self::Error> {
        self.matvec_scaled_into(F::one(), x, F::zero(), out)
    }
}

impl<F> MatVecScaledInto<DVector<F>, DVector<F>, F> for CsrMatrix<F>
where
    F: Scalar + nalgebra::Scalar + ClosedAddAssign + ClosedMulAssign,
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
        // The backend multiplies old output by beta, so zero beta alone retains NaNs.
        if beta == F::zero() {
            out.fill(F::zero());
        }
        spmm_csr_dense(
            beta,
            out.as_view_mut(),
            alpha,
            Op::NoOp(self),
            Op::NoOp(x.as_view()),
        );
        Ok(())
    }
}

impl<F> MatTransposeVecInto<DVector<F>> for CsrMatrix<F>
where
    F: Scalar + nalgebra::Scalar + ClosedAddAssign + ClosedMulAssign,
{
    fn mat_transpose_vec_into(
        &self,
        x: &DVector<F>,
        out: &mut DVector<F>,
    ) -> Result<(), Self::Error> {
        self.mat_transpose_vec_scaled_into(F::one(), x, F::zero(), out)
    }
}

impl<F> MatTransposeVecScaledInto<DVector<F>, DVector<F>, F> for CsrMatrix<F>
where
    F: Scalar + nalgebra::Scalar + ClosedAddAssign + ClosedMulAssign,
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
        // The backend multiplies old output by beta, so zero beta alone retains NaNs.
        if beta == F::zero() {
            out.fill(F::zero());
        }
        spmm_csr_dense(
            beta,
            out.as_view_mut(),
            alpha,
            Op::Transpose(self),
            Op::NoOp(x.as_view()),
        );
        Ok(())
    }
}

crate::csr_stats::impl_column_stats!(CsrMatrix<F>);
