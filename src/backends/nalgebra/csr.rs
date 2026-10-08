//! Borrowed raw rows for nalgebra-sparse CSR matrices.
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

use super::nalgebra_sparse;

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
