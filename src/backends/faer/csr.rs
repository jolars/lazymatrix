//! Borrowed raw rows for faer CSR matrices and views.
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
