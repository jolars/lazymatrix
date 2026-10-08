//! `faer` sparse backend (feature `faer`).
//!
//! Implements the vector traits on [`faer::Col`] and the operator,
//! statistics, and sparse-column traits on [`faer::sparse::SparseColMat`]. The
//! matrix–vector products delegate to faer's sparse multiplication routines
//! using borrowed views, without materializing a transposed matrix.
//! Column statistics and borrowed column
//! access use the CSC arrays directly, treating absent entries as zero.

use super::{faer, faer_traits};

use faer::sparse::SparseColMat;
use faer::sparse::linalg::matmul::sparse_dense_matmul;
use faer::{Accum, Col, Par};

use crate::SparseColumnRef;
use crate::backends::support::{
    MaybeSend, MaybeSync, collect_columns, max_or_nan, min_or_nan, range_or_nan, sparse_column_sd,
};
use crate::traits::{
    ColumnStats, MatTransposeVec, MatTransposeVecInto, MatTransposeVecScaledInto, MatVec,
    MatVecInto, MatVecScaledInto, MatrixShape, RawColumns, Scalar, SparseColumns,
};

#[cfg(feature = "parallel")]
fn parallelism() -> Par {
    Par::rayon(0)
}

#[cfg(not(feature = "parallel"))]
fn parallelism() -> Par {
    Par::Seq
}
// --- operator traits on SparseColMat (need the faer ComplexField bound) -------

impl<F> MatrixShape for SparseColMat<usize, F> {
    fn nrows(&self) -> usize {
        self.symbolic().nrows()
    }

    fn ncols(&self) -> usize {
        self.symbolic().ncols()
    }
}

impl<F: Scalar> SparseColumns<F> for SparseColMat<usize, F> {
    fn sparse_column(&self, j: usize) -> (&[usize], &[F]) {
        assert!(j < self.ncols(), "column index out of bounds");
        // A column can reserve more capacity than it actually stores.
        let range = self.col_range(j);
        (&self.row_idx()[range.clone()], &self.val()[range])
    }
}

impl<F: Scalar> RawColumns<F> for SparseColMat<usize, F> {
    type Column<'a>
        = SparseColumnRef<'a, F>
    where
        Self: 'a;

    fn raw_column(&self, j: usize) -> Self::Column<'_> {
        let (rows, values) = self.sparse_column(j);
        SparseColumnRef::new(rows, values, self.nrows())
    }
}

impl<F> MatVec<Col<F>> for SparseColMat<usize, F>
where
    F: Scalar + faer_traits::ComplexField,
{
    fn matvec(&self, x: &Col<F>) -> Result<Col<F>, Self::Error> {
        let mut y = Col::<F>::zeros(self.nrows());
        self.matvec_into(x, &mut y)?;
        Ok(y)
    }
}

impl<F> MatTransposeVec<Col<F>> for SparseColMat<usize, F>
where
    F: Scalar + faer_traits::ComplexField,
{
    fn mat_transpose_vec(&self, x: &Col<F>) -> Result<Col<F>, Self::Error> {
        let mut y = Col::<F>::zeros(self.ncols());
        self.mat_transpose_vec_into(x, &mut y)?;
        Ok(y)
    }
}

impl<F> MatVecInto<Col<F>> for SparseColMat<usize, F>
where
    F: Scalar + faer_traits::ComplexField,
{
    fn matvec_into(&self, x: &Col<F>, out: &mut Col<F>) -> Result<(), Self::Error> {
        self.matvec_scaled_into(F::one(), x, F::zero(), out)
    }
}

impl<F> MatVecScaledInto<Col<F>, Col<F>, F> for SparseColMat<usize, F>
where
    F: Scalar + faer_traits::ComplexField,
{
    fn matvec_scaled_into(
        &self,
        alpha: F,
        x: &Col<F>,
        beta: F,
        out: &mut Col<F>,
    ) -> Result<(), Self::Error> {
        assert_eq!(
            self.ncols(),
            x.nrows(),
            "matvec_scaled_into: dimension mismatch"
        );
        assert_eq!(
            self.nrows(),
            out.nrows(),
            "matvec_scaled_into: output dimension mismatch"
        );
        if alpha == F::zero() {
            crate::traits::scale_output(beta, out);
            return Ok(());
        }
        let accumulation = if beta == F::zero() {
            Accum::Replace
        } else {
            crate::traits::scale_output(beta, out);
            Accum::Add
        };
        sparse_dense_matmul(
            out.as_mat_mut(),
            accumulation,
            self.as_ref(),
            x.as_mat(),
            alpha,
            parallelism(),
        );
        Ok(())
    }
}

impl<F> MatTransposeVecInto<Col<F>> for SparseColMat<usize, F>
where
    F: Scalar + faer_traits::ComplexField,
{
    fn mat_transpose_vec_into(&self, x: &Col<F>, out: &mut Col<F>) -> Result<(), Self::Error> {
        self.mat_transpose_vec_scaled_into(F::one(), x, F::zero(), out)
    }
}

impl<F> MatTransposeVecScaledInto<Col<F>, Col<F>, F> for SparseColMat<usize, F>
where
    F: Scalar + faer_traits::ComplexField,
{
    fn mat_transpose_vec_scaled_into(
        &self,
        alpha: F,
        x: &Col<F>,
        beta: F,
        out: &mut Col<F>,
    ) -> Result<(), Self::Error> {
        assert_eq!(
            self.nrows(),
            x.nrows(),
            "mat_transpose_vec_scaled_into: dimension mismatch"
        );
        assert_eq!(
            self.ncols(),
            out.nrows(),
            "mat_transpose_vec_scaled_into: output dimension mismatch"
        );
        if alpha == F::zero() {
            crate::traits::scale_output(beta, out);
            return Ok(());
        }
        let accumulation = if beta == F::zero() {
            Accum::Replace
        } else {
            crate::traits::scale_output(beta, out);
            Accum::Add
        };
        super::transpose::multiply(self, x, out, alpha, accumulation, parallelism());
        Ok(())
    }
}

// --- column statistics over the CSC arrays ------------------------------------

impl<F> ColumnStats<F> for SparseColMat<usize, F>
where
    F: Scalar + faer_traits::ComplexField + MaybeSend + MaybeSync,
{
    fn col_means(&self) -> Result<Vec<F>, Self::Error> {
        let n = F::from_usize(self.nrows()).unwrap();
        let col_ptr = self.col_ptr();
        let vals = self.val();
        Ok(collect_columns(self.ncols(), |j| {
            let (start, end) = (col_ptr[j], col_ptr[j + 1]);
            let sum: F = vals[start..end].iter().copied().sum();
            sum / n
        }))
    }

    fn col_sds(&self) -> Result<Vec<F>, Self::Error> {
        let nrows = self.nrows();
        let col_ptr = self.col_ptr();
        let vals = self.val();
        Ok(collect_columns(self.ncols(), |j| {
            let (start, end) = (col_ptr[j], col_ptr[j + 1]);
            sparse_column_sd(&vals[start..end], nrows)
        }))
    }

    fn col_mins(&self) -> Result<Vec<F>, Self::Error> {
        let nrows = self.nrows();
        let col_ptr = self.col_ptr();
        let vals = self.val();
        Ok(collect_columns(self.ncols(), |j| {
            let (start, end) = (col_ptr[j], col_ptr[j + 1]);
            min_or_nan(
                vals[start..end]
                    .iter()
                    .copied()
                    .chain((end - start < nrows).then_some(F::zero())),
            )
        }))
    }

    fn col_ranges(&self) -> Result<Vec<F>, Self::Error> {
        let nrows = self.nrows();
        let col_ptr = self.col_ptr();
        let vals = self.val();
        Ok(collect_columns(self.ncols(), |j| {
            let (start, end) = (col_ptr[j], col_ptr[j + 1]);
            range_or_nan(
                vals[start..end]
                    .iter()
                    .copied()
                    .chain((end - start < nrows).then_some(F::zero())),
            )
        }))
    }

    fn col_maxabs(&self) -> Result<Vec<F>, Self::Error> {
        let col_ptr = self.col_ptr();
        let vals = self.val();
        Ok(collect_columns(self.ncols(), |j| {
            let (start, end) = (col_ptr[j], col_ptr[j + 1]);
            max_or_nan(vals[start..end].iter().map(|v| v.abs()))
        }))
    }

    fn col_l1(&self) -> Result<Vec<F>, Self::Error> {
        let col_ptr = self.col_ptr();
        let vals = self.val();
        Ok(collect_columns(self.ncols(), |j| {
            let (start, end) = (col_ptr[j], col_ptr[j + 1]);
            vals[start..end].iter().map(|value| value.abs()).sum()
        }))
    }

    fn col_l2(&self) -> Result<Vec<F>, Self::Error> {
        let col_ptr = self.col_ptr();
        let vals = self.val();
        Ok(collect_columns(self.ncols(), |j| {
            let (start, end) = (col_ptr[j], col_ptr[j + 1]);
            let sum_sq: F = vals[start..end].iter().map(|&v| v * v).sum();
            sum_sq.sqrt()
        }))
    }

    fn col_l2_centered(&self, centers: &[F]) -> Result<Vec<F>, Self::Error> {
        assert_eq!(
            centers.len(),
            self.ncols(),
            "col_l2_centered: length mismatch"
        );
        let nrows = self.nrows();
        let col_ptr = self.col_ptr();
        let vals = self.val();
        Ok(collect_columns(self.ncols(), |j| {
            let (start, end) = (col_ptr[j], col_ptr[j + 1]);
            let c = centers[j];
            let nnz = end - start;
            // Σ_stored (v − c)² + (n − nnz)·c²
            let stored: F = vals[start..end].iter().map(|&v| (v - c) * (v - c)).sum();
            let implicit = F::from_usize(nrows - nnz).unwrap();
            (stored + implicit * c * c).sqrt()
        }))
    }

    fn col_l1_centered(&self, centers: &[F]) -> Result<Vec<F>, Self::Error> {
        assert_eq!(
            centers.len(),
            self.ncols(),
            "col_l1_centered: length mismatch"
        );
        let nrows = self.nrows();
        let col_ptr = self.col_ptr();
        let vals = self.val();
        Ok(collect_columns(self.ncols(), |j| {
            let (start, end) = (col_ptr[j], col_ptr[j + 1]);
            let center = centers[j];
            let stored: F = vals[start..end]
                .iter()
                .map(|&value| (value - center).abs())
                .sum();
            if end - start < nrows {
                stored + F::from_usize(nrows - (end - start)).unwrap() * center.abs()
            } else {
                stored
            }
        }))
    }

    fn col_maxabs_centered(&self, centers: &[F]) -> Result<Vec<F>, Self::Error> {
        assert_eq!(
            centers.len(),
            self.ncols(),
            "col_maxabs_centered: length mismatch"
        );
        let nrows = self.nrows();
        let col_ptr = self.col_ptr();
        let vals = self.val();
        Ok(collect_columns(self.ncols(), |j| {
            let (start, end) = (col_ptr[j], col_ptr[j + 1]);
            let c = centers[j];
            // Implicit zeros contribute |0 − c| = |c|.
            if end - start < nrows {
                max_or_nan(
                    vals[start..end]
                        .iter()
                        .map(|&v| (v - c).abs())
                        .chain(std::iter::once(c.abs())),
                )
            } else {
                max_or_nan(vals[start..end].iter().map(|&v| (v - c).abs()))
            }
        }))
    }
}

impl<F> crate::MatrixErrorType for SparseColMat<usize, F> {
    type Error = std::convert::Infallible;
}

impl<F: Scalar> crate::MaterializeDense<F> for SparseColMat<usize, F> {
    fn materialize_normalized_into<O: crate::MatrixWrite<F> + ?Sized>(
        &self,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut O,
    ) -> Result<(), Self::Error> {
        crate::materialize::sparse_outer(self, true, centers, scales, out, |outer, visit| {
            let (indices, values) = self.sparse_column(outer);
            for (&index, &value) in indices.iter().zip(values) {
                visit(index, value);
            }
        });
        Ok(())
    }
}
