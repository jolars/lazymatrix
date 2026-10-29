//! `faer` sparse backend (feature `faer`).
//!
//! Implements the vector traits on [`faer::Col`] and the operator,
//! statistics, and sparse-column traits on owned CSC matrices and their borrowed
//! immutable and mutable views. Matrix–vector products delegate to faer's sparse
//! multiplication routines without materializing a transposed matrix. Column
//! borrowing takes O(1) time and exposes only occupied storage, preserving its
//! order and duplicates. Statistics include implicit zeros and combine duplicate
//! cells before computing nonlinear reductions.
//!
//! Means, standard deviations, extrema, and norms take O(nnz + ncols) time and
//! allocate only their output for sorted, unique columns. Columns with unsorted
//! or repeated row indices use O(k log k) time and O(k) workspace to combine k
//! stored entries per active column. Quantiles sort one working column at a time.

use super::{faer, faer_traits};

use faer::prelude::Reborrow;
use faer::sparse::linalg::matmul::sparse_dense_matmul;
use faer::sparse::{SparseColMat, SparseColMatMut, SparseColMatRef};
use faer::{Accum, Col, Par};

use crate::SparseColumnRef;
use crate::backends::support::{
    MaybeSend, MaybeSync, coalesced_column_values, collect_columns, max_or_nan, min_or_nan,
    range_or_nan, sparse_column_sd,
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

macro_rules! impl_storage {
    ($matrix:ty) => {
        impl<F> MatrixShape for $matrix {
            fn nrows(&self) -> usize {
                self.symbolic().nrows()
            }

            fn ncols(&self) -> usize {
                self.symbolic().ncols()
            }
        }

        impl<F: Scalar> SparseColumns<F> for $matrix {
            fn sparse_column(&self, j: usize) -> (&[usize], &[F]) {
                assert!(j < self.ncols(), "column index out of bounds");
                // A column can reserve more capacity than it actually stores.
                let matrix = self.rb();
                let range = matrix.col_range(j);
                (&matrix.row_idx()[range.clone()], &matrix.val()[range])
            }
        }

        impl<F: Scalar> RawColumns<F> for $matrix {
            type Column<'a>
                = SparseColumnRef<'a, F>
            where
                Self: 'a;

            fn raw_column(&self, j: usize) -> Self::Column<'_> {
                let (rows, values) = self.sparse_column(j);
                SparseColumnRef::new(rows, values, self.nrows())
            }
        }
    };
}

macro_rules! impl_products {
    ($matrix:ty) => {
        impl<F> MatVec<Col<F>> for $matrix
        where
            F: Scalar + faer_traits::ComplexField,
        {
            fn matvec(&self, x: &Col<F>) -> Result<Col<F>, Self::Error> {
                let mut y = Col::<F>::zeros(self.nrows());
                self.matvec_into(x, &mut y)?;
                Ok(y)
            }
        }

        impl<F> MatTransposeVec<Col<F>> for $matrix
        where
            F: Scalar + faer_traits::ComplexField,
        {
            fn mat_transpose_vec(&self, x: &Col<F>) -> Result<Col<F>, Self::Error> {
                let mut y = Col::<F>::zeros(self.ncols());
                self.mat_transpose_vec_into(x, &mut y)?;
                Ok(y)
            }
        }

        impl<F> MatVecInto<Col<F>> for $matrix
        where
            F: Scalar + faer_traits::ComplexField,
        {
            fn matvec_into(&self, x: &Col<F>, out: &mut Col<F>) -> Result<(), Self::Error> {
                self.matvec_scaled_into(F::one(), x, F::zero(), out)
            }
        }

        impl<F> MatVecScaledInto<Col<F>, Col<F>, F> for $matrix
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
                    self.rb(),
                    x.as_mat(),
                    alpha,
                    parallelism(),
                );
                Ok(())
            }
        }

        impl<F> MatTransposeVecInto<Col<F>> for $matrix
        where
            F: Scalar + faer_traits::ComplexField,
        {
            fn mat_transpose_vec_into(
                &self,
                x: &Col<F>,
                out: &mut Col<F>,
            ) -> Result<(), Self::Error> {
                self.mat_transpose_vec_scaled_into(F::one(), x, F::zero(), out)
            }
        }

        impl<F> MatTransposeVecScaledInto<Col<F>, Col<F>, F> for $matrix
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
                super::transpose::multiply(self.rb(), x, out, alpha, accumulation, parallelism());
                Ok(())
            }
        }
    };
}

// --- column statistics over the CSC arrays ------------------------------------

macro_rules! impl_column_stats {
    ($matrix:ty) => {
        impl<F> ColumnStats<F> for $matrix
        where
            F: Scalar + MaybeSend + MaybeSync,
        {
            crate::quantiles::buffered_stats_methods!(crate::quantiles::borrowed_columns);

            fn col_means(&self) -> Result<Vec<F>, Self::Error> {
                let n = F::from_usize(self.nrows()).unwrap();
                let matrix = self.rb();
                Ok(collect_columns(self.ncols(), |j| {
                    let (rows, values) = matrix.sparse_column(j);
                    let vals = coalesced_column_values(rows, values);
                    let sum: F = vals.iter().copied().sum();
                    sum / n
                }))
            }

            fn col_sds(&self, df: usize) -> Result<Vec<F>, Self::Error> {
                let nrows = self.nrows();
                let matrix = self.rb();
                Ok(collect_columns(self.ncols(), |j| {
                    let (rows, values) = matrix.sparse_column(j);
                    let vals = coalesced_column_values(rows, values);
                    sparse_column_sd(&vals, nrows, df)
                }))
            }

            fn col_mins(&self) -> Result<Vec<F>, Self::Error> {
                let nrows = self.nrows();
                let matrix = self.rb();
                Ok(collect_columns(self.ncols(), |j| {
                    let (rows, values) = matrix.sparse_column(j);
                    let vals = coalesced_column_values(rows, values);
                    min_or_nan(
                        vals.iter()
                            .copied()
                            .chain((vals.len() < nrows).then_some(F::zero())),
                    )
                }))
            }

            fn col_ranges(&self) -> Result<Vec<F>, Self::Error> {
                let nrows = self.nrows();
                let matrix = self.rb();
                Ok(collect_columns(self.ncols(), |j| {
                    let (rows, values) = matrix.sparse_column(j);
                    let vals = coalesced_column_values(rows, values);
                    range_or_nan(
                        vals.iter()
                            .copied()
                            .chain((vals.len() < nrows).then_some(F::zero())),
                    )
                }))
            }

            fn col_maxabs(&self) -> Result<Vec<F>, Self::Error> {
                let matrix = self.rb();
                Ok(collect_columns(self.ncols(), |j| {
                    let (rows, values) = matrix.sparse_column(j);
                    let vals = coalesced_column_values(rows, values);
                    max_or_nan(vals.iter().map(|v| v.abs()))
                }))
            }

            fn col_l1(&self) -> Result<Vec<F>, Self::Error> {
                let matrix = self.rb();
                Ok(collect_columns(self.ncols(), |j| {
                    let (rows, values) = matrix.sparse_column(j);
                    let vals = coalesced_column_values(rows, values);
                    vals.iter().map(|value| value.abs()).sum()
                }))
            }

            fn col_l2(&self) -> Result<Vec<F>, Self::Error> {
                let matrix = self.rb();
                Ok(collect_columns(self.ncols(), |j| {
                    let (rows, values) = matrix.sparse_column(j);
                    let vals = coalesced_column_values(rows, values);
                    let sum_sq: F = vals.iter().map(|&v| v * v).sum();
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
                let matrix = self.rb();
                Ok(collect_columns(self.ncols(), |j| {
                    let (rows, values) = matrix.sparse_column(j);
                    let vals = coalesced_column_values(rows, values);
                    let c = centers[j];
                    let nnz = vals.len();
                    let stored: F = vals.iter().map(|&v| (v - c) * (v - c)).sum();
                    let implicit = if nnz < nrows {
                        F::from_usize(nrows - nnz).unwrap() * c * c
                    } else {
                        F::zero()
                    };
                    (stored + implicit).sqrt()
                }))
            }

            fn col_l1_centered(&self, centers: &[F]) -> Result<Vec<F>, Self::Error> {
                assert_eq!(
                    centers.len(),
                    self.ncols(),
                    "col_l1_centered: length mismatch"
                );
                let nrows = self.nrows();
                let matrix = self.rb();
                Ok(collect_columns(self.ncols(), |j| {
                    let (rows, values) = matrix.sparse_column(j);
                    let vals = coalesced_column_values(rows, values);
                    let center = centers[j];
                    let stored: F = vals.iter().map(|&value| (value - center).abs()).sum();
                    if vals.len() < nrows {
                        stored + F::from_usize(nrows - vals.len()).unwrap() * center.abs()
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
                let matrix = self.rb();
                Ok(collect_columns(self.ncols(), |j| {
                    let (rows, values) = matrix.sparse_column(j);
                    let vals = coalesced_column_values(rows, values);
                    let c = centers[j];
                    if vals.len() < nrows {
                        max_or_nan(
                            vals.iter()
                                .map(|&v| (v - c).abs())
                                .chain(std::iter::once(c.abs())),
                        )
                    } else {
                        max_or_nan(vals.iter().map(|&v| (v - c).abs()))
                    }
                }))
            }
        }
    };
}

macro_rules! impl_materialization {
    ($matrix:ty) => {
        impl<F> crate::MatrixErrorType for $matrix {
            type Error = std::convert::Infallible;
        }

        impl<F: Scalar> crate::MaterializeDense<F> for $matrix {
            fn materialize_normalized_into<O: crate::MatrixWrite<F> + ?Sized>(
                &self,
                centers: Option<&[F]>,
                scales: Option<&[F]>,
                out: &mut O,
            ) -> Result<(), Self::Error> {
                crate::materialize::sparse_outer(
                    self,
                    true,
                    centers,
                    scales,
                    out,
                    |outer, visit| {
                        let (indices, values) = self.sparse_column(outer);
                        for (&index, &value) in indices.iter().zip(values) {
                            visit(index, value);
                        }
                    },
                );
                Ok(())
            }
        }

        impl<F: Scalar> crate::SubmatrixKernel<F> for $matrix {
            fn submatrix_matvec_into<
                X: crate::VectorView<F> + ?Sized,
                Y: crate::VectorViewMut<F> + ?Sized,
            >(
                &self,
                rows: std::ops::Range<usize>,
                columns: std::ops::Range<usize>,
                centers: Option<&[F]>,
                scales: Option<&[F]>,
                alpha: F,
                x: &X,
                beta: F,
                out: &mut Y,
            ) -> Result<(), Self::Error> {
                crate::submatrix_kernels::sparse_product(
                    self,
                    rows,
                    columns,
                    true,
                    |outer, f| {
                        let (indices, values) = crate::SparseColumns::sparse_column(self, outer);
                        for (&i, &v) in indices.iter().zip(values) {
                            f(i, v);
                        }
                    },
                    centers,
                    scales,
                    false,
                    alpha,
                    x,
                    beta,
                    out,
                );
                Ok(())
            }
            fn submatrix_transpose_into<
                X: crate::VectorView<F> + ?Sized,
                Y: crate::VectorViewMut<F> + ?Sized,
            >(
                &self,
                rows: std::ops::Range<usize>,
                columns: std::ops::Range<usize>,
                centers: Option<&[F]>,
                scales: Option<&[F]>,
                alpha: F,
                x: &X,
                beta: F,
                out: &mut Y,
            ) -> Result<(), Self::Error> {
                crate::submatrix_kernels::sparse_product(
                    self,
                    rows,
                    columns,
                    true,
                    |outer, f| {
                        let (indices, values) = crate::SparseColumns::sparse_column(self, outer);
                        for (&i, &v) in indices.iter().zip(values) {
                            f(i, v);
                        }
                    },
                    centers,
                    scales,
                    true,
                    alpha,
                    x,
                    beta,
                    out,
                );
                Ok(())
            }
            fn materialize_submatrix_into<O: crate::MatrixWrite<F> + ?Sized>(
                &self,
                rows: std::ops::Range<usize>,
                columns: std::ops::Range<usize>,
                centers: Option<&[F]>,
                scales: Option<&[F]>,
                out: &mut O,
            ) -> Result<(), Self::Error> {
                crate::submatrix_kernels::sparse_materialize(
                    self,
                    rows,
                    columns,
                    true,
                    |outer, f| {
                        let (indices, values) = crate::SparseColumns::sparse_column(self, outer);
                        for (&i, &v) in indices.iter().zip(values) {
                            f(i, v);
                        }
                    },
                    centers,
                    scales,
                    out,
                );
                Ok(())
            }
        }
    };
}

macro_rules! impl_csc {
    ($matrix:ty) => {
        impl_storage!($matrix);
        impl_products!($matrix);
        impl_column_stats!($matrix);
        impl_materialization!($matrix);
    };
}

impl_csc!(SparseColMat<usize, F>);
impl_csc!(SparseColMatRef<'_, usize, F>);
impl_csc!(SparseColMatMut<'_, usize, F>);

crate::indexed_rows_kernels::sparse_source!(SparseColMat<usize, F>, true, crate::SparseColumns::sparse_column);

crate::indexed_rows_kernels::sparse_source!(
    SparseColMatRef<'_, usize, F>,
    true,
    crate::SparseColumns::sparse_column
);

crate::indexed_rows_kernels::sparse_source!(
    SparseColMatMut<'_, usize, F>,
    true,
    crate::SparseColumns::sparse_column
);
