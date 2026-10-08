use super::{MatrixErrorType, Scalar};
use crate::{Centering, Normalization, Scaling};

/// Column-wise statistics of a design matrix, returned as plain `Vec<F>` of
/// length `ncols`.
///
/// Implemented per backend so that sparse matrices compute these by walking
/// stored entries — structurally absent entries are treated as zero — without
/// ever densifying a column.
///
/// All means and standard deviations use the **population** convention (divide
/// by `n`, the number of rows), matching the design-matrix normalization used by
/// the R `lazymatrix` package.
///
/// Statistics follow IEEE floating-point behavior. In particular, means,
/// standard deviations, minima, and ranges of a zero-row column are `NaN`;
/// un-centered norms of a zero-row column are zero; and a stored `NaN`
/// propagates through every statistic. Each method returns the backend error if
/// reading or decoding the data fails.
pub trait ColumnStats<F: Scalar>: MatrixErrorType {
    /// Compute the centers and raw scales requested by `spec`.
    ///
    /// Backends may override this hook to share scans between statistics. Exact
    /// zero scales are returned unchanged; [`crate::LazyMatrix::new`] replaces
    /// them with one. Each present vector must have length `ncols`.
    ///
    /// # Errors
    /// Returns the first error encountered while computing the statistics.
    fn normalization_stats(
        &self,
        spec: Normalization,
    ) -> Result<crate::NormalizationStats<F>, Self::Error> {
        let centers = match spec.center {
            Centering::None => None,
            Centering::Mean => Some(self.col_means()?),
            Centering::Min => Some(self.col_mins()?),
        };
        let scales = match spec.scale {
            Scaling::None => None,
            Scaling::Sd => Some(self.col_sds()?),
            Scaling::Range => Some(self.col_ranges()?),
            Scaling::L1 => Some(match &centers {
                Some(c) => self.col_l1_centered(c)?,
                None => self.col_l1()?,
            }),
            Scaling::L2 => Some(match &centers {
                Some(c) => self.col_l2_centered(c)?,
                None => self.col_l2()?,
            }),
            Scaling::MaxAbs => Some(match &centers {
                Some(c) => self.col_maxabs_centered(c)?,
                None => self.col_maxabs()?,
            }),
        };
        Ok((centers, scales))
    }

    /// Column means `c_j = (Σ_i x_ij) / n`.
    fn col_means(&self) -> Result<Vec<F>, Self::Error>;

    /// Column population standard deviations `√(Σ_i (x_ij − c_j)²/n)`.
    ///
    /// Centering-invariant, so this is also the standard deviation of the
    /// centered column. Implementations use a stable two-pass calculation.
    fn col_sds(&self) -> Result<Vec<F>, Self::Error>;

    /// Column minima `min_i x_ij` of the un-centered column.
    fn col_mins(&self) -> Result<Vec<F>, Self::Error>;

    /// Column ranges `max_i x_ij - min_i x_ij` of the un-centered column.
    fn col_ranges(&self) -> Result<Vec<F>, Self::Error>;

    /// Column max-absolute values `max_i |x_ij|` of the un-centered column.
    fn col_maxabs(&self) -> Result<Vec<F>, Self::Error>;

    /// Column 1-norms `‖x_j‖₁` of the un-centered column.
    fn col_l1(&self) -> Result<Vec<F>, Self::Error>;

    /// Column 2-norms `‖x_j‖₂` of the un-centered column.
    fn col_l2(&self) -> Result<Vec<F>, Self::Error>;

    /// Column 2-norms of the **centered** columns, `‖x_j − c_j·1‖₂`.
    ///
    /// Computed sparsely via the closed form
    /// `‖x_j − c_j‖₂² = Σ_stored (v − c_j)² + (n − nnz_j)·c_j²`,
    /// so it never densifies. `centers` must have length `ncols`.
    fn col_l2_centered(&self, centers: &[F]) -> Result<Vec<F>, Self::Error>;

    /// Column 1-norms of the **centered** columns, `‖x_j − c_j·1‖₁`.
    ///
    /// Computed sparsely via the closed form
    /// `Σ_stored |v − c_j| + (n − nnz_j)·|c_j|`, so it never densifies.
    /// `centers` must have length `ncols`.
    fn col_l1_centered(&self, centers: &[F]) -> Result<Vec<F>, Self::Error>;

    /// Column max-absolute values of the **centered** columns,
    /// `max_i |x_ij − c_j|`.
    ///
    /// The implicit zero entries contribute `|c_j|`, folded in alongside the
    /// stored-entry maxima. `centers` must have length `ncols`.
    fn col_maxabs_centered(&self, centers: &[F]) -> Result<Vec<F>, Self::Error>;
}

impl<M, F> ColumnStats<F> for &M
where
    M: ColumnStats<F> + ?Sized,
    F: Scalar,
{
    fn normalization_stats(
        &self,
        spec: Normalization,
    ) -> Result<crate::NormalizationStats<F>, Self::Error> {
        (**self).normalization_stats(spec)
    }

    fn col_means(&self) -> Result<Vec<F>, Self::Error> {
        (**self).col_means()
    }

    fn col_sds(&self) -> Result<Vec<F>, Self::Error> {
        (**self).col_sds()
    }

    fn col_mins(&self) -> Result<Vec<F>, Self::Error> {
        (**self).col_mins()
    }

    fn col_ranges(&self) -> Result<Vec<F>, Self::Error> {
        (**self).col_ranges()
    }

    fn col_maxabs(&self) -> Result<Vec<F>, Self::Error> {
        (**self).col_maxabs()
    }

    fn col_l1(&self) -> Result<Vec<F>, Self::Error> {
        (**self).col_l1()
    }

    fn col_l2(&self) -> Result<Vec<F>, Self::Error> {
        (**self).col_l2()
    }

    fn col_l2_centered(&self, centers: &[F]) -> Result<Vec<F>, Self::Error> {
        (**self).col_l2_centered(centers)
    }

    fn col_l1_centered(&self, centers: &[F]) -> Result<Vec<F>, Self::Error> {
        (**self).col_l1_centered(centers)
    }

    fn col_maxabs_centered(&self, centers: &[F]) -> Result<Vec<F>, Self::Error> {
        (**self).col_maxabs_centered(centers)
    }
}

impl<M, F> ColumnStats<F> for &mut M
where
    M: ColumnStats<F> + ?Sized,
    F: Scalar,
{
    fn normalization_stats(
        &self,
        spec: Normalization,
    ) -> Result<crate::NormalizationStats<F>, Self::Error> {
        (**self).normalization_stats(spec)
    }

    fn col_means(&self) -> Result<Vec<F>, Self::Error> {
        (**self).col_means()
    }

    fn col_sds(&self) -> Result<Vec<F>, Self::Error> {
        (**self).col_sds()
    }

    fn col_mins(&self) -> Result<Vec<F>, Self::Error> {
        (**self).col_mins()
    }

    fn col_ranges(&self) -> Result<Vec<F>, Self::Error> {
        (**self).col_ranges()
    }

    fn col_maxabs(&self) -> Result<Vec<F>, Self::Error> {
        (**self).col_maxabs()
    }

    fn col_l1(&self) -> Result<Vec<F>, Self::Error> {
        (**self).col_l1()
    }

    fn col_l2(&self) -> Result<Vec<F>, Self::Error> {
        (**self).col_l2()
    }

    fn col_l2_centered(&self, centers: &[F]) -> Result<Vec<F>, Self::Error> {
        (**self).col_l2_centered(centers)
    }

    fn col_l1_centered(&self, centers: &[F]) -> Result<Vec<F>, Self::Error> {
        (**self).col_l1_centered(centers)
    }

    fn col_maxabs_centered(&self, centers: &[F]) -> Result<Vec<F>, Self::Error> {
        (**self).col_maxabs_centered(centers)
    }
}
