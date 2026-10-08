use super::{MatrixErrorType, MatrixShape, Scalar, VectorView, VectorViewMut};

/// Weighted logical-column sums `out[j] = sum_i A[i,j] * weights[i]`.
///
/// Unlike a factored normalized transpose product, implementations center
/// entries before accumulation. This preserves small variations around large
/// offsets when constructing an intercept's weighted Gram cross terms.
/// Signed and nonfinite weights are preserved, including IEEE operations on
/// implicit zeros. Empty sums are zero. Every output entry is overwritten.
///
/// Dense ndarray takes O(nrows * ncols) work and constant scratch space.
/// Canonical finite CSC inputs take O(nrows + ncols + nnz) work without
/// centering, or O(nrows + ncols + nnz * log(nrows + 1)) with centering.
/// Centered CSC uses O(nrows) scratch. Noncanonical columns or nonfinite
/// arithmetic may require O(nrows * ncols + nnz) work and one working column.
/// No full design matrix is materialized.
///
/// # Errors
///
/// Returns the matrix's operational error. Output may be partial on failure.
///
/// # Panics
///
/// Panics before writing unless weights have length `nrows` and output has
/// length `ncols`.
pub trait WeightedColumnSumsInto<F: Scalar>: MatrixShape + MatrixErrorType {
    fn weighted_column_sums_into<W, O>(&self, weights: &W, out: &mut O) -> Result<(), Self::Error>
    where
        W: VectorView<F> + ?Sized,
        O: VectorViewMut<F> + ?Sized;
}

/// Backend weighted-column sums with explicit normalization.
///
/// Accumulate `((X[i,j] - centers[j]) / scales[j]) * weights[i]`, applying
/// subtraction and division only when the corresponding slice is present.
/// [`LazyMatrix`](crate::LazyMatrix) forwards its normalization here.
/// The output and error contracts are those of [`WeightedColumnSumsInto`].
///
/// # Panics
///
/// In addition to product dimensions, validate normalization slice lengths and
/// reject exact zero scales, including negative zero, before writing output.
pub trait WeightedColumnSumsKernel<F: Scalar>: MatrixShape + MatrixErrorType {
    fn weighted_column_sums_normalized_into<W, O>(
        &self,
        weights: &W,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut O,
    ) -> Result<(), Self::Error>
    where
        W: VectorView<F> + ?Sized,
        O: VectorViewMut<F> + ?Sized;
}

impl<M: WeightedColumnSumsInto<F> + ?Sized, F: Scalar> WeightedColumnSumsInto<F> for &M {
    fn weighted_column_sums_into<W, O>(&self, weights: &W, out: &mut O) -> Result<(), Self::Error>
    where
        W: VectorView<F> + ?Sized,
        O: VectorViewMut<F> + ?Sized,
    {
        (**self).weighted_column_sums_into(weights, out)
    }
}

impl<M: WeightedColumnSumsKernel<F> + ?Sized, F: Scalar> WeightedColumnSumsKernel<F> for &M {
    fn weighted_column_sums_normalized_into<W, O>(
        &self,
        weights: &W,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut O,
    ) -> Result<(), Self::Error>
    where
        W: VectorView<F> + ?Sized,
        O: VectorViewMut<F> + ?Sized,
    {
        (**self).weighted_column_sums_normalized_into(weights, centers, scales, out)
    }
}

impl<M: WeightedColumnSumsInto<F> + ?Sized, F: Scalar> WeightedColumnSumsInto<F> for &mut M {
    fn weighted_column_sums_into<W, O>(&self, weights: &W, out: &mut O) -> Result<(), Self::Error>
    where
        W: VectorView<F> + ?Sized,
        O: VectorViewMut<F> + ?Sized,
    {
        (**self).weighted_column_sums_into(weights, out)
    }
}

impl<M: WeightedColumnSumsKernel<F> + ?Sized, F: Scalar> WeightedColumnSumsKernel<F> for &mut M {
    fn weighted_column_sums_normalized_into<W, O>(
        &self,
        weights: &W,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut O,
    ) -> Result<(), Self::Error>
    where
        W: VectorView<F> + ?Sized,
        O: VectorViewMut<F> + ?Sized,
    {
        (**self).weighted_column_sums_normalized_into(weights, centers, scales, out)
    }
}
