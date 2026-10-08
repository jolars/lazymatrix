use super::{MatrixErrorType, MatrixShape, Scalar, VectorView};

/// Writable dense matrix storage, including strided mutable views.
///
/// Each call replaces one entry in O(1) time. Implementations must not depend
/// on the previous value. This capability does not require contiguous storage.
pub trait MatrixWrite<F: Scalar>: MatrixShape {
    /// Replace the entry at `(row, column)`.
    fn set(&mut self, row: usize, column: usize, value: F);
}

/// Weighted Gram product `out = Aᵀ diag(weights) A`.
///
/// Overwrites the entire symmetric `ncols × ncols` output, including both
/// triangles. Weights may be signed, zero, or nonfinite. Empty sums are zero.
/// Temporary workspace is internal; reusing output does not imply that the
/// implementation allocates no scratch storage.
///
/// Dense ndarray kernels use bounded panels and O(nrows * ncols²) work.
/// CSC kernels use O(ncols) workspace without centering, or O(nrows + ncols)
/// with centering. For sorted, unique columns and finite arithmetic, their
/// work is O(nrows + ncols * nnz + ncols²) without centering and at most
/// O(nrows + ncols * nnz * log(nrows + 1) + ncols²) with centering.
/// Centered CSC inputs with enough stored entries instead use bounded panels
/// and O(nrows * ncols²) work to avoid repeated sparse range queries.
/// Noncanonical columns or nonfinite arithmetic can require O(nrows) scratch
/// and O(nrows * ncols² + ncols * nnz) work. Output always occupies O(ncols²).
/// CSR kernels scan borrowed rows into fixed-size panels. Workspace is bounded,
/// and arithmetic costs O(nrows × ncols²). They do not convert storage.
///
/// Gram kernels run serially; an externally enabled BLAS library may use its
/// own threads.
///
/// # Errors
///
/// Returns the matrix's operational error. After an error, output may be
/// partially overwritten and must not be used as a product result.
///
/// # Panics
///
/// Panics before writing unless weights have length `nrows` and output has
/// shape `(ncols, ncols)`.
pub trait WeightedGramInto<F: Scalar>: MatrixShape + MatrixErrorType {
    fn weighted_gram_into<W, O>(&self, weights: &W, out: &mut O) -> Result<(), Self::Error>
    where
        W: VectorView<F> + ?Sized,
        O: MatrixWrite<F> + ?Sized;
}

/// Backend kernel for a weighted Gram product with explicit normalization.
///
/// Computes the product for logical entries `(X[i,j] - centers[j]) / scales[j]`,
/// using zero centers and unit scales when the corresponding slice is absent.
/// [`LazyMatrix`](crate::LazyMatrix) forwards its parameters to this capability.
/// Backends must center values before accumulating their products: correcting
/// raw cross-products afterward can erase small centered variations.
///
/// The output, weight, and error contracts are those of [`WeightedGramInto`].
/// Negative scales and nonfinite parameters are preserved. No full normalized
/// or weighted design matrix may be materialized.
///
/// # Panics
///
/// In addition to product dimension checks, implementations must reject
/// normalization slices of length other than `ncols` and exact zero scales
/// (including negative zero), before writing output.
pub trait WeightedGramKernel<F: Scalar>: MatrixShape + MatrixErrorType {
    fn weighted_gram_normalized_into<W, O>(
        &self,
        weights: &W,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut O,
    ) -> Result<(), Self::Error>
    where
        W: VectorView<F> + ?Sized,
        O: MatrixWrite<F> + ?Sized;
}

impl<M, F> WeightedGramInto<F> for &M
where
    M: WeightedGramInto<F> + ?Sized,
    F: Scalar,
{
    fn weighted_gram_into<W, O>(&self, weights: &W, out: &mut O) -> Result<(), Self::Error>
    where
        W: VectorView<F> + ?Sized,
        O: MatrixWrite<F> + ?Sized,
    {
        (**self).weighted_gram_into(weights, out)
    }
}

impl<M, F> WeightedGramKernel<F> for &M
where
    M: WeightedGramKernel<F> + ?Sized,
    F: Scalar,
{
    fn weighted_gram_normalized_into<W, O>(
        &self,
        weights: &W,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut O,
    ) -> Result<(), Self::Error>
    where
        W: VectorView<F> + ?Sized,
        O: MatrixWrite<F> + ?Sized,
    {
        (**self).weighted_gram_normalized_into(weights, centers, scales, out)
    }
}

impl<M, F> WeightedGramInto<F> for &mut M
where
    M: WeightedGramInto<F> + ?Sized,
    F: Scalar,
{
    fn weighted_gram_into<W, O>(&self, weights: &W, out: &mut O) -> Result<(), Self::Error>
    where
        W: VectorView<F> + ?Sized,
        O: MatrixWrite<F> + ?Sized,
    {
        (**self).weighted_gram_into(weights, out)
    }
}

impl<M, F> WeightedGramKernel<F> for &mut M
where
    M: WeightedGramKernel<F> + ?Sized,
    F: Scalar,
{
    fn weighted_gram_normalized_into<W, O>(
        &self,
        weights: &W,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut O,
    ) -> Result<(), Self::Error>
    where
        W: VectorView<F> + ?Sized,
        O: MatrixWrite<F> + ?Sized,
    {
        (**self).weighted_gram_normalized_into(weights, centers, scales, out)
    }
}
