/// Error shared by a matrix's fallible operations.
///
/// In-memory backends use [`std::convert::Infallible`]. Storage-backed matrices
/// use this type to report read and decoding failures across all capabilities.
pub trait MatrixErrorType {
    /// An operational failure while reading or processing matrix data.
    type Error: std::error::Error;
}

impl<M: MatrixErrorType + ?Sized> MatrixErrorType for &M {
    type Error = M::Error;
}

/// Dimensions of a matrix or linear operator.
pub trait MatrixShape {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
}

/// Matrix–vector product `A x`, returning a freshly allocated vector of length
/// `nrows`. Returns the backend error if the product cannot be completed.
pub trait MatVec<V>: MatrixShape + MatrixErrorType {
    fn matvec(&self, x: &V) -> Result<V, Self::Error>;

    /// Apply optional normalization, allowing a backend to center before summation.
    ///
    /// The default borrows unscaled inputs and clones coefficients only for scaling.
    fn matvec_normalized<F: super::Scalar>(
        &self,
        x: &V,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
    ) -> Result<V, Self::Error>
    where
        V: Clone + super::ElemDivAssign<F> + super::DotSlice<F> + super::SubScalarAssign<F>,
    {
        let mut scaled;
        let coefficients = if let Some(scales) = scales {
            scaled = x.clone();
            scaled.elem_div_assign(scales);
            &scaled
        } else {
            x
        };
        let mut out = self.matvec(coefficients)?;
        if let Some(centers) = centers {
            out.sub_scalar_assign(coefficients.dot_slice(centers));
        }
        Ok(out)
    }
}

/// Matrix–vector product `out = A x`, overwriting reusable output storage.
///
/// `X` and `Y` may differ so that backends can accept borrowed or strided
/// inputs while writing into an owned output. Implementations must not depend
/// on the previous values in `out`.
///
/// # Errors
///
/// Returns the backend error if the product cannot be completed. After an error,
/// `out` may be partially overwritten and must not be used as a product result.
///
/// # Panics
///
/// Panics unless `x` has length `ncols` and `out` has length `nrows`.
pub trait MatVecInto<X, Y = X>: MatrixShape + MatrixErrorType {
    fn matvec_into(&self, x: &X, out: &mut Y) -> Result<(), Self::Error>;

    /// Apply optional column normalization without materializing the matrix.
    ///
    /// The default uses raw products and algebraic corrections. Backends may
    /// override this method to center entries before accumulation, which avoids
    /// cancellation when column offsets are large.
    fn matvec_normalized_into<F: super::Scalar>(
        &self,
        x: &X,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut Y,
    ) -> Result<(), Self::Error>
    where
        X: Clone + super::ElemDivAssign<F> + super::DotSlice<F>,
        Y: super::SubScalarAssign<F>,
    {
        matvec_normalized_into(self, x, centers, scales, out)
    }
}

pub(crate) fn matvec_normalized_into<M, X, Y, F>(
    matrix: &M,
    x: &X,
    centers: Option<&[F]>,
    scales: Option<&[F]>,
    out: &mut Y,
) -> Result<(), M::Error>
where
    M: MatVecInto<X, Y> + ?Sized,
    F: super::Scalar,
    X: Clone + super::ElemDivAssign<F> + super::DotSlice<F>,
    Y: super::SubScalarAssign<F>,
{
    if let Some(scales) = scales {
        let mut scratch = x.clone();
        scratch.elem_div_assign(scales);
        matrix.matvec_into(&scratch, out)?;
        if let Some(centers) = centers {
            out.sub_scalar_assign(scratch.dot_slice(centers));
        }
    } else {
        matrix.matvec_into(x, out)?;
        if let Some(centers) = centers {
            out.sub_scalar_assign(x.dot_slice(centers));
        }
    }
    Ok(())
}

/// Fused product `out = alpha * A * x + beta * out`.
///
/// Exact zero `alpha` skips the product, including any storage reads. Exact zero
/// `beta` overwrites without reading prior output values. Dimensions are checked
/// even when a coefficient is zero. On an operational error, `out` may be partial.
///
/// # Panics
///
/// Panics unless `x` has length `ncols` and `out` has length `nrows`.
pub trait MatVecScaledInto<X, Y, F>: MatrixShape + MatrixErrorType {
    fn matvec_scaled_into(&self, alpha: F, x: &X, beta: F, out: &mut Y) -> Result<(), Self::Error>;

    /// Try a specialized fused normalized product before algebraic correction.
    ///
    /// `None` leaves output untouched and requests the caller's generic path.
    /// `Some` obeys the fused product's dimension, zero-coefficient, and error
    /// contracts. Dense backends use this hook to center before accumulation.
    fn try_matvec_scaled_normalized_into(
        &self,
        _alpha: F,
        _x: &X,
        _beta: F,
        _centers: Option<&[F]>,
        _scales: Option<&[F]>,
        _out: &mut Y,
    ) -> Option<Result<(), Self::Error>>
    where
        F: super::Scalar,
        X: super::VectorView<F>,
        Y: super::VectorViewMut<F>,
    {
        None
    }
}

/// Transposed matrix–vector product `Aᵀ x`, returning a freshly allocated vector
/// of length `ncols`. Returns the backend error if the product cannot be completed.
pub trait MatTransposeVec<V>: MatrixShape + MatrixErrorType {
    fn mat_transpose_vec(&self, x: &V) -> Result<V, Self::Error>;

    /// Apply a normalized transpose, allowing directly centered accumulation.
    fn mat_transpose_vec_normalized<F: super::Scalar>(
        &self,
        x: &V,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
    ) -> Result<V, Self::Error>
    where
        V: super::SumEntries<F> + super::ScaledSubSlice<F> + super::ElemDivAssign<F>,
    {
        let mut out = self.mat_transpose_vec(x)?;
        if let Some(centers) = centers {
            out.scaled_sub_slice(x.sum_entries(), centers);
        }
        if let Some(scales) = scales {
            out.elem_div_assign(scales);
        }
        Ok(out)
    }
}

/// Transposed matrix–vector product `out = Aᵀ x`, overwriting reusable output
/// storage.
///
/// `X` and `Y` may differ so that backends can accept borrowed or strided
/// inputs while writing into an owned output. Implementations must not depend
/// on the previous values in `out`.
///
/// # Errors
///
/// Returns the backend error if the product cannot be completed. After an error,
/// `out` may be partially overwritten and must not be used as a product result.
///
/// # Panics
///
/// Panics unless `x` has length `nrows` and `out` has length `ncols`.
pub trait MatTransposeVecInto<X, Y = X>: MatrixShape + MatrixErrorType {
    fn mat_transpose_vec_into(&self, x: &X, out: &mut Y) -> Result<(), Self::Error>;

    /// Apply the transpose of the optionally normalized matrix.
    ///
    /// Backends may center entries before accumulation to avoid cancellation.
    fn mat_transpose_vec_normalized_into<F: super::Scalar>(
        &self,
        x: &X,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut Y,
    ) -> Result<(), Self::Error>
    where
        X: super::SumEntries<F>,
        Y: super::ScaledSubSlice<F> + super::ElemDivAssign<F>,
    {
        mat_transpose_vec_normalized_into(self, x, centers, scales, out)
    }
}

pub(crate) fn mat_transpose_vec_normalized_into<M, X, Y, F>(
    matrix: &M,
    x: &X,
    centers: Option<&[F]>,
    scales: Option<&[F]>,
    out: &mut Y,
) -> Result<(), M::Error>
where
    M: MatTransposeVecInto<X, Y> + ?Sized,
    F: super::Scalar,
    X: super::SumEntries<F>,
    Y: super::ScaledSubSlice<F> + super::ElemDivAssign<F>,
{
    matrix.mat_transpose_vec_into(x, out)?;
    if let Some(centers) = centers {
        out.scaled_sub_slice(x.sum_entries(), centers);
    }
    if let Some(scales) = scales {
        out.elem_div_assign(scales);
    }
    Ok(())
}

/// Fused transpose product `out = alpha * Aᵀ * x + beta * out`.
///
/// Exact zero `alpha` skips the product, including any storage reads. Exact zero
/// `beta` overwrites without reading prior output values. Dimensions are checked
/// even when a coefficient is zero. On an operational error, `out` may be partial.
///
/// # Panics
///
/// Panics unless `x` has length `nrows` and `out` has length `ncols`.
pub trait MatTransposeVecScaledInto<X, Y, F>: MatrixShape + MatrixErrorType {
    fn mat_transpose_vec_scaled_into(
        &self,
        alpha: F,
        x: &X,
        beta: F,
        out: &mut Y,
    ) -> Result<(), Self::Error>;

    /// Try a specialized fused normalized transpose product.
    ///
    /// `None` leaves output untouched. `Some` obeys the fused product's
    /// dimension, zero-coefficient, and error contracts.
    fn try_mat_transpose_vec_scaled_normalized_into(
        &self,
        _alpha: F,
        _x: &X,
        _beta: F,
        _centers: Option<&[F]>,
        _scales: Option<&[F]>,
        _out: &mut Y,
    ) -> Option<Result<(), Self::Error>>
    where
        F: super::Scalar,
        X: super::VectorView<F>,
        Y: super::VectorViewMut<F>,
    {
        None
    }
}

/// Honor the fused zero-coefficient contract without reading overwritten values.
pub(crate) fn scale_output<F: super::Scalar, Y: super::VectorViewMut<F> + ?Sized>(
    beta: F,
    out: &mut Y,
) {
    if beta == F::zero() {
        for i in 0..out.len() {
            out.set(i, F::zero());
        }
    } else if beta != F::one() {
        for i in 0..out.len() {
            out.set(i, beta * out.get(i));
        }
    }
}

impl<M: MatrixShape + ?Sized> MatrixShape for &M {
    fn nrows(&self) -> usize {
        (**self).nrows()
    }

    fn ncols(&self) -> usize {
        (**self).ncols()
    }
}

impl<M, V> MatVec<V> for &M
where
    M: MatVec<V> + ?Sized,
{
    fn matvec(&self, x: &V) -> Result<V, Self::Error> {
        (**self).matvec(x)
    }

    fn matvec_normalized<F: crate::Scalar>(
        &self,
        x: &V,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
    ) -> Result<V, Self::Error>
    where
        V: Clone + crate::ElemDivAssign<F> + crate::DotSlice<F> + crate::SubScalarAssign<F>,
    {
        (**self).matvec_normalized(x, centers, scales)
    }
}

impl<M, X, Y> MatVecInto<X, Y> for &M
where
    M: MatVecInto<X, Y> + ?Sized,
{
    fn matvec_into(&self, x: &X, out: &mut Y) -> Result<(), Self::Error> {
        (**self).matvec_into(x, out)
    }
    fn matvec_normalized_into<F: super::Scalar>(
        &self,
        x: &X,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut Y,
    ) -> Result<(), Self::Error>
    where
        X: Clone + super::ElemDivAssign<F> + super::DotSlice<F>,
        Y: super::SubScalarAssign<F>,
    {
        (**self).matvec_normalized_into(x, centers, scales, out)
    }
}

impl<M, X, Y, F> MatVecScaledInto<X, Y, F> for &M
where
    M: MatVecScaledInto<X, Y, F> + ?Sized,
{
    fn matvec_scaled_into(&self, alpha: F, x: &X, beta: F, out: &mut Y) -> Result<(), Self::Error> {
        (**self).matvec_scaled_into(alpha, x, beta, out)
    }

    fn try_matvec_scaled_normalized_into(
        &self,
        alpha: F,
        x: &X,
        beta: F,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut Y,
    ) -> Option<Result<(), Self::Error>>
    where
        F: crate::Scalar,
        X: crate::VectorView<F>,
        Y: crate::VectorViewMut<F>,
    {
        (**self).try_matvec_scaled_normalized_into(alpha, x, beta, centers, scales, out)
    }
}

impl<M, V> MatTransposeVec<V> for &M
where
    M: MatTransposeVec<V> + ?Sized,
{
    fn mat_transpose_vec(&self, x: &V) -> Result<V, Self::Error> {
        (**self).mat_transpose_vec(x)
    }

    fn mat_transpose_vec_normalized<F: crate::Scalar>(
        &self,
        x: &V,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
    ) -> Result<V, Self::Error>
    where
        V: crate::SumEntries<F> + crate::ScaledSubSlice<F> + crate::ElemDivAssign<F>,
    {
        (**self).mat_transpose_vec_normalized(x, centers, scales)
    }
}

impl<M, X, Y> MatTransposeVecInto<X, Y> for &M
where
    M: MatTransposeVecInto<X, Y> + ?Sized,
{
    fn mat_transpose_vec_into(&self, x: &X, out: &mut Y) -> Result<(), Self::Error> {
        (**self).mat_transpose_vec_into(x, out)
    }
    fn mat_transpose_vec_normalized_into<F: super::Scalar>(
        &self,
        x: &X,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut Y,
    ) -> Result<(), Self::Error>
    where
        X: super::SumEntries<F>,
        Y: super::ScaledSubSlice<F> + super::ElemDivAssign<F>,
    {
        (**self).mat_transpose_vec_normalized_into(x, centers, scales, out)
    }
}

impl<M, X, Y, F> MatTransposeVecScaledInto<X, Y, F> for &M
where
    M: MatTransposeVecScaledInto<X, Y, F> + ?Sized,
{
    fn mat_transpose_vec_scaled_into(
        &self,
        alpha: F,
        x: &X,
        beta: F,
        out: &mut Y,
    ) -> Result<(), Self::Error> {
        (**self).mat_transpose_vec_scaled_into(alpha, x, beta, out)
    }

    fn try_mat_transpose_vec_scaled_normalized_into(
        &self,
        alpha: F,
        x: &X,
        beta: F,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut Y,
    ) -> Option<Result<(), Self::Error>>
    where
        F: crate::Scalar,
        X: crate::VectorView<F>,
        Y: crate::VectorViewMut<F>,
    {
        (**self).try_mat_transpose_vec_scaled_normalized_into(alpha, x, beta, centers, scales, out)
    }
}

impl<M: MatrixErrorType + ?Sized> MatrixErrorType for &mut M {
    type Error = M::Error;
}

impl<M: MatrixShape + ?Sized> MatrixShape for &mut M {
    fn nrows(&self) -> usize {
        (**self).nrows()
    }

    fn ncols(&self) -> usize {
        (**self).ncols()
    }
}

impl<M, V> MatVec<V> for &mut M
where
    M: MatVec<V> + ?Sized,
{
    fn matvec(&self, x: &V) -> Result<V, Self::Error> {
        (**self).matvec(x)
    }

    fn matvec_normalized<F: crate::Scalar>(
        &self,
        x: &V,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
    ) -> Result<V, Self::Error>
    where
        V: Clone + crate::ElemDivAssign<F> + crate::DotSlice<F> + crate::SubScalarAssign<F>,
    {
        (**self).matvec_normalized(x, centers, scales)
    }
}

impl<M, X, Y> MatVecInto<X, Y> for &mut M
where
    M: MatVecInto<X, Y> + ?Sized,
{
    fn matvec_into(&self, x: &X, out: &mut Y) -> Result<(), Self::Error> {
        (**self).matvec_into(x, out)
    }
    fn matvec_normalized_into<F: super::Scalar>(
        &self,
        x: &X,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut Y,
    ) -> Result<(), Self::Error>
    where
        X: Clone + super::ElemDivAssign<F> + super::DotSlice<F>,
        Y: super::SubScalarAssign<F>,
    {
        (**self).matvec_normalized_into(x, centers, scales, out)
    }
}

impl<M, X, Y, F> MatVecScaledInto<X, Y, F> for &mut M
where
    M: MatVecScaledInto<X, Y, F> + ?Sized,
{
    fn matvec_scaled_into(&self, alpha: F, x: &X, beta: F, out: &mut Y) -> Result<(), Self::Error> {
        (**self).matvec_scaled_into(alpha, x, beta, out)
    }

    fn try_matvec_scaled_normalized_into(
        &self,
        alpha: F,
        x: &X,
        beta: F,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut Y,
    ) -> Option<Result<(), Self::Error>>
    where
        F: crate::Scalar,
        X: crate::VectorView<F>,
        Y: crate::VectorViewMut<F>,
    {
        (**self).try_matvec_scaled_normalized_into(alpha, x, beta, centers, scales, out)
    }
}

impl<M, V> MatTransposeVec<V> for &mut M
where
    M: MatTransposeVec<V> + ?Sized,
{
    fn mat_transpose_vec(&self, x: &V) -> Result<V, Self::Error> {
        (**self).mat_transpose_vec(x)
    }

    fn mat_transpose_vec_normalized<F: crate::Scalar>(
        &self,
        x: &V,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
    ) -> Result<V, Self::Error>
    where
        V: crate::SumEntries<F> + crate::ScaledSubSlice<F> + crate::ElemDivAssign<F>,
    {
        (**self).mat_transpose_vec_normalized(x, centers, scales)
    }
}

impl<M, X, Y> MatTransposeVecInto<X, Y> for &mut M
where
    M: MatTransposeVecInto<X, Y> + ?Sized,
{
    fn mat_transpose_vec_into(&self, x: &X, out: &mut Y) -> Result<(), Self::Error> {
        (**self).mat_transpose_vec_into(x, out)
    }
    fn mat_transpose_vec_normalized_into<F: super::Scalar>(
        &self,
        x: &X,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut Y,
    ) -> Result<(), Self::Error>
    where
        X: super::SumEntries<F>,
        Y: super::ScaledSubSlice<F> + super::ElemDivAssign<F>,
    {
        (**self).mat_transpose_vec_normalized_into(x, centers, scales, out)
    }
}

impl<M, X, Y, F> MatTransposeVecScaledInto<X, Y, F> for &mut M
where
    M: MatTransposeVecScaledInto<X, Y, F> + ?Sized,
{
    fn mat_transpose_vec_scaled_into(
        &self,
        alpha: F,
        x: &X,
        beta: F,
        out: &mut Y,
    ) -> Result<(), Self::Error> {
        (**self).mat_transpose_vec_scaled_into(alpha, x, beta, out)
    }

    fn try_mat_transpose_vec_scaled_normalized_into(
        &self,
        alpha: F,
        x: &X,
        beta: F,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut Y,
    ) -> Option<Result<(), Self::Error>>
    where
        F: crate::Scalar,
        X: crate::VectorView<F>,
        Y: crate::VectorViewMut<F>,
    {
        (**self).try_mat_transpose_vec_scaled_normalized_into(alpha, x, beta, centers, scales, out)
    }
}
