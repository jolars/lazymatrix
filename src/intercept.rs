//! An implicit leading column of ones.

use std::marker::PhantomData;

use crate::{
    MatTransposeVec, MatTransposeVecInto, MatTransposeVecScaledInto, MatVec, MatVecInto,
    MatVecScaledInto, MatrixErrorType, MatrixShape, MatrixWrite, Scalar, VectorOwned, VectorView,
    VectorViewMut, WeightedColumnSumsInto, WeightedGramInto,
};

/// A design operator `[1, predictors]` with an implicit leading intercept.
///
/// Normalize predictors before wrapping them so centering cannot erase the
/// intercept. Construction takes O(1) time and allocates nothing. Ownership or
/// borrowing follows the supplied predictor operator; coefficient zero always
/// belongs to the intercept. Fitting and penalty policies remain with callers.
///
/// Products delegate to the predictor's reusable-output capabilities. Forward
/// products copy O(ncols) coefficients and broadcast the intercept after success.
/// Transpose products use O(ncols) scratch and prepend the input sum. Backend
/// normalization may allocate additional scratch. Reusing output therefore does
/// not promise allocation-free evaluation. Borrowed vector views work wherever
/// the corresponding predictor product accepts them.
/// Fused products implement [`MatVecScaledInto`] and [`MatTransposeVecScaledInto`]
/// when the predictors do. The optional workspace methods reuse the wrapper's
/// predictor buffer; the inner operator may still allocate normalization scratch.
///
/// Weighted Gram products require both [`WeightedGramInto`] and
/// [`WeightedColumnSumsInto`] on the predictors. They reuse the predictor Gram
/// output block and compute stable cross terms in a separate pass. The wrapper
/// uses O(ncols) scratch in addition to the predictor Gram and weighted-sum
/// workspaces, which run sequentially. No column of ones or full design matrix
/// is allocated.
/// All errors retain the predictor's error type; output may be partial on error.
#[derive(Clone, Copy, Debug)]
pub struct WithIntercept<M, F> {
    inner: M,
    scalar: PhantomData<F>,
}

impl<M, F> WithIntercept<M, F> {
    /// Add a leading intercept without reading or copying the predictors.
    pub fn new(predictors: M) -> Self {
        Self {
            inner: predictors,
            scalar: PhantomData,
        }
    }

    /// Borrow the predictor operator, including its normalization metadata.
    pub fn as_inner(&self) -> &M {
        &self.inner
    }

    /// Recover the predictor operator without copying.
    pub fn into_inner(self) -> M {
        self.inner
    }
}

impl<M: MatrixShape, F> MatrixShape for WithIntercept<M, F> {
    fn nrows(&self) -> usize {
        self.inner.nrows()
    }
    fn ncols(&self) -> usize {
        self.inner
            .ncols()
            .checked_add(1)
            .expect("intercept column count overflow")
    }
}

impl<M: MatrixErrorType, F> MatrixErrorType for WithIntercept<M, F> {
    type Error = M::Error;
}

impl<M: MatrixShape, F: Scalar> WithIntercept<M, F> {
    fn validate_product(&self, input_len: usize, output_len: usize, transpose: bool) {
        let (input, output) = if transpose {
            (self.nrows(), self.ncols())
        } else {
            (self.ncols(), self.nrows())
        };
        assert_eq!(input_len, input, "fused product: dimension mismatch");
        assert_eq!(
            output_len, output,
            "fused product: output dimension mismatch"
        );
    }

    /// Apply `out = alpha * [1, predictors] * x + beta * out` with reusable scratch.
    ///
    /// Scratch stores the predictor coefficients and has length `ncols() - 1`.
    /// This reuses only the wrapper's buffer: predictors may allocate their own
    /// normalization workspace. Exact zero `alpha` leaves scratch untouched and
    /// skips predictor evaluation. Exact zero `beta` ignores previous output.
    ///
    /// # Errors
    /// Returns the predictor error. Output and scratch may be partial on error;
    /// the intercept contribution is added only after the predictors succeed.
    ///
    /// # Panics
    /// Panics on incorrect input, output, or scratch lengths, even for zero `alpha`.
    pub fn matvec_scaled_with_workspace<X, Y>(
        &self,
        alpha: F,
        x: &X,
        beta: F,
        out: &mut Y,
        scratch: &mut X::Owned,
    ) -> Result<(), M::Error>
    where
        X: VectorOwned<F>,
        Y: VectorViewMut<F>,
        M: MatVecScaledInto<X::Owned, Y, F>,
    {
        self.validate_product(x.len(), out.len(), false);
        assert_eq!(
            scratch.len(),
            self.inner.ncols(),
            "fused product: scratch dimension mismatch"
        );
        if alpha == F::zero() {
            crate::traits::scale_output(beta, out);
            return Ok(());
        }
        if self.inner.ncols() == 0 {
            crate::traits::scale_output(beta, out);
        } else {
            for j in 0..scratch.len() {
                scratch.set(j, x.get(j + 1));
            }
            self.inner.matvec_scaled_into(alpha, scratch, beta, out)?;
        }
        let intercept = alpha * x.get(0);
        for i in 0..out.len() {
            out.set(i, out.get(i) + intercept);
        }
        Ok(())
    }

    /// Apply `out = alpha * [1, predictors]ᵀ * x + beta * out` with reusable scratch.
    ///
    /// Scratch stores the predictor output and has length `ncols() - 1`.
    /// Predictors may allocate additional normalization workspace. Exact zero
    /// `alpha` leaves scratch untouched and skips both predictor evaluation and
    /// the input sum. Exact zero `beta` ignores previous output.
    ///
    /// # Errors
    /// Returns the predictor error. Scratch may be partial on error, but output
    /// is updated only after the predictors succeed.
    ///
    /// # Panics
    /// Panics on incorrect input, output, or scratch lengths, even for zero `alpha`.
    pub fn mat_transpose_vec_scaled_with_workspace<X, Y>(
        &self,
        alpha: F,
        x: &X,
        beta: F,
        out: &mut Y,
        scratch: &mut Y::Owned,
    ) -> Result<(), M::Error>
    where
        X: VectorView<F>,
        Y: VectorOwned<F> + VectorViewMut<F>,
        M: MatTransposeVecScaledInto<X, Y::Owned, F>,
    {
        self.validate_product(x.len(), out.len(), true);
        assert_eq!(
            scratch.len(),
            self.inner.ncols(),
            "fused product: scratch dimension mismatch"
        );
        if alpha == F::zero() {
            crate::traits::scale_output(beta, out);
            return Ok(());
        }
        if self.inner.ncols() != 0 {
            for j in 0..scratch.len() {
                scratch.set(
                    j,
                    if beta == F::zero() {
                        F::zero()
                    } else {
                        out.get(j + 1)
                    },
                );
            }
            self.inner
                .mat_transpose_vec_scaled_into(alpha, x, beta, scratch)?;
            for j in 0..scratch.len() {
                out.set(j + 1, scratch.get(j));
            }
        }
        let intercept = alpha * x.sum();
        out.set(
            0,
            if beta == F::zero() {
                intercept
            } else {
                intercept + beta * out.get(0)
            },
        );
        Ok(())
    }
}

impl<M, F, X, Y> MatVecScaledInto<X, Y, F> for WithIntercept<M, F>
where
    F: Scalar,
    X: VectorOwned<F>,
    Y: VectorViewMut<F>,
    M: MatVecScaledInto<X::Owned, Y, F>,
{
    fn matvec_scaled_into(&self, alpha: F, x: &X, beta: F, out: &mut Y) -> Result<(), Self::Error> {
        self.validate_product(x.len(), out.len(), false);
        if alpha == F::zero() {
            crate::traits::scale_output(beta, out);
            return Ok(());
        }
        let mut scratch = X::owned_from_fn(self.inner.ncols(), |_| F::zero());
        self.matvec_scaled_with_workspace(alpha, x, beta, out, &mut scratch)
    }
}

impl<M, F, X, Y> MatTransposeVecScaledInto<X, Y, F> for WithIntercept<M, F>
where
    F: Scalar,
    X: VectorView<F>,
    Y: VectorOwned<F> + VectorViewMut<F>,
    M: MatTransposeVecScaledInto<X, Y::Owned, F>,
{
    fn mat_transpose_vec_scaled_into(
        &self,
        alpha: F,
        x: &X,
        beta: F,
        out: &mut Y,
    ) -> Result<(), Self::Error> {
        self.validate_product(x.len(), out.len(), true);
        if alpha == F::zero() {
            crate::traits::scale_output(beta, out);
            return Ok(());
        }
        let mut scratch = Y::owned_from_fn(self.inner.ncols(), |_| F::zero());
        self.mat_transpose_vec_scaled_with_workspace(alpha, x, beta, out, &mut scratch)
    }
}

impl<M, F, X, Y> MatVecInto<X, Y> for WithIntercept<M, F>
where
    F: Scalar,
    X: VectorOwned<F>,
    Y: VectorViewMut<F>,
    M: MatVecInto<X::Owned, Y>,
{
    fn matvec_into(&self, x: &X, out: &mut Y) -> Result<(), Self::Error> {
        assert_eq!(x.len(), self.ncols(), "matvec_into: dimension mismatch");
        assert_eq!(
            out.len(),
            self.nrows(),
            "matvec_into: output dimension mismatch"
        );
        let coefficients = X::owned_from_fn(self.inner.ncols(), |j| x.get(j + 1));
        self.inner.matvec_into(&coefficients, out)?;
        let intercept = x.get(0);
        for i in 0..out.len() {
            out.set(i, out.get(i) + intercept);
        }
        Ok(())
    }
}

impl<M, F, X, Y> MatTransposeVecInto<X, Y> for WithIntercept<M, F>
where
    F: Scalar,
    X: VectorView<F>,
    Y: VectorOwned<F> + VectorViewMut<F>,
    M: MatTransposeVecInto<X, Y::Owned>,
{
    fn mat_transpose_vec_into(&self, x: &X, out: &mut Y) -> Result<(), Self::Error> {
        assert_eq!(
            x.len(),
            self.nrows(),
            "mat_transpose_vec_into: dimension mismatch"
        );
        assert_eq!(
            out.len(),
            self.ncols(),
            "mat_transpose_vec_into: output dimension mismatch"
        );
        let mut predictors = Y::owned_from_fn(self.inner.ncols(), |_| F::zero());
        self.inner.mat_transpose_vec_into(x, &mut predictors)?;
        for j in 0..predictors.len() {
            out.set(j + 1, predictors.get(j));
        }
        out.set(0, x.sum());
        Ok(())
    }
}

impl<M, F, V> MatVec<V> for WithIntercept<M, F>
where
    F: Scalar,
    V: VectorOwned<F, Owned = V> + VectorViewMut<F>,
    M: MatVecInto<V>,
{
    fn matvec(&self, x: &V) -> Result<V, Self::Error> {
        assert_eq!(x.len(), self.ncols(), "matvec: dimension mismatch");
        let mut out = V::owned_from_fn(self.nrows(), |_| F::zero());
        self.matvec_into(x, &mut out)?;
        Ok(out)
    }
}

impl<M, F, V> MatTransposeVec<V> for WithIntercept<M, F>
where
    F: Scalar,
    V: VectorOwned<F, Owned = V> + VectorViewMut<F>,
    M: MatTransposeVecInto<V>,
{
    fn mat_transpose_vec(&self, x: &V) -> Result<V, Self::Error> {
        assert_eq!(
            x.len(),
            self.nrows(),
            "mat_transpose_vec: dimension mismatch"
        );
        let mut out = V::owned_from_fn(self.ncols(), |_| F::zero());
        self.mat_transpose_vec_into(x, &mut out)?;
        Ok(out)
    }
}

struct PredictorBlock<'a, O: ?Sized>(&'a mut O);

impl<O: MatrixShape + ?Sized> MatrixShape for PredictorBlock<'_, O> {
    fn nrows(&self) -> usize {
        self.0.nrows() - 1
    }
    fn ncols(&self) -> usize {
        self.0.ncols() - 1
    }
}
impl<F: Scalar, O: MatrixWrite<F> + ?Sized> MatrixWrite<F> for PredictorBlock<'_, O> {
    fn set(&mut self, i: usize, j: usize, value: F) {
        self.0.set(i + 1, j + 1, value);
    }
}

impl<M, F> WeightedGramInto<F> for WithIntercept<M, F>
where
    F: Scalar,
    M: WeightedGramInto<F> + WeightedColumnSumsInto<F>,
{
    fn weighted_gram_into<W, O>(&self, weights: &W, out: &mut O) -> Result<(), Self::Error>
    where
        W: VectorView<F> + ?Sized,
        O: MatrixWrite<F> + ?Sized,
    {
        crate::gram::validate(self.nrows(), self.ncols(), weights, None, None, out);
        self.inner
            .weighted_gram_into(weights, &mut PredictorBlock(out))?;
        let mut sums = vec![F::zero(); self.inner.ncols()];
        self.inner.weighted_column_sums_into(weights, &mut sums)?;
        // Keep the intercept block untouched until both fallible operations succeed.
        for (j, value) in sums.into_iter().enumerate() {
            out.set(0, j + 1, value);
            out.set(j + 1, 0, value);
        }
        out.set(0, 0, weights.sum());
        Ok(())
    }
}

impl<M, F> WeightedColumnSumsInto<F> for WithIntercept<M, F>
where
    F: Scalar,
    M: WeightedColumnSumsInto<F>,
{
    fn weighted_column_sums_into<W, O>(&self, weights: &W, out: &mut O) -> Result<(), Self::Error>
    where
        W: VectorView<F> + ?Sized,
        O: VectorViewMut<F> + ?Sized,
    {
        crate::weighted_sums::validate(self.nrows(), self.ncols(), weights, None, None, out);
        let mut sums = vec![F::zero(); self.inner.ncols()];
        self.inner.weighted_column_sums_into(weights, &mut sums)?;
        for (j, value) in sums.into_iter().enumerate() {
            out.set(j + 1, value);
        }
        out.set(0, weights.sum());
        Ok(())
    }
}
