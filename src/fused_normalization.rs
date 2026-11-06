//! Fused and workspace products reuse fitted row and pipeline vector kernels.

use crate::{
    LazyMatrix, MatTransposeVecInto, MatTransposeVecScaledInto, MatVecInto, MatVecScaledInto,
    MatrixErrorType, MatrixShape, NormalizationChainParams, NormalizationParams,
    RowNormalizationParams, Scalar, VectorView, VectorViewMut,
};

fn scaled_product<M, F, X, Y, E>(
    matrix: &M,
    transpose: bool,
    alpha: F,
    x: &X,
    beta: F,
    out: &mut Y,
    product: impl FnOnce(&mut Vec<F>) -> Result<(), E>,
) -> Result<(), E>
where
    M: MatrixShape,
    F: Scalar,
    X: VectorView<F>,
    Y: VectorViewMut<F>,
{
    let (input, output) = if transpose {
        (matrix.nrows(), matrix.ncols())
    } else {
        (matrix.ncols(), matrix.nrows())
    };
    assert_eq!(x.len(), input, "fused product input dimension mismatch");
    assert_eq!(out.len(), output, "fused product output dimension mismatch");
    if alpha == F::zero() {
        crate::traits::scale_output(beta, out);
        return Ok(());
    }
    // Keep the completed normalized product separate from beta's old output.
    // Scaling the final sum also preserves alpha's IEEE behavior on empty sums.
    let mut scratch = vec![F::zero(); output];
    product(&mut scratch)?;
    combine_product(alpha, beta, out, &scratch);
    Ok(())
}

fn combine_product<F: Scalar, Y: VectorViewMut<F>, W: VectorView<F>>(
    alpha: F,
    beta: F,
    out: &mut Y,
    scratch: &W,
) {
    for i in 0..out.len() {
        let value = alpha * scratch.get(i);
        out.set(
            i,
            if beta == F::zero() {
                value
            } else {
                value + beta * out.get(i)
            },
        );
    }
}

macro_rules! fused {
    ($params:ty, [$($generics:tt)*]) => {
        impl<M, F: Scalar, $($generics)*> LazyMatrix<M, $params>
        where
            M: MatrixShape + MatrixErrorType,
            $params: NormalizationParams<Scalar = F>,
        {
            /// Apply `out = alpha * A * x + beta * out` using caller-owned product scratch.
            ///
            /// `A` is the fitted normalized operator. `scratch` must have length
            /// `nrows` and may use any mutable vector representation supported by
            /// the reusable product. No product vector is allocated by this
            /// wrapper; sparse traversal and chunk decoding may allocate backend
            /// workspace. Built-in dense kernels need no additional workspace.
            ///
            /// Exact zero `alpha` skips input and storage reads and leaves scratch
            /// untouched. Exact zero `beta` ignores previous output values. All
            /// dimensions are checked before reads or writes, even for zero `alpha`.
            /// Scratch contents after a nonzero product are unspecified.
            ///
            /// # Errors
            /// Returns the backend error, leaving `out` unchanged. Scratch may
            /// contain a partial product after an error.
            ///
            /// # Panics
            /// Panics unless `x` has length `ncols`, and `out` and `scratch` have
            /// length `nrows`.
            pub fn matvec_scaled_with_workspace<X, Y, W>(
                &self,
                alpha: F,
                x: &X,
                beta: F,
                out: &mut Y,
                scratch: &mut W,
            ) -> Result<(), <Self as MatrixErrorType>::Error>
            where
                X: VectorView<F>,
                Y: VectorViewMut<F>,
                W: VectorViewMut<F>,
                Self: MatVecInto<X, W>,
            {
                assert_eq!(x.len(), self.ncols(), "workspace input dimension mismatch");
                assert_eq!(out.len(), self.nrows(), "workspace output dimension mismatch");
                assert_eq!(scratch.len(), self.nrows(), "workspace scratch dimension mismatch");
                if alpha == F::zero() {
                    crate::traits::scale_output(beta, out);
                    return Ok(());
                }
                self.matvec_into(x, scratch)?;
                combine_product(alpha, beta, out, scratch);
                Ok(())
            }

            /// Apply `out = alpha * Aᵀ * x + beta * out` using caller-owned product scratch.
            ///
            /// `A` is the fitted normalized operator. `scratch` must have length
            /// `ncols`. It holds the completed normalized product separately from
            /// the previous output. This wrapper allocates no product vector;
            /// sparse traversal and chunk decoding may allocate backend workspace.
            /// Built-in dense kernels need no additional workspace. Inputs, outputs,
            /// and scratch may have different vector representations.
            ///
            /// Exact zero `alpha` skips input and storage reads and leaves scratch
            /// untouched. Exact zero `beta` ignores previous output values. All
            /// dimensions are checked before reads or writes, even for zero `alpha`.
            /// Scratch contents after a nonzero product are unspecified.
            ///
            /// # Errors
            /// Returns the backend error, leaving `out` unchanged. Scratch may
            /// contain a partial product after an error.
            ///
            /// # Panics
            /// Panics unless `x` has length `nrows`, and `out` and `scratch` have
            /// length `ncols`.
            pub fn mat_transpose_vec_scaled_with_workspace<X, Y, W>(
                &self,
                alpha: F,
                x: &X,
                beta: F,
                out: &mut Y,
                scratch: &mut W,
            ) -> Result<(), <Self as MatrixErrorType>::Error>
            where
                X: VectorView<F>,
                Y: VectorViewMut<F>,
                W: VectorViewMut<F>,
                Self: MatTransposeVecInto<X, W>,
            {
                assert_eq!(x.len(), self.nrows(), "workspace input dimension mismatch");
                assert_eq!(out.len(), self.ncols(), "workspace output dimension mismatch");
                assert_eq!(scratch.len(), self.ncols(), "workspace scratch dimension mismatch");
                if alpha == F::zero() {
                    crate::traits::scale_output(beta, out);
                    return Ok(());
                }
                self.mat_transpose_vec_into(x, scratch)?;
                combine_product(alpha, beta, out, scratch);
                Ok(())
            }
        }

        impl<M, F: Scalar, X, Y, $($generics)*> MatVecScaledInto<X, Y, F> for LazyMatrix<M, $params>
        where
            M: MatrixShape + MatrixErrorType,
            $params: NormalizationParams<Scalar = F>,
            X: VectorView<F>,
            Y: VectorViewMut<F>,
            Self: MatVecInto<X, Vec<F>>,
        {
            fn matvec_scaled_into(&self, alpha: F, x: &X, beta: F, out: &mut Y) -> Result<(), Self::Error> {
                scaled_product(self, false, alpha, x, beta, out, |scratch| {
                    self.matvec_into(x, scratch)
                })
            }
        }
        impl<M, F: Scalar, X, Y, $($generics)*> MatTransposeVecScaledInto<X, Y, F> for LazyMatrix<M, $params>
        where
            M: MatrixShape + MatrixErrorType,
            $params: NormalizationParams<Scalar = F>,
            X: VectorView<F>,
            Y: VectorViewMut<F>,
            Self: MatTransposeVecInto<X, Vec<F>>,
        {
            fn mat_transpose_vec_scaled_into(
                &self,
                alpha: F,
                x: &X,
                beta: F,
                out: &mut Y,
            ) -> Result<(), Self::Error> {
                scaled_product(self, true, alpha, x, beta, out, |scratch| {
                    self.mat_transpose_vec_into(x, scratch)
                })
            }
        }
    };
}
fused!(RowNormalizationParams<F, S>, [S: AsRef<[F]>]);
fused!(NormalizationChainParams<A, B>, [A, B]);
