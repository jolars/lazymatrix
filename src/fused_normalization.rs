//! Fused products reuse the fitted row and pipeline vector kernels.

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
    for (i, value) in scratch.into_iter().enumerate() {
        let value = alpha * value;
        out.set(
            i,
            if beta == F::zero() {
                value
            } else {
                value + beta * out.get(i)
            },
        );
    }
    Ok(())
}

macro_rules! fused {
    ($params:ty, [$($generics:tt)*]) => {
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
