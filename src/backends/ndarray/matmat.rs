use super::ndarray;
use crate::{MatMatInto, MatTransposeMatInto, MatrixRead, MatrixWrite, Scalar};
use ndarray::{Array2, ArrayBase, Data, Ix2, s};

impl<F: Scalar, S: Data<Elem = F>> MatrixRead<F> for ArrayBase<S, Ix2> {
    fn get(&self, row: usize, column: usize) -> F {
        self[(row, column)]
    }
}

macro_rules! products {
    ($trait:ident, $product:ident, $normalized:ident, $helper:ident, $transpose:expr) => {
        impl<F: Scalar + 'static, S: Data<Elem = F>> $trait<F> for ArrayBase<S, Ix2> {
            fn $product<R, O>(&self, rhs: &R, out: &mut O) -> Result<(), Self::Error>
            where
                R: MatrixRead<F> + ?Sized,
                O: MatrixRead<F> + MatrixWrite<F> + ?Sized,
            {
                crate::matmat::dense::native(
                    self,
                    rhs,
                    out,
                    $transpose,
                    |left: &Array2<F>,
                     right: &Array2<F>,
                     output: &mut Array2<F>,
                     (rows, depth, responses),
                     first| {
                        ndarray::linalg::general_mat_mul(
                            F::one(),
                            &left.slice(s![..rows, ..depth]),
                            &right.slice(s![..depth, ..responses]),
                            if first { F::zero() } else { F::one() },
                            &mut output.slice_mut(s![..rows, ..responses]),
                        );
                    },
                );
                Ok(())
            }
            fn $normalized<R, O>(
                &self,
                rhs: &R,
                centers: Option<&[F]>,
                scales: Option<&[F]>,
                out: &mut O,
            ) -> Result<(), Self::Error>
            where
                R: MatrixRead<F> + ?Sized,
                O: MatrixRead<F> + MatrixWrite<F> + ?Sized,
            {
                if centers.is_some() {
                    crate::matmat::dense::product(self, rhs, centers, scales, out, $transpose);
                    Ok(())
                } else {
                    crate::matmat::$helper(self, rhs, centers, scales, out)
                }
            }
        }
    };
}
products!(
    MatMatInto,
    matmat_into,
    matmat_normalized_into,
    normalized,
    false
);
products!(
    MatTransposeMatInto,
    mat_transpose_mat_into,
    mat_transpose_mat_normalized_into,
    transpose_normalized,
    true
);
