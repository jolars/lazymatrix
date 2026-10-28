use super::{ClosedAddAssign, ClosedMulAssign, nalgebra, nalgebra_sparse};
use crate::{MatMatInto, MatTransposeMatInto, MatrixRead, MatrixWrite, Scalar};
use nalgebra::{DMatrix, Dim, Matrix, RawStorage};
use nalgebra_sparse::ops::{
    Op,
    serial::{spmm_csc_dense, spmm_csr_dense},
};
use nalgebra_sparse::{CscMatrix, CsrMatrix};

impl<F, R, C, S> MatrixRead<F> for Matrix<F, R, C, S>
where
    F: Scalar + nalgebra::Scalar,
    R: Dim,
    C: Dim,
    S: RawStorage<F, R, C>,
{
    fn get(&self, row: usize, column: usize) -> F {
        self[(row, column)]
    }
}

macro_rules! dense {
    ($trait:ident, $product:ident, $normalized:ident, $helper:ident, $transpose:expr) => {
        impl<F, R, C, S> $trait<F> for Matrix<F, R, C, S>
        where
            F: Scalar + nalgebra::Scalar + ClosedAddAssign + ClosedMulAssign,
            R: Dim,
            C: Dim,
            S: RawStorage<F, R, C>,
        {
            fn $product<X, O>(&self, rhs: &X, out: &mut O) -> Result<(), Self::Error>
            where
                X: MatrixRead<F> + ?Sized,
                O: MatrixRead<F> + MatrixWrite<F> + ?Sized,
            {
                crate::matmat::dense::native(
                    self,
                    rhs,
                    out,
                    $transpose,
                    |left: &DMatrix<F>,
                     right: &DMatrix<F>,
                     output: &mut DMatrix<F>,
                     (rows, depth, responses),
                     first| {
                        output.view_mut((0, 0), (rows, responses)).gemm(
                            F::one(),
                            &left.view((0, 0), (rows, depth)),
                            &right.view((0, 0), (depth, responses)),
                            if first { F::zero() } else { F::one() },
                        );
                    },
                );
                Ok(())
            }
            fn $normalized<X, O>(
                &self,
                rhs: &X,
                centers: Option<&[F]>,
                scales: Option<&[F]>,
                out: &mut O,
            ) -> Result<(), Self::Error>
            where
                X: MatrixRead<F> + ?Sized,
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
dense!(
    MatMatInto,
    matmat_into,
    matmat_normalized_into,
    normalized,
    false
);
dense!(
    MatTransposeMatInto,
    mat_transpose_mat_into,
    mat_transpose_mat_normalized_into,
    transpose_normalized,
    true
);

macro_rules! sparse {
    ($matrix:ident, $kernel:ident, $trait:ident, $product:ident, $op:ident, $transpose:expr) => {
        impl<F: Scalar + nalgebra::Scalar + ClosedAddAssign + ClosedMulAssign> $trait<F>
            for $matrix<F>
        {
            fn $product<R, O>(&self, rhs: &R, out: &mut O) -> Result<(), Self::Error>
            where
                R: MatrixRead<F> + ?Sized,
                O: MatrixRead<F> + MatrixWrite<F> + ?Sized,
            {
                crate::matmat::validate(self, rhs, out, $transpose);
                crate::matmat::sparse_panels(
                    rhs,
                    out,
                    |input: &DMatrix<F>, output: &mut DMatrix<F>, width| {
                        let rows = output.nrows();
                        $kernel(
                            F::zero(),
                            output.view_mut((0, 0), (rows, width)),
                            F::one(),
                            Op::$op(self),
                            Op::NoOp(input.columns(0, width)),
                        );
                    },
                );
                Ok(())
            }
        }
    };
}
sparse!(
    CscMatrix,
    spmm_csc_dense,
    MatMatInto,
    matmat_into,
    NoOp,
    false
);
sparse!(
    CscMatrix,
    spmm_csc_dense,
    MatTransposeMatInto,
    mat_transpose_mat_into,
    Transpose,
    true
);
sparse!(
    CsrMatrix,
    spmm_csr_dense,
    MatMatInto,
    matmat_into,
    NoOp,
    false
);
sparse!(
    CsrMatrix,
    spmm_csr_dense,
    MatTransposeMatInto,
    mat_transpose_mat_into,
    Transpose,
    true
);
