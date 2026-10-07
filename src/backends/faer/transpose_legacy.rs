use super::{faer, faer_traits};
use crate::Scalar;
use faer::sparse::SparseColMat;
use faer::{Accum, Col, Par};

pub(super) fn multiply<F: Scalar + faer_traits::ComplexField>(
    matrix: &SparseColMat<usize, F>,
    input: &Col<F>,
    output: &mut Col<F>,
    alpha: F,
    accumulation: Accum,
    parallelism: Par,
) {
    // Older faer releases only accept CSC input, so compute y^T = x^T X.
    faer::sparse::linalg::matmul::dense_sparse_matmul(
        output.as_mat_mut().transpose_mut(),
        accumulation,
        input.as_mat().transpose(),
        matrix.as_ref(),
        alpha,
        parallelism,
    );
}
