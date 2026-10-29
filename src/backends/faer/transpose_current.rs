use super::{faer, faer_traits};
use crate::Scalar;
use faer::sparse::SparseColMatRef;
use faer::{Accum, Col, Par};

pub(super) fn multiply_csr_mat<F: Scalar + faer_traits::ComplexField>(
    matrix: faer::sparse::SparseRowMatRef<'_, usize, F>,
    input: faer::MatRef<'_, F>,
    output: faer::MatMut<'_, F>,
    parallelism: Par,
) {
    faer::sparse::linalg::matmul::sparse_dense_matmul(
        output,
        Accum::Replace,
        matrix,
        input,
        F::one(),
        parallelism,
    );
}

pub(super) fn multiply_transpose_mat<F: Scalar + faer_traits::ComplexField>(
    matrix: SparseColMatRef<'_, usize, F>,
    input: faer::MatRef<'_, F>,
    output: faer::MatMut<'_, F>,
    parallelism: Par,
) {
    faer::sparse::linalg::matmul::sparse_dense_matmul(
        output,
        Accum::Replace,
        matrix.transpose(),
        input,
        F::one(),
        parallelism,
    );
}

pub(super) fn multiply_csr<F: Scalar + faer_traits::ComplexField>(
    matrix: faer::sparse::SparseRowMatRef<'_, usize, F>,
    input: &Col<F>,
    output: &mut Col<F>,
    alpha: F,
    accumulation: Accum,
    parallelism: Par,
) {
    faer::sparse::linalg::matmul::sparse_dense_matmul(
        output.as_mat_mut(),
        accumulation,
        matrix,
        input.as_mat(),
        alpha,
        parallelism,
    );
}

pub(super) fn multiply<F: Scalar + faer_traits::ComplexField>(
    matrix: SparseColMatRef<'_, usize, F>,
    input: &Col<F>,
    output: &mut Col<F>,
    alpha: F,
    accumulation: Accum,
    parallelism: Par,
) {
    faer::sparse::linalg::matmul::sparse_dense_matmul(
        output.as_mat_mut(),
        accumulation,
        matrix.transpose(),
        input.as_mat(),
        alpha,
        parallelism,
    );
}
