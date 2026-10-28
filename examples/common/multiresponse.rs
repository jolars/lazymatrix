//! Multiresponse least squares shared by the example and consumer benchmarks.

use lazymatrix::{MatMatInto, MatTransposeMatInto, MatrixOwned, MatrixRead};

pub struct Fit<D> {
    pub coefficients: D,
    pub iterations: usize,
    pub gradient_norm: f64,
}

/// Minimize `||A B - Y||_F² / (2n)` with a caller-supplied fixed step.
///
/// The caller chooses a step no larger than the inverse Lipschitz constant.
/// A zero tolerance runs a fixed number of updates for iteration benchmarks.
pub fn solve<M, D>(
    matrix: &M,
    responses: &D,
    step: f64,
    max_iterations: usize,
    tolerance: f64,
) -> Result<Fit<D>, M::Error>
where
    M: MatMatInto<f64> + MatTransposeMatInto<f64>,
    D: MatrixOwned<f64> + MatrixRead<f64>,
{
    assert_eq!(matrix.nrows(), responses.nrows());
    assert!(matrix.nrows() > 0 && step > 0. && tolerance >= 0.);
    let (n, p, k) = (matrix.nrows(), matrix.ncols(), responses.ncols());
    let mut coefficients = D::zeros(p, k);
    let mut residual = D::zeros(n, k);
    let mut gradient = D::zeros(p, k);
    for iteration in 0..=max_iterations {
        matrix.matmat_into(&coefficients, &mut residual)?;
        for i in 0..n {
            for r in 0..k {
                residual.set(i, r, residual.get(i, r) - responses.get(i, r));
            }
        }
        matrix.mat_transpose_mat_into(&residual, &mut gradient)?;
        let mut squared_norm = 0.;
        for j in 0..p {
            for r in 0..k {
                let value = gradient.get(j, r) / n as f64;
                gradient.set(j, r, value);
                squared_norm += value * value;
            }
        }
        let gradient_norm = squared_norm.sqrt();
        assert!(gradient_norm.is_finite());
        if iteration == max_iterations || (tolerance > 0. && gradient_norm <= tolerance) {
            return Ok(Fit {
                coefficients,
                iterations: iteration,
                gradient_norm,
            });
        }
        for j in 0..p {
            for r in 0..k {
                coefficients.set(j, r, coefficients.get(j, r) - step * gradient.get(j, r));
            }
        }
    }
    unreachable!()
}
