//! Gradient descent shared by the example and its complete-fit benchmarks.

use lazymatrix::{MatTransposeVecInto, MatVecInto, VectorOwned, VectorView, VectorViewMut};

#[derive(Clone, Copy)]
pub struct Options {
    pub penalty: f64,
    pub tolerance: f64,
    pub max_iterations: usize,
    pub power_iterations: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            penalty: 0.1,
            tolerance: 1e-7,
            max_iterations: 10_000,
            power_iterations: 30,
        }
    }
}

pub struct Fit<V> {
    pub coefficients: V,
    pub iterations: usize,
    pub objective: f64,
    pub gradient_norm: f64,
    pub gradient_threshold: f64,
}

fn squared_norm<V: VectorView<f64>>(vector: &V) -> f64 {
    (0..vector.len()).map(|i| vector.get(i).powi(2)).sum()
}

/// Minimize `||A beta - y||^2 / (2n) + penalty * ||beta||^2 / 2`.
///
/// All matrix access uses the operator, and every call starts from zero.
pub fn solve<M, V>(matrix: &M, response: &V, options: Options) -> Result<Fit<V>, M::Error>
where
    M: MatVecInto<V> + MatTransposeVecInto<V>,
    V: VectorOwned<f64, Owned = V> + VectorViewMut<f64>,
{
    assert_eq!(matrix.nrows(), response.len());
    assert!(matrix.nrows() > 0 && matrix.ncols() > 0);
    assert!(options.penalty >= 0.0 && options.tolerance > 0.0);
    let n = matrix.nrows() as f64;
    let p = matrix.ncols();
    let mut coefficients = V::owned_from_fn(p, |i| 1.0 + (i % 13) as f64 / 13.0);
    let mut residual = V::owned_from_fn(matrix.nrows(), |_| 0.0);
    let mut gradient = V::owned_from_fn(p, |_| 0.0);
    let mut trial = V::owned_from_fn(p, |_| 0.0);
    let mut curvature = 1.0;

    for _ in 0..options.power_iterations {
        let norm = squared_norm(&coefficients).sqrt();
        for j in 0..p {
            coefficients.set(j, coefficients.get(j) / norm);
        }
        matrix.matvec_into(&coefficients, &mut residual)?;
        matrix.mat_transpose_vec_into(&residual, &mut gradient)?;
        for j in 0..p {
            gradient.set(
                j,
                gradient.get(j) / n + options.penalty * coefficients.get(j),
            );
        }
        curvature = squared_norm(&gradient).sqrt();
        if curvature == 0.0 {
            break;
        }
        std::mem::swap(&mut coefficients, &mut gradient);
    }
    for j in 0..p {
        coefficients.set(j, 0.0);
    }
    let mut step = 1.0 / (1.1 * curvature).max(f64::EPSILON);
    let mut threshold = 0.0;
    for iteration in 0..=options.max_iterations {
        matrix.matvec_into(&coefficients, &mut residual)?;
        for i in 0..response.len() {
            residual.set(i, residual.get(i) - response.get(i));
        }
        let objective = squared_norm(&residual) / (2.0 * n)
            + options.penalty * squared_norm(&coefficients) / 2.0;
        matrix.mat_transpose_vec_into(&residual, &mut gradient)?;
        for j in 0..p {
            gradient.set(
                j,
                gradient.get(j) / n + options.penalty * coefficients.get(j),
            );
        }
        let gradient_squared = squared_norm(&gradient);
        let gradient_norm = gradient_squared.sqrt();
        if iteration == 0 {
            threshold = options.tolerance * gradient_norm.max(1.0);
        }
        assert!(objective.is_finite() && gradient_norm.is_finite());
        if gradient_norm <= threshold {
            return Ok(Fit {
                coefficients,
                iterations: iteration,
                objective,
                gradient_norm,
                gradient_threshold: threshold,
            });
        }
        assert!(
            iteration < options.max_iterations,
            "gradient descent did not converge"
        );

        // Backtracking protects against an underestimated largest eigenvalue.
        let mut accepted = false;
        for _ in 0..60 {
            for j in 0..p {
                trial.set(j, coefficients.get(j) - step * gradient.get(j));
            }
            matrix.matvec_into(&trial, &mut residual)?;
            for i in 0..response.len() {
                residual.set(i, residual.get(i) - response.get(i));
            }
            let trial_objective =
                squared_norm(&residual) / (2.0 * n) + options.penalty * squared_norm(&trial) / 2.0;
            if trial_objective
                <= objective - 0.5 * step * gradient_squared
                    + 8.0 * f64::EPSILON * objective.abs().max(1.0)
            {
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        assert!(accepted, "gradient descent line search failed");
        std::mem::swap(&mut coefficients, &mut trial);
    }
    unreachable!()
}
