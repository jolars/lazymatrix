//! Ridge regression through the smaller primal or dual normal system.
//!
//! One conjugate-gradient implementation solves `(Bᵀ B + lambda I) x = rhs`.
//! For a wide design, passing its borrowed transpose gives `A Aᵀ + lambda I`
//! without a second solver or a materialized Gram matrix. Fitted normalization
//! remains part of the design operator in either orientation.
//!
//! Run with: `cargo run --example ridge_primal_dual --features faer`

#[path = "../tests/common/backend_aliases.rs"]
mod backend_aliases;
use backend_aliases::*;

use faer::Col;
use faer::sparse::{SparseColMat, Triplet};
use lazymatrix::{
    ColumnNormalization, DotProduct, LazyMatrix, MatTransposeVecInto, MatVecInto, ScaleAssign,
    ScaledAddAssign, Transpose,
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn solve_shifted_gram<M>(operator: &M, rhs: &Col<f64>, penalty: f64) -> Result<Col<f64>, M::Error>
where
    M: MatVecInto<Col<f64>> + MatTransposeVecInto<Col<f64>>,
{
    assert_eq!(rhs.nrows(), operator.ncols());
    assert!(penalty > 0.0);
    let mut solution = Col::zeros(operator.ncols());
    let mut residual = rhs.clone();
    let mut direction = residual.clone();
    let mut intermediate = Col::zeros(operator.nrows());
    let mut product = Col::zeros(operator.ncols());
    let mut squared_norm = residual.dot(&residual);
    let threshold = 1e-20 * squared_norm.max(1.0);

    for _ in 0..1_000 {
        if squared_norm <= threshold {
            return Ok(solution);
        }
        operator.matvec_into(&direction, &mut intermediate)?;
        operator.mat_transpose_vec_into(&intermediate, &mut product)?;
        product.scaled_add_assign(penalty, &direction);
        let step = squared_norm / direction.dot(&product);
        solution.scaled_add_assign(step, &direction);
        residual.scaled_add_assign(-step, &product);
        let next_norm = residual.dot(&residual);
        direction.scale_assign(next_norm / squared_norm);
        direction.scaled_add_assign(1.0, &residual);
        squared_norm = next_norm;
    }
    panic!("conjugate gradient did not converge");
}

/// Minimize `||A beta - y||² / 2 + penalty * ||beta||² / 2`.
fn ridge<M>(design: &M, response: &Col<f64>, penalty: f64) -> Result<Col<f64>, M::Error>
where
    M: MatVecInto<Col<f64>> + MatTransposeVecInto<Col<f64>>,
{
    assert_eq!(response.nrows(), design.nrows());
    if design.ncols() <= design.nrows() {
        let mut rhs = Col::zeros(design.ncols());
        design.mat_transpose_vec_into(response, &mut rhs)?;
        solve_shifted_gram(design, &rhs, penalty)
    } else {
        let dual = solve_shifted_gram(&Transpose::new(design), response, penalty)?;
        let mut coefficients = Col::zeros(design.ncols());
        design.mat_transpose_vec_into(&dual, &mut coefficients)?;
        Ok(coefficients)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = ChaCha8Rng::seed_from_u64(0xD0A1);
    for (n, p) in [(12, 40), (40, 12)] {
        let mut triplets = Vec::new();
        for i in 0..n {
            for j in 0..p {
                if rng.random::<f64>() < 0.35 {
                    triplets.push(Triplet::new(i, j, rng.random_range(-2.0..2.0)));
                }
            }
        }
        let raw = SparseColMat::<usize, f64>::try_new_from_triplets(n, p, &triplets)?;
        let design = LazyMatrix::new(raw, ColumnNormalization::standardize())?;
        let response = Col::from_fn(n, |i| (i % 7) as f64 - 3.0);
        let penalty = 0.5;
        let coefficients = ridge(&design, &response, penalty)?;

        let mut residual = Col::zeros(n);
        design.matvec_into(&coefficients, &mut residual)?;
        residual.scaled_add_assign(-1.0, &response);
        let mut gradient = Col::zeros(p);
        design.mat_transpose_vec_into(&residual, &mut gradient)?;
        gradient.scaled_add_assign(penalty, &coefficients);
        approx::assert_abs_diff_eq!(gradient.norm_l2(), 0.0, epsilon = 1e-8);

        // Both systems must yield the same coefficients despite different dimensions.
        let mut rhs = Col::zeros(p);
        design.mat_transpose_vec_into(&response, &mut rhs)?;
        let primal = solve_shifted_gram(&design, &rhs, penalty)?;
        let dual = solve_shifted_gram(&Transpose::new(&design), &response, penalty)?;
        let mut dual_coefficients = Col::zeros(p);
        design.mat_transpose_vec_into(&dual, &mut dual_coefficients)?;
        let mut difference = primal.clone();
        difference.scaled_add_assign(-1.0, &dual_coefficients);
        approx::assert_abs_diff_eq!(difference.norm_l2(), 0.0, epsilon = 1e-8);
        let mut selected_difference = coefficients;
        selected_difference.scaled_add_assign(-1.0, &primal);
        approx::assert_abs_diff_eq!(selected_difference.norm_l2(), 0.0, epsilon = 1e-8);

        println!(
            "{n} x {p} design: solved the {} x {} {} system; coefficient difference = {:.3e}",
            n.min(p),
            n.min(p),
            if p <= n { "primal" } else { "dual" },
            difference.norm_l2(),
        );
    }
    Ok(())
}
