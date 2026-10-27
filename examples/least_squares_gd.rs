//! Least-squares by gradient descent, driven entirely through the lazy
//! or eagerly normalized operator.
//!
//! Solves `min_beta ||X_tilde beta - y||^2 / (2n)` using the same solver
//! for both representations. The shared solver also supports a ridge penalty
//! for the complete-fit benchmarks.
//!
//! Run with: `cargo run --example least_squares_gd --features faer`

#[path = "../tests/common/backend_aliases.rs"]
mod backend_aliases;
use backend_aliases::*;
#[path = "common/least_squares.rs"]
mod least_squares;

use faer::Col;
use faer::sparse::{SparseColMat, Triplet};
use lazymatrix::{Centering, LazyMatrix, MatVec, Normalization, ScaledAddAssign, Scaling};
use least_squares::{Options, solve};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (nrows, ncols, density) = (120, 10, 0.4);
    let mut rng = ChaCha8Rng::seed_from_u64(0xC0FFEE);
    let mut triplets = Vec::new();
    for i in 0..nrows {
        for j in 0..ncols {
            if rng.random::<f64>() < density {
                triplets.push(Triplet::new(i, j, rng.random_range(-2.0..2.0)));
            }
        }
    }
    let x = SparseColMat::<usize, f64>::try_new_from_triplets(nrows, ncols, &triplets)?;
    let lazy = LazyMatrix::new(x, Normalization::new(Centering::Mean, Scaling::Sd))?;
    let beta_star = Col::<f64>::from_fn(ncols, |j| ((j as f64) - 4.5) * 0.5);
    let y = lazy.matvec(&beta_star)?;
    let options = Options {
        penalty: 0.0,
        tolerance: 1e-12,
        max_iterations: 50_000,
        power_iterations: 100,
    };
    let fit = solve(&lazy, &y, options)?;
    let eager = lazy.to_eager::<faer::Mat<f64>>()?;
    let eager_fit = solve(&eager, &y, options)?;
    let mut difference = eager_fit.coefficients;
    difference.scaled_add_assign(-1.0, &fit.coefficients);
    assert!(difference.norm_l2() < 1e-8);
    println!(
        "eager dense fit converged in {} iterations",
        eager_fit.iterations
    );

    let mut coefficient_error = fit.coefficients.clone();
    coefficient_error.scaled_add_assign(-1.0, &beta_star);
    let error = coefficient_error.norm_l2();
    assert!(error < 1e-8);
    println!(
        "converged in {} iterations, gradient norm = {:.3e} <= {:.3e}",
        fit.iterations, fit.gradient_norm, fit.gradient_threshold,
    );
    println!(
        "coefficient error = {error:.3e}, mean squared loss / 2 = {:.3e}",
        fit.objective
    );
    println!("\n   j   beta*     beta_hat");
    for j in 0..ncols {
        println!(
            "  {j:>2}  {:>8.4}  {:>8.4}",
            beta_star[j], fit.coefficients[j]
        );
    }
    Ok(())
}
