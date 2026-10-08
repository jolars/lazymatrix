//! Least-squares by gradient descent, driven entirely through the lazy
//! or eagerly normalized operator.
//!
//! Solves `min_β ½‖X̃β − y‖²` first lazily, then with explicitly materialized
//! dense data, using the same solver.
//! Every matrix touch goes through [`MatVec`] / [`MatTransposeVec`]:
//!
//! * the gradient `X̃ᵀ(X̃β − y)` is one `matvec` then one `mat_transpose_vec`;
//! * the step size is `1/L` with `L ≈ λ_max(X̃ᵀX̃)` estimated by power iteration,
//!   which is itself nothing but repeated `matvec`/`mat_transpose_vec`.
//!
//! This is the cheapest possible end-to-end check that the operator is a
//! faithful linear map: with noiseless `y = X̃β*`, GD must recover `β*`.
//!
//! Run with: `cargo run --example least_squares_gd --features faer`

#[path = "../tests/common/backend_aliases.rs"]
mod backend_aliases;
use backend_aliases::*;

use faer::Col;
use faer::sparse::{SparseColMat, Triplet};
use lazymatrix::{
    Centering, DotProduct, LazyMatrix, MatTransposeVecInto, MatVec, MatVecInto, Normalization,
    ScaleAssign, ScaledAddAssign, Scaling,
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

// --- solver pieces, generic over any lazy/backend linear operator -------------

/// Estimate `λ_max(AᵀA)` (the gradient's Lipschitz constant) by power iteration
/// on `AᵀA`, using only `matvec`/`mat_transpose_vec`.
fn estimate_lipschitz<Op>(op: &Op, ncols: usize, iters: usize) -> Result<f64, Op::Error>
where
    Op: MatVecInto<Col<f64>> + MatTransposeVecInto<Col<f64>>,
{
    let mut v = Col::<f64>::from_fn(ncols, |i| 1.0 + (i as f64) * 0.01);
    let mut av = Col::<f64>::zeros(op.nrows());
    let mut atav = Col::<f64>::zeros(ncols);
    let mut lambda = 1.0;
    for _ in 0..iters {
        op.matvec_into(&v, &mut av)?;
        op.mat_transpose_vec_into(&av, &mut atav)?; // AᵀA v
        lambda = atav.norm_l2();
        if lambda == 0.0 {
            break;
        }
        atav.scale_assign(1.0 / lambda);
        std::mem::swap(&mut v, &mut atav);
    }
    Ok(lambda)
}

/// Gradient descent on `½‖Aβ − y‖²`. Returns `(β, iterations, final ‖grad‖)`.
fn gradient_descent<Op>(
    op: &Op,
    y: &Col<f64>,
    ncols: usize,
    step: f64,
    max_iter: usize,
    tol: f64,
) -> Result<(Col<f64>, usize, f64), Op::Error>
where
    Op: MatVecInto<Col<f64>> + MatTransposeVecInto<Col<f64>>,
{
    let mut beta = Col::<f64>::zeros(ncols);
    let mut resid = Col::<f64>::zeros(y.nrows());
    let mut grad = Col::<f64>::zeros(ncols);
    let mut last = f64::INFINITY;
    for k in 0..max_iter {
        op.matvec_into(&beta, &mut resid)?;
        resid.scaled_add_assign(-1.0, y); // Aβ − y
        op.mat_transpose_vec_into(&resid, &mut grad)?; // Aᵀ(Aβ − y)
        let gnorm = grad.norm_l2();
        if gnorm < tol {
            return Ok((beta, k, gnorm));
        }
        beta.scaled_add_assign(-step, &grad);
        last = gnorm;
    }
    Ok((beta, max_iter, last))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (nrows, ncols, density) = (120, 10, 0.4);
    let mut rng = ChaCha8Rng::seed_from_u64(0xC0FFEE);

    // Random sparse design matrix X.
    let mut triplets = Vec::new();
    for i in 0..nrows {
        for j in 0..ncols {
            if rng.random::<f64>() < density {
                triplets.push(Triplet::new(i, j, rng.random_range(-2.0..2.0)));
            }
        }
    }
    let x = SparseColMat::<usize, f64>::try_new_from_triplets(nrows, ncols, &triplets)?;

    // Standardize columns lazily (center + unit sd), never forming X − 1cᵀ.
    let lazy = LazyMatrix::new(x, Normalization::new(Centering::Mean, Scaling::Sd))?;

    // Ground-truth coefficients and a noiseless target y = X̃ β*.
    let beta_star = Col::<f64>::from_fn(ncols, |j| ((j as f64) - 4.5) * 0.5);
    let y = lazy.matvec(&beta_star)?;

    let l = estimate_lipschitz(&lazy, ncols, 100)?;
    let step = 1.0 / l;
    println!("estimated Lipschitz L ≈ {l:.4}, step = 1/L = {step:.4e}");

    let (beta, iters, gnorm) = gradient_descent(&lazy, &y, ncols, step, 50_000, 1e-10)?;

    // The same solver accepts explicitly materialized normalized dense data.
    let eager = lazy.to_eager::<faer::Mat<f64>>()?;
    let (eager_beta, eager_iters, _) = gradient_descent(&eager, &y, ncols, step, 50_000, 1e-10)?;
    let mut difference = eager_beta;
    difference.scaled_add_assign(-1.0, &beta);
    assert!(difference.norm_l2() < 1e-8);
    println!("eager dense fit converged in {eager_iters} iters");

    let mut coefficient_error = beta.clone();
    coefficient_error.scaled_add_assign(-1.0, &beta_star);
    let err = coefficient_error.norm_l2();
    let loss = 0.5 * {
        let mut r = lazy.matvec(&beta)?;
        r.scaled_add_assign(-1.0, &y);
        r.dot(&r)
    };
    println!("converged in {iters} iters, ‖grad‖ = {gnorm:.3e}");
    println!("‖β − β*‖ = {err:.3e}, ½‖X̃β − y‖² = {loss:.3e}");
    println!("\n   j   β*        β̂");
    for j in 0..ncols {
        println!("  {j:>2}  {:>8.4}  {:>8.4}", beta_star[j], beta[j]);
    }
    Ok(())
}
