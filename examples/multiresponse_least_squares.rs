//! Fit several responses with batched predictions and gradients.
//!
//! Run with `cargo run --example multiresponse_least_squares --features faer`.

#[path = "../tests/common/backend_aliases.rs"]
mod backend_aliases;
#[path = "common/multiresponse.rs"]
mod multiresponse;

use backend_aliases::faer::{
    Mat,
    sparse::{SparseColMat, Triplet},
};
use lazymatrix::{Centering, LazyMatrix, MatMat, Normalization, Scaling, WithIntercept};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (n, p, k) = (120, 8, 3);
    let mut rng = ChaCha8Rng::seed_from_u64(0xBA7C);
    let mut triplets = Vec::new();
    for i in 0..n {
        for j in 0..p {
            if rng.random::<f64>() < 0.4 {
                triplets.push(Triplet::new(i, j, rng.random_range(-2.0..2.0)));
            }
        }
    }
    let raw = SparseColMat::<usize, f64>::try_new_from_triplets(n, p, &triplets)?;
    let lazy = LazyMatrix::new(raw, Normalization::new(Centering::Mean, Scaling::Sd))?;
    let design = WithIntercept::new(&lazy);
    let truth = Mat::from_fn(p + 1, k, |j, r| (j as f64 - 3.) * (r + 1) as f64 / 4.);
    let response: Mat<f64> = design.matmat(&truth.as_ref())?;
    // Standardized columns and the intercept give ||A||_F² / n = p + 1.
    let step = 1. / (p + 1) as f64;
    let fit = multiresponse::solve(&design, &response, step, 10_000, 1e-9)?;
    assert!(fit.gradient_norm <= 1e-9);
    let eager = lazy.to_eager::<Mat<f64>>()?;
    let eager_fit =
        multiresponse::solve(&WithIntercept::new(eager), &response, step, 10_000, 1e-9)?;
    let mut largest_error = 0_f64;
    for j in 0..=p {
        for r in 0..k {
            largest_error = largest_error.max((fit.coefficients[(j, r)] - truth[(j, r)]).abs());
            assert!((fit.coefficients[(j, r)] - eager_fit.coefficients[(j, r)]).abs() < 1e-8);
        }
    }
    assert!(largest_error < 1e-8);
    println!(
        "fit {k} responses in {} iterations; gradient norm = {:.3e}, largest coefficient error = {largest_error:.3e}",
        fit.iterations, fit.gradient_norm
    );
    Ok(())
}
