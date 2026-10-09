//! Least-squares SGD and minibatch SGD over lazily standardized CSR rows.
//!
//! Solves `min_beta ||X_tilde beta - y||^2 / (2 n)`. A normalized row is
//! `a_i + b`, with sparse stored corrections `a_i` and common background
//! `b_j = -center_j / scale_j`. Coefficients are represented as
//! `beta = base + offset * b`, and `b.dot(beta)` is cached. Predictions and
//! updates then visit only stored row entries, despite mean centering.
//!
//! Each batch costs O(batch size + stored entries), with O(ncols) optimizer
//! workspace. Epoch shuffling uses O(nrows) indices. Once per epoch, coefficients
//! are rebased in O(ncols), and full CSR products check convergence. The solver
//! never materializes a normalized design matrix. Sampling, deferred updates,
//! and convergence policy belong here rather than in the library.
//!
//! Run both batch sizes 1 and 16:
//! `cargo run --locked --example least_squares_sgd --features faer`
//!
//! Run a particular batch size:
//! `cargo run --locked --example least_squares_sgd --features faer -- 8`

#[path = "../tests/common/backend_aliases.rs"]
mod backend_aliases;
use backend_aliases::faer;

use faer::Col;
use faer::sparse::{SparseRowMat, Triplet};
use lazymatrix::{
    Centering, LazyMatrix, LazyRow, MatTransposeVecInto, MatVec, MatVecInto, Normalization,
    Scaling, SparseRows,
};
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

struct DeferredCoefficients {
    base: Vec<f64>,
    background: Vec<f64>,
    offset: f64,
    background_dot: f64,
    background_norm_squared: f64,
    gradient: Vec<f64>,
    touched: Vec<usize>,
    seen: Vec<bool>,
}

impl DeferredCoefficients {
    fn new<M: SparseRows<f64>>(matrix: &LazyMatrix<M, f64>) -> Self {
        let background: Vec<_> = (0..matrix.ncols())
            .map(|j| {
                -matrix.centers().map_or(0.0, |centers| centers[j])
                    / matrix.scales().map_or(1.0, |scales| scales[j])
            })
            .collect();
        Self {
            base: vec![0.0; matrix.ncols()],
            background_norm_squared: background.iter().map(|b| b * b).sum(),
            background,
            offset: 0.0,
            background_dot: 0.0,
            gradient: vec![0.0; matrix.ncols()],
            touched: Vec::with_capacity(matrix.ncols()),
            seen: vec![false; matrix.ncols()],
        }
    }

    fn predict(&self, row: LazyRow<'_, f64>) -> f64 {
        self.background_dot
            + row
                .stored_corrections()
                .map(|(j, a)| a * (self.base[j] + self.offset * self.background[j]))
                .sum::<f64>()
    }

    fn update_batch<M: SparseRows<f64>>(
        &mut self,
        matrix: &LazyMatrix<M, f64>,
        y: &[f64],
        observations: &[usize],
        step: f64,
    ) {
        assert!(
            !observations.is_empty(),
            "a batch must contain observations"
        );
        let mut residual_sum = 0.0;
        for &i in observations {
            let row = matrix.row(i);
            // Predictions use the same coefficients throughout the batch.
            let residual = self.predict(row) - y[i];
            residual_sum += residual;
            for (j, correction) in row.stored_corrections() {
                if !self.seen[j] {
                    self.seen[j] = true;
                    self.touched.push(j);
                }
                self.gradient[j] += residual * correction;
            }
        }
        let rate = step / observations.len() as f64;
        for j in self.touched.drain(..) {
            let change = -rate * self.gradient[j];
            self.base[j] += change;
            self.background_dot += change * self.background[j];
            self.gradient[j] = 0.0;
            self.seen[j] = false;
        }
        let offset_change = -rate * residual_sum;
        self.offset += offset_change;
        self.background_dot += offset_change * self.background_norm_squared;
    }

    fn rebase(&mut self) -> &[f64] {
        for (value, &background) in self.base.iter_mut().zip(&self.background) {
            *value += self.offset * background;
        }
        self.offset = 0.0;
        // Periodic recomputation limits drift in the cached background dot.
        self.background_dot = self
            .base
            .iter()
            .zip(&self.background)
            .map(|(v, b)| v * b)
            .sum();
        &self.base
    }
}

struct SgdConfig {
    batch_size: usize,
    step: f64,
    max_epochs: usize,
    tolerance: f64,
}

struct Diagnostics {
    loss: f64,
    gradient_norm: f64,
}

struct Fit {
    beta: Vec<f64>,
    epochs: usize,
    diagnostics: Diagnostics,
}

fn stochastic_gradient_descent<M, E>(
    matrix: &LazyMatrix<M, f64>,
    y: &[f64],
    config: SgdConfig,
    mut evaluate: impl FnMut(&[f64]) -> Result<Diagnostics, E>,
) -> Result<Fit, E>
where
    M: SparseRows<f64>,
{
    assert_eq!(y.len(), matrix.nrows(), "response length must equal nrows");
    assert!(matrix.nrows() > 0 && config.batch_size > 0 && config.max_epochs > 0);
    assert!(config.step.is_finite() && config.step > 0.0);
    assert!(config.tolerance.is_finite() && config.tolerance > 0.0);
    let mut rng = ChaCha8Rng::seed_from_u64(0x5_6_D);
    let mut observations: Vec<_> = (0..matrix.nrows()).collect();
    let mut coefficients = DeferredCoefficients::new(matrix);
    for epoch in 1..=config.max_epochs {
        observations.shuffle(&mut rng);
        for batch in observations.chunks(config.batch_size) {
            coefficients.update_batch(matrix, y, batch, config.step);
        }
        let diagnostics = evaluate(coefficients.rebase())?;
        if diagnostics.gradient_norm < config.tolerance || epoch == config.max_epochs {
            return Ok(Fit {
                beta: coefficients.base,
                epochs: epoch,
                diagnostics,
            });
        }
    }
    unreachable!("a positive epoch limit returns a fit");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let batch_sizes = match std::env::args().nth(1) {
        Some(argument) => {
            let size: usize = argument.parse()?;
            if size == 0 {
                return Err("batch size must be positive".into());
            }
            vec![size]
        }
        None => vec![1, 16],
    };
    let (nrows, ncols) = (120, 10);
    let mut rng = ChaCha8Rng::seed_from_u64(0xC0FFEE);
    let mut triplets = Vec::new();
    for i in 0..nrows {
        for j in 0..ncols {
            if rng.random::<f64>() < 0.25 {
                triplets.push(Triplet::new(i, j, rng.random_range(0.5..2.0)));
            }
        }
    }
    let raw = SparseRowMat::<usize, f64>::try_new_from_triplets(nrows, ncols, &triplets)?;
    let matrix = LazyMatrix::new(raw, Normalization::new(Centering::Mean, Scaling::Sd))?;
    let beta_star = Col::from_fn(ncols, |j| (j as f64 - 4.5) * 0.5);
    let response = matrix.matvec(&beta_star)?;
    let y: Vec<_> = (0..nrows).map(|i| response[i]).collect();
    let background = DeferredCoefficients::new(&matrix);
    // The generated matrix is canonical, so each correction belongs to one cell.
    let max_row_norm_squared = (0..nrows)
        .map(|i| {
            background.background_norm_squared
                + matrix
                    .row(i)
                    .stored_corrections()
                    .map(|(j, a)| a * (a + 2.0 * background.background[j]))
                    .sum::<f64>()
        })
        .fold(0.0_f64, f64::max);
    // The largest squared row norm bounds each batch's gradient Lipschitz constant.
    let step = 0.5 / max_row_norm_squared;

    let mut beta = Col::zeros(ncols);
    let mut residual = Col::zeros(nrows);
    let mut gradient = Col::zeros(ncols);
    for batch_size in batch_sizes {
        let fit = stochastic_gradient_descent(
            &matrix,
            &y,
            SgdConfig {
                batch_size,
                step,
                max_epochs: 5_000,
                tolerance: 1e-9,
            },
            |values| {
                for j in 0..ncols {
                    beta[j] = values[j];
                }
                matrix.matvec_into(&beta, &mut residual)?;
                for i in 0..nrows {
                    residual[i] -= y[i];
                }
                matrix.mat_transpose_vec_into(&residual, &mut gradient)?;
                Ok::<_, std::convert::Infallible>(Diagnostics {
                    loss: (0..nrows).map(|i| residual[i] * residual[i]).sum::<f64>()
                        / (2.0 * nrows as f64),
                    gradient_norm: (0..ncols)
                        .map(|j| gradient[j] * gradient[j])
                        .sum::<f64>()
                        .sqrt()
                        / nrows as f64,
                })
            },
        )?;
        let error = fit
            .beta
            .iter()
            .enumerate()
            .map(|(j, &b)| (b - beta_star[j]).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(fit.diagnostics.gradient_norm < 1e-9, "SGD did not converge");
        assert!(error < 1e-7, "SGD did not recover the coefficients");
        println!(
            "batch size {batch_size}: {} epochs, loss = {:.3e}, gradient norm = {:.3e}, coefficient error = {error:.3e}",
            fit.epochs, fit.diagnostics.loss, fit.diagnostics.gradient_norm
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deferred_batches_match_dense_updates() {
        // The empty middle row exercises the centering background on its own.
        let raw = SparseRowMat::try_new_from_triplets(
            3,
            3,
            &[
                Triplet::new(0_usize, 0_usize, 2.0),
                Triplet::new(0, 2, -1.0),
                Triplet::new(2, 1, 4.0),
            ],
        )
        .unwrap();
        for centers in [None, Some(vec![0.5, -1.0, 2.0])] {
            for scales in [None, Some(vec![2.0, -3.0, 1.0])] {
                let matrix = LazyMatrix::from_parts(&raw, centers.clone(), scales);
                let dense = matrix.to_eager::<faer::Mat<f64>>().unwrap();
                let y = [1.0, -2.0, 0.5];
                for batch_size in [1, 2, 3, 8] {
                    let mut deferred = DeferredCoefficients::new(&matrix);
                    let mut beta = [0.0; 3];
                    for _ in 0..5 {
                        for batch in [2, 0, 1].chunks(batch_size) {
                            let mut gradient = [0.0; 3];
                            for &i in batch {
                                let prediction: f64 =
                                    (0..3).map(|j| dense.data()[(i, j)] * beta[j]).sum();
                                approx::assert_abs_diff_eq!(
                                    deferred.predict(matrix.row(i)),
                                    prediction,
                                    epsilon = 1e-12
                                );
                                let residual = prediction - y[i];
                                for j in 0..3 {
                                    gradient[j] += residual * dense.data()[(i, j)];
                                }
                            }
                            for j in 0..3 {
                                beta[j] -= 0.01 * gradient[j] / batch.len() as f64;
                            }
                            deferred.update_batch(&matrix, &y, batch, 0.01);
                            for (j, &expected) in beta.iter().enumerate() {
                                approx::assert_abs_diff_eq!(
                                    deferred.base[j] + deferred.offset * deferred.background[j],
                                    expected,
                                    epsilon = 1e-12
                                );
                            }
                        }
                        for (&actual, &expected) in deferred.rebase().iter().zip(&beta) {
                            approx::assert_abs_diff_eq!(actual, expected, epsilon = 1e-12);
                        }
                    }
                }
            }
        }
    }
}
