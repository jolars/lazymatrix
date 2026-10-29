//! Lasso coordinate descent over lazily standardized sparse columns.
//!
//! Solves `min_β ½‖y − X̃β‖² + λ‖β‖₁`, where
//! `X̃ = (X − 1cᵀ)S⁻¹`, using [`LazyMatrix::column`] for every matrix touch in
//! the solver. A centered sparse column is generally dense, so the residual is
//! represented as `base + offset·1`. Updating one coefficient then changes the
//! dense background through `offset` in O(1) and touches `base` only at the
//! column's stored rows, keeping each coordinate update O(nnz_j).
//!
//! Run with: `cargo run --example coordinate_descent --features faer`

#[path = "../tests/common/backend_aliases.rs"]
mod backend_aliases;
use backend_aliases::*;

use faer::Col;
use faer::sparse::{SparseColMat, Triplet};
use lazymatrix::{
    Centering, ColumnNormalization, LazyMatrix, LazySparseColumn, MatVec, Scaling, SparseColumns,
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[derive(Clone, Copy)]
struct ColumnSummary {
    raw_sum: f64,
    sum: f64,
    norm_squared: f64,
}

struct OffsetResidual {
    base: Vec<f64>,
    base_sum: f64,
    offset: f64,
}

impl OffsetResidual {
    fn new(values: &[f64]) -> Self {
        Self {
            base: values.to_vec(),
            base_sum: values.iter().sum(),
            offset: 0.0,
        }
    }

    fn column_dot(&self, column: LazySparseColumn<'_, f64>, column_sum: f64) -> f64 {
        column.dot_with_sum(&self.base, self.base_sum) + self.offset * column_sum
    }

    fn subtract_column(
        &mut self,
        column: LazySparseColumn<'_, f64>,
        raw_sum: f64,
        coefficient_change: f64,
    ) {
        for (row, correction) in column.stored_corrections() {
            self.base[row] -= coefficient_change * correction;
        }
        self.base_sum -= coefficient_change * raw_sum / column.scale();
        self.offset -= coefficient_change * column.implicit_value();
    }

    fn norm_squared(&self) -> f64 {
        self.base
            .iter()
            .map(|&value| {
                let residual = value + self.offset;
                residual * residual
            })
            .sum()
    }
}

struct CoordinateDescentResult {
    beta: Vec<f64>,
    residual: OffsetResidual,
    sweeps: usize,
    kkt_violation: f64,
}

fn summarize_column(column: LazySparseColumn<'_, f64>) -> ColumnSummary {
    let raw_sum = column.raw_sum();

    ColumnSummary {
        raw_sum,
        sum: column.sum(),
        norm_squared: column.norm_squared(),
    }
}

fn soft_threshold(value: f64, lambda: f64) -> f64 {
    if value > lambda {
        value - lambda
    } else if value < -lambda {
        value + lambda
    } else {
        0.0
    }
}

fn kkt_violation<M>(
    matrix: &LazyMatrix<M, f64>,
    beta: &[f64],
    residual: &OffsetResidual,
    summaries: &[ColumnSummary],
    lambda: f64,
) -> f64
where
    M: SparseColumns<f64>,
{
    (0..matrix.ncols())
        .map(|j| {
            let correlation = residual.column_dot(matrix.sparse_column(j), summaries[j].sum);
            if beta[j] > 0.0 {
                (correlation - lambda).abs()
            } else if beta[j] < 0.0 {
                (correlation + lambda).abs()
            } else {
                (correlation.abs() - lambda).max(0.0)
            }
        })
        .fold(0.0, f64::max)
}

fn coordinate_descent<M>(
    matrix: &LazyMatrix<M, f64>,
    y: &[f64],
    lambda: f64,
    tolerance: f64,
    max_sweeps: usize,
) -> CoordinateDescentResult
where
    M: SparseColumns<f64>,
{
    assert_eq!(matrix.nrows(), y.len(), "response length must equal nrows");
    let summaries: Vec<_> = (0..matrix.ncols())
        .map(|j| summarize_column(matrix.sparse_column(j)))
        .collect();
    let mut beta = vec![0.0; matrix.ncols()];
    let mut residual = OffsetResidual::new(y);

    for sweep in 1..=max_sweeps {
        for j in 0..matrix.ncols() {
            let summary = summaries[j];
            let correlation = residual.column_dot(matrix.sparse_column(j), summary.sum);
            let partial_correlation = correlation + summary.norm_squared * beta[j];
            let updated = if summary.norm_squared == 0.0 {
                0.0
            } else {
                soft_threshold(partial_correlation, lambda) / summary.norm_squared
            };
            let change = updated - beta[j];
            if change != 0.0 {
                residual.subtract_column(matrix.sparse_column(j), summary.raw_sum, change);
                beta[j] = updated;
            }
        }

        let violation = kkt_violation(matrix, &beta, &residual, &summaries, lambda);
        if violation <= tolerance {
            return CoordinateDescentResult {
                beta,
                residual,
                sweeps: sweep,
                kkt_violation: violation,
            };
        }
    }

    let violation = kkt_violation(matrix, &beta, &residual, &summaries, lambda);
    CoordinateDescentResult {
        beta,
        residual,
        sweeps: max_sweeps,
        kkt_violation: violation,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (nrows, ncols, density) = (120, 12, 0.25);
    let mut rng = ChaCha8Rng::seed_from_u64(0xC0DEC0DE);
    let mut triplets = Vec::new();
    for i in 0..nrows {
        for j in 0..ncols {
            if rng.random::<f64>() < density {
                triplets.push(Triplet::new(i, j, rng.random_range(-2.0..2.0)));
            }
        }
    }
    let data = SparseColMat::<usize, f64>::try_new_from_triplets(nrows, ncols, &triplets)?;
    let matrix = LazyMatrix::new(
        data,
        ColumnNormalization::new(Centering::Mean, Scaling::Sd { df: 0 }),
    )?;

    let beta_star = Col::from_fn(ncols, |j| match j {
        0 => 2.0,
        3 => -1.5,
        7 => 1.0,
        10 => -0.75,
        _ => 0.0,
    });
    let mut y: Vec<f64> = {
        let signal = matrix.matvec(&beta_star)?;
        (0..nrows).map(|i| signal[i]).collect()
    };
    for value in &mut y {
        *value += rng.random_range(-0.05..0.05);
    }

    let initial_residual = OffsetResidual::new(&y);
    let lambda_max = (0..ncols)
        .map(|j| {
            let column = matrix.sparse_column(j);
            initial_residual.column_dot(column, column.sum()).abs()
        })
        .fold(0.0, f64::max);
    let lambda = 0.1 * lambda_max;
    let tolerance = 1.0e-8;
    let result = coordinate_descent(&matrix, &y, lambda, tolerance, 10_000);
    assert!(
        result.kkt_violation <= tolerance,
        "coordinate descent did not satisfy the KKT tolerance"
    );

    let objective = 0.5 * result.residual.norm_squared()
        + lambda * result.beta.iter().map(|value| value.abs()).sum::<f64>();
    println!("λ = {lambda:.4}, converged in {} sweeps", result.sweeps);
    println!(
        "objective = {objective:.6}, max KKT violation = {:.3e}",
        result.kkt_violation
    );
    println!("\n   j   β*        β̂");
    for j in 0..ncols {
        println!("  {j:>2}  {:>8.4}  {:>8.4}", beta_star[j], result.beta[j]);
    }
    Ok(())
}
