//! Storage-generic primitives used by a future SLOPE solver.
//!
//! Run with: `cargo run --example slope_primitives --features faer`

#[path = "../tests/common/backend_aliases.rs"]
mod backend_aliases;
use backend_aliases::*;

use faer::Mat;
use faer::sparse::{SparseColMat, Triplet};
use lazymatrix::{ColumnNormalizationParams, Columns, LazyMatrix, LogicalColumn};

fn coordinate_derivatives<X: Columns<f64>>(
    matrix: &X,
    j: usize,
    residual: &[f64],
    weights: &[f64],
) -> (f64, f64) {
    let column = matrix.column(j);
    (
        column.weighted_dot(residual, weights),
        column.weighted_norm_squared(weights),
    )
}

fn add_active_column<X: Columns<f64>>(
    matrix: &X,
    j: usize,
    coefficient: f64,
    predictor: &mut [f64],
) {
    matrix.column(j).scaled_add_to(coefficient, predictor);
}

fn main() {
    let dense = Mat::from_fn(4, 3, |i, j| {
        let values = [
            [1.0, 0.0, 2.0],
            [0.0, -1.0, 0.0],
            [3.0, 0.0, 4.0],
            [0.0, 2.0, 0.0],
        ];
        values[i][j]
    });
    let triplets = [
        Triplet::new(0, 0, 1.0),
        Triplet::new(2, 0, 3.0),
        Triplet::new(1, 1, -1.0),
        Triplet::new(3, 1, 2.0),
        Triplet::new(0, 2, 2.0),
        Triplet::new(2, 2, 4.0),
    ];
    let sparse = SparseColMat::try_new_from_triplets(4, 3, &triplets).unwrap();
    let params = ColumnNormalizationParams::from_parts(
        3,
        Some(vec![1.0, 0.25, 1.5]),
        Some(vec![2.0, 1.5, 0.5]),
    );
    let dense = LazyMatrix::from_parts(dense, params.clone());
    let sparse = LazyMatrix::from_parts(sparse, params);
    let residual = [1.0, -0.5, 2.0, 0.25];
    let weights = [0.5, 1.0, 1.5, 2.0];

    for j in 0..dense.ncols() {
        let dense_derivatives = coordinate_derivatives(&dense, j, &residual, &weights);
        let sparse_derivatives = coordinate_derivatives(&sparse, j, &residual, &weights);
        approx::assert_abs_diff_eq!(dense_derivatives.0, sparse_derivatives.0, epsilon = 1e-12);
        approx::assert_abs_diff_eq!(dense_derivatives.1, sparse_derivatives.1, epsilon = 1e-12);
    }

    let mut dense_predictor = vec![0.0; dense.nrows()];
    let mut sparse_predictor = vec![0.0; sparse.nrows()];
    add_active_column(&dense, 2, -0.75, &mut dense_predictor);
    add_active_column(&sparse, 2, -0.75, &mut sparse_predictor);
    for (dense_value, sparse_value) in dense_predictor.iter().zip(&sparse_predictor) {
        approx::assert_abs_diff_eq!(dense_value, sparse_value, epsilon = 1e-12);
    }

    println!("dense and sparse SLOPE column primitives agree");
}
