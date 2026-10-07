//! Shared test helpers: a dense oracle and a backend-generic verification suite.
//!
//! The oracle materializes `X̃ = (X − 1cᵀ)S⁻¹` densely and runs naive
//! matrix–vector products; every backend's lazy operator is checked against it.
#![allow(dead_code)]

use lazymatrix::VectorView;
use lazymatrix::{
    Centering, ColumnStats, DotProduct, DotSlice, ElemDivAssign, L2Norm, LazyMatrix,
    LazySparseColumn, MatTransposeVec, MatTransposeVecInto, MatVec, MatVecInto, MatrixShape,
    Normalization, RawColumns, ScaleAssign, ScaledAddAssign, ScaledSubSlice, Scaling,
    SparseColumns, SparseRows, SubScalarAssign, SumEntries,
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::cell::Cell;

/// A randomly-generated sparse matrix in both dense and triplet form.
pub struct TestMatrix {
    pub nrows: usize,
    pub ncols: usize,
    pub dense: Vec<Vec<f64>>,               // n×p, row-major
    pub triplets: Vec<(usize, usize, f64)>, // stored nonzeros (row, col, val)
}

/// A backend-independent destination for coefficient-space products.
pub struct GramOutput(pub Vec<Vec<f64>>);

impl MatrixShape for GramOutput {
    fn nrows(&self) -> usize {
        self.0.len()
    }
    fn ncols(&self) -> usize {
        self.0.first().map_or(0, Vec::len)
    }
}

impl lazymatrix::MatrixWrite<f64> for GramOutput {
    fn set(&mut self, row: usize, column: usize, value: f64) {
        self.0[row][column] = value;
    }
}

pub fn assert_gram(actual: &GramOutput, normalized: &[Vec<f64>], weights: &[f64]) {
    for j in 0..actual.ncols() {
        for k in 0..actual.ncols() {
            let expected: f64 = normalized
                .iter()
                .zip(weights)
                .map(|(row, w)| (row[j] * w) * row[k])
                .sum();
            let value = actual.0[j][k];
            if expected.is_nan() {
                assert!(value.is_nan(), "({j}, {k}): {value} should be NaN");
            } else if expected.is_infinite() {
                assert_eq!(value, expected);
            } else {
                approx::assert_relative_eq!(value, expected, epsilon = 1e-10, max_relative = 1e-10);
            }
            if !value.is_nan() {
                assert_eq!(value, actual.0[k][j]);
            }
        }
    }
}

pub fn run_gram_suite<M>(build: impl Fn(&TestMatrix) -> M)
where
    M: lazymatrix::WeightedGramInto<f64>
        + lazymatrix::WeightedGramKernel<f64>
        + lazymatrix::WeightedColumnSumsInto<f64>
        + lazymatrix::WeightedColumnSumsKernel<f64>
        + ColumnStats<f64>,
{
    use lazymatrix::WeightedGramInto;
    run_intercept_gram_suite(&build);

    for (n, p, density) in [
        (19, 5, 0.2),
        (7, 3, 1.0),
        (0, 3, 0.0),
        (5, 0, 0.0),
        (5, 3, 0.0),
    ] {
        let tm = random_matrix(789, n, p, density);
        let matrix = build(&tm);
        let weights: Vec<_> = (0..n).map(|i| (i % 5) as f64 - 2.0).collect();
        let mut out = GramOutput(vec![vec![f64::NAN; p]; p]);
        matrix.weighted_gram_into(&weights, &mut out).unwrap();
        assert_gram(&out, &tm.dense, &weights);
        for center in [Centering::None, Centering::Mean, Centering::Min] {
            for scale in [
                Scaling::None,
                Scaling::Sd,
                Scaling::L1,
                Scaling::L2,
                Scaling::MaxAbs,
                Scaling::Range,
            ] {
                let lazy = LazyMatrix::new(&matrix, Normalization::new(center, scale)).unwrap();
                lazy.weighted_gram_into(&weights, &mut out).unwrap();
                assert_gram(
                    &out,
                    &materialize(&tm.dense, lazy.centers(), lazy.scales()),
                    &weights,
                );
            }
        }
        let lazy = LazyMatrix::from_parts(&matrix, Some(vec![1.25; p]), Some(vec![-2.0; p]));
        lazy.weighted_gram_into(&weights, &mut out).unwrap();
        assert_gram(
            &out,
            &materialize(&tm.dense, lazy.centers(), lazy.scales()),
            &weights,
        );
        for scale in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            let nonfinite = LazyMatrix::from_parts(&matrix, None, Some(vec![scale; p]));
            nonfinite.weighted_gram_into(&weights, &mut out).unwrap();
            assert_gram(
                &out,
                &materialize(&tm.dense, None, nonfinite.scales()),
                &weights,
            );
        }
        let bad_weights = vec![1.0; n + 1];
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                lazy.weighted_gram_into(&bad_weights, &mut out).unwrap();
            }))
            .is_err()
        );
        let mut wrong_out = GramOutput(vec![vec![42.0; p + 1]; p + 1]);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                lazy.weighted_gram_into(&weights, &mut wrong_out).unwrap();
            }))
            .is_err()
        );
        assert!(wrong_out.0.iter().flatten().all(|&v| v == 42.0));
        for (c, s) in [
            (Some(vec![0.0; p + 1]), None),
            (None, Some(vec![1.0; p + 1])),
            (None, Some(vec![-0.0; p])),
        ] {
            if p == 0 && s.as_ref().is_some_and(Vec::is_empty) {
                continue;
            }
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    matrix
                        .weighted_gram_normalized_into(
                            &weights,
                            c.as_deref(),
                            s.as_deref(),
                            &mut out,
                        )
                        .unwrap();
                }))
                .is_err()
            );
        }
    }

    let tm = random_matrix(518, 300, 37, 0.1);
    let tiled = LazyMatrix::from_parts(build(&tm), Some(vec![0.5; 37]), Some(vec![-2.0; 37]));
    let weights = random_vec(519, 300);
    let mut out = GramOutput(vec![vec![f64::NAN; 37]; 37]);
    tiled.weighted_gram_into(&weights, &mut out).unwrap();
    assert_gram(
        &out,
        &materialize(&tm.dense, tiled.centers(), tiled.scales()),
        &weights,
    );
    let mut nonfinite_weights = weights;
    nonfinite_weights[0] = f64::NAN;
    tiled
        .weighted_gram_into(&nonfinite_weights, &mut out)
        .unwrap();
    assert!(out.0.iter().flatten().all(|x| x.is_nan()));

    let dense: Vec<Vec<_>> = (0..35)
        .map(|i| {
            (0..17)
                .map(|j| 1e12 + (i % 5) as f64 - (j % 3) as f64)
                .collect()
        })
        .collect();
    let triplets = dense
        .iter()
        .enumerate()
        .flat_map(|(i, row)| row.iter().enumerate().map(move |(j, &v)| (i, j, v)))
        .collect();
    let tm = TestMatrix {
        nrows: 35,
        ncols: 17,
        dense,
        triplets,
    };
    let tiled = LazyMatrix::with_centers(build(&tm), vec![1e12; 17]);
    let mut out = GramOutput(vec![vec![f64::NAN; 17]; 17]);
    tiled.weighted_gram_into(&[1.0; 35], &mut out).unwrap();
    assert_gram(
        &out,
        &materialize(&tm.dense, tiled.centers(), None),
        &[1.0; 35],
    );

    let cases = [
        (vec![vec![1e12, -1e12]; 4], vec![1e12, -1e12], vec![1.0; 4]),
        (
            vec![
                vec![1e12 - 2.0, 1e12 + 1.0],
                vec![1e12 - 1.0, 1e12 - 2.0],
                vec![1e12 + 1.0, 1e12 + 2.0],
                vec![1e12 + 2.0, 1e12 - 1.0],
            ],
            vec![1e12; 2],
            vec![1.0; 4],
        ),
        (
            vec![vec![1e12, 1e12], vec![0.0, 0.0]],
            vec![1e12; 2],
            vec![1e16, 1.0],
        ),
        (
            vec![vec![0.0, 2.0], vec![3.0, 0.0]],
            vec![0.0; 2],
            vec![f64::INFINITY, 1.0],
        ),
        (
            vec![vec![f64::INFINITY, 0.0], vec![0.0, 1.0]],
            vec![0.0; 2],
            vec![0.0, 1.0],
        ),
        (
            vec![vec![0.0, 2.0], vec![3.0, 0.0]],
            vec![f64::NAN, f64::INFINITY],
            vec![1.0; 2],
        ),
    ];
    for (dense, centers, weights) in cases {
        let triplets = dense
            .iter()
            .enumerate()
            .flat_map(|(i, row)| {
                row.iter()
                    .enumerate()
                    .filter_map(move |(j, &v)| (v != 0.0).then_some((i, j, v)))
            })
            .collect();
        let tm = TestMatrix {
            nrows: dense.len(),
            ncols: 2,
            dense,
            triplets,
        };
        let lazy = LazyMatrix::from_parts(build(&tm), Some(centers), None);
        let mut out = GramOutput(vec![vec![f64::NAN; 2]; 2]);
        lazy.weighted_gram_into(&weights, &mut out).unwrap();
        assert_gram(
            &out,
            &materialize(&tm.dense, lazy.centers(), lazy.scales()),
            &weights,
        );
    }
}

/// Generate a reproducible sparse matrix with the given nonzero `density`.
pub fn random_matrix(seed: u64, nrows: usize, ncols: usize, density: f64) -> TestMatrix {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut dense = vec![vec![0.0; ncols]; nrows];
    let mut triplets = Vec::new();
    for (i, row) in dense.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            if rng.random::<f64>() < density {
                let v: f64 = rng.random_range(-2.0..2.0);
                if v != 0.0 {
                    *cell = v;
                    triplets.push((i, j, v));
                }
            }
        }
    }
    TestMatrix {
        nrows,
        ncols,
        dense,
        triplets,
    }
}

/// A reproducible random vector with entries in `[-1.5, 1.5)`.
pub fn random_vec(seed: u64, n: usize) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n).map(|_| rng.random_range(-1.5..1.5)).collect()
}

/// Materialize `X̃ = (X − 1cᵀ)S⁻¹` densely.
pub fn materialize(
    dense: &[Vec<f64>],
    centers: Option<&[f64]>,
    scales: Option<&[f64]>,
) -> Vec<Vec<f64>> {
    dense
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(j, &x)| {
                    let mut x = x;
                    if let Some(c) = centers {
                        x -= c[j];
                    }
                    if let Some(s) = scales {
                        x /= s[j];
                    }
                    x
                })
                .collect()
        })
        .collect()
}

pub fn dense_matvec(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter()
        .map(|row| row.iter().zip(v).map(|(a, b)| a * b).sum())
        .collect()
}

pub fn dense_tmatvec(m: &[Vec<f64>], u: &[f64]) -> Vec<f64> {
    let ncols = m.first().map_or(0, Vec::len);
    (0..ncols)
        .map(|j| m.iter().zip(u).map(|(row, &uu)| row[j] * uu).sum())
        .collect()
}

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn assert_close(a: &[f64], b: &[f64], eps: f64) {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    for (x, y) in a.iter().zip(b) {
        approx::assert_abs_diff_eq!(x, y, epsilon = eps);
    }
}

const EPS: f64 = 1e-10;

/// Exercise fused products independently of each backend's storage representation.
pub fn run_fused_suite<M, V>(build: impl Fn(&TestMatrix) -> M, to_v: impl Fn(&[f64]) -> V)
where
    M: lazymatrix::MatVecScaledInto<V, V, f64> + lazymatrix::MatTransposeVecScaledInto<V, V, f64>,
    V: lazymatrix::VectorOwned<f64, Owned = V> + lazymatrix::VectorViewMut<f64>,
{
    use lazymatrix::{MatTransposeVecScaledInto, MatVecScaledInto, WithIntercept};
    for (rows, cols, density) in [
        (7, 3, 0.4),
        (3, 7, 1.0),
        (0, 3, 0.0),
        (5, 0, 0.0),
        (0, 0, 0.0),
        (4, 3, 0.0),
    ] {
        let tm = random_matrix(219, rows, cols, density);
        let matrix = build(&tm);
        check_fused_operator(&matrix, &tm.dense, cols, &to_v);
        for centering in [false, true] {
            for scaling in [false, true] {
                let centers = centering.then(|| vec![0.75; cols]);
                let scales = scaling.then(|| {
                    (0..cols)
                        .map(|j| if j % 2 == 0 { 2.0 } else { -2.0 })
                        .collect()
                });
                let normalized = materialize(&tm.dense, centers.as_deref(), scales.as_deref());
                let lazy = LazyMatrix::from_parts(&matrix, centers, scales);
                check_fused_operator(&lazy, &normalized, cols, &to_v);
                let design = WithIntercept::new(&lazy);
                let augmented: Vec<Vec<f64>> = normalized
                    .iter()
                    .map(|row| std::iter::once(1.0).chain(row.iter().copied()).collect())
                    .collect();
                check_fused_operator(&design, &augmented, cols + 1, &to_v);

                let x = to_v(&vec![2.0; cols]);
                let u = to_v(&vec![1.0; rows]);
                let mut scratch = to_v(&vec![f64::NAN; cols]);
                let mut forward = to_v(&vec![3.0; rows]);
                let mut transpose = to_v(&vec![3.0; cols]);
                lazy.matvec_scaled_with_workspace(2.0, &x, -1.0, &mut forward, &mut scratch)
                    .unwrap();
                lazy.mat_transpose_vec_scaled_with_workspace(
                    2.0,
                    &u,
                    -1.0,
                    &mut transpose,
                    &mut scratch,
                )
                .unwrap();
                let mut expected = to_v(&vec![3.0; rows]);
                lazy.matvec_scaled_into(2.0, &x, -1.0, &mut expected)
                    .unwrap();
                assert_vector_values(&forward, &expected);
                let mut expected = to_v(&vec![3.0; cols]);
                lazy.mat_transpose_vec_scaled_into(2.0, &u, -1.0, &mut expected)
                    .unwrap();
                assert_vector_values(&transpose, &expected);

                let x = to_v(&vec![2.0; cols + 1]);
                let mut forward = to_v(&vec![3.0; rows]);
                let mut transpose = to_v(&vec![3.0; cols + 1]);
                design
                    .matvec_scaled_with_workspace(2.0, &x, -1.0, &mut forward, &mut scratch)
                    .unwrap();
                design
                    .mat_transpose_vec_scaled_with_workspace(
                        2.0,
                        &u,
                        -1.0,
                        &mut transpose,
                        &mut scratch,
                    )
                    .unwrap();
                let mut expected = to_v(&vec![3.0; rows]);
                design
                    .matvec_scaled_into(2.0, &x, -1.0, &mut expected)
                    .unwrap();
                assert_vector_values(&forward, &expected);
                let mut expected = to_v(&vec![3.0; cols + 1]);
                design
                    .mat_transpose_vec_scaled_into(2.0, &u, -1.0, &mut expected)
                    .unwrap();
                assert_vector_values(&transpose, &expected);

                let mut wrong_scratch = to_v(&vec![0.0; cols + 1]);
                assert!(
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        design
                            .matvec_scaled_with_workspace(
                                0.0,
                                &x,
                                0.0,
                                &mut forward,
                                &mut wrong_scratch,
                            )
                            .unwrap();
                    }))
                    .is_err()
                );
                assert!(
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        design
                            .mat_transpose_vec_scaled_with_workspace(
                                0.0,
                                &u,
                                0.0,
                                &mut transpose,
                                &mut wrong_scratch,
                            )
                            .unwrap();
                    }))
                    .is_err()
                );
            }
        }
    }
    let matrix = build(&TestMatrix {
        nrows: 1,
        ncols: 1,
        dense: vec![vec![2.0]],
        triplets: vec![(0, 0, 2.0)],
    });
    for (alpha, beta, expected) in [
        (f64::INFINITY, 0.0, f64::INFINITY),
        (f64::NAN, 0.0, f64::NAN),
        (1.0, f64::NAN, f64::NAN),
        (1.0, f64::INFINITY, f64::INFINITY),
    ] {
        let mut out = to_v(&[3.0]);
        matrix
            .matvec_scaled_into(alpha, &to_v(&[1.0]), beta, &mut out)
            .unwrap();
        assert_vector_values(&out, &to_v(&[expected]));
        matrix
            .mat_transpose_vec_scaled_into(alpha, &to_v(&[1.0]), beta, &mut out)
            .unwrap();
        assert_vector_values(&out, &to_v(&[expected]));
    }
}

/// Check the scalar contract on a two-row, one-column matrix containing [1, 3].
pub fn check_fused_f32<M, V>(matrix: &M, to_v: impl Fn(&[f32]) -> V)
where
    M: lazymatrix::MatVecScaledInto<V, V, f32> + lazymatrix::MatTransposeVecScaledInto<V, V, f32>,
    V: lazymatrix::VectorOwned<f32, Owned = V> + lazymatrix::VectorViewMut<f32>,
{
    use lazymatrix::WithIntercept;
    let mut out = to_v(&[3.0; 2]);
    matrix
        .matvec_scaled_into(2.0, &to_v(&[2.0]), -1.0, &mut out)
        .unwrap();
    assert_eq!([out.get(0), out.get(1)], [1.0, 9.0]);
    let mut transpose = to_v(&[3.0]);
    matrix
        .mat_transpose_vec_scaled_into(2.0, &to_v(&[1.0; 2]), -1.0, &mut transpose)
        .unwrap();
    assert_eq!(transpose.get(0), 5.0);
    let design = WithIntercept::new(LazyMatrix::from_parts(
        matrix,
        Some(vec![1.0_f32]),
        Some(vec![-2.0_f32]),
    ));
    out = to_v(&[3.0; 2]);
    let mut scratch = to_v(&[f32::NAN]);
    design
        .matvec_scaled_with_workspace(2.0, &to_v(&[3.0, 2.0]), -1.0, &mut out, &mut scratch)
        .unwrap();
    assert_eq!([out.get(0), out.get(1)], [3.0, -1.0]);
    let mut transpose = to_v(&[3.0; 2]);
    design
        .mat_transpose_vec_scaled_with_workspace(
            2.0,
            &to_v(&[1.0; 2]),
            -1.0,
            &mut transpose,
            &mut scratch,
        )
        .unwrap();
    assert_eq!([transpose.get(0), transpose.get(1)], [1.0, -5.0]);
}

fn assert_vector_values(a: &impl VectorView<f64>, b: &impl VectorView<f64>) {
    assert_eq!(a.len(), b.len());
    for j in 0..a.len() {
        let (actual, expected) = (a.get(j), b.get(j));
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else if expected.is_infinite() {
            assert_eq!(actual, expected);
        } else {
            approx::assert_abs_diff_eq!(actual, expected, epsilon = EPS);
        }
    }
}

pub fn check_fused_operator<O, V>(
    op: &O,
    dense: &[Vec<f64>],
    cols: usize,
    to_v: &impl Fn(&[f64]) -> V,
) where
    O: lazymatrix::MatVecScaledInto<V, V, f64> + lazymatrix::MatTransposeVecScaledInto<V, V, f64>,
    V: lazymatrix::VectorViewMut<f64>,
{
    let rows = op.nrows();
    let coefficients = random_vec(220, cols);
    let input = random_vec(221, rows);
    let forward = dense_matvec(dense, &coefficients);
    let transpose: Vec<f64> = (0..cols)
        .map(|j| (0..rows).map(|i| dense[i][j] * input[i]).sum())
        .collect();
    for alpha in [0.0, -0.0, 1.0, -1.0, 2.0] {
        for beta in [0.0, -0.0, 1.0, -1.0, 0.5] {
            let old = if beta == 0.0 { f64::NAN } else { 3.0 };
            let mut out = to_v(&vec![old; rows]);
            let mut trans = to_v(&vec![old; cols]);
            let x = to_v(&if alpha == 0.0 {
                vec![f64::NAN; cols]
            } else {
                coefficients.clone()
            });
            let u = to_v(&if alpha == 0.0 {
                vec![f64::NAN; rows]
            } else {
                input.clone()
            });
            op.matvec_scaled_into(alpha, &x, beta, &mut out).unwrap();
            op.mat_transpose_vec_scaled_into(alpha, &u, beta, &mut trans)
                .unwrap();
            let combine = |value| {
                if beta == 0.0 {
                    alpha * value
                } else {
                    alpha * value + beta * old
                }
            };
            assert_vector_values(
                &out,
                &to_v(&forward.iter().map(|&p| combine(p)).collect::<Vec<_>>()),
            );
            assert_vector_values(
                &trans,
                &to_v(&transpose.iter().map(|&p| combine(p)).collect::<Vec<_>>()),
            );
        }
    }
    for (x_len, out_len) in [(cols + 1, rows), (cols, rows + 1)] {
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                op.matvec_scaled_into(
                    0.0,
                    &to_v(&vec![0.0; x_len]),
                    0.0,
                    &mut to_v(&vec![0.0; out_len]),
                )
                .unwrap();
            }))
            .is_err()
        );
    }
    for (x_len, out_len) in [(rows + 1, cols), (rows, cols + 1)] {
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                op.mat_transpose_vec_scaled_into(
                    0.0,
                    &to_v(&vec![0.0; x_len]),
                    0.0,
                    &mut to_v(&vec![0.0; out_len]),
                )
                .unwrap();
            }))
            .is_err()
        );
    }
}

/// Run the full verification suite against a backend, given closures that build
/// the backend matrix `M` from a [`TestMatrix`] and convert between `Vec<f64>`
/// and the backend vector `V`.
pub fn run_backend_suite<M, V>(
    build: impl Fn(&TestMatrix) -> M,
    to_v: impl Fn(&[f64]) -> V,
    from_v: impl Fn(&V) -> Vec<f64>,
) where
    M: MatVec<V> + MatVecInto<V> + MatTransposeVec<V> + MatTransposeVecInto<V> + ColumnStats<f64>,
    V: Clone
        + lazymatrix::VectorOwned<f64, Owned = V>
        + lazymatrix::VectorViewMut<f64>
        + DotProduct<f64>
        + L2Norm<f64>
        + ScaledAddAssign<f64>
        + ScaleAssign<f64>
        + ElemDivAssign<f64>
        + DotSlice<f64>
        + SubScalarAssign<f64>
        + SumEntries<f64>
        + ScaledSubSlice<f64>,
{
    intercept_products(&build, &to_v, &from_v);
    vector_algebra(&to_v, &from_v);
    oracle_parity(&build, &to_v, &from_v);
    reusable_output_parity(&build, &to_v, &from_v);
    adjoint_identity(&build, &to_v, &from_v);
    from_parts_passthrough(&build, &to_v, &from_v);
    explicit_scales_reject_zero(&build);
    explicit_normalization_preserves_parameters(&build);
    new_matches_oracle(&build, &to_v, &from_v);
    column_stats_fixed(&build);
    column_stats_respect_sparse_storage(&build);
    normalization_statistic_dispatch(&build);
    column_sds_large_offset(&build);
    zero_scale_guard(&build, &to_v, &from_v);
    empty_and_nonfinite_stats(&build);
    shape_is_inferred(&build);
}

fn reusable_output_parity<M, V>(
    build: &impl Fn(&TestMatrix) -> M,
    to_v: &impl Fn(&[f64]) -> V,
    from_v: &impl Fn(&V) -> Vec<f64>,
) where
    M: MatVecInto<V> + MatTransposeVecInto<V>,
    V: Clone
        + ElemDivAssign<f64>
        + DotSlice<f64>
        + SubScalarAssign<f64>
        + SumEntries<f64>
        + ScaledSubSlice<f64>,
{
    let tm = random_matrix(101, 11, 7, 0.35);
    let centers = random_vec(102, tm.ncols);
    let scales: Vec<f64> = random_vec(103, tm.ncols)
        .iter()
        .map(|x| (x.abs() + 0.5).copysign(*x))
        .collect();
    let v = to_v(&random_vec(104, tm.ncols));
    let u = to_v(&random_vec(105, tm.nrows));

    for &use_c in &[false, true] {
        for &use_s in &[false, true] {
            let c = use_c.then(|| centers.clone());
            let s = use_s.then(|| scales.clone());
            let lazy = LazyMatrix::from_parts(build(&tm), c.clone(), s.clone());
            let xtilde = materialize(&tm.dense, c.as_deref(), s.as_deref());

            let mut y = to_v(&vec![123.0; tm.nrows]);
            lazy.matvec_into(&v, &mut y).unwrap();
            assert_close(&from_v(&y), &dense_matvec(&xtilde, &from_v(&v)), EPS);

            let mut z = to_v(&vec![123.0; tm.ncols]);
            lazy.mat_transpose_vec_into(&u, &mut z).unwrap();
            assert_close(&from_v(&z), &dense_tmatvec(&xtilde, &from_v(&u)), EPS);
        }
    }

    let matrix = build(&tm);
    let short_v = to_v(&vec![0.0; tm.ncols - 1]);
    let mut y = to_v(&vec![0.0; tm.nrows]);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            matrix.matvec_into(&short_v, &mut y).unwrap();
        }))
        .is_err()
    );

    let mut short_y = to_v(&vec![0.0; tm.nrows - 1]);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            matrix.matvec_into(&v, &mut short_y).unwrap();
        }))
        .is_err()
    );

    let short_u = to_v(&vec![0.0; tm.nrows - 1]);
    let mut z = to_v(&vec![0.0; tm.ncols]);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            matrix.mat_transpose_vec_into(&short_u, &mut z).unwrap();
        }))
        .is_err()
    );

    let mut short_z = to_v(&vec![0.0; tm.ncols - 1]);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            matrix.mat_transpose_vec_into(&u, &mut short_z).unwrap();
        }))
        .is_err()
    );

    let no_rows = TestMatrix {
        nrows: 0,
        ncols: 3,
        dense: Vec::new(),
        triplets: Vec::new(),
    };
    let no_rows = LazyMatrix::<_, f64>::from_parts(build(&no_rows), None, None);
    let mut empty = to_v(&[]);
    no_rows
        .matvec_into(&to_v(&[1.0, 2.0, 3.0]), &mut empty)
        .unwrap();
    assert!(from_v(&empty).is_empty());
    let mut zeros = to_v(&[123.0, 123.0, 123.0]);
    no_rows
        .mat_transpose_vec_into(&to_v(&[]), &mut zeros)
        .unwrap();
    assert_eq!(from_v(&zeros), vec![0.0; 3]);

    let no_columns = random_matrix(106, 3, 0, 0.5);
    let no_columns = LazyMatrix::<_, f64>::from_parts(build(&no_columns), None, None);
    let mut zeros = to_v(&[123.0, 123.0, 123.0]);
    no_columns.matvec_into(&to_v(&[]), &mut zeros).unwrap();
    assert_eq!(from_v(&zeros), vec![0.0; 3]);
    let mut empty = to_v(&[]);
    no_columns
        .mat_transpose_vec_into(&to_v(&[1.0, 2.0, 3.0]), &mut empty)
        .unwrap();
    assert!(from_v(&empty).is_empty());
}

pub fn vector_algebra<V>(to_v: &impl Fn(&[f64]) -> V, from_v: &impl Fn(&V) -> Vec<f64>)
where
    V: DotProduct<f64> + L2Norm<f64> + ScaledAddAssign<f64> + ScaleAssign<f64>,
{
    let a = to_v(&[1.0, -2.0, 3.0]);
    let b = to_v(&[4.0, 5.0, -6.0]);
    approx::assert_abs_diff_eq!(a.dot(&b), -24.0, epsilon = EPS);
    approx::assert_abs_diff_eq!(a.norm_l2(), 14.0_f64.sqrt(), epsilon = EPS);

    let mut updated = to_v(&[1.0, -2.0, 3.0]);
    updated.scaled_add_assign(-0.5, &b);
    assert_close(&from_v(&updated), &[-1.0, -4.5, 6.0], EPS);
    updated.scale_assign(-2.0);
    assert_close(&from_v(&updated), &[2.0, 9.0, -12.0], EPS);

    let empty = to_v(&[]);
    approx::assert_abs_diff_eq!(empty.dot(&empty), 0.0, epsilon = EPS);
    approx::assert_abs_diff_eq!(empty.norm_l2(), 0.0, epsilon = EPS);

    let short = to_v(&[1.0]);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| a.dot(&short))).is_err());
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut target = to_v(&[1.0, 2.0]);
            target.scaled_add_assign(1.0, &short);
        }))
        .is_err()
    );
}

/// Run the verification suite for backends with contiguous sparse columns.
///
/// This stays separate from [`run_backend_suite`] because future row-oriented
/// backends should still implement and test the orientation-agnostic traits.
pub fn run_sparse_columns_suite<M>(build: impl Fn(&TestMatrix) -> M)
where
    M: SparseColumns<f64>,
{
    sparse_columns_expose_raw_storage(&build);
    lazy_columns_match_dense_oracle(&build);
    lazy_column_operations_match_dense_oracle(&build);
    lazy_column_operations_handle_nonfinite_values(&build);
    empty_and_out_of_bounds_columns(&build);
    cached_sparse_products_touch_only_stored_rows(&build);
}

/// Run the verification suite for backends with contiguous sparse rows.
pub fn run_sparse_rows_suite<M: SparseRows<f64>>(build: impl Fn(&TestMatrix) -> M) {
    let explicit_zeros = TestMatrix {
        nrows: 4,
        ncols: 6,
        dense: vec![
            vec![0.0, 1.0, 0.0, 0.0, -2.0, 0.0],
            vec![0.0; 6],
            vec![3.0, 0.0, 4.0, 0.0, 0.0, 0.0],
            vec![0.0; 6],
        ],
        triplets: vec![
            (0, 1, 1.0),
            (0, 2, 0.0),
            (0, 4, -2.0),
            (2, 0, 3.0),
            (2, 2, 4.0),
            (3, 5, 0.0),
        ],
    };
    let matrix = build(&explicit_zeros);
    assert_eq!(
        matrix.sparse_row(0),
        (&[1, 2, 4][..], &[1.0, 0.0, -2.0][..])
    );
    assert_eq!(matrix.sparse_row(1), (&[][..], &[][..]));
    assert_eq!(matrix.sparse_row(3), (&[5][..], &[0.0][..]));

    for tm in [
        explicit_zeros,
        random_matrix(71, 8, 5, 0.4),
        random_matrix(72, 3, 9, 0.7),
        random_matrix(73, 0, 4, 0.0),
        random_matrix(74, 4, 0, 0.0),
        random_matrix(75, 0, 0, 0.0),
    ] {
        let matrix = build(&tm);
        assert_eq!(matrix.nrows(), tm.nrows);
        assert_eq!(matrix.ncols(), tm.ncols);
        for i in 0..tm.nrows {
            let (columns, values) = matrix.sparse_row(i);
            assert_eq!(columns.len(), values.len());
            let mut dense = vec![0.0; tm.ncols];
            for (&j, &value) in columns.iter().zip(values) {
                dense[j] += value;
            }
            assert_close(&dense, &tm.dense[i], EPS);
            let borrowed = &matrix;
            let (borrowed_columns, borrowed_values) =
                <&M as SparseRows<f64>>::sparse_row(&borrowed, i);
            assert_eq!(borrowed_columns.as_ptr(), columns.as_ptr());
            assert_eq!(borrowed_values.as_ptr(), values.as_ptr());
        }
        for i in [tm.nrows, usize::MAX] {
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| { matrix.sparse_row(i) }))
                    .is_err()
            );
        }
        lazy_rows_match_dense_oracle(&matrix, &tm);
    }
}

fn lazy_rows_match_dense_oracle<M: SparseRows<f64>>(matrix: &M, tm: &TestMatrix) {
    let centers: Vec<_> = (0..tm.ncols).map(|j| j as f64 * 0.5 - 1.25).collect();
    let scales: Vec<_> = (0..tm.ncols)
        .map(|j| if j % 2 == 0 { 0.5 } else { -2.0 })
        .collect();
    for use_center in [false, true] {
        for use_scale in [false, true] {
            let lazy = LazyMatrix::from_parts(
                matrix,
                use_center.then(|| centers.clone()),
                use_scale.then(|| scales.clone()),
            );
            let expected = materialize(&tm.dense, lazy.centers(), lazy.scales());
            for (i, expected_row) in expected.iter().enumerate() {
                let row = lazy.row(i);
                let (columns, values) = matrix.sparse_row(i);
                assert_eq!(row.len(), tm.ncols);
                assert_eq!(row.is_empty(), tm.ncols == 0);
                assert_eq!(row.column_indices().as_ptr(), columns.as_ptr());
                assert_eq!(row.values().as_ptr(), values.as_ptr());
                assert_eq!(
                    row.centers().map(<[f64]>::as_ptr),
                    lazy.centers().map(<[f64]>::as_ptr)
                );
                assert_eq!(
                    row.scales().map(<[f64]>::as_ptr),
                    lazy.scales().map(<[f64]>::as_ptr)
                );
                let mut raw = vec![0.0; row.len()];
                for (&j, &value) in row.column_indices().iter().zip(row.values()) {
                    raw[j] += value;
                }
                let direct: Vec<_> = raw
                    .iter()
                    .enumerate()
                    .map(|(j, &value)| (value - row.center(j)) / row.scale(j))
                    .collect();
                assert_close(&direct, expected_row, EPS);
                let mut affine: Vec<_> = (0..row.len()).map(|j| row.implicit_value(j)).collect();
                for (j, correction) in row.stored_corrections() {
                    affine[j] += correction;
                }
                assert_close(&affine, expected_row, EPS);
            }
            for i in [tm.nrows, usize::MAX] {
                assert!(
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| lazy.row(i))).is_err()
                );
            }
        }
    }
}

struct CountingView<'a> {
    values: &'a [f64],
    reads: Cell<usize>,
}

impl VectorView<f64> for CountingView<'_> {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn get(&self, index: usize) -> f64 {
        self.reads.set(self.reads.get() + 1);
        self.values[index]
    }
}

fn cached_sparse_products_touch_only_stored_rows<M>(build: &impl Fn(&TestMatrix) -> M)
where
    M: SparseColumns<f64>,
{
    let tm = column_view_matrix();
    let lazy = LazyMatrix::<_, f64>::from_parts(build(&tm), Some(vec![0.5; 4]), None);
    let column = lazy.sparse_column(0);
    let vector = CountingView {
        values: &[1.0, 2.0, 3.0, 4.0],
        reads: Cell::new(0),
    };
    column.dot_with_sum(&vector, 10.0);
    assert_eq!(vector.reads.get(), column.values().len());

    let weights = CountingView {
        values: &[0.5, 1.0, 1.5, 2.0],
        reads: Cell::new(0),
    };
    vector.reads.set(0);
    column.weighted_dot_with_sum(&vector, &weights, 15.0);
    assert_eq!(vector.reads.get(), column.values().len());
    assert_eq!(weights.reads.get(), column.values().len());
}

/// Run logical-column checks shared by dense and sparse storage backends.
pub fn run_logical_columns_suite<M>(build: impl Fn(&TestMatrix) -> M)
where
    M: RawColumns<f64> + ColumnStats<f64>,
{
    weighted_norms_preserve_implicit_contributions(&build);
    weighted_norms_ignore_rounded_total_weights(&build);
    weighted_norms_preserve_nonfinite_arithmetic(&build);
    weighted_norms_normalize_before_squaring(&build);
    let tm = column_view_matrix();
    let centers = vec![0.5, -1.0, 2.0, 3.0];
    let scales = vec![2.0, -4.0, 0.5, -1.5];
    let vector = vec![1.5, -2.0, 0.25, 3.0];
    let weights = vec![0.5, 2.0, 1.25, 3.0];

    for &use_center in &[false, true] {
        for &use_scale in &[false, true] {
            let active_centers = use_center.then(|| centers.clone());
            let active_scales = use_scale.then(|| scales.clone());
            let lazy =
                LazyMatrix::from_parts(build(&tm), active_centers.clone(), active_scales.clone());
            let dense = materialize(
                &tm.dense,
                active_centers.as_deref(),
                active_scales.as_deref(),
            );

            for j in 0..tm.ncols {
                let column = lazy.column(j);
                let expected: Vec<f64> = dense.iter().map(|row| row[j]).collect();
                let expected_dot = dot(&expected, &vector);
                let expected_weighted_dot = expected
                    .iter()
                    .zip(&vector)
                    .zip(&weights)
                    .map(|((&x, &v), &w)| x * v * w)
                    .sum::<f64>();
                let expected_weighted_norm = expected
                    .iter()
                    .zip(&weights)
                    .map(|(&x, &w)| w * x * x)
                    .sum::<f64>();

                approx::assert_abs_diff_eq!(column.sum(), expected.iter().sum(), epsilon = EPS);
                approx::assert_abs_diff_eq!(
                    column.norm_squared(),
                    expected.iter().map(|x| x * x).sum::<f64>(),
                    epsilon = EPS
                );
                approx::assert_abs_diff_eq!(column.dot(&vector), expected_dot, epsilon = EPS);
                approx::assert_abs_diff_eq!(
                    column.dot_with_sum(&vector, vector.iter().sum()),
                    expected_dot,
                    epsilon = EPS
                );
                approx::assert_abs_diff_eq!(
                    column.weighted_dot(&vector, &weights),
                    expected_weighted_dot,
                    epsilon = EPS
                );
                approx::assert_abs_diff_eq!(
                    column.weighted_dot_with_sum(
                        &vector,
                        &weights,
                        vector.iter().zip(&weights).map(|(v, w)| v * w).sum(),
                    ),
                    expected_weighted_dot,
                    epsilon = EPS
                );
                approx::assert_abs_diff_eq!(
                    column.weighted_norm_squared(&weights),
                    expected_weighted_norm,
                    epsilon = EPS
                );

                let mut destination = vec![1.0; tm.nrows];
                column.scaled_add_to(-0.75, &mut destination);
                let expected_destination: Vec<_> =
                    expected.iter().map(|&x| 1.0 - 0.75 * x).collect();
                assert_close(&destination, &expected_destination, EPS);
            }
        }
    }

    let matrix = build(&tm);
    let borrowed =
        LazyMatrix::new(&matrix, Normalization::new(Centering::Mean, Scaling::Sd)).unwrap();
    assert_eq!(borrowed.nrows(), tm.nrows);
    assert_eq!(borrowed.ncols(), tm.ncols);
    assert_eq!(borrowed.column(0).len(), tm.nrows);

    let lazy = LazyMatrix::<_, f64>::from_parts(build(&tm), None, None);
    let short = vec![1.0; tm.nrows - 1];
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            lazy.column(0).dot(&short)
        }))
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut destination = short.clone();
            lazy.column(0).scaled_add_to(1.0, &mut destination);
        }))
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| lazy.column(tm.ncols))).is_err()
    );

    let empty = TestMatrix {
        nrows: 0,
        ncols: 2,
        dense: Vec::new(),
        triplets: Vec::new(),
    };
    let empty = LazyMatrix::<_, f64>::from_parts(build(&empty), None, None);
    let column = empty.column(0);
    assert!(column.is_empty());
    assert_eq!(column.sum(), 0.0);
    assert_eq!(column.norm_squared(), 0.0);
    assert_eq!(column.dot(&[]), 0.0);

    let nonfinite = TestMatrix {
        nrows: 3,
        ncols: 1,
        dense: vec![vec![f64::NAN], vec![0.0], vec![f64::INFINITY]],
        triplets: vec![(0, 0, f64::NAN), (2, 0, f64::INFINITY)],
    };
    let nonfinite = LazyMatrix::<_, f64>::from_parts(build(&nonfinite), None, None);
    let column = nonfinite.column(0);
    assert!(column.sum().is_nan());
    assert!(column.dot(&[1.0, 2.0, 3.0]).is_nan());
    assert!(column.weighted_norm_squared(&[0.5, 1.0, 1.5]).is_nan());
}

fn weighted_norms_preserve_implicit_contributions<M: RawColumns<f64>>(
    build: &impl Fn(&TestMatrix) -> M,
) {
    for implicit_row in 0..3 {
        for explicit_zero in [false, true] {
            let dense: Vec<_> = (0..3)
                .map(|i| vec![if i == implicit_row { 0.0 } else { 1.0 }])
                .collect();
            let tm = TestMatrix {
                nrows: 3,
                ncols: 1,
                triplets: (0..3)
                    .filter(|&i| i != implicit_row || explicit_zero)
                    .map(|i| (i, 0, dense[i][0]))
                    .collect(),
                dense,
            };
            for scale in [1.0, -2.0] {
                let lazy = LazyMatrix::from_parts(build(&tm), Some(vec![1.0]), Some(vec![scale]));
                for small_weight in [1.0, -1.0] {
                    let mut weights = [1e16; 3];
                    weights[implicit_row] = small_weight;
                    let column = lazy.column(0);
                    let expected = small_weight / (scale * scale);
                    approx::assert_abs_diff_eq!(
                        column.weighted_norm_squared(&weights),
                        expected,
                        epsilon = EPS
                    );
                    approx::assert_abs_diff_eq!(
                        column.weighted_norm_squared_with_sum(&weights, weights.iter().sum()),
                        expected,
                        epsilon = EPS
                    );
                }
            }
        }
    }
}

fn weighted_norms_ignore_rounded_total_weights<M: RawColumns<f64>>(
    build: &impl Fn(&TestMatrix) -> M,
) {
    let dense: Vec<_> = (0..8).map(|i| vec![1e12 + (i % 3) as f64]).collect();
    let tm = TestMatrix {
        nrows: 8,
        ncols: 1,
        triplets: (0..8).map(|i| (i, 0, dense[i][0])).collect(),
        dense,
    };
    let center = 1e12 + 1.0;
    let lazy = LazyMatrix::from_parts(build(&tm), Some(vec![center]), None);
    let weights: Vec<_> = (1..=8).map(|i| 0.1 * i as f64).collect();
    let expected: f64 = (0..8)
        .map(|i| weights[i] * (tm.dense[i][0] - center).powi(2))
        .sum();
    let column = lazy.column(0);
    approx::assert_abs_diff_eq!(
        column.weighted_norm_squared(&weights),
        expected,
        epsilon = EPS
    );
    for total in [weights.iter().sum(), weights.iter().rev().sum()] {
        approx::assert_abs_diff_eq!(
            column.weighted_norm_squared_with_sum(&weights, total),
            expected,
            epsilon = EPS
        );
    }
}

fn weighted_norms_preserve_nonfinite_arithmetic<M: RawColumns<f64>>(
    build: &impl Fn(&TestMatrix) -> M,
) {
    let empty = TestMatrix {
        nrows: 0,
        ncols: 1,
        dense: Vec::new(),
        triplets: Vec::new(),
    };
    let empty = LazyMatrix::from_parts(
        build(&empty),
        Some(vec![f64::NAN]),
        Some(vec![f64::INFINITY]),
    );
    assert_eq!(empty.column(0).weighted_norm_squared(&[]), 0.0);
    assert_eq!(
        empty.column(0).weighted_norm_squared_with_sum(&[], 0.0),
        0.0
    );
    let tm = TestMatrix {
        nrows: 3,
        ncols: 1,
        dense: vec![vec![1.0], vec![0.0], vec![2.0]],
        triplets: vec![(0, 0, 1.0), (2, 0, 2.0)],
    };
    for center in [0.0, 1.0, f64::INFINITY, f64::NAN] {
        for scale in [1.0, -2.0, f64::INFINITY, f64::NAN] {
            let lazy = LazyMatrix::from_parts(build(&tm), Some(vec![center]), Some(vec![scale]));
            for weights in [
                [1.0, 1.0, 1.0],
                [f64::INFINITY, 1.0, 1.0],
                [1.0, f64::INFINITY, 1.0],
                [1.0, f64::NAN, 1.0],
                [1.0, f64::NEG_INFINITY, 1.0],
                [0.0, 0.0, 0.0],
            ] {
                let expected: f64 = (0..3)
                    .map(|i| {
                        let value = (tm.dense[i][0] - center) / scale;
                        weights[i] * value * value
                    })
                    .sum();
                let column = lazy.column(0);
                for actual in [
                    column.weighted_norm_squared(&weights),
                    column.weighted_norm_squared_with_sum(&weights, weights.iter().sum()),
                ] {
                    if expected.is_nan() {
                        assert!(actual.is_nan(), "expected NaN, got {actual}");
                    } else if expected.is_infinite() {
                        assert_eq!(actual, expected);
                    } else {
                        approx::assert_abs_diff_eq!(actual, expected, epsilon = EPS);
                    }
                }
            }
        }
    }
}

fn weighted_norms_normalize_before_squaring<M: RawColumns<f64>>(build: &impl Fn(&TestMatrix) -> M) {
    for scale in [1e200, -1e200, 1e-200, -1e-200] {
        let tm = TestMatrix {
            nrows: 3,
            ncols: 1,
            dense: vec![vec![scale], vec![0.0], vec![2.0 * scale]],
            triplets: vec![(0, 0, scale), (2, 0, 2.0 * scale)],
        };
        let lazy = LazyMatrix::from_parts(build(&tm), None, Some(vec![scale]));
        let column = lazy.column(0);
        let weights = [1.0, -2.0, 3.0];
        approx::assert_abs_diff_eq!(column.weighted_norm_squared(&weights), 13.0);
        approx::assert_abs_diff_eq!(
            column.weighted_norm_squared_with_sum(&weights, weights.iter().sum()),
            13.0
        );
    }
}

fn sparse_columns_expose_raw_storage<M>(build: &impl Fn(&TestMatrix) -> M)
where
    M: SparseColumns<f64>,
{
    let tm = column_view_matrix();
    let matrix = build(&tm);

    let (rows, values) = matrix.sparse_column(0);
    assert_eq!(rows, &[0, 2, 3]);
    assert_eq!(values, &[1.0, 0.0, -2.0]);

    let (rows, values) = matrix.sparse_column(1);
    assert!(rows.is_empty());
    assert!(values.is_empty());

    let (rows, values) = matrix.sparse_column(2);
    assert_eq!(rows, &[0, 1, 2, 3]);
    assert_eq!(values, &[4.0, 5.0, 6.0, 7.0]);

    let (rows, values) = matrix.sparse_column(3);
    assert_eq!(rows, &[1]);
    assert_eq!(values, &[0.0]);

    let lazy = LazyMatrix::from_parts(matrix, Some(vec![0.5; 4]), Some(vec![2.0; 4]));
    let column = lazy.sparse_column(0);
    assert_eq!(column.implicit_value(), -0.25);
    assert_eq!(column.raw_sum(), -1.0);
    assert_eq!(
        column.stored_corrections().collect::<Vec<_>>(),
        vec![(0, 0.5), (2, 0.0), (3, -1.0)]
    );
}

fn lazy_columns_match_dense_oracle<M>(build: &impl Fn(&TestMatrix) -> M)
where
    M: SparseColumns<f64>,
{
    let tm = column_view_matrix();
    let centers = vec![0.5, -1.0, 2.0, 3.0];
    let scales = vec![2.0, -4.0, 0.5, -1.5];

    for &use_center in &[false, true] {
        for &use_scale in &[false, true] {
            let active_centers = use_center.then(|| centers.clone());
            let active_scales = use_scale.then(|| scales.clone());
            let lazy =
                LazyMatrix::from_parts(build(&tm), active_centers.clone(), active_scales.clone());
            let dense = materialize(
                &tm.dense,
                active_centers.as_deref(),
                active_scales.as_deref(),
            );

            for j in 0..tm.ncols {
                let column = lazy.sparse_column(j);
                assert_eq!(column.len(), tm.nrows);
                assert!(!column.is_empty());
                assert_eq!(
                    column.center(),
                    active_centers.as_ref().map_or(0.0, |c| c[j])
                );
                assert_eq!(column.scale(), active_scales.as_ref().map_or(1.0, |s| s[j]));

                let expected: Vec<f64> = dense.iter().map(|row| row[j]).collect();
                assert_close(&reconstruct_column(column), &expected, EPS);
            }
        }
    }
}

fn lazy_column_operations_match_dense_oracle<M>(build: &impl Fn(&TestMatrix) -> M)
where
    M: SparseColumns<f64>,
{
    let tm = column_view_matrix();
    let centers = vec![0.5, -1.0, 2.0, 3.0];
    let scales = vec![2.0, -4.0, 0.5, -1.5];
    let vector = vec![1.5, -2.0, 0.25, 3.0];
    let weights = vec![0.5, 2.0, 1.25, 3.0];
    let vector_sum = vector.iter().sum();
    let weighted_vector_sum = vector
        .iter()
        .zip(&weights)
        .map(|(&value, &weight)| value * weight)
        .sum();
    let weight_sum = weights.iter().sum();

    for &use_center in &[false, true] {
        for &use_scale in &[false, true] {
            let active_centers = use_center.then(|| centers.clone());
            let active_scales = use_scale.then(|| scales.clone());
            let lazy =
                LazyMatrix::from_parts(build(&tm), active_centers.clone(), active_scales.clone());
            let dense = materialize(
                &tm.dense,
                active_centers.as_deref(),
                active_scales.as_deref(),
            );

            for j in 0..tm.ncols {
                let column = lazy.sparse_column(j);
                let expected: Vec<f64> = dense.iter().map(|row| row[j]).collect();
                let expected_sum: f64 = expected.iter().sum();
                let expected_norm_squared: f64 = expected.iter().map(|value| value * value).sum();
                let expected_dot = dot(&expected, &vector);
                let expected_weighted_dot: f64 = expected
                    .iter()
                    .zip(&vector)
                    .zip(&weights)
                    .map(|((&column_value, &vector_value), &weight)| {
                        column_value * vector_value * weight
                    })
                    .sum();
                let expected_weighted_norm_squared: f64 = expected
                    .iter()
                    .zip(&weights)
                    .map(|(&column_value, &weight)| weight * column_value * column_value)
                    .sum();

                approx::assert_abs_diff_eq!(column.sum(), expected_sum, epsilon = EPS);
                approx::assert_abs_diff_eq!(
                    column.norm_squared(),
                    expected_norm_squared,
                    epsilon = EPS
                );
                approx::assert_abs_diff_eq!(column.dot(&vector), expected_dot, epsilon = EPS);
                approx::assert_abs_diff_eq!(
                    column.dot_with_sum(&vector, vector_sum),
                    expected_dot,
                    epsilon = EPS
                );
                approx::assert_abs_diff_eq!(
                    column.weighted_dot(&vector, &weights),
                    expected_weighted_dot,
                    epsilon = EPS
                );
                approx::assert_abs_diff_eq!(
                    column.weighted_dot_with_sum(&vector, &weights, weighted_vector_sum),
                    expected_weighted_dot,
                    epsilon = EPS
                );
                approx::assert_abs_diff_eq!(
                    column.weighted_norm_squared(&weights),
                    expected_weighted_norm_squared,
                    epsilon = EPS
                );
                approx::assert_abs_diff_eq!(
                    column.weighted_norm_squared_with_sum(&weights, weight_sum),
                    expected_weighted_norm_squared,
                    epsilon = EPS
                );
            }
        }
    }

    let lazy = LazyMatrix::<_, f64>::from_parts(build(&tm), None, None);
    let column = lazy.sparse_column(0);
    let short = &vector[..vector.len() - 1];
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| column.dot(short))).is_err());
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            column.dot_with_sum(short, short.iter().sum())
        }))
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            column.weighted_dot(short, &weights)
        }))
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            column.weighted_dot(&vector, &weights[..weights.len() - 1])
        }))
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            column.weighted_dot_with_sum(short, &weights, 0.0)
        }))
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            column.weighted_norm_squared(&weights[..weights.len() - 1])
        }))
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            column.weighted_norm_squared_with_sum(&weights[..weights.len() - 1], 0.0)
        }))
        .is_err()
    );
}

fn lazy_column_operations_handle_nonfinite_values<M>(build: &impl Fn(&TestMatrix) -> M)
where
    M: SparseColumns<f64>,
{
    let tm = TestMatrix {
        nrows: 3,
        ncols: 1,
        dense: vec![vec![f64::NAN], vec![0.0], vec![f64::INFINITY]],
        triplets: vec![(0, 0, f64::NAN), (2, 0, f64::INFINITY)],
    };
    let lazy = LazyMatrix::<_, f64>::from_parts(build(&tm), None, None);
    let column = lazy.sparse_column(0);
    let vector = [1.0, 2.0, 3.0];
    let weights = [0.5, 1.5, 2.0];
    let weighted_vector_sum = vector
        .iter()
        .zip(&weights)
        .map(|(&value, &weight)| value * weight)
        .sum();

    assert!(column.sum().is_nan());
    assert!(column.norm_squared().is_nan());
    assert!(column.dot(&vector).is_nan());
    assert!(column.dot_with_sum(&vector, vector.iter().sum()).is_nan());
    assert!(column.weighted_dot(&vector, &weights).is_nan());
    assert!(
        column
            .weighted_dot_with_sum(&vector, &weights, weighted_vector_sum)
            .is_nan()
    );
    assert!(column.weighted_norm_squared(&weights).is_nan());
    assert!(
        column
            .weighted_norm_squared_with_sum(&weights, weights.iter().sum())
            .is_nan()
    );
}

fn empty_and_out_of_bounds_columns<M>(build: &impl Fn(&TestMatrix) -> M)
where
    M: SparseColumns<f64>,
{
    let no_rows = TestMatrix {
        nrows: 0,
        ncols: 2,
        dense: Vec::new(),
        triplets: Vec::new(),
    };
    let lazy = LazyMatrix::<_, f64>::from_parts(build(&no_rows), None, None);
    let column = lazy.sparse_column(0);
    assert_eq!(column.len(), 0);
    assert!(column.is_empty());
    assert!(column.row_indices().is_empty());
    assert!(column.values().is_empty());
    assert_eq!(column.center(), 0.0);
    assert_eq!(column.scale(), 1.0);
    assert_eq!(column.sum(), 0.0);
    assert_eq!(column.norm_squared(), 0.0);
    assert_eq!(column.dot(&[]), 0.0);
    assert_eq!(column.dot_with_sum(&[], 0.0), 0.0);
    assert_eq!(column.weighted_dot(&[], &[]), 0.0);
    assert_eq!(column.weighted_dot_with_sum(&[], &[], 0.0), 0.0);
    assert_eq!(column.weighted_norm_squared(&[]), 0.0);
    assert_eq!(column.weighted_norm_squared_with_sum(&[], 0.0), 0.0);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| lazy.sparse_column(2)));
    assert!(result.is_err());
}

fn reconstruct_column(column: LazySparseColumn<'_, f64>) -> Vec<f64> {
    let mut dense = vec![-column.center() / column.scale(); column.len()];
    for (&row, &value) in column.row_indices().iter().zip(column.values()) {
        dense[row] = (value - column.center()) / column.scale();
    }
    dense
}

fn column_view_matrix() -> TestMatrix {
    TestMatrix {
        nrows: 4,
        ncols: 4,
        dense: vec![
            vec![1.0, 0.0, 4.0, 0.0],
            vec![0.0, 0.0, 5.0, 0.0],
            vec![0.0, 0.0, 6.0, 0.0],
            vec![-2.0, 0.0, 7.0, 0.0],
        ],
        triplets: vec![
            (0, 0, 1.0),
            (2, 0, 0.0),
            (3, 0, -2.0),
            (0, 2, 4.0),
            (1, 2, 5.0),
            (2, 2, 6.0),
            (3, 2, 7.0),
            (1, 3, 0.0),
        ],
    }
}

/// Empty columns use IEEE results where a statistic is undefined, while
/// nonfinite stored values are never hidden by an aggregation.
fn empty_and_nonfinite_stats<M>(build: &impl Fn(&TestMatrix) -> M)
where
    M: ColumnStats<f64> + MatrixShape,
{
    let no_rows = TestMatrix {
        nrows: 0,
        ncols: 2,
        dense: Vec::new(),
        triplets: Vec::new(),
    };
    let matrix = build(&no_rows);
    assert!(
        matrix
            .col_means()
            .unwrap()
            .iter()
            .all(|value| value.is_nan())
    );
    assert!(matrix.col_sds().unwrap().iter().all(|value| value.is_nan()));
    assert!(
        matrix
            .col_mins()
            .unwrap()
            .iter()
            .all(|value| value.is_nan())
    );
    assert!(
        matrix
            .col_ranges()
            .unwrap()
            .iter()
            .all(|value| value.is_nan())
    );
    assert_eq!(matrix.col_maxabs().unwrap(), vec![0.0, 0.0]);
    assert_eq!(matrix.col_l1().unwrap(), vec![0.0, 0.0]);
    assert_eq!(matrix.col_l2().unwrap(), vec![0.0, 0.0]);

    let lazy = LazyMatrix::new(matrix, Normalization::new(Centering::Mean, Scaling::Sd)).unwrap();
    let (data, centers, scales) = lazy.into_parts();
    let lazy = LazyMatrix::from_parts(data, centers, scales);
    assert!(lazy.centers().unwrap().iter().all(|value| value.is_nan()));
    assert!(lazy.scales().unwrap().iter().all(|value| value.is_nan()));

    let nan_column = TestMatrix {
        nrows: 2,
        ncols: 1,
        dense: vec![vec![f64::NAN], vec![1.0]],
        triplets: vec![(0, 0, f64::NAN), (1, 0, 1.0)],
    };
    let matrix = build(&nan_column);
    assert!(matrix.col_means().unwrap()[0].is_nan());
    assert!(matrix.col_sds().unwrap()[0].is_nan());
    assert!(matrix.col_mins().unwrap()[0].is_nan());
    assert!(matrix.col_ranges().unwrap()[0].is_nan());
    assert!(matrix.col_maxabs().unwrap()[0].is_nan());
    assert!(matrix.col_l1().unwrap()[0].is_nan());
    assert!(matrix.col_l2().unwrap()[0].is_nan());
    assert!(matrix.col_l1_centered(&[0.0]).unwrap()[0].is_nan());
    assert!(matrix.col_l2_centered(&[0.0]).unwrap()[0].is_nan());
    assert!(matrix.col_maxabs_centered(&[0.0]).unwrap()[0].is_nan());

    let implicit_zero = TestMatrix {
        nrows: 2,
        ncols: 1,
        dense: vec![vec![1.0], vec![0.0]],
        triplets: vec![(0, 0, 1.0)],
    };
    assert!(
        build(&implicit_zero)
            .col_maxabs_centered(&[f64::NAN])
            .unwrap()[0]
            .is_nan()
    );
}

/// Standard deviations retain small variation around a large offset.
fn column_sds_large_offset<M>(build: &impl Fn(&TestMatrix) -> M)
where
    M: ColumnStats<f64>,
{
    let offset = 1.0e12;
    let tm = TestMatrix {
        nrows: 3,
        ncols: 1,
        dense: vec![vec![offset + 1.0], vec![offset + 2.0], vec![offset + 3.0]],
        triplets: vec![
            (0, 0, offset + 1.0),
            (1, 0, offset + 2.0),
            (2, 0, offset + 3.0),
        ],
    };

    assert_close(
        &build(&tm).col_sds().unwrap(),
        &[(2.0_f64 / 3.0).sqrt()],
        EPS,
    );
}

/// The wrapper obtains its dimensions from the backend matrix.
fn shape_is_inferred<M>(build: &impl Fn(&TestMatrix) -> M)
where
    M: MatrixShape,
{
    let tm = random_matrix(21, 7, 4, 0.3);
    let lazy = LazyMatrix::<_, f64>::from_parts(build(&tm), None, None);
    assert_eq!(lazy.nrows(), tm.nrows);
    assert_eq!(lazy.ncols(), tm.ncols);

    let no_columns = random_matrix(22, 3, 0, 0.3);
    let lazy = LazyMatrix::<_, f64>::from_parts(build(&no_columns), None, None);
    assert_eq!(lazy.nrows(), no_columns.nrows);
    assert_eq!(lazy.ncols(), 0);
}

/// (1) + (3): all four center×scale combinations match the dense oracle, for
/// both `matvec` and `mat_transpose_vec`.
fn oracle_parity<M, V>(
    build: &impl Fn(&TestMatrix) -> M,
    to_v: &impl Fn(&[f64]) -> V,
    from_v: &impl Fn(&V) -> Vec<f64>,
) where
    M: MatVec<V> + MatTransposeVec<V>,
    V: Clone
        + ElemDivAssign<f64>
        + DotSlice<f64>
        + SubScalarAssign<f64>
        + SumEntries<f64>
        + ScaledSubSlice<f64>,
{
    let tm = random_matrix(1, 11, 7, 0.35);
    let centers = random_vec(2, tm.ncols);
    let scales: Vec<f64> = random_vec(3, tm.ncols)
        .iter()
        .map(|x| (x.abs() + 0.5).copysign(*x))
        .collect();
    let v = to_v(&random_vec(4, tm.ncols));
    let u = to_v(&random_vec(5, tm.nrows));

    for &use_c in &[false, true] {
        for &use_s in &[false, true] {
            let c = use_c.then(|| centers.clone());
            let s = use_s.then(|| scales.clone());
            let lazy = LazyMatrix::from_parts(build(&tm), c.clone(), s.clone());
            let xtilde = materialize(&tm.dense, c.as_deref(), s.as_deref());

            let got = from_v(&lazy.matvec(&v).unwrap());
            let want = dense_matvec(&xtilde, &from_v(&v));
            assert_close(&got, &want, EPS);

            let got_t = from_v(&lazy.mat_transpose_vec(&u).unwrap());
            let want_t = dense_tmatvec(&xtilde, &from_v(&u));
            assert_close(&got_t, &want_t, EPS);
        }
    }
}

/// (2): the adjoint identity ⟨X̃v, u⟩ == ⟨v, X̃ᵀu⟩, independent of the oracle.
fn adjoint_identity<M, V>(
    build: &impl Fn(&TestMatrix) -> M,
    to_v: &impl Fn(&[f64]) -> V,
    from_v: &impl Fn(&V) -> Vec<f64>,
) where
    M: MatVec<V> + MatTransposeVec<V>,
    V: Clone
        + ElemDivAssign<f64>
        + DotSlice<f64>
        + SubScalarAssign<f64>
        + SumEntries<f64>
        + ScaledSubSlice<f64>,
{
    let tm = random_matrix(6, 9, 5, 0.5);
    let centers = random_vec(7, tm.ncols);
    let scales: Vec<f64> = random_vec(8, tm.ncols)
        .iter()
        .map(|x| (x.abs() + 0.3).copysign(*x))
        .collect();
    let lazy = LazyMatrix::from_parts(build(&tm), Some(centers), Some(scales));
    let v = to_v(&random_vec(9, tm.ncols));
    let u = to_v(&random_vec(10, tm.nrows));

    let xv = from_v(&lazy.matvec(&v).unwrap());
    let xtu = from_v(&lazy.mat_transpose_vec(&u).unwrap());
    let lhs = dot(&xv, &from_v(&u));
    let rhs = dot(&from_v(&v), &xtu);
    approx::assert_abs_diff_eq!(lhs, rhs, epsilon = 1e-9);
}

/// (3): `from_parts(_, None, None)` is a bit-exact backend pass-through.
fn from_parts_passthrough<M, V>(
    build: &impl Fn(&TestMatrix) -> M,
    to_v: &impl Fn(&[f64]) -> V,
    from_v: &impl Fn(&V) -> Vec<f64>,
) where
    M: MatVec<V> + MatTransposeVec<V>,
    V: Clone
        + ElemDivAssign<f64>
        + DotSlice<f64>
        + SubScalarAssign<f64>
        + SumEntries<f64>
        + ScaledSubSlice<f64>,
{
    let tm = random_matrix(11, 8, 6, 0.4);
    let v = to_v(&random_vec(12, tm.ncols));
    let u = to_v(&random_vec(13, tm.nrows));

    let bare = build(&tm);
    let bare_y = from_v(&bare.matvec(&v).unwrap());
    let bare_t = from_v(&bare.mat_transpose_vec(&u).unwrap());

    let lazy = LazyMatrix::from_parts(build(&tm), None, None);
    assert_eq!(from_v(&lazy.matvec(&v).unwrap()), bare_y);
    assert_eq!(from_v(&lazy.mat_transpose_vec(&u).unwrap()), bare_t);
}

fn explicit_scales_reject_zero<M: MatrixShape>(build: &impl Fn(&TestMatrix) -> M) {
    let tm = random_matrix(23, 2, 3, 0.5);
    for zero in [0.0, -0.0] {
        for column in 0..tm.ncols {
            for scales_only in [false, true] {
                let mut scales = vec![2.0; tm.ncols];
                scales[column] = zero;
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    if scales_only {
                        LazyMatrix::with_scales(build(&tm), scales)
                    } else {
                        LazyMatrix::from_parts(build(&tm), Some(vec![0.0; tm.ncols]), Some(scales))
                    }
                }));
                let panic = result.err().expect("explicit zero scales must panic");
                let message = panic
                    .downcast_ref::<String>()
                    .expect("panic must report the column");
                assert_eq!(
                    message,
                    &format!("scale at column {column} must be nonzero")
                );
            }
        }
    }
}

fn explicit_normalization_preserves_parameters<M: MatrixShape>(build: &impl Fn(&TestMatrix) -> M) {
    let centers = vec![
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        -0.0,
        0.0,
        -2.0,
        2.0,
    ];
    let scales = vec![
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        -2.0,
        2.0,
        f64::from_bits(1),
        -f64::from_bits(1),
    ];
    let tm = random_matrix(24, 2, scales.len(), 0.5);
    let bits = |values: &[f64]| {
        values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    };

    let lazy = LazyMatrix::from_parts(build(&tm), Some(centers.clone()), Some(scales.clone()));
    assert_eq!(bits(lazy.centers().unwrap()), bits(&centers));
    assert_eq!(bits(lazy.scales().unwrap()), bits(&scales));

    let centered = LazyMatrix::with_centers(build(&tm), centers.clone());
    assert_eq!(bits(centered.centers().unwrap()), bits(&centers));
    assert!(centered.scales().is_none());

    let scaled = LazyMatrix::with_scales(build(&tm), scales.clone());
    assert_eq!(bits(scaled.scales().unwrap()), bits(&scales));
    assert!(scaled.centers().is_none());

    let (data, centers, scales) = lazy.into_parts();
    let restored = LazyMatrix::from_parts(data, centers.clone(), scales.clone());
    assert_eq!(
        bits(restored.centers().unwrap()),
        bits(centers.as_ref().unwrap())
    );
    assert_eq!(
        bits(restored.scales().unwrap()),
        bits(scales.as_ref().unwrap())
    );
}

/// `new()` with every strategy: read back the computed centers/scales and
/// confirm the operator equals the oracle built from those same vectors.
fn new_matches_oracle<M, V>(
    build: &impl Fn(&TestMatrix) -> M,
    to_v: &impl Fn(&[f64]) -> V,
    from_v: &impl Fn(&V) -> Vec<f64>,
) where
    M: MatVec<V> + MatTransposeVec<V> + ColumnStats<f64>,
    V: Clone
        + ElemDivAssign<f64>
        + DotSlice<f64>
        + SubScalarAssign<f64>
        + SumEntries<f64>
        + ScaledSubSlice<f64>,
{
    let tm = random_matrix(14, 13, 6, 0.45);
    let v = to_v(&random_vec(15, tm.ncols));
    let u = to_v(&random_vec(16, tm.nrows));

    let centerings = [Centering::None, Centering::Mean, Centering::Min];
    let scalings = [
        Scaling::None,
        Scaling::Sd,
        Scaling::L1,
        Scaling::L2,
        Scaling::MaxAbs,
        Scaling::Range,
    ];
    for center in centerings {
        for scale in scalings {
            let spec = Normalization::new(center, scale);
            let lazy = LazyMatrix::new(build(&tm), spec).unwrap();
            let xtilde = materialize(&tm.dense, lazy.centers(), lazy.scales());

            let got = from_v(&lazy.matvec(&v).unwrap());
            assert_close(&got, &dense_matvec(&xtilde, &from_v(&v)), EPS);
            let got_t = from_v(&lazy.mat_transpose_vec(&u).unwrap());
            assert_close(&got_t, &dense_tmatvec(&xtilde, &from_v(&u)), EPS);
        }
    }
}

/// (5): `ColumnStats` against hand-computed values on a fixed tiny matrix,
/// including the centered L1/L2/maxabs implicit-zero corrections.
///
/// ```text
/// X = | 1  0  5 |
///     | 3  0  5 |
///     | 0  0  5 |
/// ```
fn column_stats_fixed<M>(build: &impl Fn(&TestMatrix) -> M)
where
    M: ColumnStats<f64>,
{
    let tm = TestMatrix {
        nrows: 3,
        ncols: 3,
        dense: vec![
            vec![1.0, 0.0, 5.0],
            vec![3.0, 0.0, 5.0],
            vec![0.0, 0.0, 5.0],
        ],
        triplets: vec![
            (0, 0, 1.0),
            (1, 0, 3.0),
            (0, 2, 5.0),
            (1, 2, 5.0),
            (2, 2, 5.0),
        ],
    };
    let m = build(&tm);

    // Column 0 = [1,3,0]; column 1 = [0,0,0]; column 2 = [5,5,5].
    assert_close(&m.col_means().unwrap(), &[4.0 / 3.0, 0.0, 5.0], EPS);
    // population sd: col0 var = 10/3 − (4/3)² = 14/9; col1 = 0; col2 constant = 0
    assert_close(
        &m.col_sds().unwrap(),
        &[(14.0_f64 / 9.0).sqrt(), 0.0, 0.0],
        EPS,
    );
    assert_close(&m.col_mins().unwrap(), &[0.0, 0.0, 5.0], EPS);
    assert_close(&m.col_ranges().unwrap(), &[3.0, 0.0, 0.0], EPS);
    assert_close(&m.col_maxabs().unwrap(), &[3.0, 0.0, 5.0], EPS);
    assert_close(&m.col_l1().unwrap(), &[4.0, 0.0, 15.0], EPS);
    assert_close(
        &m.col_l2().unwrap(),
        &[10.0_f64.sqrt(), 0.0, (75.0_f64).sqrt()],
        EPS,
    );

    let centers = m.col_means().unwrap();
    assert_close(
        &m.col_l1_centered(&centers).unwrap(),
        &[10.0 / 3.0, 0.0, 0.0],
        EPS,
    );
    // centered l2² of col0 = n·var = 3·14/9 = 14/3; col1 = 0; col2 = 0
    assert_close(
        &m.col_l2_centered(&centers).unwrap(),
        &[(14.0_f64 / 3.0).sqrt(), 0.0, 0.0],
        EPS,
    );
    // centered maxabs col0: max(|1−4/3|, |3−4/3|, implicit |0−4/3|) = 5/3
    assert_close(
        &m.col_maxabs_centered(&centers).unwrap(),
        &[5.0 / 3.0, 0.0, 0.0],
        EPS,
    );

    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            m.col_l1_centered(&centers[..2]).unwrap()
        }))
        .is_err()
    );
}

/// Structural zeros affect extrema, while a fully stored positive column does
/// not acquire an artificial zero. Explicitly stored zeros count as values.
fn column_stats_respect_sparse_storage<M>(build: &impl Fn(&TestMatrix) -> M)
where
    M: ColumnStats<f64>,
{
    let matrix = build(&column_view_matrix());

    assert_close(&matrix.col_mins().unwrap(), &[-2.0, 0.0, 4.0, 0.0], EPS);
    assert_close(&matrix.col_ranges().unwrap(), &[3.0, 0.0, 3.0, 0.0], EPS);
    assert_close(&matrix.col_l1().unwrap(), &[3.0, 0.0, 22.0, 0.0], EPS);
    assert_close(
        &matrix.col_l1_centered(&[0.5, -1.0, 2.0, 3.0]).unwrap(),
        &[4.0, 4.0, 14.0, 12.0],
        EPS,
    );
}

/// `LazyMatrix::new` selects raw or centered statistics according to the
/// normalization specification and applies the exact-zero scale guard.
fn normalization_statistic_dispatch<M>(build: &impl Fn(&TestMatrix) -> M)
where
    M: ColumnStats<f64> + MatrixShape,
{
    let tm = TestMatrix {
        nrows: 3,
        ncols: 3,
        dense: vec![
            vec![1.0, 0.0, 5.0],
            vec![3.0, 0.0, 5.0],
            vec![0.0, 0.0, 5.0],
        ],
        triplets: vec![
            (0, 0, 1.0),
            (1, 0, 3.0),
            (0, 2, 5.0),
            (1, 2, 5.0),
            (2, 2, 5.0),
        ],
    };

    let raw_l1 =
        LazyMatrix::new(build(&tm), Normalization::new(Centering::None, Scaling::L1)).unwrap();
    assert_close(raw_l1.scales().unwrap(), &[4.0, 1.0, 15.0], EPS);

    let centered_l1 =
        LazyMatrix::new(build(&tm), Normalization::new(Centering::Mean, Scaling::L1)).unwrap();
    assert_close(centered_l1.centers().unwrap(), &[4.0 / 3.0, 0.0, 5.0], EPS);
    assert_close(centered_l1.scales().unwrap(), &[10.0 / 3.0, 1.0, 1.0], EPS);

    let min_range = LazyMatrix::new(
        build(&tm),
        Normalization::new(Centering::Min, Scaling::Range),
    )
    .unwrap();
    assert_close(min_range.centers().unwrap(), &[0.0, 0.0, 5.0], EPS);
    assert_close(min_range.scales().unwrap(), &[3.0, 1.0, 1.0], EPS);
}

/// (6): a constant column (sd 0) is floored to scale 1 → finite output.
fn zero_scale_guard<M, V>(
    build: &impl Fn(&TestMatrix) -> M,
    to_v: &impl Fn(&[f64]) -> V,
    from_v: &impl Fn(&V) -> Vec<f64>,
) where
    M: MatVec<V> + MatTransposeVec<V> + ColumnStats<f64>,
    V: Clone
        + ElemDivAssign<f64>
        + DotSlice<f64>
        + SubScalarAssign<f64>
        + SumEntries<f64>
        + ScaledSubSlice<f64>,
{
    // Middle column is empty (all zero) → sd 0; last column constant → sd 0.
    let tm = TestMatrix {
        nrows: 3,
        ncols: 3,
        dense: vec![
            vec![1.0, 0.0, 4.0],
            vec![2.0, 0.0, 4.0],
            vec![3.0, 0.0, 4.0],
        ],
        triplets: vec![
            (0, 0, 1.0),
            (1, 0, 2.0),
            (2, 0, 3.0),
            (0, 2, 4.0),
            (1, 2, 4.0),
            (2, 2, 4.0),
        ],
    };
    for spec in [
        Normalization::new(Centering::Mean, Scaling::Sd),
        Normalization::new(Centering::Mean, Scaling::L1),
        Normalization::new(Centering::Min, Scaling::Range),
    ] {
        let lazy = LazyMatrix::new(build(&tm), spec).unwrap();
        let scales = lazy.scales().unwrap();
        assert_eq!(scales[1], 1.0, "empty column scale must be floored to 1");
        assert_eq!(scales[2], 1.0, "constant column scale must be floored to 1");

        let v = to_v(&random_vec(20, tm.ncols));
        let y = from_v(&lazy.matvec(&v).unwrap());
        assert!(y.iter().all(|x| x.is_finite()), "output must be finite");
    }
}

fn intercept_products<M, V>(
    build: &impl Fn(&TestMatrix) -> M,
    to_v: &impl Fn(&[f64]) -> V,
    from_v: &impl Fn(&V) -> Vec<f64>,
) where
    M: MatVecInto<V> + MatTransposeVecInto<V> + ColumnStats<f64>,
    V: Clone
        + lazymatrix::VectorOwned<f64, Owned = V>
        + lazymatrix::VectorViewMut<f64>
        + ElemDivAssign<f64>
        + DotSlice<f64>
        + SubScalarAssign<f64>
        + SumEntries<f64>
        + ScaledSubSlice<f64>,
{
    use lazymatrix::WithIntercept;
    for (n, p, density) in [
        (9, 4, 0.3),
        (5, 3, 0.0),
        (0, 3, 0.0),
        (3, 0, 0.0),
        (0, 0, 0.0),
    ] {
        let tm = random_matrix(831, n, p, density);
        let matrix = build(&tm);
        for center in [Centering::None, Centering::Mean, Centering::Min] {
            for scale in [
                Scaling::None,
                Scaling::Sd,
                Scaling::L1,
                Scaling::L2,
                Scaling::Range,
                Scaling::MaxAbs,
            ] {
                // Empty statistics can be nonfinite; finite explicit parameters
                // isolate the operator's empty-sum behavior in this fixture.
                let lazy = if n == 0 {
                    LazyMatrix::from_parts(&matrix, Some(vec![1.0; p]), Some(vec![-2.0; p]))
                } else {
                    LazyMatrix::new(&matrix, Normalization::new(center, scale)).unwrap()
                };
                let dense = with_intercept(&materialize(&tm.dense, lazy.centers(), lazy.scales()));
                let augmented = WithIntercept::new(&lazy);
                let x = random_vec(832, p + 1);
                let y = random_vec(833, n);
                let forward = augmented.matvec(&to_v(&x)).unwrap();
                let transpose = augmented.mat_transpose_vec(&to_v(&y)).unwrap();
                let expected_forward = dense_matvec(&dense, &x);
                let expected_transpose: Vec<_> = (0..p + 1)
                    .map(|j| (0..n).map(|i| dense[i][j] * y[i]).sum())
                    .collect();
                assert_close(&from_v(&forward), &expected_forward, 1e-10);
                assert_close(&from_v(&transpose), &expected_transpose, 1e-10);
                let mut out = to_v(&vec![f64::NAN; n]);
                let mut trans = to_v(&vec![f64::NAN; p + 1]);
                augmented.matvec_into(&to_v(&x), &mut out).unwrap();
                augmented
                    .mat_transpose_vec_into(&to_v(&y), &mut trans)
                    .unwrap();
                assert_close(&from_v(&out), &expected_forward, 1e-10);
                assert_close(&from_v(&trans), &expected_transpose, 1e-10);
                let lhs: f64 = from_v(&out).iter().zip(&y).map(|(a, b)| a * b).sum();
                let rhs: f64 = x.iter().zip(from_v(&trans)).map(|(a, b)| a * b).sum();
                approx::assert_abs_diff_eq!(lhs, rhs, epsilon = 1e-10);
                let raw = WithIntercept::new(&matrix);
                assert_close(
                    &from_v(&raw.matvec(&to_v(&x)).unwrap()),
                    &dense_matvec(&with_intercept(&tm.dense), &x),
                    1e-10,
                );
            }
        }
    }
}

fn with_intercept(dense: &[Vec<f64>]) -> Vec<Vec<f64>> {
    dense
        .iter()
        .map(|row| std::iter::once(1.0).chain(row.iter().copied()).collect())
        .collect()
}

fn run_intercept_gram_suite<M>(build: &impl Fn(&TestMatrix) -> M)
where
    M: lazymatrix::WeightedGramInto<f64>
        + lazymatrix::WeightedGramKernel<f64>
        + lazymatrix::WeightedColumnSumsInto<f64>
        + lazymatrix::WeightedColumnSumsKernel<f64>
        + ColumnStats<f64>,
{
    use lazymatrix::{WeightedColumnSumsInto, WeightedGramInto, WithIntercept};
    for (n, p, density) in [
        (9, 4, 0.3),
        (5, 3, 0.0),
        (0, 3, 0.0),
        (3, 0, 0.0),
        (0, 0, 0.0),
    ] {
        let tm = random_matrix(835, n, p, density);
        let matrix = build(&tm);
        let weights: Vec<_> = (0..n).map(|i| (i % 5) as f64 - 2.0).collect();
        let mut out = GramOutput(vec![vec![f64::NAN; p + 1]; p + 1]);
        WithIntercept::new(&matrix)
            .weighted_gram_into(&weights, &mut out)
            .unwrap();
        assert_gram(&out, &with_intercept(&tm.dense), &weights);
        for center in [Centering::None, Centering::Mean, Centering::Min] {
            for scale in [
                Scaling::None,
                Scaling::Sd,
                Scaling::L1,
                Scaling::L2,
                Scaling::Range,
                Scaling::MaxAbs,
            ] {
                let lazy = LazyMatrix::new(&matrix, Normalization::new(center, scale)).unwrap();
                WithIntercept::new(&lazy)
                    .weighted_gram_into(&weights, &mut out)
                    .unwrap();
                assert_gram(
                    &out,
                    &with_intercept(&materialize(&tm.dense, lazy.centers(), lazy.scales())),
                    &weights,
                );
            }
        }
    }
    for values in [
        vec![1e16 + 2.0, 1e16 + 4.0, 1e16 + 6.0],
        vec![0.0, 2.0, -3.0],
        vec![1.0, 0.0, 1.0],
        vec![f64::INFINITY, 0.0, -1.0],
    ] {
        let tm = TestMatrix {
            nrows: 3,
            ncols: 1,
            dense: values.iter().map(|&x| vec![x]).collect(),
            triplets: values
                .iter()
                .enumerate()
                .filter(|(_, x)| **x != 0.0)
                .map(|(i, &x)| (i, 0, x))
                .collect(),
        };
        let matrix = build(&tm);
        for center in [0.0, 1.0, 1e16, f64::INFINITY, f64::NAN] {
            for scale in [1.0, -2.0, f64::INFINITY, f64::NAN] {
                let lazy = LazyMatrix::from_parts(&matrix, Some(vec![center]), Some(vec![scale]));
                for weights in [
                    [0.1, 0.2, 0.3],
                    [0.0, -1.0, 2.0],
                    [1e20, 1.0, 1e20],
                    [f64::INFINITY, 1.0, 0.0],
                    [f64::NAN, 0.0, 1.0],
                ] {
                    let mut out = GramOutput(vec![vec![f64::NAN; 2]; 2]);
                    WithIntercept::new(&lazy)
                        .weighted_gram_into(&weights, &mut out)
                        .unwrap();
                    assert_gram(
                        &out,
                        &with_intercept(&materialize(&tm.dense, lazy.centers(), lazy.scales())),
                        &weights,
                    );
                    let mut sums = [f64::NAN];
                    lazy.weighted_column_sums_into(&weights, &mut sums).unwrap();
                    if out.0[0][1].is_nan() {
                        assert!(sums[0].is_nan());
                    } else {
                        assert_eq!(sums[0], out.0[0][1]);
                    }
                }
            }
        }
        if values[0] == 1e16 + 2.0 {
            let lazy = LazyMatrix::from_parts(&matrix, Some(vec![1e16]), None);
            let mut out = GramOutput(vec![vec![f64::NAN; 2]; 2]);
            WithIntercept::new(lazy)
                .weighted_gram_into(&[0.1, 0.2, 0.3], &mut out)
                .unwrap();
            approx::assert_abs_diff_eq!(out.0[0][1], 2.8, epsilon = 1e-14);
        }
        for (weights, centers, scales, size) in [
            (vec![1.0; 2], None, None, 1),
            (vec![1.0; 3], None, None, 2),
            (vec![1.0; 3], Some(vec![1.0; 2]), None, 1),
            (vec![1.0; 3], None, Some(vec![1.0; 2]), 1),
            (vec![1.0; 3], None, Some(vec![-0.0]), 1),
        ] {
            let mut sums = vec![99.0; size];
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    matrix
                        .weighted_column_sums_normalized_into(
                            &weights,
                            centers.as_deref(),
                            scales.as_deref(),
                            &mut sums,
                        )
                        .unwrap();
                }))
                .is_err()
            );
            assert!(sums.iter().all(|&x| x == 99.0));
        }
    }
}
