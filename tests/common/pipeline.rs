//! Independent eager stages and backend-shared pipeline checks.

use super::{
    TestMatrix, materialize_rows, matrix_from_dense, random_matrix, row_normalization_oracle,
};
use lazymatrix::{
    Centering, ColumnNormalization, ColumnNormalizationParams, LazyMatrix, MatTransposeVec,
    MatTransposeVecInto, MatVec, MatVecInto, MatrixErrorType, MatrixRead, NormalizationChain,
    NormalizationChainParams, NormalizationRecipe, RowNormalization, Scaling, VectorOwned,
    VectorViewMut,
};

fn apply(raw: &[Vec<f64>], ncols: usize, rows: bool, spec: RowNormalization) -> Vec<Vec<f64>> {
    if rows {
        let (centers, scales) = row_normalization_oracle(raw, spec);
        materialize_rows(raw, centers.as_deref(), scales.as_deref())
    } else {
        let columns: Vec<Vec<_>> = (0..ncols)
            .map(|j| raw.iter().map(|r| r[j]).collect())
            .collect();
        let (centers, scales) = row_normalization_oracle(&columns, spec);
        let normalized = materialize_rows(&columns, centers.as_deref(), scales.as_deref());
        (0..raw.len())
            .map(|i| (0..ncols).map(|j| normalized[j][i]).collect())
            .collect()
    }
}

fn equal(actual: f64, expected: f64) {
    if expected.is_nan() {
        assert!(actual.is_nan(), "{actual} should be NaN");
    } else if expected.is_infinite() {
        assert_eq!(actual, expected);
    } else {
        approx::assert_relative_eq!(actual, expected, epsilon = 1e-9, max_relative = 1e-9);
    }
}

fn check<O, V>(op: &O, raw: &[Vec<f64>], ncols: usize, to_v: &impl Fn(&[f64]) -> V)
where
    O: MatVec<V> + MatVecInto<V> + MatTransposeVec<V> + MatTransposeVecInto<V>,
    V: VectorOwned<f64, Owned = V> + VectorViewMut<f64>,
{
    let x: Vec<_> = (0..ncols).map(|j| 0.5 - j as f64 / 3.).collect();
    let u: Vec<_> = (0..raw.len()).map(|i| i as f64 / 4. - 0.75).collect();
    let expected: Vec<f64> = raw
        .iter()
        .map(|row| row.iter().zip(&x).map(|(a, b)| a * b).sum())
        .collect();
    let expected_t: Vec<f64> = (0..ncols)
        .map(|j| raw.iter().zip(&u).map(|(row, w)| row[j] * w).sum())
        .collect();
    let x = to_v(&x);
    let u = to_v(&u);
    let product = op.matvec(&x).unwrap();
    let transpose = op.mat_transpose_vec(&u).unwrap();
    let mut reused = V::owned_from_fn(raw.len(), |_| f64::NAN);
    let mut reused_t = V::owned_from_fn(ncols, |_| f64::NAN);
    op.matvec_into(&x, &mut reused).unwrap();
    op.mat_transpose_vec_into(&u, &mut reused_t).unwrap();
    for (i, &value) in expected.iter().enumerate() {
        equal(product.get(i), value);
        equal(reused.get(i), value);
    }
    for (j, &value) in expected_t.iter().enumerate() {
        equal(transpose.get(j), value);
        equal(reused_t.get(j), value);
    }
    if expected.iter().chain(&expected_t).all(|v| v.is_finite()) {
        let left: f64 = (0..u.len()).map(|i| u.get(i) * product.get(i)).sum();
        let right: f64 = (0..x.len()).map(|j| x.get(j) * transpose.get(j)).sum();
        equal(left, right);
    }
}

fn cases() -> Vec<TestMatrix> {
    let mut cases = vec![
        random_matrix(187, 7, 4, 0.35),
        random_matrix(812, 3, 8, 0.8),
        random_matrix(4, 0, 3, 0.),
        random_matrix(4, 4, 0, 0.),
        matrix_from_dense(vec![vec![0.; 3]; 4]),
        matrix_from_dense(vec![vec![3.; 3]; 4]),
        matrix_from_dense(vec![vec![1., f64::NAN, 2.], vec![2., 4., f64::INFINITY]]),
    ];
    cases.push(matrix_from_dense(
        (0..5)
            .map(|i| {
                (0..3)
                    .map(|j| 1e12 + (i * i + i * j + j) as f64 / 4.)
                    .collect()
            })
            .collect(),
    ));
    cases
}

/// Check both axis orders, repeated stages, and frozen and new-sample preparation.
pub fn run_dense_pipeline_suite<M, V>(build: impl Fn(&TestMatrix) -> M, to_v: impl Fn(&[f64]) -> V)
where
    M: MatrixRead<f64> + MatrixErrorType,
    V: VectorOwned<f64, Owned = V> + VectorViewMut<f64>,
{
    let specs = [
        RowNormalization::standardize(),
        RowNormalization::robust(),
        RowNormalization::min_max(),
        RowNormalization::new(Centering::None, Scaling::L1),
        RowNormalization::new(Centering::Min, Scaling::L2),
        RowNormalization::new(Centering::Mean, Scaling::MaxAbs),
        RowNormalization::new(Centering::Median, Scaling::Sd { df: 1 }),
        RowNormalization::default(),
    ];
    for tm in cases() {
        for row in specs {
            let col = ColumnNormalization::new(row.center, row.scale);
            let data = build(&tm);
            let fitted = row.then(col).fit(&data).unwrap();
            let expected = apply(&apply(&tm.dense, tm.ncols, true, row), tm.ncols, false, row);
            check(
                &fitted.view().apply_frozen(&data),
                &expected,
                tm.ncols,
                &to_v,
            );
            check(
                &fitted.view().for_new_samples(&data).unwrap(),
                &expected,
                tm.ncols,
                &to_v,
            );
            let op = LazyMatrix::new(data, col.then(row)).unwrap();
            let expected = apply(&apply(&tm.dense, tm.ncols, false, row), tm.ncols, true, row);
            check(&op, &expected, tm.ncols, &to_v);
            let prepared = op.into_parts().1.for_new_samples(build(&tm)).unwrap();
            check(&prepared, &expected, tm.ncols, &to_v);
        }
        let row = RowNormalization::new(Centering::Mean, Scaling::L2);
        let col = ColumnNormalization::standardize();
        let op = LazyMatrix::new(build(&tm), row.then(col).then(row)).unwrap();
        let expected = apply(
            &apply(
                &apply(&tm.dense, tm.ncols, true, row),
                tm.ncols,
                false,
                RowNormalization::standardize(),
            ),
            tm.ncols,
            true,
            row,
        );
        check(&op, &expected, tm.ncols, &to_v);
    }
}

type ColumnChain = NormalizationChainParams<ColumnNormalizationParams, ColumnNormalizationParams>;

pub(super) fn check_dense_pipeline_view<M, V>(matrix: &M, to_v: &impl Fn(&[f64]) -> V)
where
    M: MatrixRead<f64> + MatrixErrorType,
    V: VectorOwned<f64, Owned = V> + VectorViewMut<f64>,
{
    let raw: Vec<Vec<_>> = (0..matrix.nrows())
        .map(|i| (0..matrix.ncols()).map(|j| matrix.get(i, j)).collect())
        .collect();
    let row = RowNormalization::standardize();
    let col = ColumnNormalization::robust();
    let fitted = row.then(col).fit(matrix).unwrap();
    let expected = apply(
        &apply(&raw, matrix.ncols(), true, row),
        matrix.ncols(),
        false,
        RowNormalization::robust(),
    );
    check(
        &fitted.view().apply_frozen(matrix),
        &expected,
        matrix.ncols(),
        to_v,
    );
    let fitted = col.then(row).fit(matrix).unwrap();
    let expected = apply(
        &apply(&raw, matrix.ncols(), false, RowNormalization::robust()),
        matrix.ncols(),
        true,
        row,
    );
    check(
        &fitted.view().apply_frozen(matrix),
        &expected,
        matrix.ncols(),
        to_v,
    );
}

/// Check native column-only chains against independently materialized stages.
pub fn run_column_pipeline_suite<M, V>(build: impl Fn(&TestMatrix) -> M, to_v: impl Fn(&[f64]) -> V)
where
    M: lazymatrix::MatrixShape + MatrixErrorType,
    NormalizationChain<ColumnNormalization, ColumnNormalization>:
        NormalizationRecipe<M, ColumnChain>,
    LazyMatrix<M, ColumnChain>:
        MatVec<V> + MatVecInto<V> + MatTransposeVec<V> + MatTransposeVecInto<V>,
    V: VectorOwned<f64, Owned = V> + VectorViewMut<f64>,
{
    let specs = [
        ColumnNormalization::standardize(),
        ColumnNormalization::robust(),
        ColumnNormalization::min_max(),
        ColumnNormalization::new(Centering::Min, Scaling::L1),
        ColumnNormalization::new(Centering::Median, Scaling::L2),
        ColumnNormalization::new(Centering::Mean, Scaling::MaxAbs),
        ColumnNormalization::default(),
    ];
    for tm in cases() {
        for first in specs {
            for next in [
                ColumnNormalization::standardize(),
                ColumnNormalization::robust(),
            ] {
                let expected = apply(
                    &apply(
                        &tm.dense,
                        tm.ncols,
                        false,
                        RowNormalization::new(first.center, first.scale),
                    ),
                    tm.ncols,
                    false,
                    RowNormalization::new(next.center, next.scale),
                );
                let op = LazyMatrix::new(build(&tm), first.then(next)).unwrap();
                check(&op, &expected, tm.ncols, &to_v);
            }
        }
    }
}
