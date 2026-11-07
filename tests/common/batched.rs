use super::{
    MaterializeOutput, TestMatrix, dense_transpose, materialize, materialize_rows, random_matrix,
};
use lazymatrix::{
    Centering, ColumnNormalization, ColumnStats, LazyMatrix, MatMat, MatMatInto, MatTransposeMat,
    MatTransposeMatInto, MatrixOwned, MatrixRead, MatrixShape, Scaling, Transpose, WithIntercept,
};

fn assert_close(actual: &[f64], expected: &[f64], epsilon: f64) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        if expected.is_nan() {
            assert!(actual.is_nan(), "{actual} should be NaN");
        } else if expected.is_infinite() {
            assert_eq!(actual, expected);
        } else {
            approx::assert_abs_diff_eq!(actual, expected, epsilon = epsilon);
        }
    }
}

impl<F: lazymatrix::Scalar> MatrixRead<F> for MaterializeOutput<F> {
    fn get(&self, row: usize, column: usize) -> F {
        self.values[row * self.columns + column]
    }
}

pub fn check_batched<M>(matrix: &M, dense: &[Vec<f64>])
where
    M: MatMatInto<f64> + MatTransposeMatInto<f64>,
{
    check_batched_products(matrix, dense);
    check_batched_products(
        &Transpose::new(matrix),
        &dense_transpose(dense, matrix.ncols()),
    );
}

fn check_batched_products<M>(matrix: &M, dense: &[Vec<f64>])
where
    M: MatMatInto<f64> + MatTransposeMatInto<f64>,
{
    let (n, p) = (matrix.nrows(), matrix.ncols());
    for k in [0, 1, 3, 17, 33] {
        let mut rhs = MaterializeOutput::zeros(p, k);
        let mut u = MaterializeOutput::zeros(n, k);
        for (index, value) in rhs.values.iter_mut().enumerate() {
            *value = (index % 7) as f64 - 3.;
        }
        for (index, value) in u.values.iter_mut().enumerate() {
            *value = (index % 5) as f64 - 2.;
        }
        let input = rhs.values.clone();
        let mut expected = MaterializeOutput::zeros(n, k);
        let mut expected_t = MaterializeOutput::zeros(p, k);
        for (i, row) in dense.iter().enumerate() {
            for r in 0..k {
                expected.values[i * k + r] = (0..p).map(|j| row[j] * rhs.get(j, r)).sum();
            }
        }
        for (j, column) in expected_t.values.chunks_mut(k.max(1)).enumerate() {
            for (r, value) in column.iter_mut().enumerate() {
                *value = dense
                    .iter()
                    .enumerate()
                    .map(|(i, row)| row[j] * u.get(i, r))
                    .sum();
            }
        }
        let forward: MaterializeOutput = matrix.matmat(&rhs).unwrap();
        let transpose: MaterializeOutput = matrix.mat_transpose_mat(&u).unwrap();
        assert_eq!((forward.nrows(), forward.ncols()), (n, k));
        assert_eq!((transpose.nrows(), transpose.ncols()), (p, k));
        assert_close(&forward.values, &expected.values, 1e-9);
        assert_close(&transpose.values, &expected_t.values, 1e-9);
        let left: f64 = forward
            .values
            .iter()
            .zip(&u.values)
            .map(|(a, b)| a * b)
            .sum();
        let right: f64 = rhs
            .values
            .iter()
            .zip(&transpose.values)
            .map(|(a, b)| a * b)
            .sum();
        if forward
            .values
            .iter()
            .chain(&transpose.values)
            .all(|v| v.is_finite())
        {
            approx::assert_relative_eq!(left, right, epsilon = 1e-8, max_relative = 1e-10);
        }
        let mut reusable = forward;
        let address = reusable.values.as_ptr();
        for old in [f64::NAN, f64::INFINITY, -99.] {
            reusable.values.fill(old);
            matrix.matmat_into(&rhs, &mut reusable).unwrap();
            assert_close(&reusable.values, &expected.values, 1e-9);
            assert_eq!(reusable.values.as_ptr(), address);
            let mut reusable_t = MaterializeOutput::zeros(p, k);
            reusable_t.values.fill(old);
            matrix.mat_transpose_mat_into(&u, &mut reusable_t).unwrap();
            assert_close(&reusable_t.values, &expected_t.values, 1e-9);
        }
        assert_eq!(rhs.values, input);
    }
}

pub fn check_normalized_batches<M: lazymatrix::MatrixTraversal<f64>>(matrix: &M, raw: &[Vec<f64>]) {
    let row = LazyMatrix::from_row_parts(
        matrix,
        Some(vec![0.75; matrix.nrows()]),
        Some(vec![-2.; matrix.nrows()]),
    );
    check_batched(
        &row,
        &materialize_rows(
            raw,
            row.normalization().centers(),
            row.normalization().scales(),
        ),
    );
    let params = lazymatrix::NormalizationChainParams::from_parts(
        lazymatrix::ColumnNormalizationParams::from_parts(
            matrix.ncols(),
            Some(vec![0.5; matrix.ncols()]),
            None,
        ),
        lazymatrix::ColumnNormalizationParams::from_parts(
            matrix.ncols(),
            Some(vec![0.25; matrix.ncols()]),
            Some(vec![-2.; matrix.ncols()]),
        ),
    );
    let expected = materialize(
        &materialize(raw, params.first().centers(), None),
        params.next().centers(),
        params.next().scales(),
    );
    check_batched(&params.view().apply_frozen(matrix), &expected);
}

pub fn run_batched_suite<M>(build: impl Fn(&TestMatrix) -> M)
where
    M: MatMatInto<f64>
        + MatTransposeMatInto<f64>
        + ColumnStats<f64>
        + lazymatrix::MatrixTraversal<f64>
        + lazymatrix::RowStats<f64>,
{
    for (n, p, density) in [(7, 3, 0.4), (4, 2, 1.), (3, 2, 0.), (0, 3, 0.), (3, 0, 0.)] {
        let tm = random_matrix(0xBA7C, n, p, density);
        let matrix = build(&tm);
        check_batched(&matrix, &tm.dense);
        for spec in [
            lazymatrix::RowNormalization::default(),
            lazymatrix::RowNormalization::standardize(),
            lazymatrix::RowNormalization::robust(),
            lazymatrix::RowNormalization::new(Centering::Min, Scaling::L2),
        ] {
            let params = spec.fit(&matrix).unwrap();
            let normalized = materialize_rows(&tm.dense, params.centers(), params.scales());
            check_batched(&params.view().apply_frozen(&matrix), &normalized);
            check_batched(&params.apply_frozen(build(&tm)), &normalized);
        }
        for (center, scale) in [
            (0.75, -2.),
            (f64::INFINITY, 1.),
            (1., f64::INFINITY),
            (f64::NAN, 2.),
        ] {
            let lazy =
                LazyMatrix::from_row_parts(&matrix, Some(vec![center; n]), Some(vec![scale; n]));
            check_batched(
                &lazy,
                &materialize_rows(
                    &tm.dense,
                    lazy.normalization().centers(),
                    lazy.normalization().scales(),
                ),
            );
        }

        if n > 0 {
            for centering in [Centering::None, Centering::Mean, Centering::Min] {
                for scaling in [
                    Scaling::None,
                    Scaling::Sd { df: 0 },
                    Scaling::Sd { df: 1 },
                    Scaling::L1,
                    Scaling::L2,
                    Scaling::MaxAbs,
                    Scaling::Range,
                ] {
                    let lazy =
                        LazyMatrix::new(&matrix, ColumnNormalization::new(centering, scaling))
                            .unwrap();
                    check_batched(
                        &lazy,
                        &materialize(
                            &tm.dense,
                            lazy.normalization().centers(),
                            lazy.normalization().scales(),
                        ),
                    );
                }
            }
        }
        let lazy = LazyMatrix::from_column_parts(&matrix, Some(vec![0.75; p]), Some(vec![-2.; p]));
        let normalized = materialize(
            &tm.dense,
            lazy.normalization().centers(),
            lazy.normalization().scales(),
        );
        check_batched(&lazy, &normalized);
        let shared = lazy.normalization().view().apply(&matrix);
        check_batched(&shared, &normalized);
        let augmented: Vec<Vec<f64>> = normalized
            .iter()
            .map(|row| std::iter::once(1.).chain(row.iter().copied()).collect())
            .collect();
        check_batched(&WithIntercept::new(&lazy), &augmented);
        for transpose in [false, true] {
            let mut out = MaterializeOutput::zeros(if transpose { p } else { n }, 2);
            out.values.fill(99.);
            let rhs = MaterializeOutput::zeros(if transpose { n + 1 } else { p + 1 }, 2);
            let error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if transpose {
                    lazy.mat_transpose_mat_into(&rhs, &mut out)
                } else {
                    lazy.matmat_into(&rhs, &mut out)
                }
            }));
            assert!(error.is_err());
            assert!(out.values.iter().all(|&v| v == 99.));
        }
    }
}

pub fn check_batched_f32<M, V>(matrix: &M, to_vector: impl Fn(&[f32]) -> V)
where
    M: MatMatInto<f32>
        + MatTransposeMatInto<f32>
        + lazymatrix::MatVec<V>
        + lazymatrix::MatTransposeVec<V>
        + lazymatrix::MatrixTraversal<f32>,
    V: Clone
        + lazymatrix::VectorView<f32>
        + lazymatrix::ElemDivAssign<f32>
        + lazymatrix::DotSlice<f32>
        + lazymatrix::SubScalarAssign<f32>
        + lazymatrix::SumEntries<f32>
        + lazymatrix::ScaledSubSlice<f32>,
{
    use lazymatrix::{MatTransposeVec, MatVec, VectorView};
    fn equal(actual: f32, expected: f32) {
        if expected.is_nan() {
            assert!(actual.is_nan(), "{actual} should be NaN");
        } else if expected.is_infinite() {
            assert_eq!(actual, expected);
        } else {
            approx::assert_relative_eq!(actual, expected, epsilon = 1e-5, max_relative = 1e-5);
        }
    }
    let mut raw = vec![vec![0_f32; matrix.ncols()]; matrix.nrows()];
    matrix
        .for_each_entry(0..matrix.nrows(), 0..matrix.ncols(), |i, j, value| {
            raw[i][j] = value
        })
        .unwrap();
    for center in [0_f32, 1e38, f32::INFINITY, f32::NAN] {
        for scale in [-2_f32, 1e-38, f32::INFINITY, f32::NAN] {
            let row = LazyMatrix::from_row_parts(
                matrix,
                Some(vec![center; matrix.nrows()]),
                Some(vec![scale; matrix.nrows()]),
            );
            let expected: Vec<Vec<_>> = raw
                .iter()
                .map(|r| r.iter().map(|&v| (v - center) / scale).collect())
                .collect();
            check_f32_oracle(&row, &expected);
            let params = lazymatrix::NormalizationChainParams::from_parts(
                lazymatrix::ColumnNormalizationParams::from_parts(
                    matrix.ncols(),
                    Some(vec![center; matrix.ncols()]),
                    Some(vec![scale; matrix.ncols()]),
                ),
                lazymatrix::ColumnNormalizationParams::from_parts(
                    matrix.ncols(),
                    Some(vec![0.25_f32; matrix.ncols()]),
                    Some(vec![-2_f32; matrix.ncols()]),
                ),
            );
            let expected: Vec<Vec<_>> = expected
                .iter()
                .map(|r| r.iter().map(|&v| (v - 0.25) / -2.).collect())
                .collect();
            check_f32_oracle(&params.view().apply_frozen(matrix), &expected);
        }
    }
    for centers in [None, Some(vec![1e38_f32; matrix.ncols()])] {
        for scale in [
            None,
            Some(2_f32),
            Some(-2.),
            Some(1e-38),
            Some(f32::INFINITY),
            Some(f32::NAN),
        ] {
            let lazy = LazyMatrix::from_column_parts(
                matrix,
                centers.clone(),
                scale.map(|s| vec![s; matrix.ncols()]),
            );
            let coefficients = [
                f32::NAN,
                0.,
                1.,
                1e-38,
                f32::INFINITY,
                -1.,
                0.,
                1.,
                2.,
                3.,
                4.,
                5.,
                6.,
                7.,
                8.,
                9.,
                1.,
            ];
            let mut rhs = MaterializeOutput::<f32>::zeros(matrix.ncols(), coefficients.len());
            let mut u = MaterializeOutput::<f32>::zeros(matrix.nrows(), coefficients.len());
            for row in rhs.values.chunks_mut(coefficients.len()) {
                row.copy_from_slice(&coefficients);
            }
            for row in u.values.chunks_mut(coefficients.len()) {
                row.copy_from_slice(&coefficients);
            }
            let actual: MaterializeOutput<f32> = lazy.matmat(&rhs).unwrap();
            let transpose: MaterializeOutput<f32> = lazy.mat_transpose_mat(&u).unwrap();
            let view = Transpose::new(&lazy);
            let view_forward: MaterializeOutput<f32> = view.matmat(&u).unwrap();
            let view_transpose: MaterializeOutput<f32> = view.mat_transpose_mat(&rhs).unwrap();
            for (k, &value) in coefficients.iter().enumerate() {
                let expected = lazy
                    .matvec(&to_vector(&vec![value; matrix.ncols()]))
                    .unwrap();
                let expected_t = lazy
                    .mat_transpose_vec(&to_vector(&vec![value; matrix.nrows()]))
                    .unwrap();
                for i in 0..matrix.nrows() {
                    equal(actual.get(i, k), VectorView::get(&expected, i));
                    equal(view_transpose.get(i, k), VectorView::get(&expected, i));
                }
                for j in 0..matrix.ncols() {
                    equal(transpose.get(j, k), VectorView::get(&expected_t, j));
                    equal(view_forward.get(j, k), VectorView::get(&expected_t, j));
                }
            }
        }
    }
}

fn check_f32_oracle<M: MatMatInto<f32> + MatTransposeMatInto<f32>>(op: &M, dense: &[Vec<f32>]) {
    let coefficients = [f32::NAN, 0., 1., 1e-38, f32::INFINITY, -1., 1e38];
    for transpose in [false, true] {
        let (input, output) = if transpose {
            (op.nrows(), op.ncols())
        } else {
            (op.ncols(), op.nrows())
        };
        let mut rhs = MaterializeOutput::<f32>::zeros(input, 17);
        for i in 0..input {
            for k in 0..17 {
                rhs.values[i * 17 + k] = coefficients[(i + k) % coefficients.len()];
            }
        }
        let actual: MaterializeOutput<f32> = if transpose {
            op.mat_transpose_mat(&rhs).unwrap()
        } else {
            op.matmat(&rhs).unwrap()
        };
        assert_eq!(actual.nrows(), output);
        for (to, row) in actual.values.chunks(17).enumerate() {
            for (k, &value) in row.iter().enumerate() {
                let expected: f32 = (0..input)
                    .map(|from| {
                        (if transpose {
                            dense[from][to]
                        } else {
                            dense[to][from]
                        }) * rhs.get(from, k)
                    })
                    .sum();
                if expected.is_nan() {
                    assert!(value.is_nan());
                } else if expected.is_infinite() {
                    assert_eq!(value, expected);
                } else {
                    approx::assert_relative_eq!(
                        value,
                        expected,
                        epsilon = 1e-5,
                        max_relative = 1e-5
                    );
                }
            }
        }
    }
}
