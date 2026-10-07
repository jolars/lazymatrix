use std::{cell::Cell, fmt};

use lazymatrix::{
    LazyMatrix, MatTransposeVec, MatTransposeVecInto, MatVec, MatVecInto, MatrixErrorType,
    MatrixShape, MatrixWrite, VectorView, VectorViewMut, WeightedColumnSumsInto,
    WeightedColumnSumsKernel, WeightedGramInto, WeightedGramKernel, WithIntercept,
};

#[derive(Debug, PartialEq)]
struct ReadError;
impl fmt::Display for ReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("injected read failure")
    }
}
impl std::error::Error for ReadError {}

struct Source {
    rows: usize,
    cols: usize,
    values: Vec<f64>,
    fail: Cell<bool>,
    fail_sums: Cell<bool>,
    calls: Cell<usize>,
}

impl Source {
    fn new(rows: usize, cols: usize, values: Vec<f64>) -> Self {
        Self {
            rows,
            cols,
            values,
            fail: Cell::new(false),
            fail_sums: Cell::new(false),
            calls: Cell::new(0),
        }
    }

    fn begin(&self) -> Result<(), ReadError> {
        self.calls.set(self.calls.get() + 1);
        if self.fail.get() {
            Err(ReadError)
        } else {
            Ok(())
        }
    }

    fn logical(&self, i: usize, j: usize, c: Option<&[f64]>, s: Option<&[f64]>) -> f64 {
        (self.values[i * self.cols + j] - c.map_or(0.0, |c| c[j])) / s.map_or(1.0, |s| s[j])
    }
}

impl MatrixShape for Source {
    fn nrows(&self) -> usize {
        self.rows
    }
    fn ncols(&self) -> usize {
        self.cols
    }
}
impl MatrixErrorType for Source {
    type Error = ReadError;
}

impl MatVecInto<Vec<f64>> for Source {
    fn matvec_into(&self, x: &Vec<f64>, out: &mut Vec<f64>) -> Result<(), ReadError> {
        if !out.is_empty() {
            out[0] = 17.0;
        }
        self.begin()?;
        for (i, value) in out.iter_mut().enumerate() {
            *value = (0..self.cols)
                .map(|j| self.values[i * self.cols + j] * x[j])
                .sum();
        }
        Ok(())
    }
}
impl MatTransposeVecInto<Vec<f64>> for Source {
    fn mat_transpose_vec_into(&self, x: &Vec<f64>, out: &mut Vec<f64>) -> Result<(), ReadError> {
        if !out.is_empty() {
            out[0] = 17.0;
        }
        self.begin()?;
        for (j, value) in out.iter_mut().enumerate() {
            *value = (0..self.rows)
                .map(|i| self.values[i * self.cols + j] * x[i])
                .sum();
        }
        Ok(())
    }
}

impl<X: VectorView<f64>, Y: VectorViewMut<f64>> lazymatrix::MatVecScaledInto<X, Y, f64> for Source {
    fn matvec_scaled_into(
        &self,
        alpha: f64,
        x: &X,
        beta: f64,
        out: &mut Y,
    ) -> Result<(), ReadError> {
        if let Err(error) = self.begin() {
            if !out.is_empty() {
                out.set(0, 17.0);
            }
            return Err(error);
        }
        for i in 0..self.rows {
            let product = alpha
                * (0..self.cols)
                    .map(|j| self.values[i * self.cols + j] * x.get(j))
                    .sum::<f64>();
            out.set(
                i,
                if beta == 0.0 {
                    product
                } else {
                    product + beta * out.get(i)
                },
            );
        }
        Ok(())
    }
}

impl<X: VectorView<f64>, Y: VectorViewMut<f64>> lazymatrix::MatTransposeVecScaledInto<X, Y, f64>
    for Source
{
    fn mat_transpose_vec_scaled_into(
        &self,
        alpha: f64,
        x: &X,
        beta: f64,
        out: &mut Y,
    ) -> Result<(), ReadError> {
        if let Err(error) = self.begin() {
            if !out.is_empty() {
                out.set(0, 17.0);
            }
            return Err(error);
        }
        for j in 0..self.cols {
            let product = alpha
                * (0..self.rows)
                    .map(|i| self.values[i * self.cols + j] * x.get(i))
                    .sum::<f64>();
            out.set(
                j,
                if beta == 0.0 {
                    product
                } else {
                    product + beta * out.get(j)
                },
            );
        }
        Ok(())
    }
}

#[test]
fn fused_intercept_skips_input_scratch_and_predictors_for_zero_alpha() {
    use lazymatrix::{MatTransposeVecScaledInto, MatVecScaledInto, VectorOwned};
    struct Unreadable(usize);
    impl VectorView<f64> for Unreadable {
        fn len(&self) -> usize {
            self.0
        }
        fn get(&self, _: usize) -> f64 {
            panic!("input must not be read");
        }
    }
    impl VectorOwned<f64> for Unreadable {
        type Owned = Vec<f64>;
        fn owned_from_fn(_: usize, _: impl FnMut(usize) -> f64) -> Vec<f64> {
            panic!("scratch must not be allocated");
        }
    }
    struct WriteOnly(Vec<f64>);
    impl VectorView<f64> for WriteOnly {
        fn len(&self) -> usize {
            self.0.len()
        }
        fn get(&self, _: usize) -> f64 {
            panic!("old output must not be read");
        }
    }
    impl VectorViewMut<f64> for WriteOnly {
        fn set(&mut self, index: usize, value: f64) {
            self.0[index] = value;
        }
    }
    impl VectorOwned<f64> for WriteOnly {
        type Owned = Vec<f64>;
        fn owned_from_fn(_: usize, _: impl FnMut(usize) -> f64) -> Vec<f64> {
            panic!("scratch must not be allocated");
        }
    }
    let matrix = WithIntercept::<_, f64>::new(Source::new(2, 2, vec![f64::NAN; 4]));
    matrix.as_inner().fail.set(true);
    let mut out = vec![f64::NAN; 2];
    matrix
        .matvec_scaled_into(-0.0, &Unreadable(3), 0.0, &mut out)
        .unwrap();
    assert_eq!(out, [0.0; 2]);
    let mut transpose = vec![3.0; 3];
    matrix
        .mat_transpose_vec_scaled_into(0.0, &Unreadable(2), -2.0, &mut transpose)
        .unwrap();
    assert_eq!(transpose, [-6.0; 3]);
    let mut scratch = vec![13.0; 2];
    matrix
        .matvec_scaled_with_workspace(0.0, &Unreadable(3), 1.0, &mut out, &mut scratch)
        .unwrap();
    matrix
        .mat_transpose_vec_scaled_with_workspace(
            0.0,
            &Unreadable(2),
            1.0,
            &mut transpose,
            &mut scratch,
        )
        .unwrap();
    assert_eq!(scratch, [13.0; 2]);
    assert_eq!(matrix.as_inner().calls.get(), 0);
    let mut write_only = WriteOnly(vec![f64::NAN; 3]);
    matrix
        .mat_transpose_vec_scaled_into(0.0, &Unreadable(2), 0.0, &mut write_only)
        .unwrap();
    assert_eq!(write_only.0, [0.0; 3]);
    matrix.as_inner().fail.set(false);
    let mut scratch = vec![f64::NAN; 2];
    matrix
        .mat_transpose_vec_scaled_with_workspace(
            1.0,
            &vec![0.0; 2],
            0.0,
            &mut write_only,
            &mut scratch,
        )
        .unwrap();
    assert_eq!(write_only.0[0], 0.0);
    assert!(write_only.0[1..].iter().all(|v| v.is_nan()));
}

#[test]
fn fused_intercept_does_not_correct_partial_outputs_after_errors() {
    let matrix = WithIntercept::<_, f64>::new(Source::new(2, 2, vec![1.0; 4]));
    matrix.as_inner().fail.set(true);
    let mut out = vec![11.0, 13.0];
    let mut scratch = vec![0.0; 2];
    assert_eq!(
        matrix.matvec_scaled_with_workspace(2.0, &vec![3.0, 2.0, 4.0], 1.0, &mut out, &mut scratch),
        Err(ReadError)
    );
    assert_eq!(out, [17.0, 13.0]);
    let mut transpose = vec![11.0, 13.0, 15.0];
    assert_eq!(
        matrix.mat_transpose_vec_scaled_with_workspace(
            2.0,
            &vec![2.0, 4.0],
            1.0,
            &mut transpose,
            &mut scratch
        ),
        Err(ReadError)
    );
    assert_eq!(transpose, [11.0, 13.0, 15.0]);
    assert_eq!(scratch[0], 17.0);
}

#[test]
fn fused_intercept_only_operator_skips_predictors_and_preserves_nonfinite_values() {
    use lazymatrix::{MatTransposeVecScaledInto, MatVecScaledInto};
    let matrix = WithIntercept::<_, f64>::new(Source::new(2, 0, vec![]));
    matrix.as_inner().fail.set(true);
    let mut out = vec![f64::NAN; 2];
    matrix
        .matvec_scaled_into(f64::INFINITY, &vec![2.0], 0.0, &mut out)
        .unwrap();
    assert_eq!(out, [f64::INFINITY; 2]);
    let mut transpose = vec![f64::NAN];
    matrix
        .mat_transpose_vec_scaled_into(2.0, &vec![2.0, 4.0], 0.0, &mut transpose)
        .unwrap();
    assert_eq!(transpose, [12.0]);
    assert_eq!(matrix.as_inner().calls.get(), 0);
}
impl WeightedGramKernel<f64> for Source {
    fn weighted_gram_normalized_into<W, O>(
        &self,
        weights: &W,
        c: Option<&[f64]>,
        s: Option<&[f64]>,
        out: &mut O,
    ) -> Result<(), ReadError>
    where
        W: VectorView<f64> + ?Sized,
        O: MatrixWrite<f64> + ?Sized,
    {
        if self.cols > 0 {
            out.set(0, 0, 17.0);
        }
        self.begin()?;
        for j in 0..self.cols {
            for k in 0..self.cols {
                out.set(
                    j,
                    k,
                    (0..self.rows)
                        .map(|i| {
                            (self.logical(i, j, c, s) * weights.get(i)) * self.logical(i, k, c, s)
                        })
                        .sum(),
                );
            }
        }
        Ok(())
    }
}
impl WeightedColumnSumsKernel<f64> for Source {
    fn weighted_column_sums_normalized_into<W, O>(
        &self,
        weights: &W,
        c: Option<&[f64]>,
        s: Option<&[f64]>,
        out: &mut O,
    ) -> Result<(), ReadError>
    where
        W: VectorView<f64> + ?Sized,
        O: VectorViewMut<f64> + ?Sized,
    {
        self.begin()?;
        if self.fail_sums.get() {
            return Err(ReadError);
        }
        for j in 0..self.cols {
            out.set(
                j,
                (0..self.rows)
                    .map(|i| self.logical(i, j, c, s) * weights.get(i))
                    .sum(),
            );
        }
        Ok(())
    }
}

#[derive(Debug, PartialEq)]
struct Output([[f64; 3]; 3]);
impl MatrixShape for Output {
    fn nrows(&self) -> usize {
        3
    }
    fn ncols(&self) -> usize {
        3
    }
}
impl MatrixWrite<f64> for Output {
    fn set(&mut self, i: usize, j: usize, value: f64) {
        self.0[i][j] = value;
    }
}

#[test]
fn products_preserve_intercept_after_normalization_and_forward_errors() {
    let source = Source::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let lazy = LazyMatrix::from_parts(&source, Some(vec![1.0, 2.0]), Some(vec![2.0, 4.0]));
    let matrix = WithIntercept::new(&lazy);
    assert_eq!((matrix.nrows(), matrix.ncols()), (2, 3));
    assert!(std::ptr::eq(*matrix.as_inner(), &lazy));
    assert_eq!(matrix.matvec(&vec![3.0, 2.0, 4.0]).unwrap(), [3.0, 7.0]);
    assert_eq!(
        matrix.mat_transpose_vec(&vec![2.0, 4.0]).unwrap(),
        [6.0, 4.0, 2.0]
    );
    source.fail.set(true);
    let mut out = vec![99.0; 2];
    assert_eq!(
        matrix.matvec_into(&vec![3.0, 2.0, 4.0], &mut out),
        Err(ReadError)
    );
    assert_eq!(out, [17.0, 99.0]);
    let mut transpose = vec![99.0; 3];
    assert_eq!(
        matrix.mat_transpose_vec_into(&vec![2.0, 4.0], &mut transpose),
        Err(ReadError)
    );
    assert_eq!(transpose[0], 99.0);
    assert_eq!(matrix.matvec(&vec![3.0, 2.0, 4.0]), Err(ReadError));
    assert_eq!(matrix.mat_transpose_vec(&vec![2.0, 4.0]), Err(ReadError));
    source.fail.set(false);
    matrix.matvec_into(&vec![3.0, 2.0, 4.0], &mut out).unwrap();
    matrix
        .mat_transpose_vec_into(&vec![2.0, 4.0], &mut transpose)
        .unwrap();
    assert_eq!(out, [3.0, 7.0]);
    assert_eq!(transpose, [6.0, 4.0, 2.0]);
    assert!(std::ptr::eq(matrix.into_inner(), &lazy));
}

#[test]
fn invalid_dimensions_panic_before_backend_calls_or_writes() {
    let source = Source::new(2, 2, vec![1.0; 4]);
    let matrix = WithIntercept::<_, f64>::new(&source);
    for (x, size) in [(vec![1.0; 2], 2), (vec![1.0; 3], 3)] {
        let mut out = vec![99.0; size];
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                matrix.matvec_into(&x, &mut out).unwrap();
            }))
            .is_err()
        );
        assert!(out.iter().all(|&x| x == 99.0));
    }
    for (x, size) in [(vec![1.0; 3], 3), (vec![1.0; 2], 2)] {
        let mut out = vec![99.0; size];
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                matrix.mat_transpose_vec_into(&x, &mut out).unwrap();
            }))
            .is_err()
        );
        assert!(out.iter().all(|&x| x == 99.0));
    }
    assert_eq!(source.calls.get(), 0);
}

#[test]
fn gram_preserves_intercept_until_both_predictor_operations_succeed() {
    let source = Source::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let matrix = WithIntercept::new(LazyMatrix::from_parts(&source, Some(vec![1.0, 2.0]), None));
    let mut out = Output([[99.0; 3]; 3]);
    source.fail.set(true);
    assert_eq!(
        matrix.weighted_gram_into(&[1.0, 2.0], &mut out),
        Err(ReadError)
    );
    assert_eq!(out.0[0], [99.0; 3]);
    assert_eq!(out.0[1][1], 17.0);
    assert_eq!(source.calls.get(), 1);
    source.fail.set(false);
    source.fail_sums.set(true);
    assert_eq!(
        matrix.weighted_gram_into(&[1.0, 2.0], &mut out),
        Err(ReadError)
    );
    assert_eq!(out.0[0], [99.0; 3]);
    source.fail_sums.set(false);
    matrix.weighted_gram_into(&[1.0, 2.0], &mut out).unwrap();
    assert_eq!(out.0, [[3.0, 4.0, 4.0], [4.0, 8.0, 8.0], [4.0, 8.0, 8.0]]);
    let calls = source.calls.get();
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            matrix.weighted_gram_into(&[1.0], &mut out).unwrap();
        }))
        .is_err()
    );
    assert_eq!(source.calls.get(), calls);
    let nested = WithIntercept::new(&matrix);
    let previous = out.0;
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            nested.weighted_gram_into(&[1.0, 2.0], &mut out).unwrap();
        }))
        .is_err()
    );
    assert_eq!(out.0, previous);
    assert_eq!(source.calls.get(), calls);
    let mut sums = vec![f64::NAN; 4];
    nested
        .weighted_column_sums_into(&[1.0, 2.0], &mut sums)
        .unwrap();
    assert_eq!(sums, [3.0, 3.0, 4.0, 4.0]);
}

#[test]
fn intercept_only_and_empty_products_use_empty_sums() {
    let matrix = WithIntercept::<_, f64>::new(Source::new(3, 0, vec![]));
    assert_eq!(matrix.matvec(&vec![2.0]).unwrap(), [2.0; 3]);
    assert_eq!(
        matrix.mat_transpose_vec(&vec![1.0, -2.0, 3.0]).unwrap(),
        [2.0]
    );
    let matrix = WithIntercept::<_, f64>::new(Source::new(0, 2, vec![]));
    assert!(matrix.matvec(&vec![2.0, 3.0, 4.0]).unwrap().is_empty());
    assert_eq!(matrix.mat_transpose_vec(&vec![]).unwrap(), [0.0; 3]);
}
