use std::cell::Cell;
use std::fmt;

use lazymatrix::{
    DenseBlock, LazyMatrix, MatMat, MatMatInto, MatTransposeMat, MatTransposeMatInto,
    MatrixErrorType, MatrixOwned, MatrixRead, MatrixShape, MatrixWrite, Transpose, WithIntercept,
};

#[derive(Clone, Debug, PartialEq)]
struct Dense {
    rows: usize,
    cols: usize,
    values: Vec<f64>,
}

impl Dense {
    fn new(rows: usize, cols: usize, values: &[f64]) -> Self {
        assert_eq!(values.len(), rows * cols);
        Self {
            rows,
            cols,
            values: values.to_vec(),
        }
    }
}

impl MatrixShape for Dense {
    fn nrows(&self) -> usize {
        self.rows
    }
    fn ncols(&self) -> usize {
        self.cols
    }
}
impl MatrixRead<f64> for Dense {
    fn get(&self, i: usize, j: usize) -> f64 {
        self.values[i * self.cols + j]
    }
}
impl MatrixWrite<f64> for Dense {
    fn set(&mut self, i: usize, j: usize, value: f64) {
        self.values[i * self.cols + j] = value;
    }
}
impl MatrixOwned<f64> for Dense {
    fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            values: vec![0.0; rows.checked_mul(cols).unwrap()],
        }
    }
}

#[derive(Debug, PartialEq)]
struct ReadError;
impl fmt::Display for ReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("injected failure")
    }
}
impl std::error::Error for ReadError {}

struct Source {
    data: Dense,
    fail: Cell<bool>,
    calls: Cell<usize>,
}
impl MatrixShape for Source {
    fn nrows(&self) -> usize {
        self.data.rows
    }
    fn ncols(&self) -> usize {
        self.data.cols
    }
}
impl MatrixErrorType for Source {
    type Error = ReadError;
}

impl MatMatInto<f64> for Source {
    fn matmat_into<R, O>(&self, rhs: &R, out: &mut O) -> Result<(), ReadError>
    where
        R: MatrixRead<f64> + ?Sized,
        O: MatrixRead<f64> + MatrixWrite<f64> + ?Sized,
    {
        assert_eq!(rhs.nrows(), self.ncols());
        assert_eq!((out.nrows(), out.ncols()), (self.nrows(), rhs.ncols()));
        self.calls.set(self.calls.get() + 1);
        for i in 0..out.nrows() {
            for k in 0..out.ncols() {
                out.set(
                    i,
                    k,
                    (0..self.ncols())
                        .map(|j| self.data.get(i, j) * rhs.get(j, k))
                        .sum(),
                );
                if self.fail.get() {
                    return Err(ReadError);
                }
            }
        }
        Ok(())
    }
}
impl MatTransposeMatInto<f64> for Source {
    fn mat_transpose_mat_into<R, O>(&self, rhs: &R, out: &mut O) -> Result<(), ReadError>
    where
        R: MatrixRead<f64> + ?Sized,
        O: MatrixRead<f64> + MatrixWrite<f64> + ?Sized,
    {
        assert_eq!(rhs.nrows(), self.nrows());
        assert_eq!((out.nrows(), out.ncols()), (self.ncols(), rhs.ncols()));
        self.calls.set(self.calls.get() + 1);
        for j in 0..out.nrows() {
            for k in 0..out.ncols() {
                out.set(
                    j,
                    k,
                    (0..self.nrows())
                        .map(|i| self.data.get(i, j) * rhs.get(i, k))
                        .sum(),
                );
                if self.fail.get() {
                    return Err(ReadError);
                }
            }
        }
        Ok(())
    }
}

fn source() -> Source {
    Source {
        data: Dense::new(3, 2, &[1., 4., 2., 6., 4., 10.]),
        fail: Cell::new(false),
        calls: Cell::new(0),
    }
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn allocating_and_reusable_products_apply_fitted_parameters_once() {
    let mut source = source();
    let rhs = Dense::new(2, 2, &[2., -1., -1., 3.]);
    let input = rhs.clone();
    let u = Dense::new(3, 2, &[1., 2., -2., 0., 3., -1.]);
    let lazy = LazyMatrix::from_parts(&source, Some(vec![1., 2.]), Some(vec![2., 4.]));
    let expected = Dense::new(3, 2, &[-0.5, 1.5, 0., 2.5, 1., 4.5]);
    let actual: Dense = lazy.matmat(&rhs).unwrap();
    assert_eq!(actual, expected);
    let transpose: Dense = lazy.mat_transpose_mat(&u).unwrap();
    assert_eq!(transpose, Dense::new(2, 2, &[3.5, -1.5, 4.5, -1.]));
    let mut out = Dense::new(3, 2, &[f64::NAN; 6]);
    lazy.matmat_into(&rhs, &mut out).unwrap();
    lazy.matmat_into(&rhs, &mut out).unwrap();
    assert_eq!(out, expected);
    assert_eq!(rhs, input);
    let lazy = LazyMatrix::from_parts(&mut source, Some(vec![1., 2.]), Some(vec![2., 4.]));
    lazy.matmat_into(&rhs, &mut out).unwrap();
    assert_eq!(out, expected);
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn normalization_of_a_transpose_uses_its_own_columns() {
    let source = source();
    let lazy = LazyMatrix::from_parts(&source, Some(vec![1., 2.]), Some(vec![2., 4.]));
    let outer = LazyMatrix::from_parts(
        Transpose::new(&lazy),
        Some(vec![1., 2., 3.]),
        Some(vec![2., -1., 4.]),
    );
    let forward: Dense = outer.matmat(&Dense::new(3, 1, &[2., -1., 4.])).unwrap();
    assert_eq!(forward, Dense::new(2, 1, &[-4., -2.5]));
    let transpose: Dense = outer
        .mat_transpose_mat(&Dense::new(2, 1, &[3., -2.]))
        .unwrap();
    assert_eq!(transpose, Dense::new(3, 1, &[-1., 2.5, -0.625]));
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn intercept_uses_borrowed_predictor_rows_and_a_leading_output_row() {
    let source = source();
    let lazy = LazyMatrix::from_parts(&source, Some(vec![1., 2.]), Some(vec![2., 4.]));
    let augmented = WithIntercept::new(lazy);
    let rhs = DenseBlock::new(&[10., 20., 2., -1., -1., 3.], 3, 2);
    let actual: Dense = augmented.matmat(&rhs).unwrap();
    assert_eq!(actual, Dense::new(3, 2, &[9.5, 21.5, 10., 22.5, 11., 24.5]));
    let u = DenseBlock::new(&[1., 2., -2., 0., 3., -1.], 3, 2);
    let actual: Dense = augmented.mat_transpose_mat(&u).unwrap();
    assert_eq!(actual, Dense::new(3, 2, &[2., 1., 3.5, -1.5, 4.5, -1.]));
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn failed_products_skip_normalization_and_intercept_corrections() {
    let source = source();
    source.fail.set(true);
    let lazy = LazyMatrix::from_parts(&source, Some(vec![1., 2.]), Some(vec![2., 4.]));
    let mut out = Dense::new(3, 2, &[999.; 6]);
    let rhs = DenseBlock::new(&[2., 2., 4., 4.], 2, 2);
    assert_eq!(lazy.matmat_into(&rhs, &mut out), Err(ReadError));
    assert_eq!(out.values, [5., 999., 999., 999., 999., 999.]);
    let u = DenseBlock::new(&[1.; 6], 3, 2);
    let mut transpose = Dense::new(2, 2, &[999.; 4]);
    assert_eq!(
        lazy.mat_transpose_mat_into(&u, &mut transpose),
        Err(ReadError)
    );
    assert_eq!(transpose.values, [7., 999., 999., 999.]);
    let view = Transpose::new(&lazy);
    assert_eq!(view.matmat_into(&u, &mut transpose), Err(ReadError));
    assert_eq!(transpose.values, [7., 999., 999., 999.]);
    assert_eq!(view.mat_transpose_mat_into(&rhs, &mut out), Err(ReadError));
    assert_eq!(out.values, [5., 999., 999., 999., 999., 999.]);
    assert_eq!(view.matmat::<_, Dense>(&u), Err(ReadError));
    assert_eq!(view.mat_transpose_mat::<_, Dense>(&rhs), Err(ReadError));
    let augmented = WithIntercept::new(lazy);
    let rhs = DenseBlock::new(&[10., 10., 2., 2., 4., 4.], 3, 2);
    assert_eq!(augmented.matmat_into(&rhs, &mut out), Err(ReadError));
    assert_eq!(out.values[0], 5.);
    let mut transpose = Dense::new(3, 2, &[999.; 6]);
    assert_eq!(
        augmented.mat_transpose_mat_into(&u, &mut transpose),
        Err(ReadError)
    );
    assert_eq!(transpose.values, [999., 999., 7., 999., 999., 999.]);
    source.fail.set(false);
    augmented
        .mat_transpose_mat_into(&u, &mut transpose)
        .unwrap();
    assert_eq!(transpose.values, [3., 3., 2., 2., 3.5, 3.5]);
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn validation_precedes_writes_and_empty_batches_skip_backend_calls() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let source = source();
    let lazy = LazyMatrix::from_parts(&source, Some(vec![1., 2.]), Some(vec![2., 4.]));
    let rhs = Dense::zeros(2, 0);
    let out: Dense = lazy.matmat(&rhs).unwrap();
    assert_eq!((out.rows, out.cols), (3, 0));
    let out: Dense = lazy.mat_transpose_mat(&Dense::zeros(3, 0)).unwrap();
    assert_eq!((out.rows, out.cols), (2, 0));
    assert_eq!(source.calls.get(), 0);
    let view = Transpose::new(&lazy);
    let out: Dense = view.matmat(&Dense::zeros(3, 0)).unwrap();
    assert_eq!((out.rows, out.cols), (2, 0));
    let out: Dense = view.mat_transpose_mat(&Dense::zeros(2, 0)).unwrap();
    assert_eq!((out.rows, out.cols), (3, 0));
    assert_eq!(source.calls.get(), 0);
    let mut out = Dense::new(3, 1, &[99.; 3]);
    assert!(
        catch_unwind(AssertUnwindSafe(
            || lazy.matmat_into(&Dense::zeros(3, 1), &mut out)
        ))
        .is_err()
    );
    assert_eq!(out.values, [99.; 3]);
    for (centers, scales) in [(Some(vec![0.]), None), (None, Some(vec![-0., 1.]))] {
        assert!(
            catch_unwind(AssertUnwindSafe(|| source.matmat_normalized_into(
                &Dense::zeros(2, 1),
                centers.as_deref(),
                scales.as_deref(),
                &mut out
            )))
            .is_err()
        );
        assert_eq!(out.values, [99.; 3]);
    }
    assert_eq!(source.calls.get(), 0);
}
