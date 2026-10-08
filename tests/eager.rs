use std::convert::Infallible;

use lazymatrix::{
    DenseNormalize, EagerMatrix, LazyMatrix, MatVec, MaterializeDense, MatrixErrorType,
    MatrixOwned, MatrixShape, MatrixWrite, NormalizationParams,
};

#[derive(Clone, Debug)]
struct Dense {
    rows: usize,
    columns: usize,
    values: Vec<f64>,
}

impl MatrixShape for Dense {
    fn nrows(&self) -> usize {
        self.rows
    }
    fn ncols(&self) -> usize {
        self.columns
    }
}

impl MatrixErrorType for Dense {
    type Error = Infallible;
}

impl MatrixWrite<f64> for Dense {
    fn set(&mut self, row: usize, column: usize, value: f64) {
        self.values[row * self.columns + column] = value;
    }
}

impl MatrixOwned<f64> for Dense {
    fn zeros(rows: usize, columns: usize) -> Self {
        Self {
            rows,
            columns,
            values: vec![0.0; rows * columns],
        }
    }
}

impl MaterializeDense<f64> for Dense {
    fn materialize_normalized_into<O: MatrixWrite<f64> + ?Sized>(
        &self,
        centers: Option<&[f64]>,
        scales: Option<&[f64]>,
        out: &mut O,
    ) -> Result<(), Infallible> {
        for row in 0..self.rows {
            for col in 0..self.columns {
                let mut value = self.values[row * self.columns + col];
                if let Some(c) = centers {
                    value -= c[col];
                }
                if let Some(s) = scales {
                    value /= s[col];
                }
                out.set(row, col, value);
            }
        }
        Ok(())
    }
}

impl DenseNormalize<f64> for Dense {
    fn normalize_in_place(&mut self, centers: Option<&[f64]>, scales: Option<&[f64]>) {
        for row in self.values.chunks_mut(self.columns) {
            for (col, value) in row.iter_mut().enumerate() {
                if let Some(c) = centers {
                    *value -= c[col];
                }
                if let Some(s) = scales {
                    *value /= s[col];
                }
            }
        }
    }
}

impl MatVec<Vec<f64>> for Dense {
    fn matvec(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(self
            .values
            .chunks(self.columns)
            .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
            .collect())
    }
}

fn source() -> Dense {
    Dense {
        rows: 2,
        columns: 2,
        values: vec![2.0, 4.0, 6.0, 8.0],
    }
}

#[test]
fn materialization_retains_parameters_without_applying_them_twice() {
    fn forward<M: MatVec<Vec<f64>>, F>(
        matrix: &EagerMatrix<M, F>,
        coefficients: &Vec<f64>,
    ) -> Result<Vec<f64>, M::Error> {
        matrix.matvec(coefficients)
    }

    let source = source();
    let lazy = LazyMatrix::from_parts(&source, Some(vec![2.0, 4.0]), Some(vec![2.0, -2.0]));
    let eager: EagerMatrix<Dense> = lazy.to_eager().unwrap();
    assert_eq!(eager.data().values, [0.0, -0.0, 2.0, -2.0]);
    assert_eq!(eager.centers(), lazy.centers());
    assert_eq!(eager.scales(), lazy.scales());
    assert_eq!(forward(&eager, &vec![1.0, 2.0]).unwrap(), [0.0, -2.0]);
    assert_eq!(source.values, [2.0, 4.0, 6.0, 8.0]);
    let prediction = LazyMatrix::from_normalization(&source, eager.normalization().clone());
    assert_eq!(prediction.matvec(&vec![1.0, 2.0]).unwrap(), [0.0, -2.0]);
}

#[test]
fn consuming_conversion_reuses_dense_storage() {
    let source = source();
    let pointer = source.values.as_ptr();
    let eager = LazyMatrix::with_centers(source, vec![2.0, 4.0]).into_eager();
    assert_eq!(eager.data().values.as_ptr(), pointer);
    assert_eq!(eager.data().values, [0.0, 0.0, 4.0, 4.0]);
}

#[test]
fn reusable_output_provides_a_working_borrowed_operator() {
    let lazy = LazyMatrix::with_scales(source(), vec![2.0, -2.0]);
    let mut destination = Dense::zeros(2, 2);
    let eager = lazy.to_eager_into(&mut destination).unwrap();
    assert_eq!(eager.matvec(&vec![1.0, 2.0]).unwrap(), [-3.0, -5.0]);
    assert_eq!(eager.scales(), Some(&[2.0, -2.0][..]));
}

#[test]
fn output_dimensions_are_checked_before_writing() {
    let lazy = LazyMatrix::<_, f64>::from_parts(source(), None, None);
    let mut destination = Dense::zeros(1, 2);
    destination.values.fill(99.0);
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = lazy.to_eager_into(&mut destination);
    }));
    assert!(panic.is_err());
    assert_eq!(destination.values, [99.0, 99.0]);
}

#[test]
fn fitted_parameters_check_shape_even_when_both_axes_are_inactive() {
    let params = NormalizationParams::<f64>::from_parts(3, None, None);
    assert!(
        std::panic::catch_unwind(|| { LazyMatrix::from_normalization(source(), params) }).is_err()
    );
}

#[test]
fn reusable_output_wrapper_outlives_its_source() {
    let mut destination = Dense::zeros(2, 2);
    let eager = {
        let lazy = LazyMatrix::with_centers(source(), vec![2.0, 4.0]);
        lazy.to_eager_into(&mut destination).unwrap()
    };
    assert_eq!(eager.matvec(&vec![1.0, 1.0]).unwrap(), [0.0, 8.0]);
}

#[test]
#[should_panic(expected = "dense allocation size overflow")]
fn allocating_conversion_checks_size_before_allocating_or_reading() {
    let source = Dense {
        rows: usize::MAX,
        columns: 2,
        values: vec![],
    };
    let _ = LazyMatrix::<_, f64>::from_parts(source, None, None).to_eager::<Dense>();
}
