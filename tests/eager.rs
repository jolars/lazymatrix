use std::convert::Infallible;

use lazymatrix::{
    ColumnNormalizationParams, DenseNormalize, EagerMatrix, LazyMatrix, MatVec, MaterializeDense,
    MatrixErrorType, MatrixOwned, MatrixShape, MatrixWrite,
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
    fn forward<M: MatVec<Vec<f64>>, P>(
        matrix: &EagerMatrix<M, P>,
        coefficients: &Vec<f64>,
    ) -> Result<Vec<f64>, M::Error> {
        matrix.matvec(coefficients)
    }

    let source = source();
    let lazy = LazyMatrix::from_normalization(
        &source,
        lazymatrix::ColumnNormalizationParams::from_parts(
            2,
            Some(vec![2.0, 4.0]),
            Some(vec![2.0, -2.0]),
        ),
    );
    let eager: EagerMatrix<Dense> = lazy.to_eager().unwrap();
    assert_eq!(eager.data().values, [0.0, -0.0, 2.0, -2.0]);
    assert_eq!(
        eager.normalization().centers(),
        lazy.normalization().centers()
    );
    assert_eq!(
        eager.normalization().scales(),
        lazy.normalization().scales()
    );
    assert_eq!(forward(&eager, &vec![1.0, 2.0]).unwrap(), [0.0, -2.0]);
    assert_eq!(source.values, [2.0, 4.0, 6.0, 8.0]);
    let prediction = LazyMatrix::from_normalization(&source, eager.normalization().clone());
    assert_eq!(prediction.matvec(&vec![1.0, 2.0]).unwrap(), [0.0, -2.0]);
}

#[test]
fn consuming_conversion_reuses_dense_storage() {
    let source = source();
    let pointer = source.values.as_ptr();
    let centers = vec![2.0, 4.0];
    let center_ptr = centers.as_ptr();
    let eager = LazyMatrix::from_normalization(
        source,
        ColumnNormalizationParams::from_parts(2, Some(centers), None),
    )
    .into_eager();
    assert_eq!(eager.data().values.as_ptr(), pointer);
    assert_eq!(eager.data().values, [0.0, 0.0, 4.0, 4.0]);
    let (data, params) = eager.into_parts();
    assert_eq!(data.values.as_ptr(), pointer);
    assert_eq!(params.centers().unwrap().as_ptr(), center_ptr);
    assert_eq!(params.centers().unwrap(), &[2.0, 4.0]);
}

#[test]
fn reusable_output_provides_a_working_borrowed_operator() {
    let lazy = LazyMatrix::from_normalization(
        source(),
        lazymatrix::ColumnNormalizationParams::from_parts(2, None, Some(vec![2.0, -2.0])),
    );
    let mut destination = Dense::zeros(2, 2);
    let eager = lazy.to_eager_into(&mut destination).unwrap();
    assert_eq!(eager.matvec(&vec![1.0, 2.0]).unwrap(), [-3.0, -5.0]);
    assert_eq!(eager.normalization().scales(), Some(&[2.0, -2.0][..]));
}

#[test]
fn output_dimensions_are_checked_before_writing() {
    let lazy = {
        let normalization_data = source();
        let normalization_params = lazymatrix::ColumnNormalizationParams::<f64>::from_parts(
            lazymatrix::MatrixShape::ncols(&normalization_data),
            None,
            None,
        );
        LazyMatrix::from_normalization(normalization_data, normalization_params)
    };
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
    let params = ColumnNormalizationParams::<f64>::from_parts(3, None, None);
    assert!(
        std::panic::catch_unwind(|| { LazyMatrix::from_normalization(source(), params) }).is_err()
    );
}

#[test]
fn reusable_output_wrapper_outlives_its_source() {
    let mut destination = Dense::zeros(2, 2);
    let eager = {
        let lazy = LazyMatrix::from_normalization(
            source(),
            lazymatrix::ColumnNormalizationParams::from_parts(2, Some(vec![2.0, 4.0]), None),
        );
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
    let _ = {
        let normalization_data = source;
        let normalization_params = lazymatrix::ColumnNormalizationParams::<f64>::from_parts(
            lazymatrix::MatrixShape::ncols(&normalization_data),
            None,
            None,
        );
        LazyMatrix::from_normalization(normalization_data, normalization_params)
    }
    .to_eager::<Dense>();
}
