#![forbid(unsafe_code)]

use std::convert::Infallible;

use lazymatrix::{
    Centering, ColumnNormalization, ColumnNormalizationParams, ColumnStats, LazyMatrix,
    MatTransposeVec, MatVec, MatrixShape, QuantileRange, Scaling, SprsCsc,
};
use ndarray::{Array1, Array2, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use sprs::CsMat;

enum Storage {
    Dense(LazyMatrix<Array2<f64>>),
    Sparse(LazyMatrix<SprsCsc<CsMat<f64>>>),
}

fn infallible<T>(result: Result<T, Infallible>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => match error {},
    }
}

type Specification = (String, String, usize, (f64, f64));

fn normalized<M>(
    raw: M,
    specification: Option<Specification>,
    parameters: Option<&Params>,
) -> PyResult<LazyMatrix<M>>
where
    M: MatrixShape + ColumnStats<f64, Error = Infallible>,
{
    match (specification, parameters) {
        (None, Some(params)) => {
            if raw.ncols() != params.inner.ncols() {
                return Err(PyValueError::new_err(
                    "normalization column count must equal ncols",
                ));
            }
            Ok(LazyMatrix::from_normalization(raw, params.inner.clone()))
        }
        (Some((center, scale, df, (lower, upper))), None) => {
            let center = match center.as_str() {
                "none" => Centering::None,
                "mean" => Centering::Mean,
                "min" => Centering::Min,
                "median" => Centering::Median,
                _ => return Err(PyValueError::new_err("unknown centering method")),
            };
            if !(lower.is_finite()
                && upper.is_finite()
                && 0.0 <= lower
                && lower < upper
                && upper <= 1.0)
            {
                return Err(PyValueError::new_err(
                    "quantile bounds must satisfy 0 <= lower < upper <= 1",
                ));
            }
            let scale = match scale.as_str() {
                "none" => Scaling::None,
                "sd" => Scaling::Sd { df },
                "range" => Scaling::Range,
                "quantile_range" => Scaling::QuantileRange(QuantileRange::new(lower, upper)),
                "maxabs" => Scaling::MaxAbs,
                "l1" => Scaling::L1,
                "l2" => Scaling::L2,
                _ => return Err(PyValueError::new_err("unknown scaling method")),
            };
            Ok(infallible(LazyMatrix::new(
                raw,
                ColumnNormalization::new(center, scale),
            )))
        }
        _ => Err(PyValueError::new_err(
            "supply either a specification or fitted parameters",
        )),
    }
}

#[pyclass(frozen, module = "lazymatrix._core")]
struct Params {
    inner: ColumnNormalizationParams,
}

#[pymethods]
impl Params {
    #[new]
    fn new(
        ncols: usize,
        centers: Option<PyReadonlyArray1<'_, f64>>,
        scales: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<Self> {
        let copy =
            |array: PyReadonlyArray1<'_, f64>| array.as_array().iter().copied().collect::<Vec<_>>();
        let centers = centers.map(copy);
        let scales = scales.map(copy);
        for (name, values) in [("centers", &centers), ("scales", &scales)] {
            if values.as_ref().is_some_and(|values| values.len() != ncols) {
                return Err(PyValueError::new_err(format!(
                    "{name} must have length ncols"
                )));
            }
        }
        if let Some(values) = &scales
            && let Some(index) = values.iter().position(|&value| value == 0.0)
        {
            return Err(PyValueError::new_err(format!(
                "scale at column {index} must be nonzero"
            )));
        }
        Ok(Self {
            inner: ColumnNormalizationParams::from_parts(ncols, centers, scales),
        })
    }

    #[getter]
    fn ncols(&self) -> usize {
        self.inner.ncols()
    }

    #[getter]
    fn centers<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .centers()
            .map(|values| values.to_vec().into_pyarray(py))
    }

    #[getter]
    fn scales<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .scales()
            .map(|values| values.to_vec().into_pyarray(py))
    }
}

fn validate_rows(nrows: usize) -> PyResult<()> {
    if nrows == 0 {
        return Err(PyValueError::new_err("input must have at least one row"));
    }
    Ok(())
}

impl Storage {
    fn shape(&self) -> (usize, usize) {
        match self {
            Self::Dense(matrix) => (matrix.nrows(), matrix.ncols()),
            Self::Sparse(matrix) => (matrix.nrows(), matrix.ncols()),
        }
    }

    fn normalization(&self) -> &ColumnNormalizationParams {
        match self {
            Self::Dense(matrix) => matrix.normalization(),
            Self::Sparse(matrix) => matrix.normalization(),
        }
    }

    fn product(&self, vector: &Array1<f64>, transpose: bool) -> Array1<f64> {
        match (self, transpose) {
            (Self::Dense(matrix), false) => infallible(matrix.matvec(vector)),
            (Self::Dense(matrix), true) => infallible(matrix.mat_transpose_vec(vector)),
            (Self::Sparse(matrix), false) => infallible(matrix.matvec(vector)),
            (Self::Sparse(matrix), true) => infallible(matrix.mat_transpose_vec(vector)),
        }
    }

    fn column(&self, index: usize, transpose: bool) -> Array1<f64> {
        let (nrows, ncols) = self.shape();
        let raw = match (self, transpose) {
            (Self::Dense(matrix), false) => {
                matrix.data().column(index).to_owned().insert_axis(Axis(1))
            }
            (Self::Dense(matrix), true) => matrix.data().row(index).to_owned().insert_axis(Axis(0)),
            (Self::Sparse(matrix), false) => {
                let mut raw = Array2::zeros((nrows, 1));
                for (row, &value) in matrix.data().as_inner().outer_view(index).unwrap().iter() {
                    raw[[row, 0]] = value;
                }
                raw
            }
            (Self::Sparse(matrix), true) => Array2::from_shape_fn((1, ncols), |(_, column)| {
                matrix
                    .data()
                    .as_inner()
                    .get(index, column)
                    .copied()
                    .unwrap_or(0.0)
            }),
        };
        let fitted = self.normalization();
        let params = if transpose {
            fitted.clone()
        } else {
            ColumnNormalizationParams::from_parts(
                1,
                fitted.centers().map(|values| vec![values[index]]),
                fitted.scales().map(|values| vec![values[index]]),
            )
        };
        // Normalize only the gathered row or column, preserving the core's IEEE semantics.
        let (values, _) = LazyMatrix::from_normalization(raw, params)
            .into_eager()
            .into_parts();
        let len = if transpose { ncols } else { nrows };
        values.into_shape_with_order(len).unwrap()
    }

    fn dense(&self) -> Array2<f64> {
        match self {
            Self::Dense(matrix) => infallible(matrix.to_eager::<Array2<f64>>()).into_parts().0,
            Self::Sparse(matrix) => infallible(matrix.to_eager::<Array2<f64>>()).into_parts().0,
        }
    }
}

#[pyclass(frozen, name = "Matrix", module = "lazymatrix._core")]
struct Matrix {
    storage: Storage,
}

#[pymethods]
impl Matrix {
    #[staticmethod]
    fn from_dense(
        py: Python<'_>,
        values: PyReadonlyArray2<'_, f64>,
        specification: Option<Specification>,
        parameters: Option<&Params>,
    ) -> PyResult<Self> {
        let raw = values.as_array().to_owned();
        validate_rows(raw.nrows())?;
        py.detach(move || {
            Ok(Self {
                storage: Storage::Dense(normalized(raw, specification, parameters)?),
            })
        })
    }

    #[staticmethod]
    fn from_csc(
        py: Python<'_>,
        shape: (usize, usize),
        values: PyReadonlyArray1<'_, f64>,
        indices: PyReadonlyArray1<'_, i64>,
        indptr: PyReadonlyArray1<'_, i64>,
        specification: Option<Specification>,
        parameters: Option<&Params>,
    ) -> PyResult<Self> {
        validate_rows(shape.0)?;
        let checked_indices = |array: PyReadonlyArray1<'_, i64>| -> PyResult<Vec<usize>> {
            array
                .as_array()
                .iter()
                .map(|&index| {
                    usize::try_from(index).map_err(|_| {
                        PyValueError::new_err(
                            "sparse indices must fit nonnegative platform integers",
                        )
                    })
                })
                .collect()
        };
        let indices = checked_indices(indices)?;
        let indptr = checked_indices(indptr)?;
        let values = values.as_array().iter().copied().collect();
        let raw = CsMat::try_new_csc(shape, indptr, indices, values)
            .map_err(|(_, _, _, error)| PyValueError::new_err(error.to_string()))?;
        py.detach(move || {
            Ok(Self {
                storage: Storage::Sparse(normalized(
                    SprsCsc::try_new(raw).unwrap(),
                    specification,
                    parameters,
                )?),
            })
        })
    }

    #[getter]
    fn normalization(&self) -> Params {
        Params {
            inner: self.storage.normalization().clone(),
        }
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.storage.shape()
    }

    #[getter]
    fn centers<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.storage
            .normalization()
            .centers()
            .map(|values| values.to_vec().into_pyarray(py))
    }

    #[getter]
    fn scales<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.storage
            .normalization()
            .scales()
            .map(|values| values.to_vec().into_pyarray(py))
    }

    fn matvec<'py>(
        &self,
        py: Python<'py>,
        values: PyReadonlyArray1<'_, f64>,
        transpose: bool,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (nrows, ncols) = self.storage.shape();
        let vector = values.as_array().to_owned();
        if vector.len() != if transpose { nrows } else { ncols } {
            return Err(PyValueError::new_err("incompatible product dimensions"));
        }
        Ok(py
            .detach(|| self.storage.product(&vector, transpose))
            .into_pyarray(py))
    }

    fn matmat<'py>(
        &self,
        py: Python<'py>,
        values: PyReadonlyArray2<'_, f64>,
        transpose: bool,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (nrows, ncols) = self.storage.shape();
        let values = values.as_array().to_owned();
        let (input, output) = if transpose {
            (nrows, ncols)
        } else {
            (ncols, nrows)
        };
        if values.nrows() != input {
            return Err(PyValueError::new_err("incompatible product dimensions"));
        }
        Ok(py
            .detach(|| {
                let mut result = Array2::zeros((output, values.ncols()));
                for (index, column) in values.columns().into_iter().enumerate() {
                    result
                        .column_mut(index)
                        .assign(&self.storage.product(&column.to_owned(), transpose));
                }
                result
            })
            .into_pyarray(py))
    }

    fn column<'py>(
        &self,
        py: Python<'py>,
        index: usize,
        transpose: bool,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (nrows, ncols) = self.storage.shape();
        if index >= if transpose { nrows } else { ncols } {
            return Err(PyIndexError::new_err("column index out of bounds"));
        }
        Ok(py
            .detach(|| self.storage.column(index, transpose))
            .into_pyarray(py))
    }

    fn toarray<'py>(&self, py: Python<'py>, transpose: bool) -> Bound<'py, PyArray2<f64>> {
        py.detach(|| {
            let values = self.storage.dense();
            if transpose {
                values.reversed_axes()
            } else {
                values
            }
        })
        .into_pyarray(py)
    }
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Params>()?;
    module.add_class::<Matrix>()
}
