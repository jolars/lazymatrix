#![forbid(unsafe_code)]

use std::convert::Infallible;

use lazymatrix::{
    Centering, ColumnNormalization, ColumnNormalizationParams, ColumnStats, LazyMatrix,
    MatTransposeVec, MatVec, QuantileRange, RowNormalization, RowNormalizationParams, RowStats,
    Scaling,
};
use ndarray::{Array1, Array2, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use sprs::CsMat;

enum Storage {
    DenseColumns(LazyMatrix<Array2<f64>>),
    DenseRows(LazyMatrix<Array2<f64>, RowNormalizationParams>),
    SparseColumns(LazyMatrix<CsMat<f64>>),
    SparseRows(LazyMatrix<CsMat<f64>, RowNormalizationParams>),
}

macro_rules! with_matrix {
    ($storage:expr, $matrix:ident => $body:expr) => {
        match $storage {
            Storage::DenseColumns($matrix) => $body,
            Storage::DenseRows($matrix) => $body,
            Storage::SparseColumns($matrix) => $body,
            Storage::SparseRows($matrix) => $body,
        }
    };
}

#[derive(Clone)]
enum FittedParams {
    Columns(ColumnNormalizationParams),
    Rows(RowNormalizationParams),
}

fn infallible<T>(result: Result<T, Infallible>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => match error {},
    }
}

type Specification = (usize, String, String, usize, (f64, f64), bool);

fn fitted<M>(
    raw: &M,
    specification: Option<Specification>,
    parameters: Option<&Params>,
) -> PyResult<FittedParams>
where
    M: ColumnStats<f64, Error = Infallible> + RowStats<f64>,
{
    match (specification, parameters) {
        (None, Some(params)) => {
            match &params.inner {
                FittedParams::Columns(params) if raw.ncols() != params.ncols() => {
                    return Err(PyValueError::new_err(
                        "normalization column count must equal ncols",
                    ));
                }
                FittedParams::Rows(params) if raw.nrows() != params.nrows() => {
                    return Err(PyValueError::new_err(
                        "normalization row count must equal nrows",
                    ));
                }
                _ => {}
            }
            Ok(params.inner.clone())
        }
        (Some((axis, center, scale, df, (lower, upper), unit_variance)), None) => {
            validate_axis(axis)?;
            if axis == 0 && raw.nrows() == 0 {
                return Err(PyValueError::new_err("input must have at least one row"));
            }
            if axis == 1 && raw.ncols() == 0 {
                return Err(PyValueError::new_err("input must have at least one column"));
            }
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
            if unit_variance && (scale != "quantile_range" || !(0.0 < lower && upper < 1.0)) {
                return Err(PyValueError::new_err(
                    "unit_variance requires quantile-range scaling with bounds strictly inside (0, 1)",
                ));
            }
            let scale = match scale.as_str() {
                "none" => Scaling::None,
                "sd" => Scaling::Sd { df },
                "range" => Scaling::Range,
                "quantile_range" => Scaling::QuantileRange {
                    range: QuantileRange::new(lower, upper),
                    unit_variance,
                },
                "maxabs" => Scaling::MaxAbs,
                "l1" => Scaling::L1,
                "l2" => Scaling::L2,
                _ => return Err(PyValueError::new_err("unknown scaling method")),
            };
            Ok(if axis == 0 {
                FittedParams::Columns(infallible(ColumnNormalization::new(center, scale).fit(raw)))
            } else {
                FittedParams::Rows(infallible(RowNormalization::new(center, scale).fit(raw)))
            })
        }
        _ => Err(PyValueError::new_err(
            "supply either a specification or fitted parameters",
        )),
    }
}

#[pyclass(frozen, module = "lazymatrix._core")]
struct Params {
    inner: FittedParams,
}

#[pymethods]
impl Params {
    #[new]
    fn new(
        axis: usize,
        dimension: usize,
        centers: Option<PyReadonlyArray1<'_, f64>>,
        scales: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<Self> {
        validate_axis(axis)?;
        let (name, count) = if axis == 0 {
            ("column", "ncols")
        } else {
            ("row", "nrows")
        };
        let copy =
            |array: PyReadonlyArray1<'_, f64>| array.as_array().iter().copied().collect::<Vec<_>>();
        let centers = centers.map(copy);
        let scales = scales.map(copy);
        for (name, values) in [("centers", &centers), ("scales", &scales)] {
            if values
                .as_ref()
                .is_some_and(|values| values.len() != dimension)
            {
                return Err(PyValueError::new_err(format!(
                    "{name} must have length {count}"
                )));
            }
        }
        if let Some(values) = &scales
            && let Some(index) = values.iter().position(|&value| value == 0.0)
        {
            return Err(PyValueError::new_err(format!(
                "scale at {name} {index} must be nonzero"
            )));
        }
        Ok(Self {
            inner: if axis == 0 {
                FittedParams::Columns(ColumnNormalizationParams::from_parts(
                    dimension, centers, scales,
                ))
            } else {
                FittedParams::Rows(RowNormalizationParams::from_parts(
                    dimension, centers, scales,
                ))
            },
        })
    }

    #[getter]
    fn dimension(&self) -> usize {
        match &self.inner {
            FittedParams::Columns(params) => params.ncols(),
            FittedParams::Rows(params) => params.nrows(),
        }
    }

    #[getter]
    fn axis(&self) -> usize {
        match &self.inner {
            FittedParams::Columns(_) => 0,
            FittedParams::Rows(_) => 1,
        }
    }

    #[getter]
    fn centers<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        let values = match &self.inner {
            FittedParams::Columns(params) => params.centers(),
            FittedParams::Rows(params) => params.centers(),
        };
        values.map(|values| values.to_vec().into_pyarray(py))
    }

    #[getter]
    fn scales<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        let values = match &self.inner {
            FittedParams::Columns(params) => params.scales(),
            FittedParams::Rows(params) => params.scales(),
        };
        values.map(|values| values.to_vec().into_pyarray(py))
    }
}

fn validate_axis(axis: usize) -> PyResult<()> {
    if axis > 1 {
        return Err(PyValueError::new_err("axis must be 0 or 1"));
    }
    Ok(())
}

impl Storage {
    fn shape(&self) -> (usize, usize) {
        with_matrix!(self, matrix => (matrix.nrows(), matrix.ncols()))
    }

    fn normalization(&self) -> Params {
        Params {
            inner: match self {
                Self::DenseColumns(matrix) => FittedParams::Columns(matrix.normalization().clone()),
                Self::SparseColumns(matrix) => {
                    FittedParams::Columns(matrix.normalization().clone())
                }
                Self::DenseRows(matrix) => FittedParams::Rows(matrix.normalization().clone()),
                Self::SparseRows(matrix) => FittedParams::Rows(matrix.normalization().clone()),
            },
        }
    }

    fn centers(&self) -> Option<&[f64]> {
        with_matrix!(self, matrix => matrix.normalization().centers())
    }

    fn scales(&self) -> Option<&[f64]> {
        with_matrix!(self, matrix => matrix.normalization().scales())
    }

    fn axis(&self) -> usize {
        match self {
            Self::DenseColumns(_) | Self::SparseColumns(_) => 0,
            Self::DenseRows(_) | Self::SparseRows(_) => 1,
        }
    }

    fn product(&self, vector: &Array1<f64>, transpose: bool) -> Array1<f64> {
        with_matrix!(self, matrix => if transpose {
            infallible(matrix.mat_transpose_vec(vector))
        } else {
            infallible(matrix.matvec(vector))
        })
    }

    fn column(&self, index: usize, transpose: bool) -> Array1<f64> {
        let (nrows, ncols) = self.shape();
        let raw = match self {
            Self::DenseColumns(matrix) => dense_column(matrix.data(), index, transpose),
            Self::DenseRows(matrix) => dense_column(matrix.data(), index, transpose),
            Self::SparseColumns(matrix) => sparse_column(matrix.data(), index, transpose),
            Self::SparseRows(matrix) => sparse_column(matrix.data(), index, transpose),
        };
        let select = |values: Option<&[f64]>| {
            values.map(|values| {
                if (self.axis() == 0) == transpose {
                    values.to_vec()
                } else {
                    vec![values[index]]
                }
            })
        };
        // Normalize only the gathered row or column, preserving the core's IEEE semantics.
        let values = if self.axis() == 0 {
            let params = ColumnNormalizationParams::from_parts(
                raw.ncols(),
                select(self.centers()),
                select(self.scales()),
            );
            LazyMatrix::from_parts(raw, params)
                .into_eager()
                .into_parts()
                .0
        } else {
            let params = RowNormalizationParams::from_parts(
                raw.nrows(),
                select(self.centers()),
                select(self.scales()),
            );
            LazyMatrix::from_parts(raw, params)
                .into_eager()
                .into_parts()
                .0
        };
        let len = if transpose { ncols } else { nrows };
        values.into_shape_with_order(len).unwrap()
    }

    fn dense(&self) -> Array2<f64> {
        with_matrix!(self, matrix => infallible(matrix.to_eager::<Array2<f64>>()).into_parts().0)
    }
}

fn dense_column(raw: &Array2<f64>, index: usize, transpose: bool) -> Array2<f64> {
    if transpose {
        raw.row(index).to_owned().insert_axis(Axis(0))
    } else {
        raw.column(index).to_owned().insert_axis(Axis(1))
    }
}

fn sparse_column(raw: &CsMat<f64>, index: usize, transpose: bool) -> Array2<f64> {
    let shape = if transpose {
        (1, raw.cols())
    } else {
        (raw.rows(), 1)
    };
    if raw.is_csr() == transpose {
        let mut values = Array2::zeros(shape);
        for (offset, &value) in raw.outer_view(index).unwrap().iter() {
            values[if transpose { [0, offset] } else { [offset, 0] }] = value;
        }
        values
    } else {
        Array2::from_shape_fn(shape, |(row, column)| {
            let (row, column) = if transpose {
                (index, column)
            } else {
                (row, index)
            };
            raw.get(row, column).copied().unwrap_or(0.0)
        })
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
        py.detach(move || {
            let storage = match fitted(&raw, specification, parameters)? {
                FittedParams::Columns(params) => {
                    Storage::DenseColumns(LazyMatrix::from_parts(raw, params))
                }
                FittedParams::Rows(params) => {
                    Storage::DenseRows(LazyMatrix::from_parts(raw, params))
                }
            };
            Ok(Self { storage })
        })
    }

    #[staticmethod]
    fn from_sparse(
        py: Python<'_>,
        shape: (usize, usize),
        is_csc: bool,
        buffers: (
            PyReadonlyArray1<'_, f64>,
            PyReadonlyArray1<'_, i64>,
            PyReadonlyArray1<'_, i64>,
        ),
        specification: Option<Specification>,
        parameters: Option<&Params>,
    ) -> PyResult<Self> {
        let (values, indices, indptr) = buffers;
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
        let raw = if is_csc {
            CsMat::try_new_csc(shape, indptr, indices, values)
        } else {
            CsMat::try_new(shape, indptr, indices, values)
        }
        .map_err(|(_, _, _, error)| PyValueError::new_err(error.to_string()))?;
        py.detach(move || {
            let storage = match fitted(&raw, specification, parameters)? {
                FittedParams::Columns(params) => {
                    Storage::SparseColumns(LazyMatrix::from_parts(raw, params))
                }
                FittedParams::Rows(params) => {
                    Storage::SparseRows(LazyMatrix::from_parts(raw, params))
                }
            };
            Ok(Self { storage })
        })
    }

    #[getter]
    fn normalization(&self) -> Params {
        self.storage.normalization()
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.storage.shape()
    }

    #[getter]
    fn centers<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.storage
            .centers()
            .map(|values| values.to_vec().into_pyarray(py))
    }

    #[getter]
    fn scales<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.storage
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
