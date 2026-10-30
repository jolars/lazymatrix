//! Dense row normalization without a normalized design allocation.

use crate::{
    Centering, MatTransposeVec, MatTransposeVecInto, MatVec, MatVecInto, MatrixErrorType,
    MatrixRead, MatrixShape, RowNormalization, RowNormalizationParams, RowNormalizationStats,
    Scalar, Scaling, VectorOwned, VectorView, VectorViewMut,
};

/// A matrix presented with lazy row normalization `A = D⁻¹(X − r1ᵀ)`.
///
/// Each logical entry is `(X[i, j] - r[i]) / d[i]`, with independently optional
/// centering and scaling. The underlying matrix is never modified or copied.
/// Parameters occupy O(nrows) storage. Fitting uses at most one working row
/// for exact quantiles; vector products use O(1) additional workspace beyond
/// their output and normalize entries before accumulation.
///
/// Fitting and products require [`MatrixRead`], so all supported dense faer,
/// nalgebra, and ndarray matrices and their borrowed, potentially strided views
/// work. Products support owned vectors and reusable output views. Sparse and
/// storage-backed inputs, combined row and column normalization, fused and
/// batched products, borrowed logical rows and columns, and eager conversion
/// are not yet supported.
///
/// Fitted parameters describe particular rows. New samples need independently
/// fitted row parameters, rather than the row parameters of training samples.
/// [`crate::Transpose`] can transpose the fitted operator without refitting.
///
/// ```
/// # #[cfg(feature = "ndarray_v0_17")]
/// # {
/// use lazymatrix::{Centering, MatVec, RowNormalization, RowNormalizedMatrix, Scaling};
/// use ndarray::array;
///
/// let x = array![[3.0, 4.0], [0.0, 0.0]];
/// let spec = RowNormalization::new(Centering::None, Scaling::L2);
/// let normalized = RowNormalizedMatrix::new(x.view(), spec).unwrap();
/// let product = normalized.matvec(&array![1.0, 1.0]).unwrap();
/// assert_eq!(product, array![1.4, 0.0]);
/// assert_eq!(normalized.scales(), Some([5.0, 1.0].as_slice()));
/// # }
/// ```
///
/// Computed normalization requires dense indexed access:
///
/// ```compile_fail
/// use lazymatrix::{MatrixErrorType, MatrixShape, RowNormalization, RowNormalizedMatrix};
///
/// fn without_dense_access<M: MatrixShape + MatrixErrorType>(matrix: M) {
///     let _ = RowNormalizedMatrix::new(matrix, RowNormalization::standardize());
/// }
/// ```
#[derive(Clone, Debug)]
pub struct RowNormalizedMatrix<M, F = f64> {
    data: M,
    normalization: RowNormalizationParams<F>,
}

impl<M: MatrixShape, F: Scalar> RowNormalizedMatrix<M, F> {
    /// Construct from explicit row centers and/or scales.
    ///
    /// Negative scales and nonfinite values are preserved. Unlike [`Self::new`],
    /// this constructor rejects exact zero scales rather than replacing them.
    ///
    /// # Panics
    /// Panics if either vector does not have length `nrows`, or if a scale is
    /// positive or negative zero. A zero-scale panic reports the row index.
    pub fn from_parts(data: M, centers: Option<Vec<F>>, scales: Option<Vec<F>>) -> Self {
        let normalization = RowNormalizationParams::from_parts(data.nrows(), centers, scales);
        Self::from_normalization(data, normalization)
    }

    /// Wrap data using parameters fitted for the same rows, without refitting.
    ///
    /// Only the row count is checked. The caller determines whether reusing
    /// these row-specific parameters is meaningful for the supplied data.
    ///
    /// # Panics
    /// Panics if the matrix row count differs from the fitted row count.
    pub fn from_normalization(data: M, normalization: RowNormalizationParams<F>) -> Self {
        assert_eq!(
            data.nrows(),
            normalization.nrows(),
            "normalization row count must equal nrows"
        );
        Self {
            data,
            normalization,
        }
    }

    /// Wrap a matrix with explicit row centering only.
    ///
    /// # Panics
    /// Panics if `centers.len() != nrows`.
    pub fn with_centers(data: M, centers: Vec<F>) -> Self {
        Self::from_parts(data, Some(centers), None)
    }

    /// Wrap a matrix with explicit row scaling only.
    ///
    /// # Panics
    /// Panics if `scales.len() != nrows`, or if a scale is positive or negative zero.
    pub fn with_scales(data: M, scales: Vec<F>) -> Self {
        Self::from_parts(data, None, Some(scales))
    }

    /// Number of rows of the logical normalized matrix.
    pub fn nrows(&self) -> usize {
        self.data.nrows()
    }

    /// Number of columns of the logical normalized matrix.
    pub fn ncols(&self) -> usize {
        self.data.ncols()
    }

    /// Borrow the fitted parameters without copying.
    pub fn normalization(&self) -> &RowNormalizationParams<F> {
        &self.normalization
    }

    /// Fitted row centers, if centering is active.
    pub fn centers(&self) -> Option<&[F]> {
        self.normalization.centers()
    }

    /// Fitted row scales, if scaling is active.
    pub fn scales(&self) -> Option<&[F]> {
        self.normalization.scales()
    }

    /// Borrow the underlying unnormalized matrix.
    pub fn data(&self) -> &M {
        &self.data
    }

    /// Recover the underlying matrix and parameter vectors without copying.
    pub fn into_parts(self) -> (M, Option<Vec<F>>, Option<Vec<F>>) {
        let (centers, scales) = self.normalization.into_parts();
        (self.data, centers, scales)
    }
}

impl<M: MatrixRead<F> + MatrixErrorType, F: Scalar> RowNormalizedMatrix<M, F> {
    /// Fit row centers and scales independently according to `spec`.
    ///
    /// Norms use centered rows when centering is active. Standard deviations
    /// use a two-pass calculation with `ncols - df` as the variance divisor.
    /// Exact quantiles use one reusable working row and linear interpolation.
    /// Fitting takes O(nrows * ncols) time, or O(nrows * ncols * log(ncols))
    /// when sorting for quantiles, with O(nrows + ncols) storage at most.
    ///
    /// Exact zero scales become one, leaving zero rows unchanged under norm
    /// scaling. Nonfinite statistics retain their IEEE values. Zero-column
    /// rows have undefined means, standard deviations, minima, and ranges
    /// (`NaN`), and zero norms, matching column normalization along the other axis.
    ///
    /// The return type shares the backend error type for consistency with
    /// [`crate::LazyMatrix::new`]. Current dense fitting is infallible.
    pub fn new(data: M, spec: RowNormalization) -> Result<Self, M::Error> {
        let (centers, scales) = fit(&data, spec);
        Ok(Self::from_parts(data, centers, scales))
    }

    fn entry(&self, row: usize, column: usize) -> F {
        let mut value = self.data.get(row, column);
        if let Some(centers) = self.centers() {
            value = value - centers[row];
        }
        if let Some(scales) = self.scales() {
            value = value / scales[row];
        }
        value
    }
}

impl<M: MatrixShape, F> MatrixShape for RowNormalizedMatrix<M, F> {
    fn nrows(&self) -> usize {
        self.data.nrows()
    }

    fn ncols(&self) -> usize {
        self.data.ncols()
    }
}

impl<M: MatrixErrorType, F> MatrixErrorType for RowNormalizedMatrix<M, F> {
    type Error = M::Error;
}

impl<M, V, F> MatVec<V> for RowNormalizedMatrix<M, F>
where
    M: MatrixRead<F> + MatrixErrorType,
    F: Scalar,
    V: VectorOwned<F, Owned = V> + VectorViewMut<F>,
{
    fn matvec(&self, x: &V) -> Result<V, Self::Error> {
        assert_eq!(x.len(), self.ncols(), "matvec: dimension mismatch");
        let mut out = V::owned_from_fn(self.nrows(), |_| F::zero());
        self.matvec_into(x, &mut out)?;
        Ok(out)
    }
}

impl<M, X, Y, F> MatVecInto<X, Y> for RowNormalizedMatrix<M, F>
where
    M: MatrixRead<F> + MatrixErrorType,
    F: Scalar,
    X: VectorView<F>,
    Y: VectorViewMut<F>,
{
    fn matvec_into(&self, x: &X, out: &mut Y) -> Result<(), Self::Error> {
        assert_eq!(x.len(), self.ncols(), "matvec_into: dimension mismatch");
        assert_eq!(
            out.len(),
            self.nrows(),
            "matvec_into: output dimension mismatch"
        );
        for i in 0..self.nrows() {
            let value = (0..self.ncols()).map(|j| self.entry(i, j) * x.get(j)).sum();
            out.set(i, value);
        }
        Ok(())
    }
}

impl<M, V, F> MatTransposeVec<V> for RowNormalizedMatrix<M, F>
where
    M: MatrixRead<F> + MatrixErrorType,
    F: Scalar,
    V: VectorOwned<F, Owned = V> + VectorViewMut<F>,
{
    fn mat_transpose_vec(&self, x: &V) -> Result<V, Self::Error> {
        assert_eq!(
            x.len(),
            self.nrows(),
            "mat_transpose_vec: dimension mismatch"
        );
        let mut out = V::owned_from_fn(self.ncols(), |_| F::zero());
        self.mat_transpose_vec_into(x, &mut out)?;
        Ok(out)
    }
}

impl<M, X, Y, F> MatTransposeVecInto<X, Y> for RowNormalizedMatrix<M, F>
where
    M: MatrixRead<F> + MatrixErrorType,
    F: Scalar,
    X: VectorView<F>,
    Y: VectorViewMut<F>,
{
    fn mat_transpose_vec_into(&self, x: &X, out: &mut Y) -> Result<(), Self::Error> {
        assert_eq!(
            x.len(),
            self.nrows(),
            "mat_transpose_vec_into: dimension mismatch"
        );
        assert_eq!(
            out.len(),
            self.ncols(),
            "mat_transpose_vec_into: output dimension mismatch"
        );
        for j in 0..self.ncols() {
            let value = (0..self.nrows()).map(|i| self.entry(i, j) * x.get(i)).sum();
            out.set(j, value);
        }
        Ok(())
    }
}

fn row_values<M: MatrixRead<F> + ?Sized, F: Scalar>(
    matrix: &M,
    row: usize,
) -> impl Iterator<Item = F> + '_ {
    (0..matrix.ncols()).map(move |j| matrix.get(row, j))
}

fn extrema<F: Scalar>(values: impl Iterator<Item = F>) -> (F, F) {
    values
        .fold(None, |extrema: Option<(F, F)>, value| {
            Some(match extrema {
                None => (value, value),
                Some((min, _)) if min.is_nan() || value.is_nan() => (F::nan(), F::nan()),
                Some((min, max)) => (min.min(value), max.max(value)),
            })
        })
        .unwrap_or_else(|| (F::nan(), F::nan()))
}

fn fit<M: MatrixRead<F> + ?Sized, F: Scalar>(
    matrix: &M,
    spec: RowNormalization,
) -> RowNormalizationStats<F> {
    let (medians, ranges) = crate::quantiles::row_quantiles(matrix, spec);
    let mean = |i| row_values(matrix, i).sum::<F>() / F::from_usize(matrix.ncols()).unwrap();
    let centers = match spec.center {
        Centering::None => None,
        Centering::Mean => Some((0..matrix.nrows()).map(mean).collect::<Vec<_>>()),
        Centering::Min => Some(
            (0..matrix.nrows())
                .map(|i| extrema(row_values(matrix, i)).0)
                .collect(),
        ),
        Centering::Median => medians,
    };
    let scales = match spec.scale {
        Scaling::None => None,
        Scaling::QuantileRange(_) => ranges,
        scaling => Some(
            (0..matrix.nrows())
                .map(|i| {
                    let centered = || {
                        row_values(matrix, i).map(|value| {
                            centers.as_ref().map_or(value, |centers| value - centers[i])
                        })
                    };
                    match scaling {
                        Scaling::Sd { df } => {
                            let center = if spec.center == Centering::Mean {
                                centers.as_ref().unwrap()[i]
                            } else {
                                mean(i)
                            };
                            let sum = row_values(matrix, i)
                                .map(|value| {
                                    let deviation = value - center;
                                    deviation * deviation
                                })
                                .sum::<F>();
                            (sum / crate::normalization::sd_divisor::<F>(matrix.ncols(), df)).sqrt()
                        }
                        Scaling::Range => {
                            let (min, max) = extrema(row_values(matrix, i));
                            max - min
                        }
                        Scaling::MaxAbs => {
                            centered()
                                .map(|value| value.abs())
                                .fold(F::zero(), |max, value| {
                                    if max.is_nan() || value.is_nan() {
                                        F::nan()
                                    } else {
                                        max.max(value)
                                    }
                                })
                        }
                        Scaling::L1 => centered().map(|value| value.abs()).sum(),
                        Scaling::L2 => centered().map(|value| value * value).sum::<F>().sqrt(),
                        Scaling::None | Scaling::QuantileRange(_) => unreachable!(),
                    }
                })
                .collect::<Vec<_>>(),
        ),
    }
    .map(|mut scales| {
        for scale in &mut scales {
            if *scale == F::zero() {
                *scale = F::one();
            }
        }
        scales
    });
    (centers, scales)
}
