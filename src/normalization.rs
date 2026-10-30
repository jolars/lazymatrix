mod row;

mod private {
    pub trait Params {
        fn assert_shape(&self, nrows: usize, ncols: usize);
    }

    pub trait Recipe<M: crate::MatrixErrorType, P> {
        fn fit(self, data: &M) -> Result<P, M::Error>;
    }
}

/// Fitted normalization metadata supported by [`crate::LazyMatrix`].
///
/// The parameter type identifies the axis and scalar. This trait is sealed;
/// use [`ColumnNormalizationParams`] or [`RowNormalizationParams`] to supply
/// custom fitted values.
pub trait NormalizationParams: private::Params {
    /// Scalar used by the fitted centers and scales.
    type Scalar: crate::Scalar;
}

/// A normalization recipe supported by a backend and fitted parameter type.
///
/// [`crate::LazyMatrix::new`] selects fitting through this sealed trait. Column
/// recipes require [`crate::ColumnStats`]; row recipes require dense indexed
/// access through [`crate::MatrixRead`]. Implementing those capabilities enables
/// fitting for a downstream backend without implementing this trait directly.
pub trait NormalizationRecipe<M: crate::MatrixErrorType, P: NormalizationParams>:
    private::Recipe<M, P>
{
}

impl<M, P, R> NormalizationRecipe<M, P> for R
where
    M: crate::MatrixErrorType,
    P: NormalizationParams,
    R: private::Recipe<M, P>,
{
}

/// Computed column centers and raw scales, respectively.
///
/// An inactive normalization component is `None`. Present vectors have one
/// entry per column. Raw scales may be zero; [`crate::LazyMatrix::new`] replaces
/// exact zeros with one while preserving nonfinite values.
pub type ColumnNormalizationStats<F> = (Option<Vec<F>>, Option<Vec<F>>);

/// Computed row centers and raw scales, respectively.
///
/// An inactive component is `None`. Present vectors have one entry per row.
/// [`crate::LazyMatrix::new`] replaces exact zero scales with one
/// while preserving nonfinite values.
pub type RowNormalizationStats<F> = (Option<Vec<F>>, Option<Vec<F>>);

/// How to center each column or row.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Centering {
    /// No centering.
    #[default]
    None,
    /// Subtract the mean.
    Mean,
    /// Subtract the minimum.
    Min,
    /// Subtract the median.
    Median,
}

/// Validated probabilities defining a column's or row's quantile range.
///
/// Bounds are probabilities in `[0, 1]`, rather than percentages. The default
/// is the interquartile range, with bounds `0.25` and `0.75`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuantileRange {
    lower: f64,
    upper: f64,
}

// Private, validated bounds cannot contain NaN, so equality is an equivalence relation.
impl Eq for QuantileRange {}

impl QuantileRange {
    /// Construct a range with `0 <= lower < upper <= 1`.
    ///
    /// # Panics
    /// Panics if either bound is nonfinite or the bounds are out of order or
    /// outside `[0, 1]`.
    pub fn new(lower: f64, upper: f64) -> Self {
        assert!(
            lower.is_finite() && upper.is_finite() && 0.0 <= lower && lower < upper && upper <= 1.0,
            "quantile bounds must satisfy 0 <= lower < upper <= 1"
        );
        Self { lower, upper }
    }

    /// Lower quantile probability.
    pub fn lower(self) -> f64 {
        self.lower
    }

    /// Upper quantile probability.
    pub fn upper(self) -> f64 {
        self.upper
    }
}

impl Default for QuantileRange {
    fn default() -> Self {
        Self::new(0.25, 0.75)
    }
}

/// How to scale each column or row.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Scaling {
    /// No scaling.
    #[default]
    None,
    /// Divide by the standard deviation, using `n - df` as the variance divisor.
    ///
    /// Here `n` is `nrows` for column normalization and `ncols` for row normalization.
    /// Use `df = 0` for population standard deviation or `df = 1` for sample
    /// standard deviation. When `n <= df`, the scale is `NaN`.
    Sd {
        /// Degrees of freedom subtracted from the number of entries.
        df: usize,
    },
    /// Divide by the maximum absolute value.
    MaxAbs,
    /// Divide by the 1-norm.
    L1,
    /// Divide by the 2-norm.
    L2,
    /// Divide by the range, `max - min`.
    Range,
    /// Divide by the upper quantile minus the lower quantile.
    ///
    /// Quantiles use linear interpolation at position `(n - 1) * q`.
    QuantileRange(QuantileRange),
}

/// A column normalization specification: an independent [`Centering`] and
/// [`Scaling`] choice.
///
/// Use [`Self::standardize`], [`Self::min_max`], or [`Self::robust`] for common
/// recipes, or [`Self::new`] for a custom combination. The default performs
/// no centering or scaling.
///
/// Non-translation-invariant scales are computed from **centered** columns (see
/// [`ColumnStats::col_l1_centered`](crate::traits::ColumnStats::col_l1_centered),
/// [`ColumnStats::col_l2_centered`](crate::traits::ColumnStats::col_l2_centered),
/// and
/// [`ColumnStats::col_maxabs_centered`](crate::traits::ColumnStats::col_maxabs_centered).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ColumnNormalization {
    pub center: Centering,
    pub scale: Scaling,
}

impl ColumnNormalization {
    /// Build a specification from its centering and scaling choices.
    pub const fn new(center: Centering, scale: Scaling) -> Self {
        Self { center, scale }
    }

    /// Subtract each column's mean and divide by its population standard deviation.
    ///
    /// The variance uses `nrows` as its divisor. Fitting with
    /// [`crate::LazyMatrix::new`] replaces exact zero standard deviations with
    /// one, so constant columns become zero.
    /// For sample standard deviations, use
    /// `ColumnNormalization::new(Centering::Mean, Scaling::Sd { df: 1 })`.
    pub const fn standardize() -> Self {
        Self::new(Centering::Mean, Scaling::Sd { df: 0 })
    }

    /// Subtract each column's minimum and divide by its range.
    ///
    /// Finite, nonconstant columns span `[0, 1]` on the fitting data. Fitting
    /// replaces exact zero ranges with one, so constant columns become zero.
    /// Reusing the fitted parameters does not clip new observations to `[0, 1]`.
    pub const fn min_max() -> Self {
        Self::new(Centering::Min, Scaling::Range)
    }

    /// Subtract each column's median and divide by its interquartile range.
    ///
    /// The range uses the 25th and 75th percentiles with linear interpolation
    /// at `(nrows - 1) * q`. It has no standard-normal calibration and does not
    /// target unit variance. Fitting replaces exact zero ranges with one.
    /// Use [`Self::new`] with [`Scaling::QuantileRange`] for custom bounds.
    pub fn robust() -> Self {
        Self::new(
            Centering::Median,
            Scaling::QuantileRange(QuantileRange::default()),
        )
    }
}

/// A row normalization specification with independent [`Centering`] and
/// [`Scaling`] choices.
///
/// The default performs no centering or scaling. Norms (`L1`, `L2`, and
/// `MaxAbs`) are fitted to centered rows when centering is active. Standard
/// deviations, ranges, and quantile ranges are translation-invariant.
/// Use [`crate::LazyMatrix::new`] to fit each row independently. Fitting requires
/// [`crate::MatrixRead`] and takes O(nrows * ncols) time, or
/// O(nrows * ncols * log(ncols)) when sorting for exact linear quantiles. It uses
/// O(nrows + ncols) storage at most, including one working row for quantiles.
/// Standard deviations use a two-pass calculation with `ncols - df` as divisor.
/// Empty rows have undefined means, standard deviations, minima, and ranges
/// (`NaN`), and zero norms, matching column fitting along the other axis.
///
/// Row-normalized operators support allocating and reusable forward and
/// transpose vector products on dense matrices and strided views. Products
/// normalize entries before accumulating and use O(1) additional workspace
/// beyond their output. Sparse and storage-backed fitting, fused and batched
/// products, borrowed logical views, weighted operations, and eager conversion
/// are not yet supported. New samples require independently fitted row
/// parameters. [`crate::Transpose`] transposes the fitted operator without
/// refitting.
///
/// ```
/// # #[cfg(feature = "ndarray_v0_17")]
/// # {
/// use lazymatrix::{Centering, LazyMatrix, MatVec, RowNormalization, Scaling};
/// use ndarray::array;
///
/// let x = array![[3.0, 4.0], [0.0, 0.0]];
/// let recipe = RowNormalization::new(Centering::None, Scaling::L2);
/// let normalized = LazyMatrix::new(x.view(), recipe).unwrap();
/// let product = normalized.matvec(&array![1.0, 1.0]).unwrap();
/// assert_eq!(product, array![1.4, 0.0]);
/// assert_eq!(normalized.normalization().scales(), Some([5.0, 1.0].as_slice()));
/// # }
/// ```
///
/// Computed row normalization requires dense indexed access:
///
/// ```compile_fail
/// use lazymatrix::{LazyMatrix, MatrixErrorType, MatrixShape, RowNormalization};
/// fn without_dense_access<M: MatrixShape + MatrixErrorType>(matrix: M) {
///     let _ = LazyMatrix::new(matrix, RowNormalization::standardize());
/// }
/// ```
///
/// Row normalization does not grant fused products, even if raw storage has them:
///
/// ```compile_fail
/// use lazymatrix::{LazyMatrix, MatVecScaledInto, MatrixRead, RowNormalizationParams};
/// fn fused<M: MatrixRead<f64> + MatVecScaledInto<Vec<f64>, Vec<f64>, f64>>(
///     rows: &LazyMatrix<M, RowNormalizationParams>, x: &Vec<f64>, out: &mut Vec<f64>,
/// ) {
///     rows.matvec_scaled_into(1.0, x, 0.0, out).unwrap();
/// }
/// ```
///
/// Batched products remain unavailable:
///
/// ```compile_fail
/// use lazymatrix::{LazyMatrix, MatMatInto, MatrixRead, MatrixWrite, RowNormalizationParams};
/// fn batched<M: MatrixRead<f64> + MatMatInto<f64>, D: MatrixRead<f64> + MatrixWrite<f64>>(
///     rows: &LazyMatrix<M, RowNormalizationParams>, rhs: &D, out: &mut D,
/// ) {
///     rows.matmat_into(rhs, out).unwrap();
/// }
/// ```
///
/// Natural raw columns do not grant logical column borrowing:
///
/// ```compile_fail
/// use lazymatrix::{LazyMatrix, MatrixRead, RawColumns, RowNormalizationParams};
/// fn columns<M: MatrixRead<f64> + RawColumns<f64>>(rows: &LazyMatrix<M, RowNormalizationParams>) {
///     let _ = rows.column(0);
/// }
/// ```
///
/// Weighted operations remain unavailable:
///
/// ```compile_fail
/// use lazymatrix::{LazyMatrix, MatrixRead, MatrixWrite, RowNormalizationParams,
///                  WeightedGramInto, WeightedGramKernel};
/// fn gram<M: MatrixRead<f64> + WeightedGramKernel<f64>, D: MatrixWrite<f64>>(
///     rows: &LazyMatrix<M, RowNormalizationParams>, weights: &Vec<f64>, out: &mut D,
/// ) {
///     rows.weighted_gram_into(weights, out).unwrap();
/// }
/// ```
///
/// Eager conversion is currently available for column normalization only:
///
/// ```compile_fail
/// use lazymatrix::{LazyMatrix, MaterializeDense, MatrixRead, MatrixWrite, RowNormalizationParams};
/// fn eager<M: MatrixRead<f64> + MaterializeDense<f64>, D: MatrixWrite<f64>>(
///     rows: &LazyMatrix<M, RowNormalizationParams>, out: &mut D,
/// ) {
///     let _ = rows.to_eager_into(out);
/// }
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RowNormalization {
    /// How to center each row.
    pub center: Centering,
    /// How to scale each row.
    pub scale: Scaling,
}

impl RowNormalization {
    /// Build a specification from its centering and scaling choices.
    pub const fn new(center: Centering, scale: Scaling) -> Self {
        Self { center, scale }
    }

    /// Subtract each row's mean and divide by its population standard deviation.
    ///
    /// The variance uses `ncols` as its divisor. Fitting replaces exact zero
    /// scales with one, so finite constant rows become zero. For sample standard
    /// deviations, use `Self::new(Centering::Mean, Scaling::Sd { df: 1 })`.
    pub const fn standardize() -> Self {
        Self::new(Centering::Mean, Scaling::Sd { df: 0 })
    }

    /// Subtract each row's minimum and divide by its range.
    ///
    /// Finite, nonconstant rows span `[0, 1]` on the fitting data. Fitting
    /// replaces exact zero ranges with one, so finite constant rows become zero.
    /// Reusing parameters does not clip values to `[0, 1]`.
    pub const fn min_max() -> Self {
        Self::new(Centering::Min, Scaling::Range)
    }

    /// Subtract each row's median and divide by its interquartile range.
    ///
    /// Quantiles use linear interpolation at `(ncols - 1) * q`. There is no
    /// standard-normal calibration. Fitting replaces exact zero ranges with
    /// one. Use [`Self::new`] with [`Scaling::QuantileRange`] for custom bounds.
    pub fn robust() -> Self {
        Self::new(
            Centering::Median,
            Scaling::QuantileRange(QuantileRange::default()),
        )
    }
}

pub(crate) fn sd_divisor<F: crate::Scalar>(nrows: usize, df: usize) -> F {
    if nrows > df {
        F::from_usize(nrows - df).unwrap()
    } else {
        F::nan()
    }
}

/// Fitted column normalization, reusable on matrices with the same column count.
///
/// Present scales are nonzero. Computed normalization replaces exact zero
/// scales with one before constructing these parameters. Explicit parameters
/// preserve negative scales and nonfinite values.
#[derive(Clone, Debug, PartialEq)]
pub struct ColumnNormalizationParams<F = f64> {
    ncols: usize,
    pub(crate) centers: Option<Vec<F>>,
    pub(crate) scales: Option<Vec<F>>,
}

impl<F: crate::Scalar> ColumnNormalizationParams<F> {
    /// Construct validated explicit parameters.
    ///
    /// To wrap a matrix directly and infer its column count, use
    /// [`crate::LazyMatrix::from_column_parts`].
    ///
    /// # Panics
    /// Panics if either vector has a different length from `ncols`, or if a
    /// scale is exact positive or negative zero.
    pub fn from_parts(ncols: usize, centers: Option<Vec<F>>, scales: Option<Vec<F>>) -> Self {
        crate::gram::validate_normalization(ncols, centers.as_deref(), scales.as_deref());
        Self {
            ncols,
            centers,
            scales,
        }
    }

    /// Number of columns these parameters describe.
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Fitted column centers, if centering is active.
    pub fn centers(&self) -> Option<&[F]> {
        self.centers.as_deref()
    }

    /// Fitted column scales, if scaling is active.
    pub fn scales(&self) -> Option<&[F]> {
        self.scales.as_deref()
    }

    /// Recover the fitted centers and scales without copying.
    pub fn into_parts(self) -> ColumnNormalizationStats<F> {
        (self.centers, self.scales)
    }
}

/// Fitted normalization for particular rows of a matrix.
///
/// Present vectors have one entry per row, and scales are nonzero. Explicit
/// parameters preserve negative scales and nonfinite values. Parameters may be
/// reapplied to matrices with the same row count, but they describe the fitted
/// rows, not a transformation learned across samples. Fit new samples' row
/// parameters independently.
#[derive(Clone, Debug, PartialEq)]
pub struct RowNormalizationParams<F = f64> {
    nrows: usize,
    centers: Option<Vec<F>>,
    scales: Option<Vec<F>>,
}

impl<F: crate::Scalar> RowNormalizationParams<F> {
    /// Construct validated explicit parameters without changing their values.
    ///
    /// To wrap a matrix directly and infer its row count, use
    /// [`crate::LazyMatrix::from_row_parts`].
    ///
    /// # Panics
    /// Panics if either vector has a different length from `nrows`, or if a
    /// scale is exact positive or negative zero. A zero-scale panic reports
    /// the zero-based row index.
    pub fn from_parts(nrows: usize, centers: Option<Vec<F>>, scales: Option<Vec<F>>) -> Self {
        if let Some(centers) = &centers {
            assert_eq!(centers.len(), nrows, "centers length must equal nrows");
        }
        if let Some(scales) = &scales {
            assert_eq!(scales.len(), nrows, "scales length must equal nrows");
            for (row, &scale) in scales.iter().enumerate() {
                assert!(scale != F::zero(), "scale at row {row} must be nonzero");
            }
        }
        Self {
            nrows,
            centers,
            scales,
        }
    }

    /// Number of rows these parameters describe.
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Fitted row centers, if centering is active.
    pub fn centers(&self) -> Option<&[F]> {
        self.centers.as_deref()
    }

    /// Fitted row scales, if scaling is active.
    pub fn scales(&self) -> Option<&[F]> {
        self.scales.as_deref()
    }

    /// Recover the fitted centers and scales without copying.
    pub fn into_parts(self) -> RowNormalizationStats<F> {
        (self.centers, self.scales)
    }
}

impl<F: crate::Scalar> NormalizationParams for ColumnNormalizationParams<F> {
    type Scalar = F;
}

impl<F: crate::Scalar> private::Params for ColumnNormalizationParams<F> {
    fn assert_shape(&self, _nrows: usize, ncols: usize) {
        assert_eq!(
            ncols, self.ncols,
            "normalization column count must equal ncols"
        );
    }
}

impl<F: crate::Scalar> NormalizationParams for RowNormalizationParams<F> {
    type Scalar = F;
}

impl<F: crate::Scalar> private::Params for RowNormalizationParams<F> {
    fn assert_shape(&self, nrows: usize, _ncols: usize) {
        assert_eq!(
            nrows, self.nrows,
            "normalization row count must equal nrows"
        );
    }
}

impl<M: crate::ColumnStats<F> + crate::MatrixShape, F: crate::Scalar>
    private::Recipe<M, ColumnNormalizationParams<F>> for ColumnNormalization
{
    fn fit(self, data: &M) -> Result<ColumnNormalizationParams<F>, M::Error> {
        let (centers, scales) = data.normalization_stats(self)?;
        Ok(ColumnNormalizationParams::from_parts(
            data.ncols(),
            centers,
            scales.map(replace_zero_scales),
        ))
    }
}

impl<M: crate::MatrixRead<F> + crate::MatrixErrorType, F: crate::Scalar>
    private::Recipe<M, RowNormalizationParams<F>> for RowNormalization
{
    fn fit(self, data: &M) -> Result<RowNormalizationParams<F>, M::Error> {
        let (centers, scales) = row::fit(data, self);
        Ok(RowNormalizationParams::from_parts(
            data.nrows(),
            centers,
            scales,
        ))
    }
}

fn replace_zero_scales<F: crate::Scalar>(mut scales: Vec<F>) -> Vec<F> {
    for scale in &mut scales {
        if *scale == F::zero() {
            *scale = F::one();
        }
    }
    scales
}

#[cfg(test)]
mod tests {
    use super::QuantileRange;

    #[test]
    fn quantile_range_validates_bounds_and_defaults_to_iqr() {
        let range = QuantileRange::default();
        assert_eq!((range.lower(), range.upper()), (0.25, 0.75));
        assert_eq!(QuantileRange::new(-0.0, 1.0), QuantileRange::new(0.0, 1.0));
        for (lower, upper) in [
            (-0.1, 0.75),
            (0.25, 1.1),
            (0.5, 0.5),
            (0.75, 0.25),
            (f64::NAN, 0.75),
            (0.25, f64::NAN),
            (f64::NEG_INFINITY, 0.75),
            (0.25, f64::INFINITY),
        ] {
            assert!(std::panic::catch_unwind(|| QuantileRange::new(lower, upper)).is_err());
        }
    }
}
