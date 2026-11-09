pub(crate) mod pipeline;
pub(crate) mod row;

pub use pipeline::{
    FittedRowNormalization, NormalizationChain, NormalizationChainParams, PrepareNewSamples,
};

use std::marker::PhantomData;

mod private {
    pub trait Params<F: crate::Scalar> {
        fn assert_shape(&self, nrows: usize, ncols: usize);
        fn logical_entry(&self, value: F, row: usize, column: usize) -> F;

        // Corrections apply scales directly, avoiding cancellation and
        // nonfinite centers in differences of normalized values.
        fn scaled_correction(&self, value: F, row: usize, column: usize) -> F;
    }

    pub trait Recipe<M: crate::MatrixErrorType, P> {
        fn fit_recipe(self, data: &M) -> Result<P, M::Error>;
    }

    pub trait NewSamples<M> {}
}

/// Fitted normalization metadata supported by [`crate::LazyMatrix`].
///
/// The parameter type identifies the axis and scalar. This trait is sealed;
/// use [`ColumnNormalizationParams`] or [`RowNormalizationParams`] to supply
/// custom fitted values, or [`NormalizationChainParams`] for ordered stages.
pub trait NormalizationParams: private::Params<Self::Scalar> {
    /// Scalar used by the fitted centers and scales.
    type Scalar: crate::Scalar;

    /// Borrowed metadata retaining the same scalar and fitted stage order.
    type View<'a>: NormalizationParams<Scalar = Self::Scalar>
    where
        Self: 'a;

    /// Borrow fitted values without allocating or copying their vectors.
    fn view(&self) -> Self::View<'_>;
}

/// A normalization recipe supported by a backend and fitted parameter type.
///
/// [`crate::LazyMatrix::new`] selects fitting through this sealed trait. Column
/// recipes require [`crate::ColumnStats`]; row recipes require [`crate::RowStats`].
/// Implementing those capabilities enables fitting for a downstream backend
/// without implementing this trait directly.
/// Mixed and row-only chains require dense indexed access. Column-only chains
/// use the supported backend adapters' native scans.
pub trait NormalizationRecipe<M: crate::MatrixErrorType, P: NormalizationParams>:
    private::Recipe<M, P>
{
    /// Fit parameters without retaining or modifying the training data.
    ///
    /// # Errors
    /// Returns the backend error from fitting.
    fn fit(self, data: &M) -> Result<P, M::Error>
    where
        Self: Sized,
    {
        private::Recipe::fit_recipe(self, data)
    }
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
/// entry per column. Raw scales may be zero; [`ColumnNormalization::fit`]
/// replaces exact zeros with one while preserving nonfinite values.
pub type ColumnNormalizationStats<F> = (Option<Vec<F>>, Option<Vec<F>>);

/// Computed row centers and raw scales, respectively.
///
/// An inactive component is `None`. Present vectors have one entry per row.
/// [`RowNormalization::fit`] replaces exact zero scales with one
/// while preserving nonfinite values.
pub type RowNormalizationStats<F> = (Option<Vec<F>>, Option<Vec<F>>);

/// How to center each column or row.
///
/// These variants form a closed set of primitive choices. Additional methods
/// can use separate recipe types that fit the existing parameter types.
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
///
/// These variants form a closed set of primitive choices. Additional methods
/// can use separate recipe types that fit the existing parameter types.
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
#[non_exhaustive]
pub struct ColumnNormalization {
    pub center: Centering,
    pub scale: Scaling,
}

impl ColumnNormalization {
    /// Build a specification from its centering and scaling choices.
    pub const fn new(center: Centering, scale: Scaling) -> Self {
        Self { center, scale }
    }

    /// Fit owned column parameters without retaining or modifying the data.
    ///
    /// Uses [`crate::ColumnStats::normalization_stats`] so backends can share
    /// scans. Norm scales use centered values when centering is active. Exact
    /// zero scales become one; negative and nonfinite values are preserved.
    /// No normalized design matrix is materialized.
    ///
    /// Apply the parameters with [`ColumnNormalizationParams::apply`], borrowing
    /// them through [`ColumnNormalizationParams::view`] for reuse. Use
    /// [`Self::fit_apply`] to fit and wrap in one call. Column parameters can be
    /// reused on new samples with the same number of columns.
    ///
    /// # Errors
    /// Returns the backend error if computing normalization statistics fails.
    ///
    /// # Panics
    /// Panics if the backend returns a center or scale vector whose length
    /// differs from the number of columns.
    ///
    /// ```
    /// # #[cfg(feature = "ndarray_v0_17")]
    /// # {
    /// use lazymatrix::{ColumnNormalization, MatVec};
    /// use ndarray::array;
    ///
    /// let training = array![[1.0, 2.0], [3.0, 6.0]];
    /// let params = ColumnNormalization::standardize().fit(&training).unwrap();
    /// let validation = array![[5.0, 10.0]];
    /// let train = params.view().apply(&training);
    /// let valid = params.view().apply(&validation);
    /// assert_eq!(valid.matvec(&array![1.0, 1.0]).unwrap(), array![6.0]);
    /// # }
    /// ```
    pub fn fit<M, F>(self, data: &M) -> Result<ColumnNormalizationParams<F>, M::Error>
    where
        M: crate::ColumnStats<F> + crate::MatrixShape + ?Sized,
        F: crate::Scalar,
    {
        let (centers, scales) = data.normalization_stats(self)?;
        Ok(ColumnNormalizationParams::from_parts(
            data.ncols(),
            centers,
            scales.map(replace_zero_scales),
        ))
    }

    /// Fit column parameters and apply them lazily, owning the fitted vectors.
    ///
    /// The data handle is moved into the operator. Pass a reference or backend
    /// view to borrow the data. The result does not borrow this recipe or any
    /// temporary parameter object. This uses the same fitting path as
    /// [`Self::fit`] and [`crate::LazyMatrix::new`].
    ///
    /// # Errors
    /// Returns the backend error if computing normalization statistics fails.
    ///
    /// # Panics
    /// Panics if the backend returns parameter vectors with incorrect lengths.
    pub fn fit_apply<M, F>(
        self,
        data: M,
    ) -> Result<crate::LazyMatrix<M, ColumnNormalizationParams<F>>, M::Error>
    where
        M: crate::ColumnStats<F> + crate::MatrixShape,
        F: crate::Scalar,
    {
        Ok(self.fit(&data)?.apply(data))
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
/// [`crate::RowStats`]. Dense fitting takes O(nrows * ncols) time, or
/// O(nrows * ncols * log(ncols)) when sorting for exact linear quantiles. It uses
/// O(nrows + ncols) storage at most, including one working row for quantiles.
/// Standard deviations use a two-pass calculation with `ncols - df` as divisor.
/// Sparse fitting combines duplicate cells and includes implicit zeros. CSR
/// fitting scans storage once and buffers one coalesced row, sorting it for
/// exact quantiles. CSC fitting streams column scans with O(nrows) state;
/// quantiles use repeated scans whose count depends on the values. Both retain
/// their storage orientation and use O(nrows + ncols) total workspace, including
/// the fitted vectors and duplicate-coalescing scratch.
/// Zarr fitting streams storage chunks serially with O(nrows) state plus chunk
/// and codec buffers, including outer shards. It preserves missing chunks' fill
/// values and excludes edge padding. Standardization takes two scans, other
/// recipes without quantiles take up to three, and exact row quantiles use
/// repeated scans whose count depends on the values. A median and quantile
/// range fitted together share their scans.
/// Empty rows have undefined means, standard deviations, minima, and ranges
/// (`NaN`), and zero norms, matching column fitting along the other axis.
///
/// Row-normalized operators support allocating, reusable, and fused forward and
/// transpose vector products through [`crate::RowNormalizationKernel`] on dense
/// matrices, strided views, sparse CSC/CSR storage, and Zarr arrays. Dense
/// kernels normalize entries before accumulating and use O(1) additional
/// workspace. Sparse products combine duplicate cells before normalization and
/// account for transformed implicit zeros with O(nrows + ncols) workspace. Zarr products
/// normalize entries before accumulation and scan each storage chunk once,
/// retaining only chunk and codec buffers beyond the output and fitted vectors.
/// Fused products retain one temporary output vector and merge it only after
/// the fitted kernel succeeds. Batched forward and transpose products use
/// [`crate::MatrixTraversal`], normalizing entries before accumulation. Dense
/// and Zarr batches scan once for all responses. Sparse batches scan once per
/// panel of up to 16 responses, with O(nrows + ncols) workspace independent of
/// the response count. Borrowed logical columns require [`crate::RawColumns`],
/// and sparse column representations require [`crate::SparseColumns`]. Borrowed
/// rows require [`crate::SparseRows`]. Weighted column sums and Gram products
/// use [`crate::MatrixTraversal`], applying fitted parameters before accumulation
/// with bounded logical panels. Full-storage sums need only one scan. Eager
/// conversion is not yet supported. New samples require independently fitted
/// row parameters. [`crate::Transpose`] transposes the fitted operator
/// without refitting.
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
/// Computed row normalization requires a row-statistics capability:
///
/// ```compile_fail
/// use lazymatrix::{LazyMatrix, MatrixErrorType, MatrixShape, RowNormalization};
/// fn without_row_statistics<M: MatrixShape + MatrixErrorType>(matrix: M) {
///     let _ = LazyMatrix::new(matrix, RowNormalization::standardize());
/// }
/// ```
///
/// Fused products require the row-normalized kernel:
///
/// ```
/// use lazymatrix::{LazyMatrix, MatVecScaledInto, RowNormalizationKernel, RowNormalizationParams};
/// fn fused<M: RowNormalizationKernel<f64>>(
///     rows: &LazyMatrix<M, RowNormalizationParams>, x: &Vec<f64>, out: &mut Vec<f64>,
/// ) {
///     rows.matvec_scaled_into(1.0, x, 0.0, out).unwrap();
/// }
/// ```
///
/// Batched products require native traversal:
///
/// ```
/// use lazymatrix::{LazyMatrix, MatMatInto, MatrixRead, MatrixTraversal, MatrixWrite,
///     RowNormalizationParams};
/// fn batched<M: MatrixTraversal<f64>, D: MatrixRead<f64> + MatrixWrite<f64>>(
///     rows: &LazyMatrix<M, RowNormalizationParams>, rhs: &D, out: &mut D,
/// ) {
///     rows.matmat_into(rhs, out).unwrap();
/// }
/// ```
///
/// Borrowed logical rows require contiguous raw sparse rows:
///
/// ```
/// use lazymatrix::{LazyMatrix, RowNormalizationParams, SparseRows};
/// fn row<M: SparseRows<f64>>(rows: &LazyMatrix<M, RowNormalizationParams>) -> f64 {
///     rows.row(0).get(0)
/// }
/// ```
///
/// Borrowed logical columns require natural raw columns, without dense indexing:
///
/// ```
/// use lazymatrix::{LazyMatrix, RawColumns, RowNormalizationParams};
/// fn columns<M: RawColumns<f64>>(rows: &LazyMatrix<M, RowNormalizationParams>) -> f64 {
///     rows.column(0).get(0)
/// }
/// ```
///
/// Dense indexing alone does not grant column borrowing:
///
/// ```compile_fail
/// use lazymatrix::{LazyMatrix, MatrixRead, RowNormalizationParams};
/// fn no_columns<M: MatrixRead<f64>>(rows: &LazyMatrix<M, RowNormalizationParams>) {
///     let _ = rows.column(0);
/// }
/// ```
///
/// Raw columns do not grant access to contiguous sparse slices:
///
/// ```compile_fail
/// use lazymatrix::{LazyMatrix, RawColumns, RowNormalizationParams};
/// fn no_sparse_columns<M: RawColumns<f64>>(rows: &LazyMatrix<M, RowNormalizationParams>) {
///     let _ = rows.sparse_column(0);
/// }
/// ```
///
/// Weighted operations require native traversal, without fitting or vector kernels:
///
/// ```
/// use lazymatrix::{LazyMatrix, MatrixTraversal, MatrixWrite, RowNormalizationParams,
///                  WeightedColumnSumsInto, WeightedGramInto};
/// fn gram<M: MatrixTraversal<f64>, D: MatrixWrite<f64>>(
///     rows: &LazyMatrix<M, RowNormalizationParams>, weights: &Vec<f64>, out: &mut D,
/// ) {
///     rows.weighted_gram_into(weights, out).unwrap();
///     let mut sums = vec![0.; rows.ncols()];
///     rows.weighted_column_sums_into(weights, &mut sums).unwrap();
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
#[non_exhaustive]
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

    /// Fit owned row parameters without retaining or modifying the data.
    ///
    /// Requires [`crate::RowStats`]. Norm scales use centered values when
    /// centering is active. Exact zero scales become one, and nonfinite values
    /// are preserved. Fitting uses the bounded
    /// workspace described by [`RowNormalization`], without materializing a
    /// normalized design matrix.
    ///
    /// Apply the parameters to the same rows with
    /// [`RowNormalizationParams::apply_frozen`], borrowing them through
    /// [`RowNormalizationParams::view`] for reuse. Use [`Self::fit_apply`] to fit
    /// and wrap in one call. Fit new samples independently, even when their row
    /// count matches the fitting data.
    ///
    /// # Errors
    /// Returns the backend error if computing row statistics fails.
    ///
    /// ```
    /// # #[cfg(feature = "ndarray_v0_17")]
    /// # {
    /// use lazymatrix::{Centering, MatVec, RowNormalization, Scaling};
    /// use ndarray::array;
    ///
    /// let data = array![[3.0, 4.0], [0.0, 0.0]];
    /// let recipe = RowNormalization::new(Centering::None, Scaling::L2);
    /// let params = recipe.fit(&data).unwrap();
    /// assert_eq!(params.scales(), Some([5.0, 1.0].as_slice()));
    /// let rows = params.view().apply_frozen(&data);
    /// assert_eq!(rows.matvec(&array![1.0, 1.0]).unwrap(), array![1.4, 0.0]);
    /// # }
    /// ```
    pub fn fit<M, F>(self, data: &M) -> Result<RowNormalizationParams<F>, M::Error>
    where
        M: crate::RowStats<F> + ?Sized,
        F: crate::Scalar,
    {
        let (centers, scales) = data.row_normalization_stats(self)?;
        let scales = scales.map(replace_zero_scales);
        Ok(RowNormalizationParams::from_parts(
            data.nrows(),
            centers,
            scales,
        ))
    }

    /// Fit each input row independently and apply its parameters lazily.
    ///
    /// The result owns the fitted vectors and stores the supplied data handle.
    /// Pass a reference or backend view to borrow the data. New samples need
    /// their own call to this method or [`Self::fit`].
    ///
    /// # Errors
    /// Returns the backend error if computing row statistics fails.
    pub fn fit_apply<M, F>(
        self,
        data: M,
    ) -> Result<crate::LazyMatrix<M, RowNormalizationParams<F>>, M::Error>
    where
        M: crate::RowStats<F>,
        F: crate::Scalar,
    {
        Ok(self.fit(&data)?.apply_frozen(data))
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
///
/// The default storage owns its vectors. [`Self::view`] borrows the same values
/// without allocation; applying that view shares them across lazy operators.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColumnNormalizationParams<F = f64, S = Vec<F>> {
    ncols: usize,
    centers: Option<S>,
    scales: Option<S>,
    scalar: PhantomData<F>,
}

/// A borrowed view of validated column normalization parameters.
///
/// Created by [`ColumnNormalizationParams::view`]. Copying the view copies only
/// slice handles, never the fitted values.
pub type ColumnNormalizationParamsView<'a, F = f64> = ColumnNormalizationParams<F, &'a [F]>;

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
            scalar: PhantomData,
        }
    }
}

impl<F: crate::Scalar, S: AsRef<[F]>> ColumnNormalizationParams<F, S> {
    /// Number of columns these parameters describe.
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Fitted column centers, if centering is active.
    pub fn centers(&self) -> Option<&[F]> {
        self.centers.as_ref().map(AsRef::as_ref)
    }

    /// Fitted column scales, if scaling is active.
    pub fn scales(&self) -> Option<&[F]> {
        self.scales.as_ref().map(AsRef::as_ref)
    }

    /// Borrow the fitted values without allocating or copying them.
    ///
    /// The view retains the fitted column count, including when both
    /// normalization components are inactive.
    ///
    /// A borrowed operator cannot outlive these parameters:
    ///
    /// ```compile_fail
    /// use lazymatrix::{ColumnNormalizationParams, ColumnNormalizationParamsView,
    ///                  LazyMatrix, MatrixShape};
    /// fn escaped<M: MatrixShape>(data: M)
    ///     -> LazyMatrix<M, ColumnNormalizationParamsView<'static>>
    /// {
    ///     let params = ColumnNormalizationParams::from_parts(data.ncols(), None, None);
    ///     params.view().apply(data)
    /// }
    /// ```
    pub fn view(&self) -> ColumnNormalizationParamsView<'_, F> {
        ColumnNormalizationParams {
            ncols: self.ncols,
            centers: self.centers(),
            scales: self.scales(),
            scalar: PhantomData,
        }
    }

    /// Apply these parameters unchanged, moving their storage into a lazy operator.
    ///
    /// Use `params.view().apply(data)` to borrow the fitted values, or
    /// `params.apply(data)` to move owned vectors. The data handle may itself be
    /// an owned matrix, reference, or backend view. Application performs no
    /// fitting, data reads, or parameter copies.
    ///
    /// # Panics
    /// Panics if the data's column count differs from the fitted column count.
    pub fn apply<M: crate::MatrixShape>(self, data: M) -> crate::LazyMatrix<M, Self> {
        crate::LazyMatrix::from_parts(data, self)
    }

    /// Recover the column count, stored centers, and stored scales without copying.
    ///
    /// Owned parameters return vectors; views return borrowed slices.
    pub fn into_parts(self) -> (usize, Option<S>, Option<S>) {
        (self.ncols, self.centers, self.scales)
    }

    pub(crate) fn copy_owned(&self) -> ColumnNormalizationParams<F> {
        ColumnNormalizationParams::from_parts(
            self.ncols,
            self.centers().map(<[F]>::to_vec),
            self.scales().map(<[F]>::to_vec),
        )
    }
}

/// Fitted normalization for particular rows of a matrix.
///
/// Present vectors have one entry per row, and scales are nonzero. Explicit
/// parameters preserve negative scales and nonfinite values. Parameters may be
/// reapplied to matrices with the same row count, but they describe the fitted
/// rows, not a transformation learned across samples. Fit new samples' row
/// parameters independently.
///
/// The default storage owns its vectors. [`Self::view`] borrows the same values
/// without allocation. Both forms use [`Self::apply_frozen`] to reapply them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RowNormalizationParams<F = f64, S = Vec<F>> {
    nrows: usize,
    centers: Option<S>,
    scales: Option<S>,
    scalar: PhantomData<F>,
}

/// A borrowed view of validated parameters for particular rows.
///
/// Created by [`RowNormalizationParams::view`]. The view describes the same
/// fitted rows; fit new samples independently.
pub type RowNormalizationParamsView<'a, F = f64> = RowNormalizationParams<F, &'a [F]>;

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
        crate::row_normalization::validate_parameters(nrows, centers.as_deref(), scales.as_deref());
        Self {
            nrows,
            centers,
            scales,
            scalar: PhantomData,
        }
    }
}

impl<F: crate::Scalar, S: AsRef<[F]>> RowNormalizationParams<F, S> {
    /// Number of rows these parameters describe.
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Fitted row centers, if centering is active.
    pub fn centers(&self) -> Option<&[F]> {
        self.centers.as_ref().map(AsRef::as_ref)
    }

    /// Fitted row scales, if scaling is active.
    pub fn scales(&self) -> Option<&[F]> {
        self.scales.as_ref().map(AsRef::as_ref)
    }

    /// Borrow these rows' fitted values without allocating or copying them.
    ///
    /// Borrowed row parameters support the same batched products as owned ones:
    ///
    /// ```
    /// use lazymatrix::{MatMatInto, MatrixRead, MatrixTraversal, MatrixWrite, RowNormalizationParams};
    /// fn batched<M: MatrixTraversal<f64>, D: MatrixRead<f64> + MatrixWrite<f64>>(
    ///     params: &RowNormalizationParams, data: M, rhs: &D, out: &mut D,
    /// ) {
    ///     params.view().apply_frozen(data).matmat_into(rhs, out).unwrap();
    /// }
    /// ```
    pub fn view(&self) -> RowNormalizationParamsView<'_, F> {
        RowNormalizationParams {
            nrows: self.nrows,
            centers: self.centers(),
            scales: self.scales(),
            scalar: PhantomData,
        }
    }

    /// Reapply these rows' parameters unchanged, moving storage into a lazy operator.
    ///
    /// Use `params.view().apply_frozen(data)` to borrow the fitted values, or
    /// `params.apply_frozen(data)` to move owned vectors. This performs no fitting,
    /// data reads, or parameter copies. Fit new samples independently, even when
    /// their row count matches these parameters.
    ///
    /// # Panics
    /// Panics if the data's row count differs from the fitted row count.
    ///
    /// Row parameters deliberately require the frozen name:
    ///
    /// ```compile_fail
    /// use lazymatrix::{MatrixShape, RowNormalizationParams};
    /// fn ambiguous<M: MatrixShape>(params: RowNormalizationParams, data: M) {
    ///     let _ = params.view().apply(data);
    /// }
    /// ```
    pub fn apply_frozen<M: crate::MatrixShape>(self, data: M) -> crate::LazyMatrix<M, Self> {
        crate::LazyMatrix::from_parts(data, self)
    }

    /// Recover the row count, stored centers, and stored scales without copying.
    ///
    /// Owned parameters return vectors; views return borrowed slices.
    pub fn into_parts(self) -> (usize, Option<S>, Option<S>) {
        (self.nrows, self.centers, self.scales)
    }

    pub(crate) fn copy_owned(&self) -> RowNormalizationParams<F> {
        RowNormalizationParams::from_parts(
            self.nrows,
            self.centers().map(<[F]>::to_vec),
            self.scales().map(<[F]>::to_vec),
        )
    }
}

impl<F: crate::Scalar, S: AsRef<[F]>> NormalizationParams for ColumnNormalizationParams<F, S> {
    type Scalar = F;
    type View<'a>
        = ColumnNormalizationParamsView<'a, F>
    where
        Self: 'a;

    fn view(&self) -> Self::View<'_> {
        ColumnNormalizationParams::view(self)
    }
}

impl<F: crate::Scalar, S: AsRef<[F]>> private::Params<F> for ColumnNormalizationParams<F, S> {
    fn logical_entry(&self, value: F, _row: usize, column: usize) -> F {
        crate::materialize::normalized(value, column, self.centers(), self.scales())
    }
    fn scaled_correction(&self, value: F, _row: usize, column: usize) -> F {
        self.scales().map_or(value, |scales| value / scales[column])
    }

    fn assert_shape(&self, _nrows: usize, ncols: usize) {
        assert_eq!(
            ncols, self.ncols,
            "normalization column count must equal ncols"
        );
    }
}

impl<F: crate::Scalar, S: AsRef<[F]>> NormalizationParams for RowNormalizationParams<F, S> {
    type Scalar = F;
    type View<'a>
        = RowNormalizationParamsView<'a, F>
    where
        Self: 'a;

    fn view(&self) -> Self::View<'_> {
        RowNormalizationParams::view(self)
    }
}

impl<F: crate::Scalar, S: AsRef<[F]>> private::Params<F> for RowNormalizationParams<F, S> {
    fn logical_entry(&self, value: F, row: usize, _column: usize) -> F {
        crate::materialize::normalized(value, row, self.centers(), self.scales())
    }
    fn scaled_correction(&self, value: F, row: usize, _column: usize) -> F {
        self.scales().map_or(value, |scales| value / scales[row])
    }

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
    fn fit_recipe(self, data: &M) -> Result<ColumnNormalizationParams<F>, M::Error> {
        ColumnNormalization::fit(self, data)
    }
}

impl<M: crate::RowStats<F>, F: crate::Scalar> private::Recipe<M, RowNormalizationParams<F>>
    for RowNormalization
{
    fn fit_recipe(self, data: &M) -> Result<RowNormalizationParams<F>, M::Error> {
        RowNormalization::fit(self, data)
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
