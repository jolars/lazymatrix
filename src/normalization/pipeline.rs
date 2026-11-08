//! Ordered recipes, fitted metadata, and statically selected execution.

use super::{
    ColumnNormalization, ColumnNormalizationParams, NormalizationParams, NormalizationRecipe,
    RowNormalization, RowNormalizationParams, private,
};
use crate::materialize::normalized;
use crate::{LazyMatrix, MatrixErrorType, MatrixRead, MatrixShape, Scalar};
use num_traits::Zero;
use private::Params as _;
use std::marker::PhantomData;

mod batched;
mod dense;
mod products;
pub(crate) use products::product_rectangle;
pub(crate) mod source;
mod stats;

#[cfg(any(
    feature = "faer_all",
    feature = "nalgebra_all",
    feature = "sprs_all",
    feature = "zarrs_all"
))]
pub(crate) fn raw_stats<M: Source<F>, F: Scalar>(
    data: &M,
    spec: ColumnNormalization,
) -> Result<super::ColumnNormalizationStats<F>, M::Error> {
    use crate::ColumnStats;
    stats::Logical::new(data, &Identity(PhantomData)).normalization_stats(spec)
}

pub(crate) use source::Source;
#[cfg(feature = "faer_all")]
pub(crate) use source::dense_source;
#[cfg(any(feature = "faer_all", feature = "nalgebra_all", feature = "sprs_all"))]
pub(crate) use source::sparse;
#[cfg(any(feature = "faer_all", feature = "nalgebra_all"))]
pub(crate) use source::sparse_source;

/// Two normalization recipes applied from left to right.
///
/// Chains may contain either axis, repeated axes, and further chains.
/// Each stage is fitted on the logical output of its predecessors. Later
/// stages can change statistics established by earlier stages.
///
/// Row stages require dense indexed access throughout the sequence:
///
/// ```compile_fail
/// use lazymatrix::{ColumnNormalization, ColumnStats, MatrixShape, RowNormalization};
/// fn unsupported<M: ColumnStats<f64> + MatrixShape>(data: &M) {
///     let _ = RowNormalization::standardize()
///         .then(ColumnNormalization::standardize()).fit::<_,
///             lazymatrix::NormalizationChainParams<
///                 lazymatrix::FittedRowNormalization<lazymatrix::RowNormalizationParams>,
///                 lazymatrix::ColumnNormalizationParams>>(data);
/// }
/// ```
///
/// ```
/// # #[cfg(feature = "ndarray_v0_17")]
/// # {
/// use lazymatrix::{ColumnNormalization, RowNormalization};
/// use ndarray::array;
///
/// let training = array![[1., 3., 2.], [7., 3., 8.], [2., 8., 5.]];
/// let validation = array![[100., 120., 90.]];
/// let recipe = RowNormalization::standardize()
///     .then(ColumnNormalization::standardize());
/// let fitted = recipe.fit(&training).unwrap();
/// let train = fitted.view().apply_frozen(training.view());
/// let valid = fitted.view().for_new_samples(validation.view()).unwrap();
/// assert_eq!(valid.normalization().first().parameters().nrows(), 1);
/// assert_eq!(train.normalization().next(), valid.normalization().next());
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NormalizationChain<A, B> {
    first: A,
    next: B,
}

macro_rules! then {
    ($type:ty) => {
        impl $type {
            /// Append a recipe, applying it after this recipe.
            pub fn then<R>(self, next: R) -> NormalizationChain<Self, R> {
                NormalizationChain { first: self, next }
            }
        }
    };
}
then!(ColumnNormalization);
then!(RowNormalization);

impl<A, B> NormalizationChain<A, B> {
    /// Append a recipe after every stage in this chain.
    pub fn then<R>(self, next: R) -> NormalizationChain<Self, R> {
        NormalizationChain { first: self, next }
    }

    /// Inspect the first recipe or prefix.
    pub fn first(&self) -> &A {
        &self.first
    }

    /// Inspect the next recipe or suffix.
    pub fn next(&self) -> &B {
        &self.next
    }

    /// Fit every stage without retaining the training data.
    ///
    /// Mixed chains require dense indexed access. Column-only chains use
    /// native traversal on supported dense, sparse, and chunked backends.
    ///
    /// # Errors
    /// Returns the backend error from a statistics scan.
    pub fn fit<M, P>(self, data: &M) -> Result<P, M::Error>
    where
        M: MatrixShape + MatrixErrorType,
        P: NormalizationParams,
        Self: NormalizationRecipe<M, P>,
    {
        NormalizationRecipe::fit(self, data)
    }

    /// Fit the chain and move its fitted values into a lazy operator.
    ///
    /// # Errors
    /// Returns the backend error from fitting.
    pub fn fit_apply<M, P>(self, data: M) -> Result<LazyMatrix<M, P>, M::Error>
    where
        M: MatrixShape + MatrixErrorType,
        P: NormalizationParams,
        Self: NormalizationRecipe<M, P>,
    {
        let params = self.fit(&data)?;
        Ok(LazyMatrix::from_parts(data, params))
    }
}

/// A row recipe retained alongside the parameters fitted for particular rows.
///
/// Retaining the recipe lets a chain prepare new samples without reusing
/// training rows' fitted values. The underlying parameter type remains
/// RowNormalizationParams, with owned vectors or borrowed slices.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FittedRowNormalization<P> {
    recipe: RowNormalization,
    parameters: P,
}

impl<P> FittedRowNormalization<P> {
    /// Pair an explicit or computed set of row parameters with its refitting recipe.
    ///
    /// This performs no fitting and leaves the supplied parameters unchanged.
    pub fn from_parts(recipe: RowNormalization, parameters: P) -> Self {
        Self { recipe, parameters }
    }

    /// The recipe to fit when preparing new samples.
    pub fn recipe(&self) -> RowNormalization {
        self.recipe
    }

    /// The saved parameters for these rows.
    pub fn parameters(&self) -> &P {
        &self.parameters
    }

    /// Recover the row recipe and saved parameters.
    pub fn into_parts(self) -> (RowNormalization, P) {
        (self.recipe, self.parameters)
    }
}

impl<F: Scalar, S: AsRef<[F]>> FittedRowNormalization<RowNormalizationParams<F, S>> {
    /// Borrow the saved row parameters without copying their values.
    pub fn view(&self) -> FittedRowNormalization<super::RowNormalizationParamsView<'_, F>> {
        FittedRowNormalization::from_parts(self.recipe, self.parameters.view())
    }
}

/// Fitted stages retained in the same order and tree shape as their recipes.
///
/// Column stages contain ColumnNormalizationParams. Row stages contain
/// FittedRowNormalization, retaining the recipe needed to prepare new samples.
/// Application consumes owned stages or explicit borrowed views.
/// Forward and transpose batched products preserve every stage in order.
/// Column-only batches use native traversal; mixed batches require dense
/// indexed storage. Neither path materializes the normalized design.
/// Borrowed logical columns require only [`crate::RawColumns`], and sparse
/// representations require only [`crate::SparseColumns`], even for explicit
/// mixed chains. These views borrow all stages without copying or refitting.
///
/// Generic consumers can use the public column capabilities:
///
/// ```
/// use lazymatrix::{Columns, LogicalColumn};
/// fn column_sum<M: Columns<f64>>(matrix: &M, j: usize) -> f64 {
///     matrix.column(j).sum()
/// }
/// ```
///
/// Explicit chains can borrow columns independently of fitting and products:
///
/// ```
/// use lazymatrix::{ColumnNormalizationParams, FittedRowNormalization, LazyMatrix,
///     NormalizationChainParams, RawColumns, RowNormalizationParams};
/// fn entry<M: RawColumns<f64>>(op: &LazyMatrix<M, NormalizationChainParams<
///     FittedRowNormalization<RowNormalizationParams>, ColumnNormalizationParams>>,
///     i: usize, j: usize,
/// ) -> f64 {
///     op.column(j).get(i)
/// }
/// ```
///
/// Mixed-chain column borrowing still requires natural raw columns:
///
/// ```compile_fail
/// use lazymatrix::{ColumnNormalizationParams, FittedRowNormalization, LazyMatrix,
///     MatrixRead, NormalizationChainParams, RowNormalizationParams};
/// fn no_columns<M: MatrixRead<f64>>(op: &LazyMatrix<M, NormalizationChainParams<
///     FittedRowNormalization<RowNormalizationParams>, ColumnNormalizationParams>>,
/// ) {
///     let _ = op.column(0);
/// }
/// ```
///
/// Chains deliberately require explicit application names:
///
/// ```compile_fail
/// use lazymatrix::{ColumnNormalizationParams, FittedRowNormalization,
///     MatrixShape, NormalizationChainParams, RowNormalizationParams};
/// fn ambiguous<M: MatrixShape>(
///     params: NormalizationChainParams<
///         FittedRowNormalization<RowNormalizationParams>, ColumnNormalizationParams>,
///     data: M,
/// ) {
///     let _ = params.apply(data);
/// }
/// ```
///
/// Mixed chains support batched products on dense indexed storage:
///
/// ```
/// use lazymatrix::{ColumnNormalization, MatMatInto, MatrixErrorType,
///     MatrixRead, MatrixWrite, RowNormalization};
/// fn batched<M: MatrixRead<f64> + MatrixErrorType,
///     D: MatrixRead<f64> + MatrixWrite<f64>>(data: M, rhs: &D, out: &mut D)
/// {
///     let op = RowNormalization::standardize()
///         .then(ColumnNormalization::standardize()).fit_apply(data).unwrap();
///     op.matmat_into(rhs, out).unwrap();
/// }
/// ```
///
/// Preparing new samples cannot extend borrowed column parameters' lifetime:
///
/// ```compile_fail
/// use lazymatrix::{ColumnNormalizationParams, ColumnNormalizationParamsView,
///     FittedRowNormalization, LazyMatrix, MatrixErrorType, MatrixRead,
///     NormalizationChainParams, RowNormalization, RowNormalizationParams};
/// fn escaped<M: MatrixRead<f64> + MatrixErrorType>(data: M)
///     -> LazyMatrix<M, NormalizationChainParams<
///         FittedRowNormalization<RowNormalizationParams>,
///         ColumnNormalizationParamsView<'static>>>
/// {
///     let rows = FittedRowNormalization::from_parts(RowNormalization::standardize(),
///         RowNormalizationParams::from_parts(data.nrows(), None, None));
///     let cols = ColumnNormalizationParams::from_parts(data.ncols(), None, None);
///     let params = NormalizationChainParams::from_parts(rows, cols);
///     params.view().for_new_samples(data).unwrap()
/// }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalizationChainParams<A, B> {
    first: A,
    next: B,
}

impl<A, B> NormalizationChainParams<A, B> {
    /// Assemble explicit or previously fitted stages without fitting or copying.
    ///
    /// Matrix dimensions are validated when the chain is applied.
    pub fn from_parts(first: A, next: B) -> Self
    where
        Self: NormalizationParams,
    {
        Self { first, next }
    }

    /// Inspect the first fitted stage or prefix.
    pub fn first(&self) -> &A {
        &self.first
    }

    /// Inspect the next fitted stage or suffix.
    pub fn next(&self) -> &B {
        &self.next
    }

    /// Recover both fitted stages without copying.
    pub fn into_parts(self) -> (A, B) {
        (self.first, self.next)
    }

    /// Borrow all saved parameter vectors without allocation or copying.
    pub fn view(&self) -> <Self as NormalizationParams>::View<'_>
    where
        Self: NormalizationParams,
    {
        NormalizationParams::view(self)
    }

    /// Apply every saved stage unchanged to matching rows and columns.
    ///
    /// This performs no fitting, data reads, or parameter copies.
    ///
    /// # Panics
    /// Panics if any fitted axis has a different dimension from the data.
    pub fn apply_frozen<M: MatrixShape>(self, data: M) -> LazyMatrix<M, Self>
    where
        Self: NormalizationParams,
    {
        LazyMatrix::from_parts(data, self)
    }

    /// Prepare new samples, refitting row stages and reusing column stages.
    ///
    /// Row fitting occurs after the preceding prepared stages. Owned column
    /// parameters move into the result; a view lends its training columns to
    /// the result. Newly fitted row parameters are owned in both cases.
    ///
    /// # Errors
    /// Returns the backend error from preparation.
    ///
    /// # Panics
    /// Panics before reading data if a saved column count differs.
    pub fn for_new_samples<M>(
        self,
        data: M,
    ) -> Result<LazyMatrix<M, <Self as PrepareNewSamples<M>>::Params>, M::Error>
    where
        M: MatrixShape + MatrixErrorType,
        Self: PrepareNewSamples<M>,
    {
        PrepareNewSamples::for_new_samples(self, data)
    }
}

/// Fitted column parameters or chains that can prepare new samples.
///
/// Column parameters are reused unchanged. Chains reuse their column stages
/// and refit each row stage after its prepared predecessors. This trait is
/// sealed; downstream code can use it as a bound without depending on the
/// execution dispatch of a particular chain.
///
/// Bare row parameters describe particular rows and do not retain a recipe:
///
/// ```compile_fail
/// use lazymatrix::{MatrixErrorType, MatrixShape, PrepareNewSamples,
///     RowNormalizationParams};
/// fn prepare<M: MatrixShape + MatrixErrorType>(params: RowNormalizationParams, data: M) {
///     let _ = PrepareNewSamples::for_new_samples(params, data);
/// }
/// ```
pub trait PrepareNewSamples<M>: NormalizationParams + private::NewSamples<M> + Sized
where
    M: MatrixShape + MatrixErrorType,
{
    /// Parameters retained or fitted for the new samples.
    type Params: NormalizationParams<Scalar = Self::Scalar>;

    /// Prepare samples, preserving column parameters and refitting row stages.
    ///
    /// Owned columns move into the result; borrowed columns retain their
    /// original lifetime. Newly fitted row vectors are owned.
    ///
    /// # Errors
    /// Returns the backend error from fitting a row stage.
    ///
    /// # Panics
    /// Panics before reading data if a saved column count differs.
    fn for_new_samples(self, data: M) -> Result<LazyMatrix<M, Self::Params>, M::Error>;
}

// These traits seal the recursive representation while keeping method return
// types inferable without exposing backend dispatch in fitted parameter types.
pub trait Stage: NormalizationParams + Sized {
    type Kind;
    type StageView<'a>: Stage<Scalar = Self::Scalar, Kind = Self::Kind>
    where
        Self: 'a;
    type NewRows: Stage<Scalar = Self::Scalar, Kind = Self::Kind>;

    fn stage_view(&self) -> Self::StageView<'_>;
    fn entry(&self, value: Self::Scalar, row: usize, column: usize) -> Self::Scalar;
    fn assert_columns(&self, ncols: usize);
    fn is_identity(&self) -> bool {
        false
    }
    fn prepare_dense<M, P>(self, data: &M, prefix: &P) -> Result<Self::NewRows, M::Error>
    where
        M: MatrixRead<Self::Scalar> + MatrixErrorType,
        P: Stage<Scalar = Self::Scalar>;
}

pub struct Columns;
pub struct Rows;

pub trait Combine<K> {
    type Output;
}
impl Combine<Columns> for Columns {
    type Output = Columns;
}
impl Combine<Rows> for Columns {
    type Output = Rows;
}
impl Combine<Columns> for Rows {
    type Output = Rows;
}
impl Combine<Rows> for Rows {
    type Output = Rows;
}

impl<F: Scalar, S: AsRef<[F]>> Stage for ColumnNormalizationParams<F, S> {
    type Kind = Columns;
    type StageView<'a>
        = super::ColumnNormalizationParamsView<'a, F>
    where
        Self: 'a;
    type NewRows = Self;

    fn stage_view(&self) -> Self::StageView<'_> {
        self.view()
    }
    fn entry(&self, value: F, _row: usize, column: usize) -> F {
        normalized(value, column, self.centers(), self.scales())
    }
    fn assert_columns(&self, ncols: usize) {
        self.assert_shape(0, ncols);
    }
    fn prepare_dense<M, P>(self, _data: &M, _prefix: &P) -> Result<Self, M::Error>
    where
        M: MatrixRead<F> + MatrixErrorType,
        P: Stage<Scalar = F>,
    {
        Ok(self)
    }
}

impl<F: Scalar, S: AsRef<[F]>> NormalizationParams
    for FittedRowNormalization<RowNormalizationParams<F, S>>
{
    type Scalar = F;
    type View<'a>
        = FittedRowNormalization<super::RowNormalizationParamsView<'a, F>>
    where
        Self: 'a;

    fn view(&self) -> Self::View<'_> {
        FittedRowNormalization::view(self)
    }
}
impl<F: Scalar, S: AsRef<[F]>> private::Params<F>
    for FittedRowNormalization<RowNormalizationParams<F, S>>
{
    fn logical_entry(&self, value: F, row: usize, column: usize) -> F {
        self.parameters.logical_entry(value, row, column)
    }
    fn scaled_correction(&self, value: F, row: usize, column: usize) -> F {
        self.parameters.scaled_correction(value, row, column)
    }

    fn assert_shape(&self, nrows: usize, ncols: usize) {
        self.parameters.assert_shape(nrows, ncols);
    }
}
impl<F: Scalar, S: AsRef<[F]>> Stage for FittedRowNormalization<RowNormalizationParams<F, S>> {
    type Kind = Rows;
    type StageView<'a>
        = FittedRowNormalization<super::RowNormalizationParamsView<'a, F>>
    where
        Self: 'a;
    type NewRows = FittedRowNormalization<RowNormalizationParams<F>>;

    fn stage_view(&self) -> Self::StageView<'_> {
        self.view()
    }
    fn entry(&self, value: F, row: usize, _column: usize) -> F {
        normalized(
            value,
            row,
            self.parameters.centers(),
            self.parameters.scales(),
        )
    }
    fn assert_columns(&self, _ncols: usize) {}
    fn prepare_dense<M, P>(self, data: &M, prefix: &P) -> Result<Self::NewRows, M::Error>
    where
        M: MatrixRead<F> + MatrixErrorType,
        P: Stage<Scalar = F>,
    {
        let logical = dense::Logical { data, prefix };
        Ok(FittedRowNormalization::from_parts(
            self.recipe,
            self.recipe.fit(&logical)?,
        ))
    }
}

impl<A, B> NormalizationParams for NormalizationChainParams<A, B>
where
    A: Stage,
    B: Stage<Scalar = A::Scalar>,
    A::Kind: Combine<B::Kind>,
{
    type Scalar = A::Scalar;
    type View<'a>
        = NormalizationChainParams<A::StageView<'a>, B::StageView<'a>>
    where
        Self: 'a;

    fn view(&self) -> Self::View<'_> {
        NormalizationChainParams {
            first: self.first.stage_view(),
            next: self.next.stage_view(),
        }
    }
}
impl<A, B> private::Params<A::Scalar> for NormalizationChainParams<A, B>
where
    A: Stage,
    B: Stage<Scalar = A::Scalar>,
    A::Kind: Combine<B::Kind>,
{
    fn logical_entry(&self, value: A::Scalar, row: usize, column: usize) -> A::Scalar {
        self.entry(value, row, column)
    }
    fn scaled_correction(&self, value: A::Scalar, row: usize, column: usize) -> A::Scalar {
        self.next.scaled_correction(
            self.first.scaled_correction(value, row, column),
            row,
            column,
        )
    }

    fn assert_shape(&self, nrows: usize, ncols: usize) {
        self.first.assert_shape(nrows, ncols);
        self.next.assert_shape(nrows, ncols);
    }
}
impl<A, B> Stage for NormalizationChainParams<A, B>
where
    A: Stage,
    B: Stage<Scalar = A::Scalar>,
    A::Kind: Combine<B::Kind>,
{
    type Kind = <A::Kind as Combine<B::Kind>>::Output;
    type StageView<'a>
        = NormalizationChainParams<A::StageView<'a>, B::StageView<'a>>
    where
        Self: 'a;
    type NewRows = NormalizationChainParams<A::NewRows, B::NewRows>;

    fn stage_view(&self) -> Self::StageView<'_> {
        self.view()
    }
    fn entry(&self, value: Self::Scalar, row: usize, column: usize) -> Self::Scalar {
        self.next
            .entry(self.first.entry(value, row, column), row, column)
    }
    fn assert_columns(&self, ncols: usize) {
        self.first.assert_columns(ncols);
        self.next.assert_columns(ncols);
    }
    fn prepare_dense<M, P>(self, data: &M, prefix: &P) -> Result<Self::NewRows, M::Error>
    where
        M: MatrixRead<Self::Scalar> + MatrixErrorType,
        P: Stage<Scalar = Self::Scalar>,
    {
        let first = self.first.prepare_dense(data, prefix)?;
        let combined = Prefix(prefix, &first);
        let next = self.next.prepare_dense(data, &combined)?;
        Ok(NormalizationChainParams { first, next })
    }
}

struct Identity<F>(PhantomData<F>);
impl<F: Scalar> NormalizationParams for Identity<F> {
    type Scalar = F;
    type View<'a>
        = Self
    where
        Self: 'a;

    fn view(&self) -> Self {
        Self(PhantomData)
    }
}
impl<F: Scalar> private::Params<F> for Identity<F> {
    fn logical_entry(&self, value: F, _row: usize, _column: usize) -> F {
        value
    }
    fn scaled_correction(&self, value: F, _row: usize, _column: usize) -> F {
        value
    }

    fn assert_shape(&self, _nrows: usize, _ncols: usize) {}
}
impl<F: Scalar> Stage for Identity<F> {
    type Kind = Columns;
    type StageView<'a>
        = Self
    where
        Self: 'a;
    type NewRows = Self;
    fn stage_view(&self) -> Self {
        Self(PhantomData)
    }
    fn entry(&self, value: F, _row: usize, _column: usize) -> F {
        value
    }
    fn assert_columns(&self, _ncols: usize) {}
    fn is_identity(&self) -> bool {
        true
    }
    fn prepare_dense<M, P>(self, _data: &M, _prefix: &P) -> Result<Self, M::Error>
    where
        M: MatrixRead<F> + MatrixErrorType,
        P: Stage<Scalar = F>,
    {
        Ok(self)
    }
}

// A fitting prefix need not expose a new public parameter tree.
struct Prefix<'a, A, B>(&'a A, &'a B);
impl<A: Stage, B: Stage<Scalar = A::Scalar>> private::Params<A::Scalar> for Prefix<'_, A, B> {
    fn logical_entry(&self, value: A::Scalar, row: usize, column: usize) -> A::Scalar {
        self.entry(value, row, column)
    }
    fn scaled_correction(&self, value: A::Scalar, row: usize, column: usize) -> A::Scalar {
        self.1
            .scaled_correction(self.0.scaled_correction(value, row, column), row, column)
    }

    fn assert_shape(&self, nrows: usize, ncols: usize) {
        self.0.assert_shape(nrows, ncols);
        self.1.assert_shape(nrows, ncols);
    }
}
impl<A: Stage, B: Stage<Scalar = A::Scalar>> NormalizationParams for Prefix<'_, A, B> {
    type Scalar = A::Scalar;
    type View<'a>
        = Prefix<'a, A, B>
    where
        Self: 'a;

    fn view(&self) -> Self::View<'_> {
        Prefix(self.0, self.1)
    }
}
impl<A: Stage, B: Stage<Scalar = A::Scalar>> Stage for Prefix<'_, A, B> {
    type Kind = Rows;
    type StageView<'a>
        = Prefix<'a, A, B>
    where
        Self: 'a;
    type NewRows = Self;
    fn stage_view(&self) -> Self::StageView<'_> {
        Prefix(self.0, self.1)
    }
    fn entry(&self, value: Self::Scalar, row: usize, column: usize) -> Self::Scalar {
        self.1.entry(self.0.entry(value, row, column), row, column)
    }
    fn assert_columns(&self, ncols: usize) {
        self.0.assert_columns(ncols);
        self.1.assert_columns(ncols);
    }
    fn prepare_dense<M, P>(self, _data: &M, _prefix: &P) -> Result<Self, M::Error>
    where
        M: MatrixRead<Self::Scalar> + MatrixErrorType,
        P: Stage<Scalar = Self::Scalar>,
    {
        Ok(self)
    }
}

pub trait RecipeType<F: Scalar> {
    type Params: Stage<Scalar = F>;
}
impl<F: Scalar> RecipeType<F> for ColumnNormalization {
    type Params = ColumnNormalizationParams<F>;
}
impl<F: Scalar> RecipeType<F> for RowNormalization {
    type Params = FittedRowNormalization<RowNormalizationParams<F>>;
}
impl<A, B, F> RecipeType<F> for NormalizationChain<A, B>
where
    F: Scalar,
    A: RecipeType<F>,
    B: RecipeType<F>,
    <A::Params as Stage>::Kind: Combine<<B::Params as Stage>::Kind>,
{
    type Params = NormalizationChainParams<A::Params, B::Params>;
}

pub trait Fit<M: MatrixErrorType, F: Scalar> {
    type Params: Stage<Scalar = F>;
    fn fit(self, data: &M) -> Result<Self::Params, M::Error>;
}
impl<R, M, F> Fit<M, F> for R
where
    M: MatrixShape + MatrixErrorType,
    F: Scalar,
    R: RecipeType<F>,
    <R::Params as Stage>::Kind: FitKind<M, F, R>,
{
    type Params = R::Params;
    fn fit(self, data: &M) -> Result<Self::Params, M::Error> {
        <R::Params as Stage>::Kind::fit(self, data)
    }
}

pub trait FitKind<M: MatrixErrorType, F: Scalar, R: RecipeType<F>> {
    fn fit(recipe: R, data: &M) -> Result<R::Params, M::Error>;
}
impl<M, F, R> FitKind<M, F, R> for Rows
where
    M: MatrixRead<F> + MatrixErrorType,
    F: Scalar,
    R: dense::FitDense<M, F>,
{
    fn fit(recipe: R, data: &M) -> Result<R::Params, M::Error> {
        recipe.fit_dense(data, &Identity(PhantomData))
    }
}
impl<M, F, R> FitKind<M, F, R> for Columns
where
    M: Source<F>,
    F: Scalar,
    R: stats::FitColumns<M, F>,
{
    fn fit(recipe: R, data: &M) -> Result<R::Params, M::Error> {
        recipe.fit_columns(data, &Identity(PhantomData))
    }
}

impl<M, A, B, P, Q> private::Recipe<M, NormalizationChainParams<P, Q>> for NormalizationChain<A, B>
where
    M: MatrixShape + MatrixErrorType,
    P: Stage,
    Q: Stage<Scalar = P::Scalar>,
    P::Kind: Combine<Q::Kind>,
    Self: Fit<M, P::Scalar, Params = NormalizationChainParams<P, Q>>,
{
    fn fit_recipe(self, data: &M) -> Result<NormalizationChainParams<P, Q>, M::Error> {
        Fit::fit(self, data)
    }
}

pub trait Prepare<M: MatrixErrorType, P: Stage> {
    fn prepare(params: P, data: &M) -> Result<P::NewRows, M::Error>;
}
impl<M, P> Prepare<M, P> for Rows
where
    P: Stage,
    M: MatrixRead<P::Scalar> + MatrixErrorType,
{
    fn prepare(params: P, data: &M) -> Result<P::NewRows, M::Error> {
        params.prepare_dense(data, &Identity(PhantomData))
    }
}
pub trait ReuseColumns: Stage {
    fn reuse(self) -> Self::NewRows;
}
impl<F: Scalar, S: AsRef<[F]>> ReuseColumns for ColumnNormalizationParams<F, S> {
    fn reuse(self) -> Self {
        self
    }
}
impl<A, B> ReuseColumns for NormalizationChainParams<A, B>
where
    A: ReuseColumns<Kind = Columns>,
    B: ReuseColumns<Scalar = A::Scalar, Kind = Columns>,
{
    fn reuse(self) -> Self::NewRows {
        NormalizationChainParams {
            first: self.first.reuse(),
            next: self.next.reuse(),
        }
    }
}
impl<M, P> Prepare<M, P> for Columns
where
    P: ReuseColumns,
    M: MatrixShape + MatrixErrorType,
{
    fn prepare(params: P, _data: &M) -> Result<P::NewRows, M::Error> {
        Ok(params.reuse())
    }
}

impl<M, F, S> private::NewSamples<M> for ColumnNormalizationParams<F, S>
where
    M: MatrixShape + MatrixErrorType,
    F: Scalar,
    S: AsRef<[F]>,
{
}

impl<M, F, S> PrepareNewSamples<M> for ColumnNormalizationParams<F, S>
where
    M: MatrixShape + MatrixErrorType,
    F: Scalar,
    S: AsRef<[F]>,
{
    type Params = Self;

    fn for_new_samples(self, data: M) -> Result<LazyMatrix<M, Self>, M::Error> {
        Ok(LazyMatrix::from_parts(data, self))
    }
}

impl<M, A, B> private::NewSamples<M> for NormalizationChainParams<A, B>
where
    M: MatrixShape + MatrixErrorType,
    Self: Stage,
    <Self as Stage>::Kind: Prepare<M, Self>,
{
}

impl<M, A, B> PrepareNewSamples<M> for NormalizationChainParams<A, B>
where
    M: MatrixShape + MatrixErrorType,
    Self: Stage,
    <Self as Stage>::Kind: Prepare<M, Self>,
{
    type Params = <Self as Stage>::NewRows;

    fn for_new_samples(self, data: M) -> Result<LazyMatrix<M, Self::Params>, M::Error> {
        self.assert_columns(data.ncols());
        let prepared = <Self as Stage>::Kind::prepare(self, &data)?;
        Ok(LazyMatrix::from_parts(data, prepared))
    }
}

pub trait Execute<M: MatrixErrorType, P: Stage> {
    fn product<
        X: crate::VectorView<P::Scalar> + ?Sized,
        Y: crate::VectorViewMut<P::Scalar> + ?Sized,
    >(
        data: &M,
        params: &P,
        transpose: bool,
        x: &X,
        out: &mut Y,
    ) -> Result<(), M::Error>;
}

macro_rules! products {
    ($alloc:ident, $allocate:ident, $into:ident, $write:ident, $transpose:expr, $size:ident) => {
        impl<M, A, B, V> crate::$alloc<V> for LazyMatrix<M, NormalizationChainParams<A, B>>
        where
            M: MatrixShape + MatrixErrorType,
            A: Stage,
            B: Stage<Scalar = A::Scalar>,
            A::Kind: Combine<B::Kind>,
            <NormalizationChainParams<A, B> as Stage>::Kind:
                Execute<M, NormalizationChainParams<A, B>>,
            V: crate::VectorOwned<A::Scalar, Owned = V> + crate::VectorViewMut<A::Scalar>,
        {
            fn $allocate(&self, x: &V) -> Result<V, M::Error> {
                let input = if $transpose {
                    self.nrows()
                } else {
                    self.ncols()
                };
                assert_eq!(
                    <V as crate::VectorView<A::Scalar>>::len(x),
                    input,
                    "pipeline input dimension mismatch"
                );
                let mut out = V::owned_from_fn(self.$size(), |_| A::Scalar::zero());
                crate::$into::$write(self, x, &mut out)?;
                Ok(out)
            }
        }
        impl<M, A, B, X, Y> crate::$into<X, Y> for LazyMatrix<M, NormalizationChainParams<A, B>>
        where
            M: MatrixShape + MatrixErrorType,
            A: Stage,
            B: Stage<Scalar = A::Scalar>,
            A::Kind: Combine<B::Kind>,
            <NormalizationChainParams<A, B> as Stage>::Kind:
                Execute<M, NormalizationChainParams<A, B>>,
            X: crate::VectorView<A::Scalar>,
            Y: crate::VectorViewMut<A::Scalar>,
        {
            fn $write(&self, x: &X, out: &mut Y) -> Result<(), M::Error> {
                <NormalizationChainParams<A, B> as Stage>::Kind::product(
                    self.data(),
                    self.normalization(),
                    $transpose,
                    x,
                    out,
                )
            }
        }
    };
}
products!(MatVec, matvec, MatVecInto, matvec_into, false, nrows);
products!(
    MatTransposeVec,
    mat_transpose_vec,
    MatTransposeVecInto,
    mat_transpose_vec_into,
    true,
    ncols
);
