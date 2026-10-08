use crate::traits::*;
use crate::{LazyColumn, NormalizationParams};

/// An eagerly normalized dense matrix with its original fitted parameters.
///
/// The underlying data already contains normalized values. Operations delegate
/// directly to it without applying the retained parameters again. Materializing
/// entries changes arithmetic order relative to lazy products, so results may
/// differ through rounding and IEEE nonfinite arithmetic. Available capabilities
/// and operational errors come from the selected dense backend, rather than
/// the original input. Statistics describe the normalized data.
#[derive(Clone, Debug)]
pub struct EagerMatrix<M, F = f64> {
    data: M,
    normalization: NormalizationParams<F>,
}

impl<M: MatrixShape, F: Scalar> EagerMatrix<M, F> {
    pub(crate) fn from_normalized(data: M, normalization: NormalizationParams<F>) -> Self {
        Self {
            data,
            normalization,
        }
    }

    /// Number of rows of the normalized matrix.
    pub fn nrows(&self) -> usize {
        self.data.nrows()
    }
    /// Number of columns of the normalized matrix.
    pub fn ncols(&self) -> usize {
        self.data.ncols()
    }
    /// Borrow the already normalized backend data.
    pub fn data(&self) -> &M {
        &self.data
    }
    /// Borrow the original fitted normalization, for example for prediction.
    pub fn normalization(&self) -> &NormalizationParams<F> {
        &self.normalization
    }
    /// Original fitted centers, already applied to the data.
    pub fn centers(&self) -> Option<&[F]> {
        self.normalization.centers()
    }
    /// Original fitted scales, already applied to the data.
    pub fn scales(&self) -> Option<&[F]> {
        self.normalization.scales()
    }
    /// Recover normalized data and original fitted center and scale vectors.
    pub fn into_parts(self) -> (M, Option<Vec<F>>, Option<Vec<F>>) {
        let (centers, scales) = self.normalization.into_parts();
        (self.data, centers, scales)
    }
    /// Borrow a logical column of the already normalized data.
    ///
    /// Its center and scale are zero and one, respectively. The fitted
    /// parameters returned by the matrix accessors are metadata only.
    pub fn column(&self, j: usize) -> LazyColumn<M::Column<'_>, F>
    where
        M: RawColumns<F>,
    {
        assert!(j < self.ncols(), "column index out of bounds");
        LazyColumn::new(self.data.raw_column(j), F::zero(), F::one())
    }
}

impl<M: RawColumns<F>, F: Scalar> Columns<F> for EagerMatrix<M, F> {
    type Column<'a>
        = LazyColumn<M::Column<'a>, F>
    where
        Self: 'a;
    fn column(&self, j: usize) -> Self::Column<'_> {
        EagerMatrix::column(self, j)
    }
}

impl<M: MatrixErrorType, T> MatrixErrorType for EagerMatrix<M, T> {
    type Error = M::Error;
}

impl<M: MatrixShape, T> MatrixShape for EagerMatrix<M, T> {
    fn nrows(&self) -> usize {
        self.data.nrows()
    }

    fn ncols(&self) -> usize {
        self.data.ncols()
    }
}

impl<M, V, T> MatVec<V> for EagerMatrix<M, T>
where
    M: MatVec<V>,
{
    fn matvec(&self, x: &V) -> Result<V, Self::Error> {
        self.data.matvec(x)
    }
}

impl<M, X, Y, T> MatVecInto<X, Y> for EagerMatrix<M, T>
where
    M: MatVecInto<X, Y>,
{
    fn matvec_into(&self, x: &X, out: &mut Y) -> Result<(), Self::Error> {
        self.data.matvec_into(x, out)
    }
    fn matvec_normalized_into<F: Scalar>(
        &self,
        x: &X,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut Y,
    ) -> Result<(), Self::Error>
    where
        X: Clone + ElemDivAssign<F> + DotSlice<F>,
        Y: SubScalarAssign<F>,
    {
        self.data.matvec_normalized_into(x, centers, scales, out)
    }
}

impl<M, X, Y, F, T> MatVecScaledInto<X, Y, F> for EagerMatrix<M, T>
where
    M: MatVecScaledInto<X, Y, F>,
{
    fn matvec_scaled_into(&self, alpha: F, x: &X, beta: F, out: &mut Y) -> Result<(), Self::Error> {
        self.data.matvec_scaled_into(alpha, x, beta, out)
    }
}

impl<M, V, T> MatTransposeVec<V> for EagerMatrix<M, T>
where
    M: MatTransposeVec<V>,
{
    fn mat_transpose_vec(&self, x: &V) -> Result<V, Self::Error> {
        self.data.mat_transpose_vec(x)
    }
}

impl<M, X, Y, T> MatTransposeVecInto<X, Y> for EagerMatrix<M, T>
where
    M: MatTransposeVecInto<X, Y>,
{
    fn mat_transpose_vec_into(&self, x: &X, out: &mut Y) -> Result<(), Self::Error> {
        self.data.mat_transpose_vec_into(x, out)
    }
    fn mat_transpose_vec_normalized_into<F: Scalar>(
        &self,
        x: &X,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut Y,
    ) -> Result<(), Self::Error>
    where
        X: SumEntries<F>,
        Y: ScaledSubSlice<F> + ElemDivAssign<F>,
    {
        self.data
            .mat_transpose_vec_normalized_into(x, centers, scales, out)
    }
}

impl<M, X, Y, F, T> MatTransposeVecScaledInto<X, Y, F> for EagerMatrix<M, T>
where
    M: MatTransposeVecScaledInto<X, Y, F>,
{
    fn mat_transpose_vec_scaled_into(
        &self,
        alpha: F,
        x: &X,
        beta: F,
        out: &mut Y,
    ) -> Result<(), Self::Error> {
        self.data.mat_transpose_vec_scaled_into(alpha, x, beta, out)
    }
}

impl<M, F, T> RawColumns<F> for EagerMatrix<M, T>
where
    M: RawColumns<F>,
    F: Scalar,
{
    type Column<'a>
        = M::Column<'a>
    where
        Self: 'a;

    fn raw_column(&self, j: usize) -> Self::Column<'_> {
        self.data.raw_column(j)
    }
}

impl<M, F, T> ColumnStats<F> for EagerMatrix<M, T>
where
    M: ColumnStats<F>,
    F: Scalar,
{
    fn normalization_stats(
        &self,
        spec: Normalization,
    ) -> Result<crate::NormalizationStats<F>, Self::Error> {
        self.data.normalization_stats(spec)
    }

    fn col_means(&self) -> Result<Vec<F>, Self::Error> {
        self.data.col_means()
    }

    fn col_sds(&self) -> Result<Vec<F>, Self::Error> {
        self.data.col_sds()
    }

    fn col_mins(&self) -> Result<Vec<F>, Self::Error> {
        self.data.col_mins()
    }

    fn col_ranges(&self) -> Result<Vec<F>, Self::Error> {
        self.data.col_ranges()
    }

    fn col_maxabs(&self) -> Result<Vec<F>, Self::Error> {
        self.data.col_maxabs()
    }

    fn col_l1(&self) -> Result<Vec<F>, Self::Error> {
        self.data.col_l1()
    }

    fn col_l2(&self) -> Result<Vec<F>, Self::Error> {
        self.data.col_l2()
    }

    fn col_l2_centered(&self, centers: &[F]) -> Result<Vec<F>, Self::Error> {
        self.data.col_l2_centered(centers)
    }

    fn col_l1_centered(&self, centers: &[F]) -> Result<Vec<F>, Self::Error> {
        self.data.col_l1_centered(centers)
    }

    fn col_maxabs_centered(&self, centers: &[F]) -> Result<Vec<F>, Self::Error> {
        self.data.col_maxabs_centered(centers)
    }
}

impl<M, F, T> WeightedGramInto<F> for EagerMatrix<M, T>
where
    M: WeightedGramInto<F>,
    F: Scalar,
{
    fn weighted_gram_into<W, O>(&self, weights: &W, out: &mut O) -> Result<(), Self::Error>
    where
        W: VectorView<F> + ?Sized,
        O: MatrixWrite<F> + ?Sized,
    {
        self.data.weighted_gram_into(weights, out)
    }
}

impl<M, F, T> WeightedGramKernel<F> for EagerMatrix<M, T>
where
    M: WeightedGramKernel<F>,
    F: Scalar,
{
    fn weighted_gram_normalized_into<W, O>(
        &self,
        weights: &W,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut O,
    ) -> Result<(), Self::Error>
    where
        W: VectorView<F> + ?Sized,
        O: MatrixWrite<F> + ?Sized,
    {
        self.data
            .weighted_gram_normalized_into(weights, centers, scales, out)
    }
}

impl<M: WeightedColumnSumsInto<F>, F: Scalar, T> WeightedColumnSumsInto<F> for EagerMatrix<M, T> {
    fn weighted_column_sums_into<W, O>(&self, weights: &W, out: &mut O) -> Result<(), Self::Error>
    where
        W: VectorView<F> + ?Sized,
        O: VectorViewMut<F> + ?Sized,
    {
        self.data.weighted_column_sums_into(weights, out)
    }
}

impl<M: WeightedColumnSumsKernel<F>, F: Scalar, T> WeightedColumnSumsKernel<F>
    for EagerMatrix<M, T>
{
    fn weighted_column_sums_normalized_into<W, O>(
        &self,
        weights: &W,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut O,
    ) -> Result<(), Self::Error>
    where
        W: VectorView<F> + ?Sized,
        O: VectorViewMut<F> + ?Sized,
    {
        self.data
            .weighted_column_sums_normalized_into(weights, centers, scales, out)
    }
}

impl<M: MaterializeDense<F>, F: Scalar, T> MaterializeDense<F> for EagerMatrix<M, T> {
    fn materialize_normalized_into<O: MatrixWrite<F> + ?Sized>(
        &self,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut O,
    ) -> Result<(), Self::Error> {
        self.data.materialize_normalized_into(centers, scales, out)
    }
}
