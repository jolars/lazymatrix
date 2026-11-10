use crate::column::{LazyColumn, LazySparseColumn, SparseColumnRef};
use crate::normalization::{
    ColumnNormalizationParams, NormalizationParams, NormalizationRecipe, RowNormalizationParams,
};
use crate::row::LazyRow;
use crate::traits::{
    Columns, DotSlice, ElemDivAssign, MatTransposeVec, MatTransposeVecInto,
    MatTransposeVecScaledInto, MatVec, MatVecInto, MatVecScaledInto, MatrixShape, RawColumns,
    Scalar, ScaledSubSlice, SparseColumns, SparseRows, SubScalarAssign, SumEntries, VectorOwned,
    VectorView, VectorViewMut,
};

mod columns;
mod eager;
mod row;

/// A matrix presented with lazy row, column, or ordered normalization.
///
/// The underlying matrix `data` is never modified or densified; the centers and
/// scales are folded into the matrix–vector products on the fly. See the
/// [crate-level documentation](crate) for the math.
///
/// The fitted parameter type selects the axis and scalar. Column normalization
/// retains the backend's existing operator capabilities. Row fitting uses
/// [`crate::RowStats`], and vector products use [`crate::RowNormalizationKernel`]
/// over dense, sparse, and chunked Zarr storage. See [`crate::RowNormalization`] for the row
/// capability limits. Ordered recipes use [`crate::NormalizationChainParams`].
/// Fitting and products for mixed and row-only chains require dense storage;
/// column-only chains also support sparse and chunked storage. Borrowed rows
/// require only [`SparseRows`], and columns require [`RawColumns`] or
/// [`SparseColumns`], including with explicit mixed chains. Every stage remains
/// visible through [`Self::normalization`].
///
/// Access centers and scales through [`Self::normalization`]. Each component is
/// independently optional. Negative scales and nonfinite values are preserved.
/// Supply explicit vectors with [`Self::from_column_parts`] or
/// [`Self::from_row_parts`]; each constructor infers the dimension from `data`.
///
/// ```
/// # #[cfg(feature = "ndarray_v0_17")]
/// # {
/// use lazymatrix::{ColumnNormalization, LazyMatrix, RowNormalization};
/// use ndarray::array;
///
/// let data = array![[1.0, 3.0], [10.0, 14.0]];
/// let columns = LazyMatrix::new(data.view(), ColumnNormalization::standardize()).unwrap();
/// let rows = LazyMatrix::new(data.view(), RowNormalization::standardize()).unwrap();
/// assert_eq!(columns.normalization().centers(), Some([5.5, 8.5].as_slice()));
/// assert_eq!(rows.normalization().centers(), Some([2.0, 12.0].as_slice()));
/// let (raw, params) = rows.into_parts();
/// let rows = LazyMatrix::from_parts(raw, params);
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct LazyMatrix<M, P = ColumnNormalizationParams<f64>> {
    data: M,
    normalization: P,
}

impl<M: MatrixShape, P: NormalizationParams> LazyMatrix<M, P> {
    /// Fit normalization parameters according to a built-in or custom recipe.
    ///
    /// Built-in scales use centered values when centering is active. Exact
    /// computed zero scales become one; negative and nonfinite values are
    /// preserved. Built-in column recipes use [`crate::ColumnStats`], while
    /// row recipes use [`crate::RowStats`]. Custom recipes implement
    /// [`NormalizationRecipe`] and return supported fitted parameter types.
    /// To fit parameters separately, use [`crate::ColumnNormalization::fit`]
    /// or [`crate::RowNormalization::fit`].
    ///
    /// # Errors
    /// Returns the recipe's fitting error. Built-in recipes use the backend error.
    ///
    /// # Panics
    /// Panics if the fitted parameter dimensions do not match the matrix.
    pub fn new<R>(data: M, recipe: R) -> Result<Self, R::Error>
    where
        M: crate::MatrixErrorType,
        R: NormalizationRecipe<M, P>,
    {
        let normalization = recipe.fit(&data)?;
        Ok(Self::from_parts(data, normalization))
    }

    /// Wrap raw data using explicit or previously fitted parameters.
    ///
    /// Parameters are applied unchanged, without fitting or copying. Column
    /// parameters can be reused on new samples with the same column count. Row
    /// parameters describe particular rows; fit new samples independently.
    /// This constructor reverses [`Self::into_parts`]. To supply center and
    /// scale vectors directly, use [`Self::from_column_parts`] or
    /// [`Self::from_row_parts`].
    ///
    /// # Panics
    /// Panics if the matrix dimension along the fitted axis differs from the
    /// parameter count. Constructing parameter objects rejects exact zero scales.
    pub fn from_parts(data: M, normalization: P) -> Self {
        normalization.assert_shape(data.nrows(), data.ncols());
        Self {
            data,
            normalization,
        }
    }
}

impl<M, P> LazyMatrix<M, P> {
    /// Borrow the fitted parameters without copying.
    pub fn normalization(&self) -> &P {
        &self.normalization
    }

    /// Borrow the underlying unnormalized matrix.
    pub fn data(&self) -> &M {
        &self.data
    }

    /// Recover the underlying matrix and fitted parameters without copying.
    /// Rebuild the operator with [`Self::from_parts`].
    pub fn into_parts(self) -> (M, P) {
        (self.data, self.normalization)
    }
}

impl<M: MatrixShape, P> LazyMatrix<M, P> {
    /// Number of rows of the logical normalized matrix.
    pub fn nrows(&self) -> usize {
        self.data.nrows()
    }

    /// Number of columns of the logical normalized matrix.
    pub fn ncols(&self) -> usize {
        self.data.ncols()
    }
}

impl<M: MatrixShape, F: Scalar> LazyMatrix<M, RowNormalizationParams<F>> {
    /// Wrap raw data using explicit row centers and scales.
    ///
    /// The row count comes from `data`. Centers and scales are independently
    /// optional and are preserved unchanged, without fitting or copying.
    /// Negative scales and nonfinite values are allowed.
    ///
    /// # Panics
    /// Panics if either vector has a different length from `data.nrows()`, or
    /// if a scale is exact positive or negative zero.
    pub fn from_row_parts(data: M, centers: Option<Vec<F>>, scales: Option<Vec<F>>) -> Self {
        let normalization = RowNormalizationParams::from_parts(data.nrows(), centers, scales);
        Self::from_parts(data, normalization)
    }
}

impl<M: MatrixShape, F: Scalar> LazyMatrix<M, ColumnNormalizationParams<F>> {
    /// Wrap raw data using explicit column centers and scales.
    ///
    /// The column count comes from `data`. Centers and scales are independently
    /// optional and are preserved unchanged, without fitting or copying.
    /// Negative scales and nonfinite values are allowed.
    ///
    /// # Panics
    /// Panics if either vector has a different length from `data.ncols()`, or
    /// if a scale is exact positive or negative zero.
    pub fn from_column_parts(data: M, centers: Option<Vec<F>>, scales: Option<Vec<F>>) -> Self {
        let normalization = ColumnNormalizationParams::from_parts(data.ncols(), centers, scales);
        Self::from_parts(data, normalization)
    }
}

impl<M: MatrixShape, F: Scalar, S: AsRef<[F]>> LazyMatrix<M, ColumnNormalizationParams<F, S>> {
    /// Borrow one lazily normalized column without copying.
    pub fn column(&self, j: usize) -> LazyColumn<M::Column<'_>, F>
    where
        M: RawColumns<F>,
    {
        assert!(j < self.ncols(), "column index out of bounds");
        LazyColumn::new(
            self.data.raw_column(j),
            self.normalization.centers().map_or_else(F::zero, |c| c[j]),
            self.normalization.scales().map_or_else(F::one, |s| s[j]),
        )
    }

    /// Borrow one lazily normalized CSC column with its sparse representation.
    pub fn sparse_column(&self, j: usize) -> LazySparseColumn<'_, F>
    where
        M: SparseColumns<F>,
    {
        assert!(j < self.ncols(), "column index out of bounds");
        let (row_indices, values) = self.data.sparse_column(j);
        LazyColumn::new(
            SparseColumnRef::new(row_indices, values, self.nrows()),
            self.normalization.centers().map_or_else(F::zero, |c| c[j]),
            self.normalization.scales().map_or_else(F::one, |s| s[j]),
        )
    }
}

impl<M: MatrixShape, P: NormalizationParams> LazyMatrix<M, P> {
    /// Borrow one lazily normalized sparse row without copying.
    ///
    /// This takes O(1) time and allocates nothing. The view borrows the raw
    /// stored column indices and values and the saved normalization parameters.
    /// Row and composed normalization apply every stage in its saved order.
    /// This also supports explicit chains on sparse storage without requiring
    /// the backend capabilities used to fit or multiply that chain.
    /// Centering generally makes the logical row dense; [`LazyRow`] exposes
    /// its sparse-plus-affine representation explicitly.
    ///
    /// ```
    /// # #[cfg(feature = "sprs_all")]
    /// # {
    /// use lazymatrix::{LazyMatrix, SprsCsr};
    /// use sprs::CsMat;
    ///
    /// let x = CsMat::new((1, 3), vec![0, 2], vec![0, 2], vec![1.0, 0.0]);
    /// let csr = SprsCsr::try_new(x.view()).unwrap();
    /// let lazy = LazyMatrix::from_column_parts(
    ///     csr, Some(vec![0.5, -1.0, 2.0]), Some(vec![2.0; 3]),
    /// );
    /// let row = lazy.row(0);
    /// assert_eq!(row.len(), 3);
    /// assert_eq!(row.column_indices(), &[0, 2]);
    /// assert_eq!(row.values(), &[1.0, 0.0]);
    /// assert_eq!(row.implicit_value(1), 0.5);
    /// assert_eq!(row.stored_corrections().collect::<Vec<_>>(), vec![(0, 0.5), (2, 0.0)]);
    /// # }
    /// ```
    ///
    /// Generic consumers need only public metadata and storage bounds:
    ///
    /// ```
    /// use lazymatrix::{LazyMatrix, NormalizationParams, SparseRows};
    /// fn value<P: NormalizationParams, M: SparseRows<P::Scalar>>(
    ///     matrix: &LazyMatrix<M, P>, i: usize, j: usize,
    /// ) -> P::Scalar {
    ///     matrix.row(i).get(j)
    /// }
    /// ```
    ///
    /// Row access requires contiguous sparse-row storage:
    ///
    /// ```compile_fail
    /// use lazymatrix::{LazyMatrix, MatrixShape};
    ///
    /// fn row_without_sparse_rows<M: MatrixShape>(matrix: &LazyMatrix<M>) {
    ///     let _ = matrix.row(0);
    /// }
    /// ```
    ///
    /// The same storage capability is required with row parameters:
    ///
    /// ```compile_fail
    /// use lazymatrix::{LazyMatrix, MatrixRead, RowNormalizationParams};
    /// fn no_sparse_rows<M: MatrixRead<f64>>(matrix: &LazyMatrix<M, RowNormalizationParams>) {
    ///     let _ = matrix.row(0);
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.nrows()`.
    pub fn row(&self, i: usize) -> LazyRow<'_, P::Scalar, P>
    where
        M: SparseRows<P::Scalar>,
    {
        assert!(i < self.nrows(), "row index out of bounds");
        let (column_indices, values) = self.data.sparse_row(i);
        LazyRow::new(column_indices, values, self.ncols(), &self.normalization, i)
    }
}

impl<M, P> MatrixShape for LazyMatrix<M, P>
where
    M: MatrixShape,
{
    fn nrows(&self) -> usize {
        self.data.nrows()
    }

    fn ncols(&self) -> usize {
        self.data.ncols()
    }
}

impl<M, F, S: AsRef<[F]>> Columns<F> for LazyMatrix<M, ColumnNormalizationParams<F, S>>
where
    F: Scalar,
    M: RawColumns<F>,
{
    type Column<'a>
        = LazyColumn<M::Column<'a>, F>
    where
        Self: 'a;

    fn column(&self, j: usize) -> Self::Column<'_> {
        LazyMatrix::<M, ColumnNormalizationParams<F, S>>::column(self, j)
    }
}

impl<M, V, F, S: AsRef<[F]>> MatVec<V> for LazyMatrix<M, ColumnNormalizationParams<F, S>>
where
    F: Scalar,
    M: MatVec<V>,
    V: Clone + ElemDivAssign<F> + DotSlice<F> + SubScalarAssign<F>,
{
    /// `X̃ v = X (S⁻¹ v) − 1 · (cᵀ S⁻¹ v)`.
    fn matvec(&self, v: &V) -> Result<V, Self::Error> {
        self.data
            .matvec_normalized(v, self.normalization.centers(), self.normalization.scales())
    }
}

impl<M, X, Y, F, S: AsRef<[F]>> MatVecInto<X, Y> for LazyMatrix<M, ColumnNormalizationParams<F, S>>
where
    F: Scalar,
    M: MatVecInto<X, Y>,
    X: Clone + ElemDivAssign<F> + DotSlice<F>,
    Y: SubScalarAssign<F>,
{
    /// `out = X̃ v = X (S⁻¹ v) − 1 · (cᵀ S⁻¹ v)`.
    fn matvec_into(&self, v: &X, out: &mut Y) -> Result<(), Self::Error> {
        self.data.matvec_normalized_into(
            v,
            self.normalization.centers(),
            self.normalization.scales(),
            out,
        )
    }
}

impl<M, V, F, S: AsRef<[F]>> MatTransposeVec<V> for LazyMatrix<M, ColumnNormalizationParams<F, S>>
where
    F: Scalar,
    M: MatTransposeVec<V>,
    V: SumEntries<F> + ScaledSubSlice<F> + ElemDivAssign<F>,
{
    /// `X̃ᵀ u = S⁻¹ (Xᵀ u − c · Σu)`.
    fn mat_transpose_vec(&self, u: &V) -> Result<V, Self::Error> {
        self.data.mat_transpose_vec_normalized(
            u,
            self.normalization.centers(),
            self.normalization.scales(),
        )
    }
}

impl<M, X, Y, F, S: AsRef<[F]>> MatTransposeVecInto<X, Y>
    for LazyMatrix<M, ColumnNormalizationParams<F, S>>
where
    F: Scalar,
    M: MatTransposeVecInto<X, Y>,
    X: SumEntries<F>,
    Y: ScaledSubSlice<F> + ElemDivAssign<F>,
{
    /// `out = X̃ᵀ u = S⁻¹ (Xᵀ u − c · Σu)`.
    fn mat_transpose_vec_into(&self, u: &X, out: &mut Y) -> Result<(), Self::Error> {
        self.data.mat_transpose_vec_normalized_into(
            u,
            self.normalization.centers(),
            self.normalization.scales(),
            out,
        )
    }
}

impl<M, F: Scalar, S: AsRef<[F]>> LazyMatrix<M, ColumnNormalizationParams<F, S>>
where
    M: MatrixShape,
{
    /// Apply `out = alpha * X̃ * x + beta * out` using caller-owned coefficient scratch.
    ///
    /// `scratch` must have length `ncols`. Backends may use it for column scaling
    /// or leave it unused. Its contents after the call are unspecified.
    pub fn matvec_scaled_with_workspace<X, Y>(
        &self,
        alpha: F,
        x: &X,
        beta: F,
        out: &mut Y,
        scratch: &mut X::Owned,
    ) -> Result<(), M::Error>
    where
        X: VectorOwned<F>,
        Y: VectorViewMut<F>,
        M: MatVecScaledInto<X, Y, F> + MatVecScaledInto<X::Owned, Y, F>,
    {
        assert_eq!(
            x.len(),
            self.ncols(),
            "matvec_scaled_into: dimension mismatch"
        );
        assert_eq!(
            out.len(),
            self.nrows(),
            "matvec_scaled_into: output dimension mismatch"
        );
        assert_eq!(
            scratch.len(),
            self.ncols(),
            "matvec_scaled_into: scratch dimension mismatch"
        );
        if let Some(result) = self.data.try_matvec_scaled_normalized_into(
            alpha,
            x,
            beta,
            self.normalization.centers(),
            self.normalization.scales(),
            out,
        ) {
            return result;
        }
        if let Some(scales) = self.normalization.scales().filter(|_| alpha != F::zero()) {
            for (j, &scale) in scales.iter().enumerate() {
                scratch.set(j, x.get(j) / scale);
            }
            self.data.matvec_scaled_into(alpha, scratch, beta, out)?;
            if let Some(centers) = self.normalization.centers() {
                let correction: F = (0..self.ncols()).map(|j| scratch.get(j) * centers[j]).sum();
                for i in 0..out.len() {
                    out.set(i, out.get(i) - alpha * correction);
                }
            }
        } else {
            self.data.matvec_scaled_into(alpha, x, beta, out)?;
            if let Some(centers) = self.normalization.centers().filter(|_| alpha != F::zero()) {
                let correction: F = (0..self.ncols()).map(|j| x.get(j) * centers[j]).sum();
                for i in 0..out.len() {
                    out.set(i, out.get(i) - alpha * correction);
                }
            }
        }
        Ok(())
    }

    /// Apply `out = alpha * X̃ᵀ * x + beta * out` using caller-owned scratch.
    ///
    /// `scratch` must have length `ncols`. Backends may use it to keep the raw
    /// transpose product separate from previous output values, or leave it
    /// unused. Its contents after the call are unspecified.
    pub fn mat_transpose_vec_scaled_with_workspace<X, Y>(
        &self,
        alpha: F,
        x: &X,
        beta: F,
        out: &mut Y,
        scratch: &mut Y::Owned,
    ) -> Result<(), M::Error>
    where
        X: VectorView<F>,
        Y: VectorOwned<F> + VectorViewMut<F>,
        M: MatTransposeVecScaledInto<X, Y, F> + MatTransposeVecScaledInto<X, Y::Owned, F>,
    {
        assert_eq!(
            x.len(),
            self.nrows(),
            "mat_transpose_vec_scaled_into: dimension mismatch"
        );
        assert_eq!(
            out.len(),
            self.ncols(),
            "mat_transpose_vec_scaled_into: output dimension mismatch"
        );
        assert_eq!(
            scratch.len(),
            self.ncols(),
            "mat_transpose_vec_scaled_into: scratch dimension mismatch"
        );
        if let Some(result) = self.data.try_mat_transpose_vec_scaled_normalized_into(
            alpha,
            x,
            beta,
            self.normalization.centers(),
            self.normalization.scales(),
            out,
        ) {
            return result;
        }
        if let Some(scales) = self.normalization.scales().filter(|_| alpha != F::zero()) {
            self.data
                .mat_transpose_vec_scaled_into(F::one(), x, F::zero(), scratch)?;
            let total = if self.normalization.centers().is_some() {
                x.sum()
            } else {
                F::zero()
            };
            for (j, &scale) in scales.iter().enumerate() {
                let center = self.normalization.centers().map_or_else(F::zero, |c| c[j]);
                let product = alpha * ((scratch.get(j) - center * total) / scale);
                out.set(
                    j,
                    if beta == F::zero() {
                        product
                    } else {
                        product + beta * out.get(j)
                    },
                );
            }
        } else {
            self.data
                .mat_transpose_vec_scaled_into(alpha, x, beta, out)?;
            if let Some(centers) = self.normalization.centers().filter(|_| alpha != F::zero()) {
                let total = x.sum();
                for (j, &center) in centers.iter().enumerate() {
                    out.set(j, out.get(j) - alpha * center * total);
                }
            }
        }
        Ok(())
    }
}

impl<M, X, Y, F, S: AsRef<[F]>> MatVecScaledInto<X, Y, F>
    for LazyMatrix<M, ColumnNormalizationParams<F, S>>
where
    F: Scalar,
    X: VectorOwned<F>,
    Y: VectorViewMut<F>,
    M: MatVecScaledInto<X, Y, F> + MatVecScaledInto<X::Owned, Y, F>,
{
    fn matvec_scaled_into(&self, alpha: F, x: &X, beta: F, out: &mut Y) -> Result<(), Self::Error> {
        assert_eq!(
            x.len(),
            self.ncols(),
            "matvec_scaled_into: dimension mismatch"
        );
        assert_eq!(
            out.len(),
            self.nrows(),
            "matvec_scaled_into: output dimension mismatch"
        );
        if let Some(result) = self.data.try_matvec_scaled_normalized_into(
            alpha,
            x,
            beta,
            self.normalization.centers(),
            self.normalization.scales(),
            out,
        ) {
            return result;
        }
        if self.normalization.scales().is_some() && alpha != F::zero() {
            let mut scratch = X::owned_from_fn(self.ncols(), |_| F::zero());
            self.matvec_scaled_with_workspace(alpha, x, beta, out, &mut scratch)
        } else {
            self.data.matvec_scaled_into(alpha, x, beta, out)?;
            if let Some(centers) = self.normalization.centers().filter(|_| alpha != F::zero()) {
                let correction: F = (0..self.ncols()).map(|j| x.get(j) * centers[j]).sum();
                for i in 0..out.len() {
                    out.set(i, out.get(i) - alpha * correction);
                }
            }
            Ok(())
        }
    }
}

impl<M, X, Y, F, S: AsRef<[F]>> MatTransposeVecScaledInto<X, Y, F>
    for LazyMatrix<M, ColumnNormalizationParams<F, S>>
where
    F: Scalar,
    X: VectorView<F>,
    Y: VectorOwned<F> + VectorViewMut<F>,
    M: MatTransposeVecScaledInto<X, Y, F> + MatTransposeVecScaledInto<X, Y::Owned, F>,
{
    fn mat_transpose_vec_scaled_into(
        &self,
        alpha: F,
        x: &X,
        beta: F,
        out: &mut Y,
    ) -> Result<(), Self::Error> {
        assert_eq!(
            x.len(),
            self.nrows(),
            "mat_transpose_vec_scaled_into: dimension mismatch"
        );
        assert_eq!(
            out.len(),
            self.ncols(),
            "mat_transpose_vec_scaled_into: output dimension mismatch"
        );
        if let Some(result) = self.data.try_mat_transpose_vec_scaled_normalized_into(
            alpha,
            x,
            beta,
            self.normalization.centers(),
            self.normalization.scales(),
            out,
        ) {
            return result;
        }
        if self.normalization.scales().is_some() && alpha != F::zero() {
            let mut scratch = Y::owned_from_fn(self.ncols(), |_| F::zero());
            self.mat_transpose_vec_scaled_with_workspace(alpha, x, beta, out, &mut scratch)
        } else {
            self.data
                .mat_transpose_vec_scaled_into(alpha, x, beta, out)?;
            if let Some(centers) = self.normalization.centers().filter(|_| alpha != F::zero()) {
                let total = x.sum();
                for (j, &center) in centers.iter().enumerate() {
                    out.set(j, out.get(j) - alpha * center * total);
                }
            }
            Ok(())
        }
    }
}

impl<M: crate::MatrixErrorType, P> crate::MatrixErrorType for LazyMatrix<M, P> {
    type Error = M::Error;
}

impl<M, F, S: AsRef<[F]>> crate::WeightedGramInto<F>
    for LazyMatrix<M, ColumnNormalizationParams<F, S>>
where
    F: Scalar,
    M: crate::WeightedGramKernel<F>,
{
    fn weighted_gram_into<W, O>(&self, weights: &W, out: &mut O) -> Result<(), Self::Error>
    where
        W: crate::VectorView<F> + ?Sized,
        O: crate::MatrixWrite<F> + ?Sized,
    {
        crate::gram::validate(
            self.nrows(),
            self.ncols(),
            weights,
            self.normalization.centers(),
            self.normalization.scales(),
            out,
        );
        self.data.weighted_gram_normalized_into(
            weights,
            self.normalization.centers(),
            self.normalization.scales(),
            out,
        )
    }
}

impl<M, F, S: AsRef<[F]>> crate::WeightedColumnSumsInto<F>
    for LazyMatrix<M, ColumnNormalizationParams<F, S>>
where
    F: Scalar,
    M: crate::WeightedColumnSumsKernel<F>,
{
    fn weighted_column_sums_into<W, O>(&self, weights: &W, out: &mut O) -> Result<(), Self::Error>
    where
        W: crate::VectorView<F> + ?Sized,
        O: crate::VectorViewMut<F> + ?Sized,
    {
        crate::weighted_sums::validate(
            self.nrows(),
            self.ncols(),
            weights,
            self.normalization.centers(),
            self.normalization.scales(),
            out,
        );
        self.data.weighted_column_sums_normalized_into(
            weights,
            self.normalization.centers(),
            self.normalization.scales(),
            out,
        )
    }
}

impl<M, P> LazyMatrix<M, P>
where
    M: MatrixShape + crate::MatrixErrorType,
    P: eager::Materialize<M>,
{
    /// Allocate normalized dense storage in the selected backend.
    ///
    /// Sparse and storage-backed inputs are explicitly materialized. Fitted
    /// metadata is copied into owned vectors, including when this operator
    /// borrows its parameters. Statistics are not recomputed. This requires O(nrows *
    /// ncols) output storage, plus backend workspace. Allocation size overflow
    /// panics; allocation failures follow the selected backend's behavior.
    ///
    /// Column normalization requires [`crate::MaterializeDense`]. Row and chain
    /// normalization require [`crate::MatrixTraversal`], including explicit
    /// mixed chains on sparse or chunked storage. They apply stages in order
    /// with one scan, coalescing duplicates before normalization and including
    /// transformed implicit zeros. Workspace is limited to traversal buffers.
    ///
    /// # Errors
    /// Returns the source error if reading the matrix fails.
    pub fn to_eager<D: crate::MatrixOwned<P::Scalar>>(
        &self,
    ) -> Result<crate::EagerMatrix<D, P::Owned>, M::Error> {
        crate::materialize::validate_allocation::<P::Scalar>(self.nrows(), self.ncols());
        let mut data = D::zeros(self.nrows(), self.ncols());
        self.materialize_into(&mut data)?;
        Ok(crate::EagerMatrix::from_normalized(
            data,
            self.normalization.copy_params(),
        ))
    }

    /// Overwrite dense storage and return a normalized operator borrowing it.
    ///
    /// No matrix allocation is made. Fitted parameters are cloned, and the
    /// backend may allocate workspace. The wrapper borrows only the output.
    ///
    /// The output remains exclusively borrowed while the wrapper is in use:
    ///
    /// ```compile_fail
    /// use lazymatrix::{LazyMatrix, MaterializeDense, MatrixWrite};
    /// fn cannot_reuse_output<M: MaterializeDense<f64>, D: MatrixWrite<f64>>(
    ///     matrix: &LazyMatrix<M>, out: &mut D,
    /// ) {
    ///     let eager = matrix.to_eager_into(out).unwrap();
    ///     out.set(0, 0, 0.0);
    ///     let _ = eager.nrows();
    /// }
    /// ```
    ///
    /// # Errors
    /// On a source error, output may be partial and no wrapper is returned.
    ///
    /// # Panics
    /// Panics before writing if the output shape differs from the input shape.
    pub fn to_eager_into<'a, D: crate::MatrixWrite<P::Scalar> + ?Sized>(
        &self,
        out: &'a mut D,
    ) -> Result<crate::EagerMatrix<&'a mut D, P::Owned>, M::Error> {
        self.materialize_into(out)?;
        Ok(crate::EagerMatrix::from_normalized(
            out,
            self.normalization.copy_params(),
        ))
    }

    fn materialize_into<D: crate::MatrixWrite<P::Scalar> + ?Sized>(
        &self,
        out: &mut D,
    ) -> Result<(), M::Error> {
        crate::materialize::validate_output(self, out, None, None);
        self.normalization.assert_shape(self.nrows(), self.ncols());
        self.normalization.materialize(&self.data, out)
    }
}

impl<M, P> LazyMatrix<M, P>
where
    M: MatrixShape,
    P: eager::InPlace<M>,
{
    /// Consume writable dense data, normalizing it in its existing allocation.
    ///
    /// Mutable views modify their borrowed storage. All statistics have already
    /// been fitted, and no new statistics or full matrix allocation are needed.
    /// The fitted parameter storage is retained unchanged, including any borrows.
    /// Row and chain conversion additionally require [`crate::MatrixRead`] and
    /// apply every stage in order to each entry without workspace allocation.
    ///
    /// Borrowed immutable and sparse storage do not support in-place conversion:
    ///
    /// ```compile_fail
    /// use lazymatrix::{LazyMatrix, MatrixShape};
    /// fn needs_writable_dense_storage<M: MatrixShape>(matrix: LazyMatrix<M>) {
    ///     let _ = matrix.into_eager();
    /// }
    /// ```
    pub fn into_eager(mut self) -> crate::EagerMatrix<M, P> {
        self.normalization.assert_shape(self.nrows(), self.ncols());
        self.normalization.normalize(&mut self.data);
        crate::EagerMatrix::from_normalized(self.data, self.normalization)
    }
}
