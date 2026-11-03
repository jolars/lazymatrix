//! Borrowed rectangular restrictions of logical operators.

use num_traits::{One, Zero};
use std::ops::Range;

use crate::{
    ColumnNormalizationParams, Columns, EagerMatrix, LazyColumn, LazyMatrix, MatTransposeVec,
    MatTransposeVecInto, MatTransposeVecScaledInto, MatVec, MatVecInto, MatVecScaledInto,
    MatrixErrorType, MatrixOwned, MatrixShape, MatrixWrite, RawColumn, RawColumns,
    RectangularProducts, Scalar, SparseColumnRef, SparseColumns, SparseRows, SubmatrixKernel,
    VectorOwned, VectorView, VectorViewMut, WeightedColumnSumsInto, WeightedGramInto,
};

/// A borrowed restriction of an already normalized matrix to contiguous ranges.
///
/// Construction and nested selection take O(1) time, allocate nothing, and
/// borrow the original logical operator. Selecting rows never refits
/// normalization. All entry indices exposed by this view are local.
/// Row normalization retains the parameters fitted across the original columns.
/// Mixed chains preserve their saved stage order.
///
/// Vector products require [`RectangularProducts`] on the parent. Column
/// normalization additionally retains borrowed logical views, weighted
/// operations, and eager conversion. Parent fitted parameters remain available
/// through [`Self::as_inner`].
///
/// Dense products visit only selected entries. Sparse products scan stored
/// entries in selected outer rows or columns, filtering the other axis. They
/// use O(selected rows + selected columns) workspace, never a dense design copy.
/// Column filtering scans the parent column; row filtering scans the parent row.
///
/// Zarr products, eager conversion, and weighted column sums read each
/// intersecting chunk once, serially, excluding values outside the selection.
/// Products use O(selected output length) scratch. Decoded storage chunks and
/// codec workspaces, including outer shards, are additional to that scratch.
/// Empty rectangles and products with zero `alpha` perform no storage reads.
/// Zarr storage does not provide borrowed columns or rows.
///
/// [`WeightedColumnSumsInto`] and [`WeightedGramInto`] accept weights indexed
/// by local rows and normalize entries before accumulating. Gram products use
/// two fixed 32-by-32 panels and a fixed accumulator, in addition to backend
/// workspace and the O(selected columns²) output. Dense arithmetic costs
/// O(selected rows * selected columns²). Sparse kernels revisit stored entries,
/// and Zarr kernels may decode chunks repeatedly for different panels. Weighted
/// sums on in-memory sources use the same bounded panel traversal.
///
/// This is a logical operator, not a contiguous sparse storage capability.
/// Filtered sparse views expose iterators instead of rebased index slices.
///
/// ```compile_fail
/// use lazymatrix::{LazyMatrix, SparseColumns, Submatrix};
/// fn needs_slices<M: SparseColumns<f64>>() {}
/// fn restricted_columns<M: SparseColumns<f64>>(_: &LazyMatrix<M>) {
///     needs_slices::<Submatrix<'_, LazyMatrix<M>>>();
/// }
/// ```
///
/// ```compile_fail
/// use lazymatrix::{LazyMatrix, SparseRows, Submatrix};
/// fn needs_slices<M: SparseRows<f64>>() {}
/// fn restricted_rows<M: SparseRows<f64>>(_: &LazyMatrix<M>) {
///     needs_slices::<Submatrix<'_, LazyMatrix<M>>>();
/// }
/// ```
///
/// Row-normalized rectangles do not acquire column borrowing or eager conversion:
///
/// ```compile_fail
/// use lazymatrix::{LazyMatrix, MatrixShape, RowNormalizationParams};
/// fn columns<M: MatrixShape>(rows: &LazyMatrix<M, RowNormalizationParams>) {
///     let _ = rows.submatrix(0..0, 0..0).column(0);
/// }
/// ```
///
/// ```compile_fail
/// use lazymatrix::{LazyMatrix, MatrixOwned, MatrixShape, RowNormalizationParams};
/// fn eager<M: MatrixShape, D: MatrixOwned<f64>>(
///     rows: &LazyMatrix<M, RowNormalizationParams>,
/// ) {
///     let _ = rows.submatrix(0..0, 0..0).to_eager::<D>();
/// }
/// ```
#[derive(Debug)]
pub struct Submatrix<'a, O: ?Sized> {
    inner: &'a O,
    rows: Range<usize>,
    columns: Range<usize>,
}

impl<O: ?Sized> Clone for Submatrix<'_, O> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner,
            rows: self.rows.clone(),
            columns: self.columns.clone(),
        }
    }
}

pub(crate) fn validate_ranges(
    shape: &(impl MatrixShape + ?Sized),
    rows: &Range<usize>,
    columns: &Range<usize>,
) {
    assert!(
        rows.start <= rows.end && rows.end <= shape.nrows(),
        "submatrix row range out of bounds"
    );
    assert!(
        columns.start <= columns.end && columns.end <= shape.ncols(),
        "submatrix column range out of bounds"
    );
}

impl<M: MatrixShape, P> LazyMatrix<M, P> {
    /// Borrow a rectangle of this fitted operator without refitting parameters.
    ///
    /// Row parameters retain the values fitted across the original columns.
    /// Empty ranges are valid, and construction reads no entries.
    ///
    /// # Panics
    /// Panics on reversed or out-of-bounds ranges.
    pub fn submatrix(&self, rows: Range<usize>, columns: Range<usize>) -> Submatrix<'_, Self> {
        Submatrix::new(self, rows, columns)
    }
}

impl<'a, O: MatrixShape + ?Sized> Submatrix<'a, O> {
    /// Restrict a logical operator without reading it or fitting new parameters.
    ///
    /// Products require the parent's corresponding rectangular capability.
    ///
    /// # Panics
    /// Panics on reversed or out-of-bounds ranges.
    pub fn new(inner: &'a O, rows: Range<usize>, columns: Range<usize>) -> Self {
        validate_ranges(inner, &rows, &columns);
        Self {
            inner,
            rows,
            columns,
        }
    }

    /// Select a rectangle in local coordinates, retaining the original parent.
    ///
    /// # Panics
    /// Panics on reversed or out-of-bounds ranges.
    pub fn submatrix(&self, rows: Range<usize>, columns: Range<usize>) -> Self {
        validate_ranges(self, &rows, &columns);
        Self {
            inner: self.inner,
            rows: self.rows.start + rows.start..self.rows.start + rows.end,
            columns: self.columns.start + columns.start..self.columns.start + columns.end,
        }
    }
}

impl<'a, O: ?Sized> Submatrix<'a, O> {
    /// Borrow the original logical operator, including any fitted parameters.
    pub fn as_inner(&self) -> &'a O {
        self.inner
    }
    /// Selected rows in the original parent's coordinates.
    pub fn row_range(&self) -> Range<usize> {
        self.rows.clone()
    }
    /// Selected columns in the original parent's coordinates.
    pub fn column_range(&self) -> Range<usize> {
        self.columns.clone()
    }
    /// Number of selected rows.
    pub fn nrows(&self) -> usize {
        self.rows.len()
    }
    /// Number of selected columns.
    pub fn ncols(&self) -> usize {
        self.columns.len()
    }
}

impl<'a, M: MatrixShape, F: Scalar, S: AsRef<[F]>>
    Submatrix<'a, LazyMatrix<M, ColumnNormalizationParams<F, S>>>
{
    #[inline]
    fn selected_centers(&self) -> Option<&'a [F]> {
        self.inner
            .normalization()
            .centers()
            .map(|v| &v[self.columns.clone()])
    }
    #[inline]
    fn selected_scales(&self) -> Option<&'a [F]> {
        self.inner
            .normalization()
            .scales()
            .map(|v| &v[self.columns.clone()])
    }
    /// Borrow a logical column, filtering and rebasing its stored row indices.
    ///
    /// Filtering takes O(parent stored entries) per traversal and allocates
    /// nothing. The parent must provide natural column borrowing.
    ///
    /// # Panics
    /// Panics if `j >= self.ncols()`.
    pub fn column(&self, j: usize) -> LazyColumn<FilteredRawColumn<M::Column<'_>>, F>
    where
        M: RawColumns<F>,
    {
        assert!(j < self.ncols(), "column index out of bounds");
        LazyColumn::new(
            FilteredRawColumn {
                raw: self.inner.data().raw_column(self.columns.start + j),
                rows: self.rows.clone(),
            },
            self.selected_centers().map_or_else(F::zero, |c| c[j]),
            self.selected_scales().map_or_else(F::one, |s| s[j]),
        )
    }
    /// Borrow a filtered sparse column with locally indexed entry iterators.
    ///
    /// # Panics
    /// Panics if `j >= self.ncols()`.
    pub fn sparse_column(&self, j: usize) -> SubmatrixSparseColumn<'_, F>
    where
        M: SparseColumns<F>,
    {
        assert!(j < self.ncols(), "column index out of bounds");
        let (indices, values) = self.inner.data().sparse_column(self.columns.start + j);
        LazyColumn::new(
            FilteredRawColumn {
                raw: SparseColumnRef::new(indices, values, self.inner.data().nrows()),
                rows: self.rows.clone(),
            },
            self.selected_centers().map_or_else(F::zero, |c| c[j]),
            self.selected_scales().map_or_else(F::one, |s| s[j]),
        )
    }
    /// Borrow a filtered normalized sparse row without copying its entries.
    ///
    /// Stored-entry iteration takes O(parent stored row entries), preserves
    /// order and duplicates, and rebases column indices to the selection.
    ///
    /// # Panics
    /// Panics if `i >= self.nrows()`.
    pub fn row(&self, i: usize) -> SubmatrixRow<'_, F>
    where
        M: SparseRows<F>,
    {
        assert!(i < self.nrows(), "row index out of bounds");
        let (indices, values) = self.inner.data().sparse_row(self.rows.start + i);
        SubmatrixRow {
            indices,
            values,
            columns: self.columns.clone(),
            centers: self.selected_centers(),
            scales: self.selected_scales(),
        }
    }
}

impl<O: ?Sized> MatrixShape for Submatrix<'_, O> {
    fn nrows(&self) -> usize {
        self.rows.len()
    }
    fn ncols(&self) -> usize {
        self.columns.len()
    }
}
impl<O: MatrixErrorType + ?Sized> MatrixErrorType for Submatrix<'_, O> {
    type Error = O::Error;
}
impl<M: RawColumns<F>, F: Scalar, S: AsRef<[F]>> Columns<F>
    for Submatrix<'_, LazyMatrix<M, ColumnNormalizationParams<F, S>>>
{
    type Column<'b>
        = LazyColumn<FilteredRawColumn<M::Column<'b>>, F>
    where
        Self: 'b;
    fn column(&self, j: usize) -> Self::Column<'_> {
        Submatrix::column(self, j)
    }
}

impl<M: SubmatrixKernel<F>, F: Scalar, S: AsRef<[F]>> WeightedColumnSumsInto<F>
    for Submatrix<'_, LazyMatrix<M, ColumnNormalizationParams<F, S>>>
{
    #[inline]
    fn weighted_column_sums_into<W, O>(&self, weights: &W, out: &mut O) -> Result<(), Self::Error>
    where
        W: VectorView<F> + ?Sized,
        O: VectorViewMut<F> + ?Sized,
    {
        self.inner.data().submatrix_weighted_column_sums_into(
            self.rows.clone(),
            self.columns.clone(),
            self.selected_centers(),
            self.selected_scales(),
            weights,
            out,
        )
    }
}

impl<M: SubmatrixKernel<F>, F: Scalar, S: AsRef<[F]>> WeightedGramInto<F>
    for Submatrix<'_, LazyMatrix<M, ColumnNormalizationParams<F, S>>>
{
    #[inline]
    fn weighted_gram_into<W, O>(&self, weights: &W, out: &mut O) -> Result<(), Self::Error>
    where
        W: VectorView<F> + ?Sized,
        O: MatrixWrite<F> + ?Sized,
    {
        self.inner.data().submatrix_weighted_gram_into(
            self.rows.clone(),
            self.columns.clone(),
            self.selected_centers(),
            self.selected_scales(),
            weights,
            out,
        )
    }
}

impl<M: SubmatrixKernel<F>, F: Scalar, S: AsRef<[F]>>
    Submatrix<'_, LazyMatrix<M, ColumnNormalizationParams<F, S>>>
{
    fn fitted_parameters(&self) -> ColumnNormalizationParams<F> {
        ColumnNormalizationParams::from_parts(
            self.ncols(),
            self.selected_centers().map(<[F]>::to_vec),
            self.selected_scales().map(<[F]>::to_vec),
        )
    }
    /// Explicitly allocate and materialize this rectangle in a dense backend.
    ///
    /// Requires O(selected rows * selected columns) output storage. Only the
    /// selected fitted parameters are cloned; no statistics are recomputed.
    /// Returns the source error on failure.
    pub fn to_eager<D: MatrixOwned<F>>(
        &self,
    ) -> Result<EagerMatrix<D, ColumnNormalizationParams<F>>, M::Error> {
        crate::materialize::validate_allocation::<F>(self.nrows(), self.ncols());
        let mut data = D::zeros(self.nrows(), self.ncols());
        self.inner.data().materialize_submatrix_into(
            self.rows.clone(),
            self.columns.clone(),
            self.selected_centers(),
            self.selected_scales(),
            &mut data,
        )?;
        Ok(EagerMatrix::from_normalized(data, self.fitted_parameters()))
    }
    /// Overwrite dense output and return an eager operator borrowing it.
    ///
    /// Fitted metadata is cloned. In-memory kernels may allocate one working
    /// row or column; Zarr kernels decode one intersecting chunk at a time.
    /// Output may be partial on a source error.
    ///
    /// # Panics
    /// Panics before writing if the output shape differs from this rectangle.
    pub fn to_eager_into<'b, D: MatrixWrite<F> + ?Sized>(
        &self,
        out: &'b mut D,
    ) -> Result<EagerMatrix<&'b mut D, ColumnNormalizationParams<F>>, M::Error> {
        crate::materialize::validate_output(
            self,
            out,
            self.selected_centers(),
            self.selected_scales(),
        );
        self.inner.data().materialize_submatrix_into(
            self.rows.clone(),
            self.columns.clone(),
            self.selected_centers(),
            self.selected_scales(),
            out,
        )?;
        Ok(EagerMatrix::from_normalized(out, self.fitted_parameters()))
    }
}

macro_rules! products {
    ($alloc:ident, $alloc_method:ident, $into:ident, $into_method:ident, $scaled:ident, $scaled_method:ident, $kernel:ident, $input:ident, $output:ident) => {
        impl<O: RectangularProducts + ?Sized, V> $alloc<V> for Submatrix<'_, O>
        where
            V: VectorOwned<O::Scalar, Owned = V> + VectorViewMut<O::Scalar>,
        {
            fn $alloc_method(&self, x: &V) -> Result<V, Self::Error> {
                assert_eq!(x.len(), self.$input(), "submatrix input dimension mismatch");
                let mut out = V::owned_from_fn(self.$output(), |_| O::Scalar::zero());
                self.$into_method(x, &mut out)?;
                Ok(out)
            }
        }
        impl<O: RectangularProducts + ?Sized, X, Y> $into<X, Y> for Submatrix<'_, O>
        where
            X: VectorView<O::Scalar>,
            Y: VectorViewMut<O::Scalar>,
        {
            fn $into_method(&self, x: &X, out: &mut Y) -> Result<(), Self::Error> {
                self.$scaled_method(O::Scalar::one(), x, O::Scalar::zero(), out)
            }
        }
        impl<O: RectangularProducts + ?Sized, X, Y> $scaled<X, Y, O::Scalar> for Submatrix<'_, O>
        where
            X: VectorView<O::Scalar>,
            Y: VectorViewMut<O::Scalar>,
        {
            fn $scaled_method(
                &self,
                alpha: O::Scalar,
                x: &X,
                beta: O::Scalar,
                out: &mut Y,
            ) -> Result<(), Self::Error> {
                assert_eq!(x.len(), self.$input(), "submatrix input dimension mismatch");
                assert_eq!(
                    out.len(),
                    self.$output(),
                    "submatrix output dimension mismatch"
                );
                if alpha == O::Scalar::zero() {
                    crate::traits::scale_output(beta, out);
                    return Ok(());
                }
                self.inner
                    .$kernel(self.rows.clone(), self.columns.clone(), alpha, x, beta, out)
            }
        }
    };
}
products!(
    MatVec,
    matvec,
    MatVecInto,
    matvec_into,
    MatVecScaledInto,
    matvec_scaled_into,
    rectangle_matvec_into,
    ncols,
    nrows
);
products!(
    MatTransposeVec,
    mat_transpose_vec,
    MatTransposeVecInto,
    mat_transpose_vec_into,
    MatTransposeVecScaledInto,
    mat_transpose_vec_scaled_into,
    rectangle_transpose_into,
    nrows,
    ncols
);

/// A raw column adapter filtering a contiguous range and rebasing row indices.
///
/// Each traversal scans the parent column without gathering or allocating.
#[derive(Clone, Debug)]
pub struct FilteredRawColumn<C> {
    raw: C,
    rows: Range<usize>,
}
impl<C: RawColumn<F>, F: Scalar> RawColumn<F> for FilteredRawColumn<C> {
    fn len(&self) -> usize {
        self.rows.len()
    }
    fn stored_len(&self) -> usize {
        let mut len = 0;
        self.for_each_stored(|_, _| len += 1);
        len
    }
    fn for_each_stored(&self, mut f: impl FnMut(usize, F)) {
        self.raw.for_each_stored(|i, value| {
            if self.rows.contains(&i) {
                f(i - self.rows.start, value);
            }
        });
    }
}

/// A normalized sparse column with filtered, locally indexed stored entries.
pub type SubmatrixSparseColumn<'a, F> = LazyColumn<FilteredRawColumn<SparseColumnRef<'a, F>>, F>;
impl<F: Scalar> SubmatrixSparseColumn<'_, F> {
    /// Raw stored entries, preserving storage order, duplicates, and zeros.
    pub fn stored_entries(&self) -> impl Iterator<Item = (usize, F)> + '_ {
        let raw = self.raw();
        raw.raw
            .row_indices()
            .iter()
            .copied()
            .zip(raw.raw.values().iter().copied())
            .filter(|(i, _)| raw.rows.contains(i))
            .map(|(i, value)| (i - raw.rows.start, value))
    }
    /// Logical value of an implicit zero.
    pub fn implicit_value(&self) -> F {
        -self.center() / self.scale()
    }
    /// Scaled stored corrections to the implicit value.
    pub fn stored_corrections(&self) -> impl Iterator<Item = (usize, F)> + '_ {
        self.stored_entries()
            .map(|(i, value)| (i, value / self.scale()))
    }
}

/// A borrowed filtered sparse row with the selected fitted column parameters.
///
/// Entries and corrections preserve duplicates. Sum duplicate raw values before
/// applying normalization when reconstructing logical cells.
#[derive(Clone, Debug)]
pub struct SubmatrixRow<'a, F> {
    indices: &'a [usize],
    values: &'a [F],
    columns: Range<usize>,
    centers: Option<&'a [F]>,
    scales: Option<&'a [F]>,
}
impl<'a, F: Scalar> SubmatrixRow<'a, F> {
    /// Number of logical selected columns.
    pub fn len(&self) -> usize {
        self.columns.len()
    }
    /// Whether the row has zero logical columns.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Borrow selected centers, or `None` when inactive.
    pub fn centers(&self) -> Option<&'a [F]> {
        self.centers
    }
    /// Borrow selected scales, or `None` when inactive.
    pub fn scales(&self) -> Option<&'a [F]> {
        self.scales
    }
    /// Effective center at a local column index. Panics if out of bounds.
    pub fn center(&self, j: usize) -> F {
        assert!(j < self.len(), "column index out of bounds");
        self.centers.map_or_else(F::zero, |c| c[j])
    }
    /// Effective scale at a local column index. Panics if out of bounds.
    pub fn scale(&self, j: usize) -> F {
        assert!(j < self.len(), "column index out of bounds");
        self.scales.map_or_else(F::one, |s| s[j])
    }
    /// Logical background at a local column index. Panics if out of bounds.
    pub fn implicit_value(&self, j: usize) -> F {
        -self.center(j) / self.scale(j)
    }
    /// Raw stored entries with local indices, preserving order and duplicates.
    pub fn stored_entries(&self) -> impl Iterator<Item = (usize, F)> + '_ {
        self.indices
            .iter()
            .copied()
            .zip(self.values.iter().copied())
            .filter(|(j, _)| self.columns.contains(j))
            .map(|(j, value)| (j - self.columns.start, value))
    }
    /// Scaled stored corrections to the column-dependent background.
    pub fn stored_corrections(&self) -> impl Iterator<Item = (usize, F)> + '_ {
        self.stored_entries()
            .map(|(j, value)| (j, value / self.scale(j)))
    }
}
