//! Borrowed rectangular restrictions of fitted normalized matrices.

use std::ops::Range;

use crate::{
    Columns, EagerMatrix, LazyColumn, LazyMatrix, MatTransposeVec, MatTransposeVecInto,
    MatTransposeVecScaledInto, MatVec, MatVecInto, MatVecScaledInto, MatrixErrorType, MatrixOwned,
    MatrixShape, MatrixWrite, NormalizationParams, RawColumn, RawColumns, Scalar, SparseColumnRef,
    SparseColumns, SparseRows, SubmatrixKernel, VectorOwned, VectorView, VectorViewMut,
};

/// A borrowed restriction of an already normalized matrix to contiguous ranges.
///
/// Construction and nested selection take O(1) time, allocate nothing, and
/// borrow the original data and fitted parameters. Selecting rows never refits
/// normalization. All entry indices exposed by this view are local.
///
/// Dense products visit only selected entries. Sparse products scan stored
/// entries in selected outer rows or columns, filtering the other axis. They
/// use O(selected rows + selected columns) workspace, never a dense design copy.
/// Column filtering scans the parent column; row filtering scans the parent row.
///
/// This is a logical operator, not a contiguous sparse storage capability.
/// Filtered sparse views expose iterators instead of rebased index slices.
///
/// ```compile_fail
/// use lazymatrix::{LazyMatrix, SparseColumns, Submatrix};
/// fn needs_slices<M: SparseColumns<f64>>() {}
/// fn restricted_columns<M: SparseColumns<f64>>(_: &LazyMatrix<M>) {
///     needs_slices::<Submatrix<'_, M>>();
/// }
/// ```
///
/// ```compile_fail
/// use lazymatrix::{LazyMatrix, SparseRows, Submatrix};
/// fn needs_slices<M: SparseRows<f64>>() {}
/// fn restricted_rows<M: SparseRows<f64>>(_: &LazyMatrix<M>) {
///     needs_slices::<Submatrix<'_, M>>();
/// }
/// ```
#[derive(Debug)]
pub struct Submatrix<'a, M, F = f64> {
    parent: &'a LazyMatrix<M, F>,
    rows: Range<usize>,
    columns: Range<usize>,
}

impl<M, F> Clone for Submatrix<'_, M, F> {
    fn clone(&self) -> Self {
        Self {
            parent: self.parent,
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

impl<M: MatrixShape, F: Scalar> LazyMatrix<M, F> {
    /// Borrow a rectangle of this normalized matrix without refitting parameters.
    ///
    /// Empty ranges are allowed. Construction reads no entries and allocates
    /// nothing. To fit normalization on a subset instead, select raw data first
    /// and pass it to [`LazyMatrix::new`].
    ///
    /// # Panics
    /// Panics on reversed or out-of-bounds ranges.
    pub fn submatrix(&self, rows: Range<usize>, columns: Range<usize>) -> Submatrix<'_, M, F> {
        validate_ranges(self, &rows, &columns);
        Submatrix {
            parent: self,
            rows,
            columns,
        }
    }
}

impl<'a, M: MatrixShape, F: Scalar> Submatrix<'a, M, F> {
    /// Number of selected rows.
    pub fn nrows(&self) -> usize {
        self.rows.len()
    }
    /// Number of selected columns.
    pub fn ncols(&self) -> usize {
        self.columns.len()
    }
    /// Borrow the selected columns' original centers.
    pub fn centers(&self) -> Option<&'a [F]> {
        self.parent.centers().map(|v| &v[self.columns.clone()])
    }
    /// Borrow the selected columns' original scales.
    pub fn scales(&self) -> Option<&'a [F]> {
        self.parent.scales().map(|v| &v[self.columns.clone()])
    }
    /// Select a rectangle using coordinates relative to this view.
    ///
    /// # Panics
    /// Panics on reversed or out-of-bounds ranges.
    pub fn submatrix(&self, rows: Range<usize>, columns: Range<usize>) -> Self {
        validate_ranges(self, &rows, &columns);
        Self {
            parent: self.parent,
            rows: self.rows.start + rows.start..self.rows.start + rows.end,
            columns: self.columns.start + columns.start..self.columns.start + columns.end,
        }
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
                raw: self.parent.data().raw_column(self.columns.start + j),
                rows: self.rows.clone(),
            },
            self.centers().map_or_else(F::zero, |c| c[j]),
            self.scales().map_or_else(F::one, |s| s[j]),
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
        let (indices, values) = self.parent.data().sparse_column(self.columns.start + j);
        LazyColumn::new(
            FilteredRawColumn {
                raw: SparseColumnRef::new(indices, values, self.parent.nrows()),
                rows: self.rows.clone(),
            },
            self.centers().map_or_else(F::zero, |c| c[j]),
            self.scales().map_or_else(F::one, |s| s[j]),
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
        let (indices, values) = self.parent.data().sparse_row(self.rows.start + i);
        SubmatrixRow {
            indices,
            values,
            columns: self.columns.clone(),
            centers: self.centers(),
            scales: self.scales(),
        }
    }
}

impl<M, F> MatrixShape for Submatrix<'_, M, F> {
    fn nrows(&self) -> usize {
        self.rows.len()
    }
    fn ncols(&self) -> usize {
        self.columns.len()
    }
}
impl<M: MatrixErrorType, F> MatrixErrorType for Submatrix<'_, M, F> {
    type Error = M::Error;
}
impl<M: RawColumns<F>, F: Scalar> Columns<F> for Submatrix<'_, M, F> {
    type Column<'b>
        = LazyColumn<FilteredRawColumn<M::Column<'b>>, F>
    where
        Self: 'b;
    fn column(&self, j: usize) -> Self::Column<'_> {
        Submatrix::column(self, j)
    }
}

impl<M: SubmatrixKernel<F>, F: Scalar> Submatrix<'_, M, F> {
    fn fitted_parameters(&self) -> NormalizationParams<F> {
        NormalizationParams::from_parts(
            self.ncols(),
            self.centers().map(<[F]>::to_vec),
            self.scales().map(<[F]>::to_vec),
        )
    }
    /// Explicitly allocate and materialize this rectangle in a dense backend.
    ///
    /// Requires O(selected rows * selected columns) output storage. Only the
    /// selected fitted parameters are cloned; no statistics are recomputed.
    /// Returns the source error on failure.
    pub fn to_eager<D: MatrixOwned<F>>(&self) -> Result<EagerMatrix<D, F>, M::Error> {
        crate::materialize::validate_allocation::<F>(self.nrows(), self.ncols());
        let mut data = D::zeros(self.nrows(), self.ncols());
        self.parent.data().materialize_submatrix_into(
            self.rows.clone(),
            self.columns.clone(),
            self.centers(),
            self.scales(),
            &mut data,
        )?;
        Ok(EagerMatrix::from_normalized(data, self.fitted_parameters()))
    }
    /// Overwrite dense output and return an eager operator borrowing it.
    ///
    /// Fitted metadata is cloned, and kernels may allocate one working row or
    /// column. Output may be partial on a source error.
    ///
    /// # Panics
    /// Panics before writing if the output shape differs from this rectangle.
    pub fn to_eager_into<'b, D: MatrixWrite<F> + ?Sized>(
        &self,
        out: &'b mut D,
    ) -> Result<EagerMatrix<&'b mut D, F>, M::Error> {
        crate::materialize::validate_output(self, out, self.centers(), self.scales());
        self.parent.data().materialize_submatrix_into(
            self.rows.clone(),
            self.columns.clone(),
            self.centers(),
            self.scales(),
            out,
        )?;
        Ok(EagerMatrix::from_normalized(out, self.fitted_parameters()))
    }
}

macro_rules! products {
    ($alloc:ident, $alloc_method:ident, $into:ident, $into_method:ident, $scaled:ident, $scaled_method:ident, $kernel:ident, $input:ident, $output:ident) => {
        impl<M, V, F> $alloc<V> for Submatrix<'_, M, F>
        where
            M: SubmatrixKernel<F>,
            F: Scalar,
            V: VectorOwned<F, Owned = V> + VectorViewMut<F>,
        {
            fn $alloc_method(&self, x: &V) -> Result<V, Self::Error> {
                assert_eq!(x.len(), self.$input(), "submatrix input dimension mismatch");
                let mut out = V::owned_from_fn(self.$output(), |_| F::zero());
                self.$into_method(x, &mut out)?;
                Ok(out)
            }
        }
        impl<M, X, Y, F> $into<X, Y> for Submatrix<'_, M, F>
        where
            M: SubmatrixKernel<F>,
            F: Scalar,
            X: VectorView<F>,
            Y: VectorViewMut<F>,
        {
            fn $into_method(&self, x: &X, out: &mut Y) -> Result<(), Self::Error> {
                self.$scaled_method(F::one(), x, F::zero(), out)
            }
        }
        impl<M, X, Y, F> $scaled<X, Y, F> for Submatrix<'_, M, F>
        where
            M: SubmatrixKernel<F>,
            F: Scalar,
            X: VectorView<F>,
            Y: VectorViewMut<F>,
        {
            fn $scaled_method(
                &self,
                alpha: F,
                x: &X,
                beta: F,
                out: &mut Y,
            ) -> Result<(), Self::Error> {
                assert_eq!(x.len(), self.$input(), "submatrix input dimension mismatch");
                assert_eq!(
                    out.len(),
                    self.$output(),
                    "submatrix output dimension mismatch"
                );
                if alpha == F::zero() {
                    crate::traits::scale_output(beta, out);
                    return Ok(());
                }
                self.parent.data().$kernel(
                    self.rows.clone(),
                    self.columns.clone(),
                    self.centers(),
                    self.scales(),
                    alpha,
                    x,
                    beta,
                    out,
                )
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
    submatrix_matvec_into,
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
    submatrix_transpose_into,
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
