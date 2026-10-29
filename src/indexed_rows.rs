//! Raw row restrictions for fitting normalization without copying a design.

use std::marker::PhantomData;

use crate::{
    IndexedRowSource, MatTransposeVec, MatTransposeVecInto, MatTransposeVecScaledInto, MatVec,
    MatVecInto, MatVecScaledInto, MatrixErrorType, MatrixShape, RawColumn, RawColumns, Scalar,
    SparseRows, VectorOwned, VectorView, VectorViewMut,
};

mod stats;

/// A raw matrix restricted to strictly increasing, unique row indices.
///
/// Unlike [`crate::Submatrix`], this selects data **before** normalization.
/// Pass it to [`crate::LazyMatrix::new`] to fit on selected rows alone. Local
/// row `i` maps to `row_indices()[i]` in the source. All columns are retained.
/// Empty selections are allowed and follow [`crate::ColumnStats`]' empty-column
/// rules. Invalid indices, duplicates, and descending indices panic at construction.
///
/// The wrapper owns `M` and borrows the index slice. Pass `&matrix` or a backend
/// view to borrow storage without copying. Construction checks indices in O(k)
/// time for k selected rows and allocates nothing. Cloning only clones `M` and
/// the borrowed index slice; cloning an owned matrix can copy its storage.
///
/// Products and statistics use native source traversal. Dense scans cost
/// O(k * ncols), CSR scans cost O(ncols + k + selected stored entries), and
/// CSC scans cost O(k + ncols + stored entries * log(k + 1)). Sparse scans combine duplicate
/// cells using O(k) CSC or O(ncols) CSR scratch. They never copy the design or
/// convert its storage orientation. Reusable raw products need no additional
/// scratch beyond traversal. Fused products with nonzero beta and centered dense fused products use
/// O(output length) scratch. Nested raw selections allocate O(k) index mapping
/// per scan.
/// Statistics use O(ncols) state in addition to traversal scratch. SD uses two
/// scans; quantiles use exact, data-dependent repeated scans, sharing scans for
/// median and quantile range in [`crate::ColumnStats::normalization_stats`].
///
/// [`RawColumns`] is available only if the source naturally borrows columns.
/// Creating a filtered column allocates nothing; each traversal scans the
/// parent column and maps selected indices by binary search. [`SparseRows`]
/// retains the source's contiguous row slices, order, and duplicate entries.
/// Selected CSC indices cannot be borrowed as contiguous local index slices:
///
/// ```compile_fail
/// use lazymatrix::{IndexedRows, SparseColumns};
/// fn needs_slices<M: SparseColumns<f64>>() {}
/// fn selection<M: SparseColumns<f64>>() {
///     needs_slices::<IndexedRows<'_, M>>();
/// }
/// ```
#[derive(Clone, Copy, Debug)]
pub struct IndexedRows<'a, M, F = f64> {
    source: M,
    rows: &'a [usize],
    scalar: PhantomData<F>,
}

impl<'a, M: MatrixShape, F> IndexedRows<'a, M, F> {
    /// Restrict raw storage to the supplied rows without reading entries.
    ///
    /// # Panics
    /// Panics unless indices are strictly increasing, unique, and in bounds.
    pub fn new(source: M, rows: &'a [usize]) -> Self
    where
        F: Scalar,
        M: IndexedRowSource<F>,
    {
        assert!(
            rows.windows(2).all(|pair| pair[0] < pair[1]),
            "row indices must be strictly increasing and unique"
        );
        assert!(
            rows.last().is_none_or(|&i| i < source.nrows()),
            "row index out of bounds"
        );
        Self {
            source,
            rows,
            scalar: PhantomData,
        }
    }

    /// Borrow the source, which may itself be a borrowed backend view.
    pub fn source(&self) -> &M {
        &self.source
    }

    /// Source row indices, in local row order.
    pub fn row_indices(&self) -> &'a [usize] {
        self.rows
    }

    /// Recover the source without copying.
    pub fn into_source(self) -> M {
        self.source
    }
}

impl<M: MatrixShape, F> MatrixShape for IndexedRows<'_, M, F> {
    fn nrows(&self) -> usize {
        self.rows.len()
    }
    fn ncols(&self) -> usize {
        self.source.ncols()
    }
}

impl<M: MatrixErrorType, F> MatrixErrorType for IndexedRows<'_, M, F> {
    type Error = M::Error;
}

/// A borrowed raw column filtered to local selected row indices.
///
/// Traversal preserves source order and duplicates, takes O(parent stored
/// entries * log(k + 1)) time, and allocates nothing. `stored_len` traverses the
/// column with the same cost. The logical length is the number of selected rows.
#[derive(Clone, Copy, Debug)]
pub struct IndexedRawColumn<'a, C> {
    raw: C,
    rows: &'a [usize],
}

impl<F: Scalar, C: RawColumn<F>> RawColumn<F> for IndexedRawColumn<'_, C> {
    fn len(&self) -> usize {
        self.rows.len()
    }
    fn stored_len(&self) -> usize {
        let mut count = 0;
        self.for_each_stored(|_, _| count += 1);
        count
    }
    fn for_each_stored(&self, mut visit: impl FnMut(usize, F)) {
        if self.rows.is_empty() {
            return;
        }
        self.raw.for_each_stored(|i, value| {
            if let Ok(local) = self.rows.binary_search(&i) {
                visit(local, value);
            }
        });
    }
}

impl<M: RawColumns<F>, F: Scalar> RawColumns<F> for IndexedRows<'_, M, F> {
    type Column<'b>
        = IndexedRawColumn<'b, M::Column<'b>>
    where
        Self: 'b;
    fn raw_column(&self, j: usize) -> Self::Column<'_> {
        assert!(j < self.ncols(), "column index out of bounds");
        IndexedRawColumn {
            raw: self.source.raw_column(j),
            rows: self.rows,
        }
    }
}

impl<M: SparseRows<F>, F: Scalar> SparseRows<F> for IndexedRows<'_, M, F> {
    fn sparse_row(&self, i: usize) -> (&[usize], &[F]) {
        self.source.sparse_row(self.rows[i])
    }
}

impl<M: IndexedRowSource<F>, F: Scalar> IndexedRowSource<F> for IndexedRows<'_, M, F> {
    const IS_DENSE: bool = M::IS_DENSE;
    fn for_each_indexed_entry(
        &self,
        rows: &[usize],
        visit: impl FnMut(usize, usize, F),
    ) -> Result<(), Self::Error> {
        // Nested selections map local coordinates explicitly, without copying entries.
        let mapped: Vec<_> = rows.iter().map(|&i| self.rows[i]).collect();
        self.source.for_each_indexed_entry(&mapped, visit)
    }
}

impl<M: IndexedRowSource<F>, F: Scalar> IndexedRows<'_, M, F> {
    fn product<N: Scalar, X: VectorView<F>, Y: VectorViewMut<F>>(
        &self,
        transpose: bool,
        x: &X,
        out: &mut Y,
        centers: Option<&[N]>,
        scales: Option<&[N]>,
    ) -> Result<(), M::Error> {
        let (input, output) = if transpose {
            (self.nrows(), self.ncols())
        } else {
            (self.ncols(), self.nrows())
        };
        assert_eq!(x.len(), input, "indexed rows input dimension mismatch");
        assert_eq!(out.len(), output, "indexed rows output dimension mismatch");
        crate::gram::validate_normalization(self.ncols(), centers, scales);
        for i in 0..output {
            out.set(i, F::zero());
        }
        if self.nrows() == 0 || self.ncols() == 0 {
            return Ok(());
        }
        self.source
            .for_each_indexed_entry(self.rows, |i, j, value| {
                let value = centers.map_or(value, |c| value - F::from(c[j]).unwrap());
                let value = scales.map_or(value, |s| value / F::from(s[j]).unwrap());
                let (to, from) = if transpose { (j, i) } else { (i, j) };
                out.set(to, out.get(to) + value * x.get(from));
            })
    }
}

macro_rules! products {
    ($alloc:ident, $alloc_fn:ident, $norm_alloc:ident, $into:ident, $into_fn:ident,
     $norm_into:ident, $scaled:ident, $scaled_fn:ident, $try_norm:ident,
     $transpose:expr, $output:ident, [$($bounds:tt)*], [$($input_bounds:tt)*], [$($output_bounds:tt)*]) => {
        impl<M: IndexedRowSource<F>, F: Scalar, V> $alloc<V> for IndexedRows<'_, M, F>
        where V: VectorOwned<F, Owned = V> + VectorViewMut<F> {
            fn $alloc_fn(&self, x: &V) -> Result<V, Self::Error> {
                let mut out = V::owned_from_fn(self.$output(), |_| F::zero());
                self.$into_fn(x, &mut out)?;
                Ok(out)
            }
            fn $norm_alloc<N: Scalar>(&self, x: &V, centers: Option<&[N]>, scales: Option<&[N]>)
                -> Result<V, Self::Error>
            where V: $($bounds)* {
                let mut out = V::owned_from_fn(self.$output(), |_| F::zero());
                self.$norm_into(x, centers, scales, &mut out)?;
                Ok(out)
            }
        }
        impl<M: IndexedRowSource<F>, F: Scalar, X: VectorView<F>, Y: VectorViewMut<F>>
            $into<X, Y> for IndexedRows<'_, M, F> {
            fn $into_fn(&self, x: &X, out: &mut Y) -> Result<(), Self::Error> {
                self.product::<F, _, _>($transpose, x, out, None, None)
            }
            fn $norm_into<N: Scalar>(&self, x: &X, centers: Option<&[N]>, scales: Option<&[N]>, out: &mut Y)
                -> Result<(), Self::Error>
            where X: $($input_bounds)*, Y: $($output_bounds)* {
                if M::IS_DENSE && centers.is_some() {
                    self.product($transpose, x, out, centers, scales)
                } else {
                    crate::traits::$norm_into(self, x, centers, scales, out)
                }
            }
        }
        impl<M: IndexedRowSource<F>, F: Scalar, X: VectorView<F>, Y: VectorViewMut<F>>
            $scaled<X, Y, F> for IndexedRows<'_, M, F> {
            fn $scaled_fn(&self, alpha: F, x: &X, beta: F, out: &mut Y) -> Result<(), Self::Error> {
                let input = if $transpose { self.nrows() } else { self.ncols() };
                assert_eq!(x.len(), input, "indexed rows input dimension mismatch");
                assert_eq!(out.len(), self.$output(), "indexed rows output dimension mismatch");
                if alpha == F::zero() {
                    crate::traits::scale_output(beta, out);
                    return Ok(());
                }
                if beta == F::zero() {
                    self.$into_fn(x, out)?;
                    for i in 0..out.len() { out.set(i, alpha * out.get(i)); }
                } else {
                    let mut scratch = vec![F::zero(); out.len()];
                    self.$into_fn(x, &mut scratch)?;
                    for (i, value) in scratch.into_iter().enumerate() {
                        out.set(i, alpha * value + beta * out.get(i));
                    }
                }
                Ok(())
            }
            fn $try_norm(&self, alpha: F, x: &X, beta: F, centers: Option<&[F]>, scales: Option<&[F]>, out: &mut Y)
                -> Option<Result<(), Self::Error>> {
                if !M::IS_DENSE || centers.is_none() { return None; }
                let mut scratch = vec![F::zero(); out.len()];
                let result = if alpha == F::zero() {
                    let input = if $transpose { self.nrows() } else { self.ncols() };
                    assert_eq!(x.len(), input, "indexed rows input dimension mismatch");
                    assert_eq!(out.len(), self.$output(), "indexed rows output dimension mismatch");
                    crate::traits::scale_output(beta, out);
                    Ok(())
                } else { self.product($transpose, x, &mut scratch, centers, scales).map(|()| {
                    for (i, value) in scratch.into_iter().enumerate() {
                        out.set(i, if beta == F::zero() { alpha * value } else { alpha * value + beta * out.get(i) });
                    }
                }) };
                Some(result)
            }
        }
    };
}

products!(
    MatVec,
    matvec,
    matvec_normalized,
    MatVecInto,
    matvec_into,
    matvec_normalized_into,
    MatVecScaledInto,
    matvec_scaled_into,
    try_matvec_scaled_normalized_into,
    false,
    nrows,
    [Clone + crate::ElemDivAssign<N> + crate::DotSlice<N> + crate::SubScalarAssign<N>],
    [Clone + crate::ElemDivAssign<N> + crate::DotSlice<N>],
    [crate::SubScalarAssign<N>]
);
products!(
    MatTransposeVec,
    mat_transpose_vec,
    mat_transpose_vec_normalized,
    MatTransposeVecInto,
    mat_transpose_vec_into,
    mat_transpose_vec_normalized_into,
    MatTransposeVecScaledInto,
    mat_transpose_vec_scaled_into,
    try_mat_transpose_vec_scaled_normalized_into,
    true,
    ncols,
    [crate::SumEntries<N> + crate::ScaledSubSlice<N> + crate::ElemDivAssign<N>],
    [crate::SumEntries<N>],
    [crate::ScaledSubSlice<N> + crate::ElemDivAssign<N>]
);
