use std::ops::Deref;

use sprs::{CsMatBase, CsVecBase, CsVecViewI, SpIndex};

use crate::{
    ColumnStats, MatTransposeVec, MatTransposeVecInto, MatTransposeVecScaledInto, MatVec,
    MatVecInto, MatVecScaledInto, MatrixShape, RawColumn, RawColumns, Scalar, SparseColumns,
};

/// A checked CSC matrix or view that supports contiguous borrowed columns.
///
/// sprs stores CSC and CSR orientation in a runtime flag on the same type.
/// This wrapper checks that flag once, in [`try_new`](Self::try_new), so column
/// access never needs a gather or an orientation conversion. Construction and
/// column borrowing take O(1) time and allocate nothing. The wrapper retains
/// ownership of the supplied matrix or view and exposes no mutable access that
/// could change its orientation.
///
/// Operators, statistics, and logical columns accept any sprs index type.
/// Borrowing row indices as slices requires `usize` to match [`SparseColumns`].
///
/// ```
/// # #[cfg(feature = "sprs_all")]
/// # {
/// use lazymatrix::{Centering, LazyMatrix, MatVec, Normalization, Scaling, SprsCsc};
/// use sprs::CsMat;
///
/// let x = CsMat::new_csc((3, 2), vec![0, 2, 3], vec![0, 2, 1], vec![1.0, 3.0, 2.0]);
/// let csc = SprsCsc::try_new(x.view()).unwrap();
/// let lazy = LazyMatrix::new(csc, Normalization::new(Centering::Mean, Scaling::Sd)).unwrap();
/// assert_eq!(lazy.matvec(&vec![1.0, -1.0]).unwrap().len(), 3);
/// assert_eq!(lazy.sparse_column(0).row_indices(), &[0, 2]);
/// # }
/// ```
///
/// A plain sprs matrix has no column-borrowing capability, since it could be CSR:
///
/// ```compile_fail
/// use lazymatrix::SparseColumns;
/// fn needs_columns<M: SparseColumns<f64>>() {}
/// needs_columns::<sprs::CsMat<f64>>();
/// ```
///
/// ```compile_fail
/// use lazymatrix::RawColumns;
/// fn needs_columns<M: RawColumns<f64>>() {}
/// needs_columns::<sprs::CsMat<f64>>();
/// ```
///
/// Indices of another width cannot provide a borrowed `&[usize]`:
///
/// ```compile_fail
/// use lazymatrix::{SparseColumns, SprsCsc};
/// fn needs_columns<M: SparseColumns<f64>>() {}
/// needs_columns::<SprsCsc<sprs::CsMatI<f64, u32>>>();
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SprsCsc<M> {
    inner: M,
}

impl<F, I, IP, IS, DS, Iptr> SprsCsc<CsMatBase<F, I, IP, IS, DS, Iptr>>
where
    I: SpIndex,
    Iptr: SpIndex,
    IP: Deref<Target = [Iptr]>,
    IS: Deref<Target = [I]>,
    DS: Deref<Target = [F]>,
{
    /// Wrap CSC storage without copying, or return the original CSR input.
    ///
    /// To convert CSR storage explicitly, call sprs's `to_csc` or `into_csc`
    /// before this constructor. Borrow existing data by passing `matrix.view()`.
    pub fn try_new(
        matrix: CsMatBase<F, I, IP, IS, DS, Iptr>,
    ) -> Result<Self, CsMatBase<F, I, IP, IS, DS, Iptr>> {
        if matrix.is_csc() {
            Ok(Self { inner: matrix })
        } else {
            Err(matrix)
        }
    }
}

impl<M> SprsCsc<M> {
    /// Borrow the original matrix or view.
    pub fn as_inner(&self) -> &M {
        &self.inner
    }

    /// Recover the original matrix or view without copying.
    pub fn into_inner(self) -> M {
        self.inner
    }
}

impl<M: MatrixShape> MatrixShape for SprsCsc<M> {
    fn nrows(&self) -> usize {
        self.inner.nrows()
    }
    fn ncols(&self) -> usize {
        self.inner.ncols()
    }
}

impl<F, IP, IS, DS, Iptr> SparseColumns<F> for SprsCsc<CsMatBase<F, usize, IP, IS, DS, Iptr>>
where
    F: Scalar,
    Iptr: SpIndex,
    IP: Deref<Target = [Iptr]>,
    IS: Deref<Target = [usize]>,
    DS: Deref<Target = [F]>,
{
    fn sparse_column(&self, j: usize) -> (&[usize], &[F]) {
        assert!(j < self.ncols(), "column index out of bounds");
        // sprs adjusts these ranges for views whose first pointer is nonzero.
        let range = self.inner.indptr().outer_inds_sz(j);
        (
            &self.inner.indices()[range.clone()],
            &self.inner.data()[range],
        )
    }
}

impl<F, I, IP, IS, DS, Iptr> RawColumns<F> for SprsCsc<CsMatBase<F, I, IP, IS, DS, Iptr>>
where
    F: Scalar,
    I: SpIndex,
    Iptr: SpIndex,
    IP: Deref<Target = [Iptr]>,
    IS: Deref<Target = [I]>,
    DS: Deref<Target = [F]>,
{
    type Column<'a>
        = CsVecViewI<'a, F, I>
    where
        Self: 'a;

    fn raw_column(&self, j: usize) -> Self::Column<'_> {
        self.inner
            .outer_view(j)
            .expect("column index out of bounds")
    }
}

impl<F, I, IS, DS> RawColumn<F> for CsVecBase<IS, DS, F, I>
where
    F: Scalar,
    I: SpIndex,
    IS: Deref<Target = [I]>,
    DS: Deref<Target = [F]>,
{
    fn len(&self) -> usize {
        self.dim()
    }

    fn stored_len(&self) -> usize {
        self.nnz()
    }

    fn for_each_stored(&self, mut f: impl FnMut(usize, F)) {
        for (row, &value) in self.iter() {
            f(row, value);
        }
    }
}

impl<M: MatVec<V>, V> MatVec<V> for SprsCsc<M> {
    fn matvec(&self, x: &V) -> Result<V, Self::Error> {
        self.inner.matvec(x)
    }
}

impl<M: MatTransposeVec<V>, V> MatTransposeVec<V> for SprsCsc<M> {
    fn mat_transpose_vec(&self, x: &V) -> Result<V, Self::Error> {
        self.inner.mat_transpose_vec(x)
    }
}

impl<M: MatVecInto<X, Y>, X, Y> MatVecInto<X, Y> for SprsCsc<M> {
    fn matvec_into(&self, x: &X, out: &mut Y) -> Result<(), Self::Error> {
        self.inner.matvec_into(x, out)
    }
}

impl<M: MatTransposeVecInto<X, Y>, X, Y> MatTransposeVecInto<X, Y> for SprsCsc<M> {
    fn mat_transpose_vec_into(&self, x: &X, out: &mut Y) -> Result<(), Self::Error> {
        self.inner.mat_transpose_vec_into(x, out)
    }
}

impl<M: MatVecScaledInto<X, Y, F>, X, Y, F> MatVecScaledInto<X, Y, F> for SprsCsc<M> {
    fn matvec_scaled_into(&self, alpha: F, x: &X, beta: F, out: &mut Y) -> Result<(), Self::Error> {
        self.inner.matvec_scaled_into(alpha, x, beta, out)
    }
}

impl<M: MatTransposeVecScaledInto<X, Y, F>, X, Y, F> MatTransposeVecScaledInto<X, Y, F>
    for SprsCsc<M>
{
    fn mat_transpose_vec_scaled_into(
        &self,
        alpha: F,
        x: &X,
        beta: F,
        out: &mut Y,
    ) -> Result<(), Self::Error> {
        self.inner
            .mat_transpose_vec_scaled_into(alpha, x, beta, out)
    }
}

impl<M: ColumnStats<F>, F: Scalar> ColumnStats<F> for SprsCsc<M> {
    fn normalization_stats(
        &self,
        spec: crate::Normalization,
    ) -> Result<crate::NormalizationStats<F>, Self::Error> {
        self.inner.normalization_stats(spec)
    }

    fn col_means(&self) -> Result<Vec<F>, Self::Error> {
        self.inner.col_means()
    }
    fn col_sds(&self) -> Result<Vec<F>, Self::Error> {
        self.inner.col_sds()
    }
    fn col_mins(&self) -> Result<Vec<F>, Self::Error> {
        self.inner.col_mins()
    }
    fn col_ranges(&self) -> Result<Vec<F>, Self::Error> {
        self.inner.col_ranges()
    }
    fn col_maxabs(&self) -> Result<Vec<F>, Self::Error> {
        self.inner.col_maxabs()
    }
    fn col_l1(&self) -> Result<Vec<F>, Self::Error> {
        self.inner.col_l1()
    }
    fn col_l2(&self) -> Result<Vec<F>, Self::Error> {
        self.inner.col_l2()
    }
    fn col_l1_centered(&self, centers: &[F]) -> Result<Vec<F>, Self::Error> {
        self.inner.col_l1_centered(centers)
    }
    fn col_l2_centered(&self, centers: &[F]) -> Result<Vec<F>, Self::Error> {
        self.inner.col_l2_centered(centers)
    }
    fn col_maxabs_centered(&self, centers: &[F]) -> Result<Vec<F>, Self::Error> {
        self.inner.col_maxabs_centered(centers)
    }
}

impl<M: crate::MatrixErrorType> crate::MatrixErrorType for SprsCsc<M> {
    type Error = M::Error;
}
