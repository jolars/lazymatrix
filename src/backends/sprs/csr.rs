use std::ops::Deref;

use sprs::{CsMatBase, SpIndex};

use crate::{
    ColumnStats, MatTransposeVec, MatTransposeVecInto, MatTransposeVecScaledInto, MatVec,
    MatVecInto, MatVecScaledInto, MatrixShape, Scalar, SparseRows,
};

/// A checked CSR matrix or view that supports contiguous borrowed rows.
///
/// sprs stores CSC and CSR orientation in a runtime flag on the same type.
/// This wrapper checks that flag once, in [`try_new`](Self::try_new), so row
/// access never needs a gather or an orientation conversion. Construction and
/// row borrowing take O(1) time and allocate nothing. The wrapper retains
/// ownership of the supplied matrix or view and exposes no mutable access that
/// could change its orientation.
///
/// Operators and statistics accept any sprs index type. Borrowing column indices
/// as slices requires `usize` to match [`SparseRows`]. Pointer indices may use
/// any supported width.
///
/// ```
/// use lazymatrix::{SparseRows, SprsCsr};
/// use sprs::CsMat;
///
/// let x = CsMat::new((2, 3), vec![0, 2, 3], vec![0, 2, 1], vec![1.0, 0.0, 2.0]);
/// let csr = SprsCsr::try_new(x.view()).unwrap();
/// let (columns, values) = csr.sparse_row(0);
/// assert_eq!(columns, &[0, 2]);
/// assert_eq!(values, &[1.0, 0.0]);
/// ```
///
/// A plain sprs matrix has no row-borrowing capability, since it could be CSC:
///
/// ```compile_fail
/// use lazymatrix::SparseRows;
/// fn needs_rows<M: SparseRows<f64>>() {}
/// needs_rows::<sprs::CsMat<f64>>();
/// ```
///
/// Checked CSC storage cannot borrow rows:
///
/// ```compile_fail
/// use lazymatrix::{SparseRows, SprsCsc};
/// fn needs_rows<M: SparseRows<f64>>() {}
/// needs_rows::<SprsCsc<sprs::CsMat<f64>>>();
/// ```
///
/// Indices of another width cannot provide a borrowed `&[usize]`:
///
/// ```compile_fail
/// use lazymatrix::{SparseRows, SprsCsr};
/// fn needs_rows<M: SparseRows<f64>>() {}
/// needs_rows::<SprsCsr<sprs::CsMatI<f64, u32>>>();
/// ```
///
/// Checked CSR storage does not provide contiguous columns:
///
/// ```compile_fail
/// use lazymatrix::{SparseColumns, SprsCsr};
/// fn needs_columns<M: SparseColumns<f64>>() {}
/// needs_columns::<SprsCsr<sprs::CsMat<f64>>>();
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SprsCsr<M> {
    inner: M,
}

impl<F, I, IP, IS, DS, Iptr> SprsCsr<CsMatBase<F, I, IP, IS, DS, Iptr>>
where
    I: SpIndex,
    Iptr: SpIndex,
    IP: Deref<Target = [Iptr]>,
    IS: Deref<Target = [I]>,
    DS: Deref<Target = [F]>,
{
    /// Wrap CSR storage without copying, or return the original CSC input.
    ///
    /// To convert CSC storage explicitly, call sprs's `to_csr` or `into_csr`
    /// before this constructor. Borrow existing data by passing `matrix.view()`.
    pub fn try_new(
        matrix: CsMatBase<F, I, IP, IS, DS, Iptr>,
    ) -> Result<Self, CsMatBase<F, I, IP, IS, DS, Iptr>> {
        if matrix.is_csr() {
            Ok(Self { inner: matrix })
        } else {
            Err(matrix)
        }
    }
}

impl<M> SprsCsr<M> {
    /// Borrow the original matrix or view.
    pub fn as_inner(&self) -> &M {
        &self.inner
    }

    /// Recover the original matrix or view without copying.
    pub fn into_inner(self) -> M {
        self.inner
    }
}

impl<M: MatrixShape> MatrixShape for SprsCsr<M> {
    fn nrows(&self) -> usize {
        self.inner.nrows()
    }

    fn ncols(&self) -> usize {
        self.inner.ncols()
    }
}

impl<F, IP, IS, DS, Iptr> SparseRows<F> for SprsCsr<CsMatBase<F, usize, IP, IS, DS, Iptr>>
where
    F: Scalar,
    Iptr: SpIndex,
    IP: Deref<Target = [Iptr]>,
    IS: Deref<Target = [usize]>,
    DS: Deref<Target = [F]>,
{
    fn sparse_row(&self, i: usize) -> (&[usize], &[F]) {
        assert!(i < self.nrows(), "row index out of bounds");
        // sprs adjusts these ranges for views whose first pointer is nonzero.
        let range = self.inner.indptr().outer_inds_sz(i);
        (
            &self.inner.indices()[range.clone()],
            &self.inner.data()[range],
        )
    }
}

impl<M: MatVec<V>, V> MatVec<V> for SprsCsr<M> {
    fn matvec(&self, x: &V) -> Result<V, Self::Error> {
        self.inner.matvec(x)
    }
}

impl<M: MatTransposeVec<V>, V> MatTransposeVec<V> for SprsCsr<M> {
    fn mat_transpose_vec(&self, x: &V) -> Result<V, Self::Error> {
        self.inner.mat_transpose_vec(x)
    }
}

impl<M: MatVecInto<X, Y>, X, Y> MatVecInto<X, Y> for SprsCsr<M> {
    fn matvec_into(&self, x: &X, out: &mut Y) -> Result<(), Self::Error> {
        self.inner.matvec_into(x, out)
    }
}

impl<M: MatTransposeVecInto<X, Y>, X, Y> MatTransposeVecInto<X, Y> for SprsCsr<M> {
    fn mat_transpose_vec_into(&self, x: &X, out: &mut Y) -> Result<(), Self::Error> {
        self.inner.mat_transpose_vec_into(x, out)
    }
}

impl<M: MatVecScaledInto<X, Y, F>, X, Y, F> MatVecScaledInto<X, Y, F> for SprsCsr<M> {
    fn matvec_scaled_into(&self, alpha: F, x: &X, beta: F, out: &mut Y) -> Result<(), Self::Error> {
        self.inner.matvec_scaled_into(alpha, x, beta, out)
    }
}

impl<M: MatTransposeVecScaledInto<X, Y, F>, X, Y, F> MatTransposeVecScaledInto<X, Y, F>
    for SprsCsr<M>
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

impl<M: ColumnStats<F>, F: Scalar> ColumnStats<F> for SprsCsr<M> {
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

impl<M: crate::MatrixErrorType> crate::MatrixErrorType for SprsCsr<M> {
    type Error = M::Error;
}

impl<M: crate::MaterializeDense<F>, F: Scalar> crate::MaterializeDense<F> for SprsCsr<M> {
    fn materialize_normalized_into<O: crate::MatrixWrite<F> + ?Sized>(
        &self,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut O,
    ) -> Result<(), Self::Error> {
        self.inner.materialize_normalized_into(centers, scales, out)
    }
}
