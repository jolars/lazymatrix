use crate::column::{LazyColumn, LazySparseColumn, SparseColumnRef};
use crate::normalization::Normalization;
use crate::row::LazyRow;
use crate::traits::{
    ColumnStats, Columns, DotSlice, ElemDivAssign, MatTransposeVec, MatTransposeVecInto,
    MatTransposeVecScaledInto, MatVec, MatVecInto, MatVecScaledInto, MatrixShape, RawColumns,
    Scalar, ScaledSubSlice, SparseColumns, SparseRows, SubScalarAssign, SumEntries, VectorOwned,
    VectorView, VectorViewMut,
};

/// A matrix presented with lazy column normalization `X̃ = (X − 1cᵀ)S⁻¹`.
///
/// The underlying matrix `data` is never modified or densified; the centers and
/// scales are folded into the matrix–vector products on the fly. See the
/// [crate-level documentation](crate) for the math.
///
/// `centers` and `scales` are each `None` when that axis of normalization is
/// inactive. When present, each has length `ncols`, and no scale equals zero.
/// Negative scales and nonfinite normalization parameters are allowed.
#[derive(Clone, Debug)]
pub struct LazyMatrix<M, F = f64> {
    data: M,
    centers: Option<Vec<F>>,
    scales: Option<Vec<F>>,
}

impl<M, F: Scalar> LazyMatrix<M, F>
where
    M: MatrixShape,
{
    /// Construct from an explicit center and/or scale vector.
    ///
    /// Parameters are preserved unchanged, including negative scales and
    /// nonfinite values. Unlike [`Self::new`], this constructor rejects exact
    /// zero scales rather than replacing them with one.
    ///
    /// # Panics
    /// Panics if a provided `centers`/`scales` vector does not have length
    /// `ncols`, or if a scale is `+0.0` or `-0.0`. The zero-scale panic reports
    /// the zero-based column index.
    pub fn from_parts(data: M, centers: Option<Vec<F>>, scales: Option<Vec<F>>) -> Self {
        let ncols = data.ncols();
        if let Some(c) = &centers {
            assert_eq!(c.len(), ncols, "centers length must equal ncols");
        }
        if let Some(s) = &scales {
            assert_eq!(s.len(), ncols, "scales length must equal ncols");
            for (column, &scale) in s.iter().enumerate() {
                assert!(
                    scale != F::zero(),
                    "scale at column {column} must be nonzero"
                );
            }
        }
        Self {
            data,
            centers,
            scales,
        }
    }

    /// Wrap a matrix with column centering only.
    ///
    /// Centers are preserved unchanged, including nonfinite values.
    ///
    /// # Panics
    /// Panics if `centers.len() != ncols`.
    pub fn with_centers(data: M, centers: Vec<F>) -> Self {
        Self::from_parts(data, Some(centers), None)
    }

    /// Wrap a matrix with column scaling only.
    ///
    /// Scales are preserved unchanged, including negative and nonfinite values.
    /// Exact zero scales are rejected, as in [`Self::from_parts`].
    ///
    /// # Panics
    /// Panics if `scales.len() != ncols`, or if a scale is `+0.0` or `-0.0`.
    /// The zero-scale panic reports the zero-based column index.
    pub fn with_scales(data: M, scales: Vec<F>) -> Self {
        Self::from_parts(data, None, Some(scales))
    }

    /// Number of rows of the logical normalized matrix.
    pub fn nrows(&self) -> usize {
        self.data.nrows()
    }

    /// Number of columns of the logical normalized matrix.
    pub fn ncols(&self) -> usize {
        self.data.ncols()
    }

    /// The column centers `c`, if centering is active.
    pub fn centers(&self) -> Option<&[F]> {
        self.centers.as_deref()
    }

    /// The column scales `s`, if scaling is active.
    pub fn scales(&self) -> Option<&[F]> {
        self.scales.as_deref()
    }

    /// Borrow the underlying (un-normalized) matrix.
    pub fn data(&self) -> &M {
        &self.data
    }

    /// Borrow one lazily normalized column without copying.
    pub fn column(&self, j: usize) -> LazyColumn<M::Column<'_>, F>
    where
        M: RawColumns<F>,
    {
        assert!(j < self.ncols(), "column index out of bounds");
        LazyColumn::new(
            self.data.raw_column(j),
            self.centers.as_ref().map_or_else(F::zero, |c| c[j]),
            self.scales.as_ref().map_or_else(F::one, |s| s[j]),
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
            self.centers.as_ref().map_or_else(F::zero, |c| c[j]),
            self.scales.as_ref().map_or_else(F::one, |s| s[j]),
        )
    }

    /// Borrow one lazily normalized sparse row without copying.
    ///
    /// This takes O(1) time and allocates nothing. The view borrows the raw
    /// stored column indices and values and the full normalization slices.
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
    /// let lazy = LazyMatrix::from_parts(csr, Some(vec![0.5, -1.0, 2.0]), Some(vec![2.0; 3]));
    /// let row = lazy.row(0);
    /// assert_eq!(row.len(), 3);
    /// assert_eq!(row.column_indices(), &[0, 2]);
    /// assert_eq!(row.values(), &[1.0, 0.0]);
    /// assert_eq!(row.implicit_value(1), 0.5);
    /// assert_eq!(row.stored_corrections().collect::<Vec<_>>(), vec![(0, 0.5), (2, 0.0)]);
    /// # }
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
    /// # Panics
    ///
    /// Panics if `i >= self.nrows()`.
    pub fn row(&self, i: usize) -> LazyRow<'_, F>
    where
        M: SparseRows<F>,
    {
        assert!(i < self.nrows(), "row index out of bounds");
        let (column_indices, values) = self.data.sparse_row(i);
        LazyRow::new(
            column_indices,
            values,
            self.ncols(),
            self.centers(),
            self.scales(),
        )
    }

    /// Consume the wrapper, returning the underlying matrix and the
    /// center/scale vectors.
    pub fn into_parts(self) -> (M, Option<Vec<F>>, Option<Vec<F>>) {
        (self.data, self.centers, self.scales)
    }
}

impl<M, F: Scalar> LazyMatrix<M, F>
where
    M: ColumnStats<F> + MatrixShape,
{
    /// Construct by **computing** the centers and scales from `data` according
    /// to `spec`.
    ///
    /// When both centering and scaling are requested, scales are computed from
    /// the *centered* columns. Standard deviation and range are
    /// centering-invariant; `L1`, `L2`, and `MaxAbs` use sparse closed-form
    /// centered variants.
    ///
    /// An exact zero scale (e.g. a constant column whose standard deviation is
    /// zero) is replaced with `1`, so the resulting operator never divides by
    /// zero. Nonfinite statistics retain their IEEE values and propagate through
    /// subsequent operations.
    ///
    /// # Errors
    /// Returns the backend error if computing normalization statistics fails.
    pub fn new(data: M, spec: Normalization) -> Result<Self, M::Error> {
        let (centers, scales) = data.normalization_stats(spec)?;
        Ok(Self::from_parts(
            data,
            centers,
            scales.map(replace_zero_scales),
        ))
    }
}

impl<M, F> MatrixShape for LazyMatrix<M, F>
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

impl<M, F> Columns<F> for LazyMatrix<M, F>
where
    F: Scalar,
    M: RawColumns<F>,
{
    type Column<'a>
        = LazyColumn<M::Column<'a>, F>
    where
        Self: 'a;

    fn column(&self, j: usize) -> Self::Column<'_> {
        LazyMatrix::column(self, j)
    }
}

/// Replace exact zero entries with `1`, leaving nonfinite values untouched.
///
/// Mirrors the zero-variance guard used in standard penalized-regression
/// preprocessing: a constant column has scale `0`, which would otherwise produce
/// a division by zero; replacing it with `1` makes that column a no-op under
/// scaling.
fn replace_zero_scales<F: Scalar>(mut scales: Vec<F>) -> Vec<F> {
    let one = F::one();
    let zero = F::zero();
    for s in &mut scales {
        if *s == zero {
            *s = one;
        }
    }
    scales
}

impl<M, V, F> MatVec<V> for LazyMatrix<M, F>
where
    F: Scalar,
    M: MatVec<V>,
    V: Clone + ElemDivAssign<F> + DotSlice<F> + SubScalarAssign<F>,
{
    /// `X̃ v = X (S⁻¹ v) − 1 · (cᵀ S⁻¹ v)`.
    fn matvec(&self, v: &V) -> Result<V, Self::Error> {
        // The forward op clones `v` because it mutates it into `S⁻¹v`. The
        // transpose op below does NOT clone `u`: it reads `Σu` first, then only
        // reads `u` through the backend product.
        let mut w = v.clone();
        if let Some(s) = &self.scales {
            w.elem_div_assign(s); // w = S⁻¹ v
        }
        let mut y = self.data.matvec(&w)?;
        if let Some(c) = &self.centers {
            y.sub_scalar_assign(w.dot_slice(c)); // y −= 1 · (cᵀ w)
        }
        Ok(y)
    }
}

impl<M, X, Y, F> MatVecInto<X, Y> for LazyMatrix<M, F>
where
    F: Scalar,
    M: MatVecInto<X, Y>,
    X: Clone + ElemDivAssign<F> + DotSlice<F>,
    Y: SubScalarAssign<F>,
{
    /// `out = X̃ v = X (S⁻¹ v) − 1 · (cᵀ S⁻¹ v)`.
    fn matvec_into(&self, v: &X, out: &mut Y) -> Result<(), Self::Error> {
        if let Some(s) = &self.scales {
            let mut w = v.clone();
            w.elem_div_assign(s);
            self.data.matvec_into(&w, out)?;
            if let Some(c) = &self.centers {
                out.sub_scalar_assign(w.dot_slice(c));
            }
        } else {
            self.data.matvec_into(v, out)?;
            if let Some(c) = &self.centers {
                out.sub_scalar_assign(v.dot_slice(c));
            }
        }
        Ok(())
    }
}

impl<M, V, F> MatTransposeVec<V> for LazyMatrix<M, F>
where
    F: Scalar,
    M: MatTransposeVec<V>,
    V: SumEntries<F> + ScaledSubSlice<F> + ElemDivAssign<F>,
{
    /// `X̃ᵀ u = S⁻¹ (Xᵀ u − c · Σu)`.
    fn mat_transpose_vec(&self, u: &V) -> Result<V, Self::Error> {
        let total = if self.centers.is_some() {
            u.sum_entries()
        } else {
            F::zero()
        };
        let mut t = self.data.mat_transpose_vec(u)?;
        if let Some(c) = &self.centers {
            t.scaled_sub_slice(total, c); // t −= Σu · c
        }
        if let Some(s) = &self.scales {
            t.elem_div_assign(s); // t = S⁻¹ t
        }
        Ok(t)
    }
}

impl<M, X, Y, F> MatTransposeVecInto<X, Y> for LazyMatrix<M, F>
where
    F: Scalar,
    M: MatTransposeVecInto<X, Y>,
    X: SumEntries<F>,
    Y: ScaledSubSlice<F> + ElemDivAssign<F>,
{
    /// `out = X̃ᵀ u = S⁻¹ (Xᵀ u − c · Σu)`.
    fn mat_transpose_vec_into(&self, u: &X, out: &mut Y) -> Result<(), Self::Error> {
        let total = if self.centers.is_some() {
            u.sum_entries()
        } else {
            F::zero()
        };
        self.data.mat_transpose_vec_into(u, out)?;
        if let Some(c) = &self.centers {
            out.scaled_sub_slice(total, c);
        }
        if let Some(s) = &self.scales {
            out.elem_div_assign(s);
        }
        Ok(())
    }
}

impl<M, F: Scalar> LazyMatrix<M, F>
where
    M: MatrixShape,
{
    /// Apply `out = alpha * X̃ * x + beta * out` using caller-owned coefficient scratch.
    ///
    /// `scratch` must have length `ncols`. It is used when column scaling is
    /// active, and its contents after the call are unspecified on error.
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
        if let Some(scales) = self.scales.as_ref().filter(|_| alpha != F::zero()) {
            for (j, &scale) in scales.iter().enumerate() {
                scratch.set(j, x.get(j) / scale);
            }
            self.data.matvec_scaled_into(alpha, scratch, beta, out)?;
            if let Some(centers) = &self.centers {
                let correction: F = (0..self.ncols()).map(|j| scratch.get(j) * centers[j]).sum();
                for i in 0..out.len() {
                    out.set(i, out.get(i) - alpha * correction);
                }
            }
        } else {
            self.data.matvec_scaled_into(alpha, x, beta, out)?;
            if let Some(centers) = self.centers.as_ref().filter(|_| alpha != F::zero()) {
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
    /// `scratch` must have length `ncols`. Scaling with nonzero `alpha` uses
    /// this buffer to keep the previous `out` values separate from the raw
    /// transpose product.
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
        if let Some(scales) = self.scales.as_ref().filter(|_| alpha != F::zero()) {
            self.data
                .mat_transpose_vec_scaled_into(F::one(), x, F::zero(), scratch)?;
            let total = if self.centers.is_some() {
                x.sum()
            } else {
                F::zero()
            };
            for (j, &scale) in scales.iter().enumerate() {
                let center = self.centers.as_ref().map_or_else(F::zero, |c| c[j]);
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
            if let Some(centers) = self.centers.as_ref().filter(|_| alpha != F::zero()) {
                let total = x.sum();
                for (j, &center) in centers.iter().enumerate() {
                    out.set(j, out.get(j) - alpha * center * total);
                }
            }
        }
        Ok(())
    }
}

impl<M, X, Y, F> MatVecScaledInto<X, Y, F> for LazyMatrix<M, F>
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
        if self.scales.is_some() && alpha != F::zero() {
            let mut scratch = X::owned_from_fn(self.ncols(), |_| F::zero());
            self.matvec_scaled_with_workspace(alpha, x, beta, out, &mut scratch)
        } else {
            self.data.matvec_scaled_into(alpha, x, beta, out)?;
            if let Some(centers) = self.centers.as_ref().filter(|_| alpha != F::zero()) {
                let correction: F = (0..self.ncols()).map(|j| x.get(j) * centers[j]).sum();
                for i in 0..out.len() {
                    out.set(i, out.get(i) - alpha * correction);
                }
            }
            Ok(())
        }
    }
}

impl<M, X, Y, F> MatTransposeVecScaledInto<X, Y, F> for LazyMatrix<M, F>
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
        if self.scales.is_some() && alpha != F::zero() {
            let mut scratch = Y::owned_from_fn(self.ncols(), |_| F::zero());
            self.mat_transpose_vec_scaled_with_workspace(alpha, x, beta, out, &mut scratch)
        } else {
            self.data
                .mat_transpose_vec_scaled_into(alpha, x, beta, out)?;
            if let Some(centers) = self.centers.as_ref().filter(|_| alpha != F::zero()) {
                let total = x.sum();
                for (j, &center) in centers.iter().enumerate() {
                    out.set(j, out.get(j) - alpha * center * total);
                }
            }
            Ok(())
        }
    }
}

impl<M: crate::MatrixErrorType, F> crate::MatrixErrorType for LazyMatrix<M, F> {
    type Error = M::Error;
}

impl<M, F> crate::WeightedGramInto<F> for LazyMatrix<M, F>
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
            self.centers(),
            self.scales(),
            out,
        );
        self.data
            .weighted_gram_normalized_into(weights, self.centers(), self.scales(), out)
    }
}

impl<M, F> crate::WeightedColumnSumsInto<F> for LazyMatrix<M, F>
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
            self.centers(),
            self.scales(),
            out,
        );
        self.data
            .weighted_column_sums_normalized_into(weights, self.centers(), self.scales(), out)
    }
}
