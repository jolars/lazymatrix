use super::{MatrixShape, Scalar, VectorView, VectorViewMut};

/// A borrowed raw column supplied by a matrix backend.
///
/// `for_each_stored` visits every stored entry, including explicitly stored
/// zeros. Dense implementations visit every row. Sparse implementations omit
/// structural zeros.
pub trait RawColumn<F: Scalar> {
    fn len(&self) -> usize;
    fn stored_len(&self) -> usize;
    fn for_each_stored(&self, f: impl FnMut(usize, F));

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn raw_sum(&self) -> F {
        let mut sum = F::zero();
        self.for_each_stored(|_, value| sum = sum + value);
        sum
    }

    /// Add `raw_multiplier * raw_column + offset` to a dense destination.
    ///
    /// Sparse implementations take O(nnz) when `offset == 0`, and O(n + nnz)
    /// otherwise. Dense implementations take O(n).
    fn affine_add_to<V>(&self, raw_multiplier: F, offset: F, destination: &mut V)
    where
        V: VectorViewMut<F> + ?Sized,
    {
        assert_eq!(
            destination.len(),
            self.len(),
            "destination length must equal column length"
        );
        if offset != F::zero() {
            for row in 0..destination.len() {
                destination.set(row, destination.get(row) + offset);
            }
        }
        self.for_each_stored(|row, value| {
            destination.set(row, destination.get(row) + raw_multiplier * value);
        });
    }
}

/// Borrowed raw-column access for a backend matrix.
pub trait RawColumns<F: Scalar>: MatrixShape {
    type Column<'a>: RawColumn<F>
    where
        Self: 'a;

    fn raw_column(&self, j: usize) -> Self::Column<'_>;
}

/// Operations on one logical, possibly normalized column.
pub trait LogicalColumn<F: Scalar> {
    fn len(&self) -> usize;
    fn center(&self) -> F;
    fn scale(&self) -> F;
    fn sum(&self) -> F;
    fn norm_squared(&self) -> F;
    fn dot<V: VectorView<F> + ?Sized>(&self, vector: &V) -> F;
    fn dot_with_sum<V: VectorView<F> + ?Sized>(&self, vector: &V, vector_sum: F) -> F;
    fn weighted_dot<V, W>(&self, vector: &V, weights: &W) -> F
    where
        V: VectorView<F> + ?Sized,
        W: VectorView<F> + ?Sized;
    fn weighted_dot_with_sum<V, W>(&self, vector: &V, weights: &W, weighted_vector_sum: F) -> F
    where
        V: VectorView<F> + ?Sized,
        W: VectorView<F> + ?Sized;
    /// Sum weighted squares of logical entries, including implicit zeros.
    fn weighted_norm_squared<W: VectorView<F> + ?Sized>(&self, weights: &W) -> F;
    /// Accept a cached total, which implementations may ignore for stability.
    ///
    /// [`LazyColumn`](crate::LazyColumn) uses the same O(n + nnz) calculation
    /// as its uncached method to preserve small implicit-zero contributions.
    fn weighted_norm_squared_with_sum<W: VectorView<F> + ?Sized>(
        &self,
        weights: &W,
        weight_sum: F,
    ) -> F;
    fn scaled_add_to<V: VectorViewMut<F> + ?Sized>(&self, alpha: F, destination: &mut V);

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Borrowed logical-column access for a matrix or matrix-like operator.
pub trait Columns<F: Scalar>: MatrixShape {
    type Column<'a>: LogicalColumn<F>
    where
        Self: 'a;

    fn column(&self, j: usize) -> Self::Column<'_>;
}

/// Borrowed access to columns stored contiguously in sparse form.
///
/// Implement this capability only when a backend can return the stored row
/// indices and values for one column as slices without gathering or copying.
/// Structurally absent entries are implicit zeros.
pub trait SparseColumns<F: Scalar>: MatrixShape {
    /// Return the stored `(row index, raw value)` slices for column `j`.
    ///
    /// The two slices have equal length and corresponding entries. Explicitly
    /// stored zeros remain present.
    ///
    /// # Panics
    ///
    /// Panics if `j >= self.ncols()`.
    fn sparse_column(&self, j: usize) -> (&[usize], &[F]);
}

impl<M, F> RawColumns<F> for &M
where
    M: RawColumns<F> + ?Sized,
    F: Scalar,
{
    type Column<'a>
        = M::Column<'a>
    where
        Self: 'a;

    fn raw_column(&self, j: usize) -> Self::Column<'_> {
        (**self).raw_column(j)
    }
}

impl<M, F> SparseColumns<F> for &M
where
    M: SparseColumns<F> + ?Sized,
    F: Scalar,
{
    fn sparse_column(&self, j: usize) -> (&[usize], &[F]) {
        (**self).sparse_column(j)
    }
}

impl<M, F> RawColumns<F> for &mut M
where
    M: RawColumns<F> + ?Sized,
    F: Scalar,
{
    type Column<'a>
        = M::Column<'a>
    where
        Self: 'a;

    fn raw_column(&self, j: usize) -> Self::Column<'_> {
        (**self).raw_column(j)
    }
}

impl<M, F> SparseColumns<F> for &mut M
where
    M: SparseColumns<F> + ?Sized,
    F: Scalar,
{
    fn sparse_column(&self, j: usize) -> (&[usize], &[F]) {
        (**self).sparse_column(j)
    }
}
