use crate::{
    LogicalColumn, NormalizationParams, RawColumn, Scalar, SparseColumnRef, VectorView,
    VectorViewMut,
};
use num_traits::Zero;

/// A borrowed column with row or ordered normalization.
///
/// The view borrows raw storage and fitted parameters without copying, fitting,
/// gathering, or decoding. Logical entries combine raw duplicates in storage
/// order before applying each saved stage. Borrowing takes O(1) time.
///
/// Reductions and dense updates visit every logical row, including implicit
/// zeros, in O(nrows * stages + stored entries) time. Dense slices and sorted,
/// unique sparse columns use constant workspace. Other storage orders and
/// duplicates use one O(nrows) working column to coalesce raw values. The view
/// itself and individual reads allocate nothing.
///
/// Sparse views expose their generally dense, row-dependent background through
/// [`Self::implicit_value`] and scaled raw corrections through
/// [`Self::stored_corrections`]. Adding those parts may differ from direct
/// normalization through rounding or nonfinite arithmetic. Use [`Self::get`] or
/// [`Self::normalize_value`] for direct stage-by-stage evaluation.
///
/// A view cannot outlive the operator or its fitted parameters:
///
/// ```compile_fail
/// use lazymatrix::{LazyMatrix, RawColumns};
/// fn escaped<M: RawColumns<f64>>(data: M) -> f64 {
///     let column = {
///         let op = LazyMatrix::from_row_parts(data, None, None);
///         op.column(0)
///     };
///     column.get(0)
/// }
/// ```
#[derive(Debug)]
pub struct NormalizedColumn<'a, C, P> {
    raw: C,
    normalization: &'a P,
    column: usize,
}

impl<C: Copy, P> Copy for NormalizedColumn<'_, C, P> {}

impl<C: Clone, P> Clone for NormalizedColumn<'_, C, P> {
    fn clone(&self) -> Self {
        Self {
            raw: self.raw.clone(),
            normalization: self.normalization,
            column: self.column,
        }
    }
}

impl<'a, C, P> NormalizedColumn<'a, C, P> {
    pub(crate) fn new(raw: C, normalization: &'a P, column: usize) -> Self {
        Self {
            raw,
            normalization,
            column,
        }
    }

    /// Borrow the backend's raw column view.
    pub fn raw(&self) -> &C {
        &self.raw
    }

    /// Borrow the saved parameters for the complete matrix.
    pub fn normalization(&self) -> &'a P {
        self.normalization
    }
}

impl<C, P> NormalizedColumn<'_, C, P>
where
    P: NormalizationParams,
    C: RawColumn<P::Scalar>,
{
    /// Logical length, including structurally absent entries.
    pub fn len(&self) -> usize {
        self.raw.len()
    }

    /// Whether the column has no logical rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Apply every saved stage to a coalesced raw value in row `row`.
    ///
    /// This takes O(stages) time and allocates nothing. Sum raw duplicates
    /// before calling this method.
    ///
    /// # Panics
    /// Panics if `row >= self.len()`.
    pub fn normalize_value(&self, row: usize, raw: P::Scalar) -> P::Scalar {
        assert!(row < self.len(), "row index out of bounds");
        self.normalization.logical_entry(raw, row, self.column)
    }

    /// Read a logical entry, combining raw duplicates in storage order first.
    ///
    /// Dense slices take O(stages) time; other raw columns take
    /// O(stored entries + stages) time. This allocates nothing and preserves
    /// stored signed zeros before normalization.
    ///
    /// # Panics
    /// Panics if `row >= self.len()`.
    pub fn get(&self, row: usize) -> P::Scalar {
        assert!(row < self.len(), "row index out of bounds");
        let raw = if let Some(values) = self.raw.dense_values() {
            values[row]
        } else {
            let mut raw = None;
            self.raw.for_each_stored(|i, value| {
                if i == row {
                    raw = Some(raw.map_or(value, |sum| sum + value));
                }
            });
            raw.unwrap_or_else(P::Scalar::zero)
        };
        self.normalize_value(row, raw)
    }

    /// Logical value at a structurally absent row, in O(stages) time.
    ///
    /// # Panics
    /// Panics if `row >= self.len()`.
    pub fn implicit_value(&self, row: usize) -> P::Scalar {
        self.normalize_value(row, P::Scalar::zero())
    }

    fn for_each(&self, mut visit: impl FnMut(usize, P::Scalar)) {
        if let Some(values) = self.raw.dense_values() {
            for (row, &value) in values.iter().enumerate() {
                visit(row, self.normalize_value(row, value));
            }
            return;
        }

        let mut previous = None;
        let mut canonical = true;
        self.raw.for_each_stored(|row, _| {
            if previous.is_some_and(|last| row <= last) {
                canonical = false;
            }
            previous = Some(row);
        });
        if !canonical {
            // Track presence separately from zero so a lone stored -0.0 keeps
            // its sign. Coalescing must finish before updating a destination.
            let mut values = vec![None; self.len()];
            self.raw.for_each_stored(|row, value| {
                values[row] = Some(values[row].map_or(value, |sum| sum + value));
            });
            for (row, value) in values.into_iter().enumerate() {
                visit(
                    row,
                    self.normalize_value(row, value.unwrap_or_else(P::Scalar::zero)),
                );
            }
            return;
        }

        let mut next = 0;
        self.raw.for_each_stored(|row, value| {
            for i in next..row {
                visit(i, self.implicit_value(i));
            }
            visit(row, self.normalize_value(row, value));
            next = row + 1;
        });
        for row in next..self.len() {
            visit(row, self.implicit_value(row));
        }
    }

    fn sum_map(&self, mut map: impl FnMut(usize, P::Scalar) -> P::Scalar) -> P::Scalar {
        let mut sum = P::Scalar::zero();
        self.for_each(|row, value| sum = sum + map(row, value));
        sum
    }

    /// Sum logical entries, including transformed implicit zeros.
    ///
    /// See the [type documentation](Self) for time and workspace bounds.
    pub fn sum(&self) -> P::Scalar {
        self.sum_map(|_, value| value)
    }

    /// Sum squared logical entries after applying every stage in order.
    ///
    /// See the [type documentation](Self) for time and workspace bounds.
    pub fn norm_squared(&self) -> P::Scalar {
        self.sum_map(|_, value| value * value)
    }

    /// Dot product with a vector, including transformed implicit zeros.
    ///
    /// See the [type documentation](Self) for time and workspace bounds.
    ///
    /// # Panics
    /// Panics if `vector.len() != self.len()`.
    pub fn dot<V: VectorView<P::Scalar> + ?Sized>(&self, vector: &V) -> P::Scalar {
        assert_eq!(
            vector.len(),
            self.len(),
            "vector length must equal column length"
        );
        self.sum_map(|row, value| value * vector.get(row))
    }

    /// Dot product accepting a cached vector sum for [`LogicalColumn`] parity.
    ///
    /// The sum is ignored because a scalar total cannot account for
    /// row-dependent normalization. Complexity and panics match [`Self::dot`].
    pub fn dot_with_sum<V: VectorView<P::Scalar> + ?Sized>(
        &self,
        vector: &V,
        _vector_sum: P::Scalar,
    ) -> P::Scalar {
        self.dot(vector)
    }

    /// Compute `sum_i weights[i] * self[i] * vector[i]` on logical entries.
    ///
    /// See the [type documentation](Self) for time and workspace bounds.
    ///
    /// # Panics
    /// Panics unless `vector.len() == weights.len() == self.len()`.
    pub fn weighted_dot<V, W>(&self, vector: &V, weights: &W) -> P::Scalar
    where
        V: VectorView<P::Scalar> + ?Sized,
        W: VectorView<P::Scalar> + ?Sized,
    {
        assert_eq!(
            vector.len(),
            self.len(),
            "vector length must equal column length"
        );
        assert_eq!(
            weights.len(),
            self.len(),
            "weights length must equal column length"
        );
        self.sum_map(|row, value| value * weights.get(row) * vector.get(row))
    }

    /// Weighted dot product accepting a cached sum for [`LogicalColumn`] parity.
    ///
    /// The cached total is ignored. Complexity and panics match
    /// [`Self::weighted_dot`].
    pub fn weighted_dot_with_sum<V, W>(
        &self,
        vector: &V,
        weights: &W,
        _weighted_vector_sum: P::Scalar,
    ) -> P::Scalar
    where
        V: VectorView<P::Scalar> + ?Sized,
        W: VectorView<P::Scalar> + ?Sized,
    {
        self.weighted_dot(vector, weights)
    }

    /// Sum `weights[i] * self[i]^2`, including implicit zeros and signed weights.
    ///
    /// Normalizes before squaring and visits implicit rows directly, preserving
    /// small contributions and IEEE nonfinite arithmetic. See the
    /// [type documentation](Self) for time and workspace bounds.
    ///
    /// # Panics
    /// Panics if `weights.len() != self.len()`.
    pub fn weighted_norm_squared<W: VectorView<P::Scalar> + ?Sized>(
        &self,
        weights: &W,
    ) -> P::Scalar {
        assert_eq!(
            weights.len(),
            self.len(),
            "weights length must equal column length"
        );
        self.sum_map(|row, value| weights.get(row) * value * value)
    }

    /// Weighted squared norm accepting a cached total for [`LogicalColumn`] parity.
    ///
    /// The cached total is ignored to preserve implicit-row contributions.
    /// Complexity and panics match [`Self::weighted_norm_squared`].
    pub fn weighted_norm_squared_with_sum<W: VectorView<P::Scalar> + ?Sized>(
        &self,
        weights: &W,
        _weight_sum: P::Scalar,
    ) -> P::Scalar {
        self.weighted_norm_squared(weights)
    }

    /// Add `alpha` times each logical entry to a dense destination.
    ///
    /// Normalizes each coalesced value before multiplying by `alpha`. See the
    /// [type documentation](Self) for time and workspace bounds.
    ///
    /// # Panics
    /// Panics if `destination.len() != self.len()`.
    pub fn scaled_add_to<V: VectorViewMut<P::Scalar> + ?Sized>(
        &self,
        alpha: P::Scalar,
        destination: &mut V,
    ) {
        assert_eq!(
            destination.len(),
            self.len(),
            "destination length must equal column length"
        );
        self.for_each(|row, value| destination.set(row, destination.get(row) + alpha * value));
    }
}

impl<'a, F: Scalar, P: NormalizationParams<Scalar = F>>
    NormalizedColumn<'a, SparseColumnRef<'a, F>, P>
{
    /// Raw stored row indices in storage order, retaining duplicates.
    pub fn row_indices(&self) -> &'a [usize] {
        self.raw.row_indices()
    }

    /// Raw stored values, including explicitly stored zeros.
    pub fn values(&self) -> &'a [F] {
        self.raw.values()
    }

    /// Sum raw stored values without applying normalization.
    pub fn raw_sum(&self) -> F {
        self.raw.raw_sum()
    }

    /// Stored corrections as `(row, scaled raw value)`, without allocating.
    ///
    /// Preserves storage order, duplicates, and explicit zeros. Each correction
    /// applies scales in stage order, taking O(stages) time. See the
    /// [type documentation](Self) for numerical limitations.
    pub fn stored_corrections(&self) -> impl Iterator<Item = (usize, F)> + '_ {
        self.row_indices()
            .iter()
            .copied()
            .zip(self.values())
            .map(|(row, &value)| {
                (
                    row,
                    self.normalization
                        .scaled_correction(value, row, self.column),
                )
            })
    }
}

impl<C, P> LogicalColumn<P::Scalar> for NormalizedColumn<'_, C, P>
where
    P: NormalizationParams,
    C: RawColumn<P::Scalar>,
{
    fn len(&self) -> usize {
        self.len()
    }
    fn get(&self, row: usize) -> P::Scalar {
        self.get(row)
    }
    fn sum(&self) -> P::Scalar {
        self.sum()
    }
    fn norm_squared(&self) -> P::Scalar {
        self.norm_squared()
    }
    fn dot<V: VectorView<P::Scalar> + ?Sized>(&self, vector: &V) -> P::Scalar {
        self.dot(vector)
    }
    fn dot_with_sum<V: VectorView<P::Scalar> + ?Sized>(
        &self,
        vector: &V,
        sum: P::Scalar,
    ) -> P::Scalar {
        self.dot_with_sum(vector, sum)
    }
    fn weighted_dot<V, W>(&self, vector: &V, weights: &W) -> P::Scalar
    where
        V: VectorView<P::Scalar> + ?Sized,
        W: VectorView<P::Scalar> + ?Sized,
    {
        self.weighted_dot(vector, weights)
    }
    fn weighted_dot_with_sum<V, W>(&self, vector: &V, weights: &W, sum: P::Scalar) -> P::Scalar
    where
        V: VectorView<P::Scalar> + ?Sized,
        W: VectorView<P::Scalar> + ?Sized,
    {
        self.weighted_dot_with_sum(vector, weights, sum)
    }
    fn weighted_norm_squared<W: VectorView<P::Scalar> + ?Sized>(&self, weights: &W) -> P::Scalar {
        self.weighted_norm_squared(weights)
    }
    fn weighted_norm_squared_with_sum<W: VectorView<P::Scalar> + ?Sized>(
        &self,
        weights: &W,
        sum: P::Scalar,
    ) -> P::Scalar {
        self.weighted_norm_squared_with_sum(weights, sum)
    }
    fn scaled_add_to<V: VectorViewMut<P::Scalar> + ?Sized>(
        &self,
        alpha: P::Scalar,
        destination: &mut V,
    ) {
        self.scaled_add_to(alpha, destination);
    }
}
