use crate::{ColumnNormalizationParams, NormalizationParams, Scalar};

/// A borrowed sparse row with lazy row, column, or composed normalization.
///
/// Raw indices and values preserve storage order, duplicates, and explicit zeros.
/// The view borrows fitted parameters without copying or refitting them. Logical
/// entries apply every saved stage in order after summing duplicate raw values.
///
/// The sparse-plus-affine representation consists of a generally dense
/// background, exposed by [`Self::implicit_value`], and stored corrections,
/// exposed by [`Self::stored_corrections`]. Adding these parts can differ from
/// direct normalization because of rounding or nonfinite arithmetic. Use
/// [`Self::get`] or [`Self::normalize_value`] for direct stage-by-stage evaluation.
///
/// A view cannot outlive the operator and its borrowed parameters:
///
/// ```compile_fail
/// use lazymatrix::{LazyMatrix, SparseRows};
/// fn escaped<M: SparseRows<f64>>(data: M) -> f64 {
///     let row = {
///         let op = LazyMatrix::from_row_parts(data, None, Some(vec![2.0]));
///         op.row(0)
///     };
///     row.get(0)
/// }
/// ```
#[derive(Debug)]
pub struct LazyRow<'a, F, P = ColumnNormalizationParams<F>> {
    column_indices: &'a [usize],
    values: &'a [F],
    len: usize,
    normalization: &'a P,
    row: usize,
}

impl<F, P> Copy for LazyRow<'_, F, P> {}

impl<F, P> Clone for LazyRow<'_, F, P> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, F, P> LazyRow<'a, F, P> {
    pub(crate) fn new(
        column_indices: &'a [usize],
        values: &'a [F],
        len: usize,
        normalization: &'a P,
        row: usize,
    ) -> Self {
        Self {
            column_indices,
            values,
            len,
            normalization,
            row,
        }
    }

    /// Column indices of the raw stored entries, in backend storage order.
    ///
    /// Unsorted and duplicate indices remain present.
    pub fn column_indices(&self) -> &'a [usize] {
        self.column_indices
    }

    /// Raw stored values corresponding to [`Self::column_indices`].
    ///
    /// Explicitly stored zeros remain present.
    pub fn values(&self) -> &'a [F] {
        self.values
    }

    /// Logical length of the row, including structurally absent entries.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the logical row has no columns.
    ///
    /// A row with no stored entries can still have a nonzero logical length.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Borrow the saved parameters for the complete matrix.
    pub fn normalization(&self) -> &'a P {
        self.normalization
    }
}

impl<'a, F: Scalar, S: AsRef<[F]>> LazyRow<'a, F, ColumnNormalizationParams<F, S>> {
    /// Borrow the full column-center slice, or `None` when centering is inactive.
    pub fn centers(&self) -> Option<&'a [F]> {
        self.normalization.centers()
    }

    /// Borrow the full column-scale slice, or `None` when scaling is inactive.
    pub fn scales(&self) -> Option<&'a [F]> {
        self.normalization.scales()
    }

    /// Effective center of column `j`, or zero when centering is inactive.
    ///
    /// This takes O(1) time.
    ///
    /// # Panics
    ///
    /// Panics if `j >= self.len()`.
    pub fn center(&self, j: usize) -> F {
        assert!(j < self.len, "column index out of bounds");
        self.centers().map_or_else(F::zero, |centers| centers[j])
    }

    /// Effective scale of column `j`, or one when scaling is inactive.
    ///
    /// This takes O(1) time.
    ///
    /// # Panics
    ///
    /// Panics if `j >= self.len()`.
    pub fn scale(&self, j: usize) -> F {
        assert!(j < self.len, "column index out of bounds");
        self.scales().map_or_else(F::one, |scales| scales[j])
    }
}

impl<F: Scalar, P: NormalizationParams<Scalar = F>> LazyRow<'_, F, P> {
    /// Apply the saved stages to a coalesced raw value in column `j`.
    ///
    /// This takes O(number of stages) time and allocates nothing. Sum raw
    /// duplicates before calling this method.
    ///
    /// # Panics
    /// Panics if `j >= self.len()`.
    pub fn normalize_value(&self, j: usize, raw: F) -> F {
        assert!(j < self.len, "column index out of bounds");
        self.normalization.logical_entry(raw, self.row, j)
    }

    /// Read a logical entry, summing raw duplicates in storage order first.
    ///
    /// This scans O(stored entries), then applies O(number of stages)
    /// normalization work, without allocating. Structurally absent entries
    /// start at zero; stored signed zeros are preserved before normalization.
    ///
    /// # Panics
    /// Panics if `j >= self.len()`.
    pub fn get(&self, j: usize) -> F {
        assert!(j < self.len, "column index out of bounds");
        let raw = self
            .column_indices
            .iter()
            .zip(self.values)
            .filter(|(column, _)| **column == j)
            .map(|(_, &value)| value)
            .reduce(|sum, value| sum + value)
            .unwrap_or_else(F::zero);
        self.normalize_value(j, raw)
    }

    /// Logical value at a structurally absent column.
    ///
    /// This applies every stage to zero in O(number of stages) time.
    ///
    /// # Panics
    /// Panics if `j >= self.len()`.
    pub fn implicit_value(&self, j: usize) -> F {
        self.normalize_value(j, F::zero())
    }

    /// Stored corrections to the background as `(column, scaled raw value)`.
    ///
    /// The iterator preserves storage order, duplicates, and explicit zeros.
    /// Each correction applies only the scales, in stage order. Consuming it
    /// takes O(stored entries * number of stages) time and allocates nothing.
    /// See the [type documentation](Self) for numerical limitations.
    pub fn stored_corrections(&self) -> impl Iterator<Item = (usize, F)> + '_ {
        self.column_indices
            .iter()
            .copied()
            .zip(self.values)
            .map(|(j, &value)| (j, self.normalization.scaled_correction(value, self.row, j)))
    }
}
