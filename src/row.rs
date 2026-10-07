use crate::Scalar;

/// A borrowed sparse row with lazy column normalization.
///
/// The raw column indices and values, and the optional centers and scales, are
/// borrowed without copying. Each logical entry is `(raw[j] - center[j]) /
/// scale[j]`, with effective centers of zero and scales of one when inactive.
/// If storage contains duplicate column indices, sum their raw values before
/// applying normalization.
///
/// The sparse-plus-affine representation consists of a generally dense
/// background `-center[j] / scale[j]` and stored corrections `value / scale[j]`.
/// A centered logical row can therefore be dense even when its raw row is empty.
/// [`Self::implicit_value`] and [`Self::stored_corrections`] expose these parts
/// without allocating a dense row. Adding the parts can differ from direct
/// normalization because of floating-point rounding or nonfinite arithmetic.
#[derive(Clone, Copy, Debug)]
pub struct LazyRow<'a, F> {
    column_indices: &'a [usize],
    values: &'a [F],
    len: usize,
    centers: Option<&'a [F]>,
    scales: Option<&'a [F]>,
}

impl<'a, F> LazyRow<'a, F> {
    pub(crate) fn new(
        column_indices: &'a [usize],
        values: &'a [F],
        len: usize,
        centers: Option<&'a [F]>,
        scales: Option<&'a [F]>,
    ) -> Self {
        Self {
            column_indices,
            values,
            len,
            centers,
            scales,
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

    /// Borrow the full column-center slice, or `None` when centering is inactive.
    pub fn centers(&self) -> Option<&'a [F]> {
        self.centers
    }

    /// Borrow the full column-scale slice, or `None` when scaling is inactive.
    pub fn scales(&self) -> Option<&'a [F]> {
        self.scales
    }
}

impl<F: Scalar> LazyRow<'_, F> {
    /// Effective center of column `j`, or zero when centering is inactive.
    ///
    /// This takes O(1) time.
    ///
    /// # Panics
    ///
    /// Panics if `j >= self.len()`.
    pub fn center(&self, j: usize) -> F {
        assert!(j < self.len, "column index out of bounds");
        self.centers.map_or_else(F::zero, |centers| centers[j])
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
        self.scales.map_or_else(F::one, |scales| scales[j])
    }

    /// Logical value at a structurally absent column, `-center(j) / scale(j)`.
    ///
    /// This takes O(1) time.
    ///
    /// # Panics
    ///
    /// Panics if `j >= self.len()`.
    pub fn implicit_value(&self, j: usize) -> F {
        -self.center(j) / self.scale(j)
    }

    /// Stored corrections to the background as `(column, raw_value / scale)`.
    ///
    /// The iterator preserves storage order, duplicates, and explicit zeros.
    /// Consuming it takes O(stored entries) time and allocates nothing. These
    /// corrections exclude centering; see the [type documentation](Self) for
    /// the relationship to direct logical normalization.
    pub fn stored_corrections(&self) -> impl Iterator<Item = (usize, F)> + '_ {
        self.column_indices
            .iter()
            .copied()
            .zip(self.values)
            .map(|(j, &value)| (j, value / self.scale(j)))
    }
}
