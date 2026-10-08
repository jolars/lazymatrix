/// Computed column centers and raw scales, respectively.
///
/// An inactive normalization component is `None`. Present vectors have one
/// entry per column. Raw scales may be zero; [`crate::LazyMatrix::new`] replaces
/// exact zeros with one while preserving nonfinite values.
pub type NormalizationStats<F> = (Option<Vec<F>>, Option<Vec<F>>);

/// How to center each column.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Centering {
    /// No centering.
    #[default]
    None,
    /// Subtract the column mean.
    Mean,
    /// Subtract the column minimum.
    Min,
}

/// How to scale each column.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Scaling {
    /// No scaling.
    #[default]
    None,
    /// Divide by the (population) standard deviation.
    Sd,
    /// Divide by the maximum absolute value.
    MaxAbs,
    /// Divide by the 1-norm.
    L1,
    /// Divide by the 2-norm.
    L2,
    /// Divide by the range, `max - min`.
    Range,
}

/// A full normalization specification: an independent [`Centering`] and
/// [`Scaling`] choice.
///
/// When both are active, scales are computed from the **centered** columns
/// Non-translation-invariant scales are computed from centered columns (see
/// [`ColumnStats::col_l1_centered`](crate::traits::ColumnStats::col_l1_centered),
/// [`ColumnStats::col_l2_centered`](crate::traits::ColumnStats::col_l2_centered),
/// and
/// [`ColumnStats::col_maxabs_centered`](crate::traits::ColumnStats::col_maxabs_centered).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Normalization {
    pub center: Centering,
    pub scale: Scaling,
}

impl Normalization {
    /// Build a specification from its two axes.
    pub fn new(center: Centering, scale: Scaling) -> Self {
        Self { center, scale }
    }
}

/// Fitted column normalization, reusable on matrices with the same column count.
///
/// Present scales are nonzero. Computed normalization replaces exact zero
/// scales with one before constructing these parameters. Explicit parameters
/// preserve negative scales and nonfinite values.
#[derive(Clone, Debug, PartialEq)]
pub struct NormalizationParams<F = f64> {
    ncols: usize,
    pub(crate) centers: Option<Vec<F>>,
    pub(crate) scales: Option<Vec<F>>,
}

impl<F: crate::Scalar> NormalizationParams<F> {
    /// Construct validated explicit parameters.
    ///
    /// # Panics
    /// Panics if either vector has a different length from `ncols`, or if a
    /// scale is exact positive or negative zero.
    pub fn from_parts(ncols: usize, centers: Option<Vec<F>>, scales: Option<Vec<F>>) -> Self {
        crate::gram::validate_normalization(ncols, centers.as_deref(), scales.as_deref());
        Self {
            ncols,
            centers,
            scales,
        }
    }

    /// Number of columns these parameters describe.
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Fitted column centers, if centering is active.
    pub fn centers(&self) -> Option<&[F]> {
        self.centers.as_deref()
    }

    /// Fitted column scales, if scaling is active.
    pub fn scales(&self) -> Option<&[F]> {
        self.scales.as_deref()
    }

    /// Recover the fitted centers and scales without copying.
    pub fn into_parts(self) -> NormalizationStats<F> {
        (self.centers, self.scales)
    }
}
