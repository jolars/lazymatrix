use std::fmt;

use super::{ColumnNormalization, ColumnNormalizationParams, NormalizationRecipe};
use crate::{ColumnStats, LazyMatrix, MatrixShape, Scalar};

/// Feature bounds must be finite, strictly increasing, and have finite width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidFeatureRange;

impl fmt::Display for InvalidFeatureRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("feature bounds must be finite and strictly increasing with finite width")
    }
}

impl std::error::Error for InvalidFeatureRange {}

/// Failure while fitting a custom min-max range.
#[derive(Debug)]
#[non_exhaustive]
pub enum MinMaxFitError<E> {
    /// Statistics could not be read or computed.
    Backend(E),
    /// The fitted affine parameters overflowed or the divisor underflowed to zero.
    Unrepresentable {
        /// Zero-based column index.
        column: usize,
    },
}

impl<E: fmt::Display> fmt::Display for MinMaxFitError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Backend(error) => error.fmt(f),
            Self::Unrepresentable { column } => {
                write!(
                    f,
                    "min-max parameters are unrepresentable for column {column}"
                )
            }
        }
    }
}

impl<E: std::error::Error> std::error::Error for MinMaxFitError<E> {}

/// Fit column minima and ranges to a specified output interval.
///
/// Constant training columns map to the lower bound. Fitted parameters
/// extrapolate on new samples. The stored center can round toward the
/// training minimum when the minimum is large relative to the column spread.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MinMaxNormalization<F = f64> {
    lower: F,
    upper: F,
}

impl<F: Scalar> MinMaxNormalization<F> {
    /// Construct a recipe from finite, increasing bounds with finite width.
    ///
    /// # Errors
    /// Returns an error if the bounds or their difference are invalid.
    pub fn new(lower: F, upper: F) -> Result<Self, InvalidFeatureRange> {
        if lower.is_finite() && upper.is_finite() && lower < upper && (upper - lower).is_finite() {
            Ok(Self { lower, upper })
        } else {
            Err(InvalidFeatureRange)
        }
    }

    /// Fit reusable column parameters using the backend's min-max statistics.
    ///
    /// # Errors
    /// Returns backend failures and errors encoding finite fitted statistics.
    pub fn fit<M>(self, data: &M) -> Result<ColumnNormalizationParams<F>, MinMaxFitError<M::Error>>
    where
        M: ColumnStats<F> + MatrixShape + ?Sized,
    {
        let params = ColumnNormalization::min_max()
            .fit(data)
            .map_err(MinMaxFitError::Backend)?;
        if self.lower == F::zero() && self.upper == F::one() {
            return Ok(params);
        }

        let (ncols, centers, scales) = params.into_parts();
        let mut centers = centers.expect("min-max statistics must contain minima");
        let mut scales = scales.expect("min-max statistics must contain ranges");
        let width = self.upper - self.lower;
        for (column, (minimum, range)) in centers.iter_mut().zip(&mut scales).enumerate() {
            let scale = *range / width;
            let center = if self.lower == F::zero() {
                *minimum
            } else {
                (-self.lower).mul_add(scale, *minimum)
            };
            if (range.is_finite() && (scale == F::zero() || !scale.is_finite()))
                || (minimum.is_finite() && scale.is_finite() && !center.is_finite())
            {
                return Err(MinMaxFitError::Unrepresentable { column });
            }
            *minimum = center;
            *range = scale;
        }
        Ok(ColumnNormalizationParams::from_parts(
            ncols,
            Some(centers),
            Some(scales),
        ))
    }

    /// Fit and wrap a matrix in a lazy operator.
    ///
    /// # Errors
    /// Returns the same errors as [`Self::fit`].
    pub fn fit_apply<M>(
        self,
        data: M,
    ) -> Result<LazyMatrix<M, ColumnNormalizationParams<F>>, MinMaxFitError<M::Error>>
    where
        M: ColumnStats<F> + MatrixShape,
    {
        Ok(self.fit(&data)?.apply(data))
    }
}

impl<M, F> NormalizationRecipe<M, ColumnNormalizationParams<F>> for MinMaxNormalization<F>
where
    M: ColumnStats<F> + MatrixShape + ?Sized,
    F: Scalar,
{
    type Error = MinMaxFitError<M::Error>;

    fn fit(self, data: &M) -> Result<ColumnNormalizationParams<F>, Self::Error> {
        MinMaxNormalization::fit(self, data)
    }
}

#[cfg(all(test, feature = "ndarray_v0_17"))]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    #[test]
    fn validates_and_fits_f32() {
        assert!(MinMaxNormalization::new(f32::NAN, 1.0).is_err());
        assert!(MinMaxNormalization::new(-f32::MAX, f32::MAX).is_err());
        let data = array![[4.0_f32], [8.0]];
        let recipe = MinMaxNormalization::new(-2.0, 2.0).unwrap();
        let params = recipe.fit(&data).unwrap();
        assert_eq!(params.centers().unwrap(), [6.0]);
        assert_eq!(params.scales().unwrap(), [1.0]);
        let tiny = MinMaxNormalization::new(0.0, f32::MIN_POSITIVE).unwrap();
        assert!(matches!(
            tiny.fit(&data),
            Err(MinMaxFitError::Unrepresentable { column: 0 })
        ));

        let empty = Array2::<f32>::zeros((0, 1));
        let nan = array![[f32::NAN]];
        for data in [&empty, &nan] {
            let params = recipe.fit(data).unwrap();
            assert!(params.centers().unwrap()[0].is_nan());
            assert!(params.scales().unwrap()[0].is_nan());
        }
        let default = MinMaxNormalization::new(0.0, 1.0).unwrap();
        assert_eq!(
            default.fit(&data).unwrap(),
            ColumnNormalization::min_max().fit(&data).unwrap()
        );
    }
}
