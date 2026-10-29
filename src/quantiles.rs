//! Exact order statistics shared by buffered and streaming backends.

use crate::{
    Centering, ColumnStats, Normalization, NormalizationStats, QuantileRange, Scalar, Scaling,
};

#[cfg(any(
    test,
    feature = "faer_all",
    feature = "nalgebra_all",
    feature = "ndarray_all",
    feature = "sprs_all"
))]
mod buffered;
#[cfg(any(feature = "faer_all", feature = "nalgebra_all"))]
pub(crate) use buffered::borrowed_columns;
#[cfg(feature = "sprs_all")]
pub(crate) use buffered::columns;
#[cfg(any(
    feature = "faer_all",
    feature = "nalgebra_all",
    feature = "ndarray_all"
))]
pub(crate) use buffered::dense_columns;

mod streaming;
#[cfg(any(feature = "faer_all", feature = "nalgebra_all", feature = "sprs_all"))]
pub(crate) use streaming::sparse_rows;
pub(crate) use streaming::stream;

#[cfg(any(
    feature = "faer_all",
    feature = "nalgebra_all",
    feature = "ndarray_all"
))]
macro_rules! buffered_stats_methods {
    ($compute:path) => {
        fn normalization_stats(
            &self,
            spec: crate::Normalization,
        ) -> Result<crate::NormalizationStats<F>, Self::Error> {
            crate::quantiles::fit(self, spec, |request| Ok($compute(self, request)))
        }

        fn col_medians(&self) -> Result<Vec<F>, Self::Error> {
            Ok($compute(self, crate::quantiles::Request::median())
                .0
                .unwrap())
        }

        fn col_quantile_ranges(&self, range: crate::QuantileRange) -> Result<Vec<F>, Self::Error> {
            Ok($compute(self, crate::quantiles::Request::range(range))
                .1
                .unwrap())
        }
    };
}
#[cfg(any(
    feature = "faer_all",
    feature = "nalgebra_all",
    feature = "ndarray_all"
))]
pub(crate) use buffered_stats_methods;

#[derive(Clone, Copy)]
pub(crate) struct Request {
    median: bool,
    range: Option<QuantileRange>,
}

impl Request {
    pub(crate) fn median() -> Self {
        Self {
            median: true,
            range: None,
        }
    }

    pub(crate) fn range(range: QuantileRange) -> Self {
        Self {
            median: false,
            range: Some(range),
        }
    }

    fn from_spec(spec: Normalization) -> Self {
        Self {
            median: spec.center == Centering::Median,
            range: match spec.scale {
                Scaling::QuantileRange(range) => Some(range),
                _ => None,
            },
        }
    }

    fn positions(self, nrows: usize) -> Vec<Position> {
        let mut positions = Vec::with_capacity(3);
        if self.median {
            positions.push(Position::new(nrows, 0.5));
        }
        if let Some(range) = self.range {
            positions.push(Position::new(nrows, range.lower()));
            positions.push(Position::new(nrows, range.upper()));
        }
        positions
    }

    fn output<F>(self, ncols: usize) -> NormalizationStats<F> {
        (
            self.median.then(|| Vec::with_capacity(ncols)),
            self.range.map(|_| Vec::with_capacity(ncols)),
        )
    }

    fn push<F: Scalar>(
        self,
        output: &mut NormalizationStats<F>,
        positions: &[Position],
        values: &[F],
    ) {
        if let Some(medians) = output.0.as_mut() {
            medians.push(interpolate(values[0], values[1], positions[0].fraction));
        }
        if let Some(ranges) = output.1.as_mut() {
            let offset = usize::from(self.median);
            let i = 2 * offset;
            ranges.push(quantile_range(
                positions[offset],
                positions[offset + 1],
                [values[i], values[i + 1], values[i + 2], values[i + 3]],
            ));
        }
    }
}

pub(crate) fn fit<M, F>(
    matrix: &M,
    spec: Normalization,
    compute: impl FnOnce(Request) -> Result<NormalizationStats<F>, M::Error>,
) -> Result<NormalizationStats<F>, M::Error>
where
    M: ColumnStats<F> + ?Sized,
    F: Scalar,
{
    let request = Request::from_spec(spec);
    let quantiles = if request.median || request.range.is_some() {
        Some(compute(request)?)
    } else {
        None
    };
    crate::traits::normalization_stats_with_quantiles(matrix, spec, quantiles)
}

#[derive(Clone, Copy)]
struct Position {
    lower: usize,
    upper: usize,
    fraction: f64,
}

impl Position {
    fn new(nrows: usize, probability: f64) -> Self {
        // Integer ranks must remain exact even for f32 data or very tall sparse columns.
        let (mantissa, exponent, _) = num_traits::Float::integer_decode(probability);
        let product = u128::from(mantissa) * nrows.saturating_sub(1) as u128;
        let shift = u32::from(exponent.unsigned_abs());
        let (lower, remainder) = if shift >= 128 {
            (0, product)
        } else {
            (
                (product >> shift) as usize,
                product & ((1_u128 << shift) - 1),
            )
        };
        // Splitting the power avoids underflow for subnormal probabilities.
        let fraction =
            (remainder as f64 * 2.0_f64.powi(i32::from(exponent) + 64)) * 2.0_f64.powi(-64);
        Self {
            lower,
            upper: lower + usize::from(remainder != 0),
            fraction,
        }
    }
}

fn weighted<F: Scalar>(value: F, weight: f64) -> F {
    if weight == 0.0 {
        return F::zero();
    }
    let coefficient = F::from_f64(weight).unwrap();
    if coefficient >= F::min_positive_value() {
        return value * coefficient;
    }
    // A tiny probability can still have a representable product with a large value.
    let (mantissa, exponent, _) = num_traits::Float::integer_decode(weight);
    let bits = 64 - mantissa.leading_zeros();
    let mut result = value * F::from_f64(mantissa as f64 / 2.0_f64.powi(bits as i32)).unwrap();
    let mut power = i32::from(exponent) + bits as i32;
    while power < -64 {
        result = result * F::from_f64(2.0_f64.powi(-64)).unwrap();
        power += 64;
    }
    result * F::from_f64(2.0_f64.powi(power)).unwrap()
}

fn weighted_gap<F: Scalar>(lower: F, upper: F, weight: f64) -> F {
    if weight == 0.0 {
        return F::zero();
    }
    let gap = upper - lower;
    if gap.is_infinite() && lower.is_finite() && upper.is_finite() {
        weighted(upper, weight) - weighted(lower, weight)
    } else {
        weighted(gap, weight)
    }
}

fn interpolate<F: Scalar>(lower: F, upper: F, fraction: f64) -> F {
    if fraction == 0.0 || lower == upper {
        lower
    } else if fraction == 1.0 {
        upper
    } else if !lower.is_finite() || !upper.is_finite() || (lower < F::zero() && upper > F::zero()) {
        weighted(lower, 1.0 - fraction) + weighted(upper, fraction)
    } else if fraction <= 0.5 {
        lower + weighted_gap(lower, upper, fraction)
    } else {
        upper - weighted_gap(lower, upper, 1.0 - fraction)
    }
}

fn quantile_range<F: Scalar>(lower: Position, upper: Position, values: [F; 4]) -> F {
    let [a, b, c, d] = values;
    if values.iter().any(|value| !value.is_finite()) {
        return interpolate(c, d, upper.fraction) - interpolate(a, b, lower.fraction);
    }
    // Subtracting two interpolated large offsets can erase their entire spread.
    if lower.lower == upper.lower {
        weighted_gap(a, d, upper.fraction - lower.fraction)
    } else {
        weighted_gap(a, b, 1.0 - lower.fraction) + (c - b) + weighted_gap(c, d, upper.fraction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn buffered<F: Scalar>(values: &[F], request: Request) -> NormalizationStats<F> {
        super::buffered::columns(values.len(), 1, request, |_, entries| {
            entries.extend(values.iter().copied().enumerate());
        })
    }

    fn streamed<F: Scalar>(values: &[F], request: Request) -> NormalizationStats<F> {
        let mut scans = 0;
        super::streaming::stream(values.len(), 1, request, |visit| {
            scans += 1;
            assert!(scans <= values.len() + 1);
            for &value in values {
                visit(0, value, 1);
            }
            Ok::<_, std::convert::Infallible>(())
        })
        .unwrap()
    }

    fn oracle(values: &[f64], q: f64) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let position = (sorted.len() - 1) as f64 * q;
        let lo = position.floor() as usize;
        let hi = position.ceil() as usize;
        let fraction = position - lo as f64;
        (1.0 - fraction) * sorted[lo] + fraction * sorted[hi]
    }

    #[test]
    fn exact_selection_matches_independent_sorting() {
        let mut rng = ChaCha8Rng::seed_from_u64(127);
        for nrows in [1, 2, 3, 4, 5, 12, 31, 100] {
            for _ in 0..15 {
                let values: Vec<_> = (0..nrows)
                    .map(|_| rng.random_range(-50..=50) as f64)
                    .collect();
                for range in [
                    QuantileRange::default(),
                    QuantileRange::new(0.0, 1.0),
                    QuantileRange::new(0.0, 0.1),
                    QuantileRange::new(0.1, 0.9),
                    QuantileRange::new(0.8, 1.0),
                ] {
                    let request = Request {
                        median: true,
                        range: Some(range),
                    };
                    for output in [buffered(&values, request), streamed(&values, request)] {
                        approx::assert_abs_diff_eq!(
                            output.0.unwrap()[0],
                            oracle(&values, 0.5),
                            epsilon = 1e-12
                        );
                        approx::assert_abs_diff_eq!(
                            output.1.unwrap()[0],
                            oracle(&values, range.upper()) - oracle(&values, range.lower()),
                            epsilon = 1e-12
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[cfg(any(
        feature = "faer_all",
        feature = "nalgebra_all",
        feature = "ndarray_all"
    ))]
    fn zero_columns_need_no_row_workspace() {
        struct Empty;
        impl crate::MatrixShape for Empty {
            fn nrows(&self) -> usize {
                usize::MAX
            }
            fn ncols(&self) -> usize {
                0
            }
        }
        impl crate::MatrixRead<f64> for Empty {
            fn get(&self, _: usize, _: usize) -> f64 {
                unreachable!()
            }
        }
        let request = Request {
            median: true,
            range: Some(QuantileRange::default()),
        };
        assert_eq!(
            super::buffered::dense_columns(&Empty, request),
            (Some(vec![]), Some(vec![]))
        );
    }

    #[test]
    fn ranges_preserve_large_offsets_and_avoid_intermediate_overflow() {
        for values in [
            vec![1e16, 1e16 + 2.0, 1e16 + 4.0, 1e16 + 6.0],
            vec![-f64::MAX, f64::MAX],
        ] {
            let expected = if values.len() == 4 { 3.0 } else { f64::MAX };
            for output in [
                buffered(&values, Request::range(QuantileRange::default())),
                streamed(&values, Request::range(QuantileRange::default())),
            ] {
                assert_eq!(output.1.unwrap()[0], expected);
            }
        }
        for output in [
            buffered(&[-f64::MAX, f64::MAX], Request::median()),
            streamed(&[-f64::MAX, f64::MAX], Request::median()),
        ] {
            assert_eq!(output.0.unwrap()[0], 0.0);
        }
    }

    #[test]
    fn f32_quantiles_keep_small_spreads_and_tiny_probabilities() {
        let values = [1e8_f32, 1e8 + 8.0, 1e8 + 16.0, 1e8 + 24.0];
        let request = Request::range(QuantileRange::new(0.125, 0.875));
        for output in [buffered(&values, request), streamed(&values, request)] {
            assert_eq!(output.1.unwrap()[0], 18.0);
        }
        let request = Request::range(QuantileRange::new(0.0, 1e-50));
        for output in [
            buffered(&[0.0_f32, 1e38], request),
            streamed(&[0.0_f32, 1e38], request),
        ] {
            approx::assert_relative_eq!(output.1.unwrap()[0], 1e-12_f32, max_relative = 1e-6);
        }
    }

    #[test]
    fn empty_nonfinite_and_adjacent_values_have_defined_results() {
        let request = Request {
            median: true,
            range: Some(QuantileRange::default()),
        };
        for values in [
            vec![],
            vec![f64::NAN, 1.0],
            vec![f64::NEG_INFINITY, f64::INFINITY],
            vec![f64::INFINITY, f64::INFINITY],
            vec![f64::NEG_INFINITY, f64::NEG_INFINITY],
            vec![0.0, f64::from_bits(1)],
            vec![-f64::from_bits(1), 0.0],
            vec![f64::MAX, f64::from_bits(f64::MAX.to_bits() - 1)],
        ] {
            let expected = buffered(&values, request);
            let actual = streamed(&values, request);
            for (actual, expected) in actual
                .0
                .into_iter()
                .flatten()
                .chain(actual.1.into_iter().flatten())
                .zip(
                    expected
                        .0
                        .into_iter()
                        .flatten()
                        .chain(expected.1.into_iter().flatten()),
                )
            {
                assert!(actual == expected || actual.is_nan() && expected.is_nan());
            }
        }
    }

    #[test]
    fn sparse_selection_coalesces_duplicates_and_counts_implicit_zeros() {
        let request = Request {
            median: true,
            range: Some(QuantileRange::default()),
        };
        let output = super::buffered::columns(6, 1, request, |_, entries| {
            entries.extend([(2, -3.0), (0, 2.0), (2, 4.0), (0, -2.0), (4, 5.0), (1, 0.0)]);
        });
        assert_eq!(output.0.unwrap(), [0.0]);
        assert_eq!(output.1.unwrap(), [0.75]);
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn integer_ranks_remain_exact_above_f64_integer_precision() {
        let nrows = (1_usize << 54) + 2;
        let position = Position::new(nrows, 0.5);
        assert_eq!(position.lower, 1 << 53);
        assert_eq!(position.upper, (1 << 53) + 1);
        assert_eq!(position.fraction, 0.5);
        let output = super::streaming::stream(nrows, 1, Request::median(), |visit| {
            visit(0, -1.0_f32, nrows / 2);
            visit(0, 1.0_f32, nrows / 2);
            Ok::<_, std::convert::Infallible>(())
        })
        .unwrap();
        assert_eq!(output.0.unwrap(), [0.0]);
    }
}
