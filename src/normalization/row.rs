//! Row fitting with bounded workspace.

use crate::{Centering, MatrixRead, RowNormalization, RowNormalizationStats, Scalar, Scaling};

mod traversal;
pub(crate) use traversal::{Transposed, rectangle_product};
#[cfg(any(
    feature = "faer_all",
    feature = "nalgebra_all",
    feature = "sprs_all",
    feature = "zarrs_all"
))]
pub(crate) use traversal::{product as traversal_product, stats as traversal_stats};

fn row_values<M: MatrixRead<F> + ?Sized, F: Scalar>(
    matrix: &M,
    row: usize,
) -> impl Iterator<Item = F> + '_ {
    (0..matrix.ncols()).map(move |j| matrix.get(row, j))
}

fn extrema<F: Scalar>(values: impl Iterator<Item = F>) -> (F, F) {
    values
        .fold(None, |extrema: Option<(F, F)>, value| {
            Some(match extrema {
                None => (value, value),
                Some((min, _)) if min.is_nan() || value.is_nan() => (F::nan(), F::nan()),
                Some((min, max)) => (min.min(value), max.max(value)),
            })
        })
        .unwrap_or_else(|| (F::nan(), F::nan()))
}

pub(crate) fn fit<M: MatrixRead<F> + ?Sized, F: Scalar>(
    matrix: &M,
    spec: RowNormalization,
) -> RowNormalizationStats<F> {
    spec.scale.validate_quantile_range();
    let (medians, ranges) = crate::quantiles::row_quantiles(matrix, spec);
    let mean = |i| row_values(matrix, i).sum::<F>() / F::from_usize(matrix.ncols()).unwrap();
    let centers = match spec.center {
        Centering::None => None,
        Centering::Mean => Some((0..matrix.nrows()).map(mean).collect::<Vec<_>>()),
        Centering::Min => Some(
            (0..matrix.nrows())
                .map(|i| extrema(row_values(matrix, i)).0)
                .collect(),
        ),
        Centering::Median => medians,
    };
    let scales = match spec.scale {
        Scaling::None => None,
        Scaling::QuantileRange { .. } => ranges,
        scaling => Some(
            (0..matrix.nrows())
                .map(|i| {
                    let centered = || {
                        row_values(matrix, i).map(|value| {
                            centers.as_ref().map_or(value, |centers| value - centers[i])
                        })
                    };
                    match scaling {
                        Scaling::Sd { df } => {
                            let center = if spec.center == Centering::Mean {
                                centers.as_ref().unwrap()[i]
                            } else {
                                mean(i)
                            };
                            let sum = row_values(matrix, i)
                                .map(|value| {
                                    let deviation = value - center;
                                    deviation * deviation
                                })
                                .sum::<F>();
                            (sum / crate::normalization::sd_divisor::<F>(matrix.ncols(), df)).sqrt()
                        }
                        Scaling::Range => {
                            let (min, max) = extrema(row_values(matrix, i));
                            max - min
                        }
                        Scaling::MaxAbs => {
                            centered()
                                .map(|value| value.abs())
                                .fold(F::zero(), |max, value| {
                                    if max.is_nan() || value.is_nan() {
                                        F::nan()
                                    } else {
                                        max.max(value)
                                    }
                                })
                        }
                        Scaling::L1 => centered().map(|value| value.abs()).sum(),
                        Scaling::L2 => centered().map(|value| value * value).sum::<F>().sqrt(),
                        Scaling::None | Scaling::QuantileRange { .. } => unreachable!(),
                    }
                })
                .collect::<Vec<_>>(),
        ),
    };
    (
        centers,
        scales.map(|values| spec.scale.calibrate_quantile_scales(values)),
    )
}
