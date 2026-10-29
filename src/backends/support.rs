use crate::traits::Scalar;

/// `Send` when parallelism is enabled, and unconstrained otherwise.
#[cfg(all(
    feature = "parallel",
    any(
        feature = "faer_all",
        feature = "nalgebra_all",
        feature = "ndarray_all",
        feature = "sprs_all"
    )
))]
pub(crate) trait MaybeSend: Send {}

#[cfg(all(
    feature = "parallel",
    any(
        feature = "faer_all",
        feature = "nalgebra_all",
        feature = "ndarray_all",
        feature = "sprs_all"
    )
))]
impl<T: Send + ?Sized> MaybeSend for T {}

#[cfg(all(
    not(feature = "parallel"),
    any(
        feature = "faer_all",
        feature = "nalgebra_all",
        feature = "ndarray_all",
        feature = "sprs_all"
    )
))]
pub(crate) trait MaybeSend {}

#[cfg(all(
    not(feature = "parallel"),
    any(
        feature = "faer_all",
        feature = "nalgebra_all",
        feature = "ndarray_all",
        feature = "sprs_all"
    )
))]
impl<T: ?Sized> MaybeSend for T {}

/// `Sync` when parallelism is enabled, and unconstrained otherwise.
#[cfg(all(
    feature = "parallel",
    any(
        feature = "faer_all",
        feature = "nalgebra_all",
        feature = "ndarray_all",
        feature = "sprs_all"
    )
))]
pub(crate) trait MaybeSync: Sync {}

#[cfg(all(
    feature = "parallel",
    any(
        feature = "faer_all",
        feature = "nalgebra_all",
        feature = "ndarray_all",
        feature = "sprs_all"
    )
))]
impl<T: Sync + ?Sized> MaybeSync for T {}

#[cfg(all(
    not(feature = "parallel"),
    any(
        feature = "faer_all",
        feature = "nalgebra_all",
        feature = "ndarray_all",
        feature = "sprs_all"
    )
))]
pub(crate) trait MaybeSync {}

#[cfg(all(
    not(feature = "parallel"),
    any(
        feature = "faer_all",
        feature = "nalgebra_all",
        feature = "ndarray_all",
        feature = "sprs_all"
    )
))]
impl<T: ?Sized> MaybeSync for T {}

#[cfg(all(
    feature = "parallel",
    any(
        feature = "faer_all",
        feature = "nalgebra_all",
        feature = "ndarray_all",
        feature = "sprs_all"
    )
))]
pub(crate) fn collect_columns<T, Map>(ncols: usize, map: Map) -> Vec<T>
where
    T: MaybeSend,
    Map: Fn(usize) -> T + MaybeSend + MaybeSync,
{
    use rayon::prelude::*;
    (0..ncols).into_par_iter().map(map).collect()
}

#[cfg(all(
    not(feature = "parallel"),
    any(
        feature = "faer_all",
        feature = "nalgebra_all",
        feature = "ndarray_all",
        feature = "sprs_all"
    )
))]
pub(crate) fn collect_columns<T, Map>(ncols: usize, map: Map) -> Vec<T>
where
    T: MaybeSend,
    Map: Fn(usize) -> T + MaybeSend + MaybeSync,
{
    (0..ncols).map(map).collect()
}

impl<F> Scalar for F where
    F: num_traits::Float + num_traits::FromPrimitive + std::iter::Sum + std::fmt::Debug
{
}

/// Borrows canonical CSC values or sums duplicate cells in one working column.
#[cfg(feature = "faer_all")]
pub(crate) fn coalesced_column_values<'a, F: Scalar>(
    rows: &[usize],
    values: &'a [F],
) -> std::borrow::Cow<'a, [F]> {
    if rows.windows(2).all(|pair| pair[0] < pair[1]) {
        return std::borrow::Cow::Borrowed(values);
    }
    let mut entries: Vec<_> = rows.iter().copied().zip(values.iter().copied()).collect();
    // Stable sorting retains the storage order when summing a duplicate cell.
    entries.sort_by_key(|&(row, _)| row);
    let mut length = 0;
    for i in 0..entries.len() {
        let (row, value) = entries[i];
        if length != 0 && entries[length - 1].0 == row {
            entries[length - 1].1 = entries[length - 1].1 + value;
        } else {
            entries[length] = (row, value);
            length += 1;
        }
    }
    std::borrow::Cow::Owned(entries[..length].iter().map(|&(_, value)| value).collect())
}

/// Computes a standard deviation from stored values and implicit zeros.
#[cfg(any(feature = "faer_all", feature = "nalgebra_all"))]
pub(crate) fn sparse_column_sd<F: Scalar>(values: &[F], nrows: usize, df: usize) -> F {
    let n = F::from_usize(nrows).unwrap();
    let mean = values.iter().copied().sum::<F>() / n;
    let stored_squared_deviations = values
        .iter()
        .map(|&value| {
            let deviation = value - mean;
            deviation * deviation
        })
        .sum::<F>();
    let implicit_count = F::from_usize(nrows - values.len()).unwrap();
    let variance = (stored_squared_deviations + implicit_count * mean * mean)
        / crate::normalization::sd_divisor::<F>(nrows, df);
    variance.sqrt()
}

/// Returns the maximum of nonnegative values without masking `NaN` entries.
#[cfg(any(
    feature = "faer_all",
    feature = "nalgebra_all",
    feature = "ndarray_all",
    feature = "sprs_all"
))]
pub(crate) fn max_or_nan<F: Scalar>(values: impl Iterator<Item = F>) -> F {
    values.fold(F::zero(), |maximum, value| {
        if maximum.is_nan() || value.is_nan() {
            F::nan()
        } else if value > maximum {
            value
        } else {
            maximum
        }
    })
}

/// Returns the minimum value, propagating `NaN` and treating an empty input as
/// undefined.
#[cfg(any(
    feature = "faer_all",
    feature = "nalgebra_all",
    feature = "ndarray_all",
    feature = "sprs_all"
))]
pub(crate) fn min_or_nan<F: Scalar>(values: impl Iterator<Item = F>) -> F {
    values
        .fold(None, |minimum: Option<F>, value| {
            Some(match minimum {
                None => value,
                Some(minimum) if minimum.is_nan() || value.is_nan() => F::nan(),
                Some(minimum) => minimum.min(value),
            })
        })
        .unwrap_or_else(F::nan)
}

/// Returns `max - min`, propagating `NaN` and treating an empty input as
/// undefined.
#[cfg(any(
    feature = "faer_all",
    feature = "nalgebra_all",
    feature = "ndarray_all",
    feature = "sprs_all"
))]
pub(crate) fn range_or_nan<F: Scalar>(values: impl Iterator<Item = F>) -> F {
    values
        .fold(None, |extrema: Option<(F, F)>, value| {
            Some(match extrema {
                None => (value, value),
                Some((minimum, maximum))
                    if minimum.is_nan() || maximum.is_nan() || value.is_nan() =>
                {
                    (F::nan(), F::nan())
                }
                Some((minimum, maximum)) => (minimum.min(value), maximum.max(value)),
            })
        })
        .map_or_else(F::nan, |(minimum, maximum)| maximum - minimum)
}
