//! Shared column reductions over borrowed CSR rows.

use crate::backends::support::{max_or_nan, min_or_nan, range_or_nan};
use crate::{Scalar, SparseRows};

fn reduce<F, M, T>(
    matrix: &M,
    init: impl Fn(usize) -> T,
    fold: impl Fn(usize, T, F) -> T,
    finish: impl Fn(usize, T, usize) -> F,
) -> Vec<F>
where
    F: Scalar,
    M: SparseRows<F> + ?Sized,
    T: Copy,
{
    let mut states: Vec<_> = (0..matrix.ncols()).map(init).collect();
    let mut counts = vec![0; matrix.ncols()];
    let mut scratch = None;
    for i in 0..matrix.nrows() {
        let (columns, values) = matrix.sparse_row(i);
        if columns.windows(2).all(|pair| pair[0] < pair[1]) {
            for (&j, &value) in columns.iter().zip(values) {
                states[j] = fold(j, states[j], value);
                counts[j] += 1;
            }
        } else {
            // Nonlinear statistics must see the sum of duplicates at each cell.
            let (sums, seen, touched) = scratch.get_or_insert_with(|| {
                (
                    vec![F::zero(); matrix.ncols()],
                    vec![false; matrix.ncols()],
                    Vec::new(),
                )
            });
            for (&j, &value) in columns.iter().zip(values) {
                if seen[j] {
                    sums[j] = sums[j] + value;
                } else {
                    sums[j] = value;
                    seen[j] = true;
                    touched.push(j);
                }
            }
            for j in touched.drain(..) {
                states[j] = fold(j, states[j], sums[j]);
                counts[j] += 1;
                seen[j] = false;
            }
        }
    }
    states
        .into_iter()
        .enumerate()
        .map(|(j, state)| finish(j, state, matrix.nrows() - counts[j]))
        .collect()
}

pub(crate) fn means<F: Scalar, M: SparseRows<F> + ?Sized>(matrix: &M) -> Vec<F> {
    let n = F::from_usize(matrix.nrows()).unwrap();
    reduce(
        matrix,
        |_| F::zero(),
        |_, sum, value| sum + value,
        |_, sum, _| sum / n,
    )
}

pub(crate) fn sds<F: Scalar, M: SparseRows<F> + ?Sized>(matrix: &M) -> Vec<F> {
    let centers = means(matrix);
    let n = F::from_usize(matrix.nrows()).unwrap();
    reduce(
        matrix,
        |_| F::zero(),
        |j, sum, value| {
            let delta = value - centers[j];
            sum + delta * delta
        },
        |j, sum, missing| {
            let implicit = if missing == 0 {
                F::zero()
            } else {
                F::from_usize(missing).unwrap() * centers[j] * centers[j]
            };
            ((sum + implicit) / n).sqrt()
        },
    )
}

pub(crate) fn mins<F: Scalar, M: SparseRows<F> + ?Sized>(matrix: &M) -> Vec<F> {
    reduce(
        matrix,
        |_| None,
        |_, minimum: Option<F>, value| Some(min_or_nan(minimum.into_iter().chain([value]))),
        |_, minimum, missing| {
            min_or_nan(
                minimum
                    .into_iter()
                    .chain((missing > 0).then_some(F::zero())),
            )
        },
    )
}

pub(crate) fn ranges<F: Scalar, M: SparseRows<F> + ?Sized>(matrix: &M) -> Vec<F> {
    reduce(
        matrix,
        |_| None,
        |_, extrema: Option<(F, F)>, value| {
            Some(match extrema {
                None => (value, value),
                Some((lo, hi)) if lo.is_nan() || hi.is_nan() || value.is_nan() => {
                    (F::nan(), F::nan())
                }
                Some((lo, hi)) => (lo.min(value), hi.max(value)),
            })
        },
        |_, extrema, missing| {
            range_or_nan(
                extrema
                    .into_iter()
                    .flat_map(|(lo, hi)| [lo, hi])
                    .chain((missing > 0).then_some(F::zero())),
            )
        },
    )
}

#[derive(Clone, Copy)]
pub(crate) enum Norm {
    L1,
    L2,
    MaxAbs,
}

pub(crate) fn norms<F: Scalar, M: SparseRows<F> + ?Sized>(
    matrix: &M,
    norm: Norm,
    centers: Option<&[F]>,
) -> Vec<F> {
    if let Some(centers) = centers {
        assert_eq!(centers.len(), matrix.ncols(), "center length mismatch");
    }
    let center = |j: usize| centers.map_or_else(F::zero, |centers| centers[j]);
    let accumulate = |sum: F, value: F| match norm {
        Norm::L1 => sum + value.abs(),
        Norm::L2 => sum + value * value,
        Norm::MaxAbs => max_or_nan([sum, value.abs()].into_iter()),
    };
    reduce(
        matrix,
        |_| F::zero(),
        |j, sum, value| {
            let value = if centers.is_some() {
                value - center(j)
            } else {
                value
            };
            accumulate(sum, value)
        },
        |j, sum, missing| {
            let sum = if centers.is_some() && missing > 0 {
                let c = center(j);
                let count = F::from_usize(missing).unwrap();
                match norm {
                    Norm::L1 => sum + count * c.abs(),
                    Norm::L2 => sum + count * c * c,
                    Norm::MaxAbs => accumulate(sum, c),
                }
            } else {
                sum
            };
            match norm {
                Norm::L2 => sum.sqrt(),
                _ => sum,
            }
        },
    )
}

macro_rules! impl_column_stats {
    ($matrix:ty) => {
        impl<F: crate::Scalar> crate::ColumnStats<F> for $matrix {
            fn col_means(&self) -> Result<Vec<F>, Self::Error> {
                Ok(crate::csr_stats::means(self))
            }
            fn col_sds(&self) -> Result<Vec<F>, Self::Error> {
                Ok(crate::csr_stats::sds(self))
            }
            fn col_mins(&self) -> Result<Vec<F>, Self::Error> {
                Ok(crate::csr_stats::mins(self))
            }
            fn col_ranges(&self) -> Result<Vec<F>, Self::Error> {
                Ok(crate::csr_stats::ranges(self))
            }
            fn col_l1(&self) -> Result<Vec<F>, Self::Error> {
                Ok(crate::csr_stats::norms(
                    self,
                    crate::csr_stats::Norm::L1,
                    None,
                ))
            }
            fn col_l2(&self) -> Result<Vec<F>, Self::Error> {
                Ok(crate::csr_stats::norms(
                    self,
                    crate::csr_stats::Norm::L2,
                    None,
                ))
            }
            fn col_maxabs(&self) -> Result<Vec<F>, Self::Error> {
                Ok(crate::csr_stats::norms(
                    self,
                    crate::csr_stats::Norm::MaxAbs,
                    None,
                ))
            }
            fn col_l1_centered(&self, centers: &[F]) -> Result<Vec<F>, Self::Error> {
                Ok(crate::csr_stats::norms(
                    self,
                    crate::csr_stats::Norm::L1,
                    Some(centers),
                ))
            }
            fn col_l2_centered(&self, centers: &[F]) -> Result<Vec<F>, Self::Error> {
                Ok(crate::csr_stats::norms(
                    self,
                    crate::csr_stats::Norm::L2,
                    Some(centers),
                ))
            }
            fn col_maxabs_centered(&self, centers: &[F]) -> Result<Vec<F>, Self::Error> {
                Ok(crate::csr_stats::norms(
                    self,
                    crate::csr_stats::Norm::MaxAbs,
                    Some(centers),
                ))
            }
        }
    };
}
pub(crate) use impl_column_stats;
