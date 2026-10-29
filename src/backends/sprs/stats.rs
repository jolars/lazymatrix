use std::ops::Deref;

use sprs::{CsMatBase, CsMatViewI, SpIndex};

use crate::backends::support::{
    MaybeSend, MaybeSync, collect_columns, max_or_nan, min_or_nan, range_or_nan,
};
use crate::{ColumnStats, Scalar};

fn quantiles<F: Scalar, I: SpIndex, Iptr: SpIndex>(
    matrix: CsMatViewI<'_, F, I, Iptr>,
    request: crate::quantiles::Request,
) -> crate::ColumnNormalizationStats<F> {
    if matrix.is_csc() {
        crate::quantiles::columns(matrix.rows(), matrix.cols(), request, |j, entries| {
            entries.extend(
                matrix
                    .outer_view(j)
                    .unwrap()
                    .iter()
                    .map(|(i, &value)| (i, value)),
            );
        })
    } else {
        crate::quantiles::sparse_rows(matrix.rows(), matrix.cols(), request, |i, visit| {
            for (j, &value) in matrix.outer_view(i).unwrap().iter() {
                visit(j, value);
            }
        })
    }
}

// CSR accumulates all columns in one scan; CSC reduces independent borrowed
// slices and can parallelize without allocating a converted sparse matrix.
fn reduce_columns<F, I, Iptr, T>(
    matrix: CsMatViewI<'_, F, I, Iptr>,
    init: impl Fn(usize) -> T + MaybeSend + MaybeSync,
    fold: impl Fn(usize, T, F) -> T + MaybeSend + MaybeSync,
    finish: impl Fn(usize, T, usize) -> F + MaybeSend + MaybeSync,
) -> Vec<F>
where
    F: Scalar + MaybeSend + MaybeSync,
    I: SpIndex,
    Iptr: SpIndex,
    T: Copy,
{
    if matrix.is_csc() {
        collect_columns(matrix.cols(), |j| {
            let range = matrix.indptr().outer_inds_sz(j);
            let values = &matrix.data()[range];
            let state = values.iter().fold(init(j), |state, &v| fold(j, state, v));
            finish(j, state, matrix.rows() - values.len())
        })
    } else {
        let mut states: Vec<_> = (0..matrix.cols()).map(&init).collect();
        let mut counts = vec![0; matrix.cols()];
        for row in matrix.outer_iterator() {
            for (j, &value) in row.iter() {
                states[j] = fold(j, states[j], value);
                counts[j] += 1;
            }
        }
        states
            .into_iter()
            .enumerate()
            .map(|(j, state)| finish(j, state, matrix.rows() - counts[j]))
            .collect()
    }
}

impl<F, I, IP, IS, DS, Iptr> ColumnStats<F> for CsMatBase<F, I, IP, IS, DS, Iptr>
where
    F: Scalar + MaybeSend + MaybeSync,
    I: SpIndex,
    Iptr: SpIndex,
    IP: Deref<Target = [Iptr]>,
    IS: Deref<Target = [I]>,
    DS: Deref<Target = [F]>,
{
    fn normalization_stats(
        &self,
        spec: crate::ColumnNormalization,
    ) -> Result<crate::ColumnNormalizationStats<F>, Self::Error> {
        crate::quantiles::fit(self, spec, |request| Ok(quantiles(self.view(), request)))
    }

    fn col_medians(&self) -> Result<Vec<F>, Self::Error> {
        Ok(quantiles(self.view(), crate::quantiles::Request::median())
            .0
            .unwrap())
    }

    fn col_quantile_ranges(&self, range: crate::QuantileRange) -> Result<Vec<F>, Self::Error> {
        Ok(
            quantiles(self.view(), crate::quantiles::Request::range(range))
                .1
                .unwrap(),
        )
    }

    fn col_means(&self) -> Result<Vec<F>, Self::Error> {
        let n = F::from_usize(self.rows()).unwrap();
        Ok(reduce_columns(
            self.view(),
            |_| F::zero(),
            |_, sum, v| sum + v,
            |_, sum, _| sum / n,
        ))
    }

    fn col_sds(&self, df: usize) -> Result<Vec<F>, Self::Error> {
        let means = self.col_means()?;
        let n = crate::normalization::sd_divisor::<F>(self.rows(), df);
        Ok(reduce_columns(
            self.view(),
            |_| F::zero(),
            |j, sum, v| {
                let delta = v - means[j];
                sum + delta * delta
            },
            |j, sum, missing| {
                let implicit = if missing == 0 {
                    F::zero()
                } else {
                    F::from_usize(missing).unwrap() * means[j] * means[j]
                };
                ((sum + implicit) / n).sqrt()
            },
        ))
    }

    fn col_mins(&self) -> Result<Vec<F>, Self::Error> {
        Ok(reduce_columns(
            self.view(),
            |_| None,
            |_, minimum: Option<F>, v| Some(min_or_nan(minimum.into_iter().chain([v]))),
            |_, minimum, missing| {
                min_or_nan(
                    minimum
                        .into_iter()
                        .chain((missing > 0).then_some(F::zero())),
                )
            },
        ))
    }

    fn col_ranges(&self) -> Result<Vec<F>, Self::Error> {
        Ok(reduce_columns(
            self.view(),
            |_| None,
            |_, extrema: Option<(F, F)>, v| {
                Some(match extrema {
                    None => (v, v),
                    Some((lo, hi)) if lo.is_nan() || hi.is_nan() || v.is_nan() => {
                        (F::nan(), F::nan())
                    }
                    Some((lo, hi)) => (lo.min(v), hi.max(v)),
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
        ))
    }

    fn col_maxabs(&self) -> Result<Vec<F>, Self::Error> {
        Ok(reduce_columns(
            self.view(),
            |_| F::zero(),
            |_, maximum, v| max_or_nan([maximum, v.abs()].into_iter()),
            |_, maximum, _| maximum,
        ))
    }

    fn col_l1(&self) -> Result<Vec<F>, Self::Error> {
        Ok(reduce_columns(
            self.view(),
            |_| F::zero(),
            |_, sum, v| sum + v.abs(),
            |_, sum, _| sum,
        ))
    }

    fn col_l2(&self) -> Result<Vec<F>, Self::Error> {
        Ok(reduce_columns(
            self.view(),
            |_| F::zero(),
            |_, sum, v| sum + v * v,
            |_, sum, _| sum.sqrt(),
        ))
    }

    fn col_l1_centered(&self, centers: &[F]) -> Result<Vec<F>, Self::Error> {
        assert_eq!(
            centers.len(),
            self.cols(),
            "col_l1_centered: length mismatch"
        );
        Ok(reduce_columns(
            self.view(),
            |_| F::zero(),
            |j, sum, v| sum + (v - centers[j]).abs(),
            |j, sum, missing| {
                if missing == 0 {
                    sum
                } else {
                    sum + F::from_usize(missing).unwrap() * centers[j].abs()
                }
            },
        ))
    }

    fn col_l2_centered(&self, centers: &[F]) -> Result<Vec<F>, Self::Error> {
        assert_eq!(
            centers.len(),
            self.cols(),
            "col_l2_centered: length mismatch"
        );
        Ok(reduce_columns(
            self.view(),
            |_| F::zero(),
            |j, sum, v| {
                let delta = v - centers[j];
                sum + delta * delta
            },
            |j, sum, missing| {
                let implicit = if missing == 0 {
                    F::zero()
                } else {
                    F::from_usize(missing).unwrap() * centers[j] * centers[j]
                };
                (sum + implicit).sqrt()
            },
        ))
    }

    fn col_maxabs_centered(&self, centers: &[F]) -> Result<Vec<F>, Self::Error> {
        assert_eq!(
            centers.len(),
            self.cols(),
            "col_maxabs_centered: length mismatch"
        );
        Ok(reduce_columns(
            self.view(),
            |_| F::zero(),
            |j, maximum, v| max_or_nan([maximum, (v - centers[j]).abs()].into_iter()),
            |j, maximum, missing| {
                max_or_nan(
                    std::iter::once(maximum).chain((missing > 0).then_some(centers[j].abs())),
                )
            },
        ))
    }
}
