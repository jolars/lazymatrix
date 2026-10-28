use super::{ZarrArray, ZarrMatrix, ZarrMatrixError};
use crate::{Centering, ColumnStats, MatrixShape, Normalization, Scalar, Scaling};

#[derive(Clone, Copy)]
struct Reduction<F> {
    sum: F,
    extrema: Option<(F, F)>,
    norm: F,
}

impl<F: Scalar> Reduction<F> {
    fn new() -> Self {
        Self {
            sum: F::zero(),
            extrema: None,
            norm: F::zero(),
        }
    }

    fn extrema(&mut self, value: F) {
        self.extrema = Some(match self.extrema {
            Some((min, _)) if min.is_nan() || value.is_nan() => (F::nan(), F::nan()),
            Some((min, max)) => (min.min(value), max.max(value)),
            None => (value, value),
        });
    }

    fn min(&self) -> F {
        self.extrema.map_or_else(F::nan, |(min, _)| min)
    }
    fn range(&self) -> F {
        self.extrema.map_or_else(F::nan, |(min, max)| max - min)
    }
}

fn accumulate_norm<F: Scalar>(sum: F, value: F, scale: Scaling) -> F {
    match scale {
        Scaling::L1 => sum + value.abs(),
        Scaling::L2 | Scaling::Sd => sum + value * value,
        Scaling::MaxAbs if sum.is_nan() || value.is_nan() => F::nan(),
        Scaling::MaxAbs => sum.max(value.abs()),
        _ => sum,
    }
}

fn finish_norm<F: Scalar>(sum: F, scale: Scaling, rows: usize) -> F {
    match scale {
        Scaling::L2 => sum.sqrt(),
        Scaling::Sd => (sum / F::from_usize(rows).unwrap()).sqrt(),
        _ => sum,
    }
}

impl<A: ZarrArray<F>, F: Scalar> ZarrMatrix<A, F> {
    fn quantiles(
        &self,
        request: crate::quantiles::Request,
    ) -> Result<crate::NormalizationStats<F>, ZarrMatrixError> {
        crate::quantiles::stream(self.nrows(), self.ncols(), request, |visit| {
            self.for_each_value(|_, j, value| visit(j, value, 1))
        })
    }

    fn reductions(
        &self,
        mean: bool,
        extrema: bool,
        norm: Scaling,
    ) -> Result<Vec<Reduction<F>>, ZarrMatrixError> {
        let mut stats = vec![Reduction::new(); self.ncols()];
        self.for_each_value(|_, col, value| {
            let state = &mut stats[col];
            if mean {
                state.sum = state.sum + value;
            }
            if extrema {
                state.extrema(value);
            }
            state.norm = accumulate_norm(state.norm, value, norm);
        })?;
        Ok(stats)
    }

    fn column_norms(
        &self,
        centers: Option<&[F]>,
        scale: Scaling,
    ) -> Result<Vec<F>, ZarrMatrixError> {
        if let Some(centers) = centers {
            assert_eq!(
                centers.len(),
                self.ncols(),
                "center length must equal ncols"
            );
        }
        let mut sums = vec![F::zero(); self.ncols()];
        self.for_each_value(|_, col, value| {
            let value = centers.map_or(value, |centers| value - centers[col]);
            sums[col] = accumulate_norm(sums[col], value, scale);
        })?;
        Ok(sums
            .into_iter()
            .map(|sum| finish_norm(sum, scale, self.nrows()))
            .collect())
    }
}

impl<A: ZarrArray<F>, F: Scalar> ColumnStats<F> for ZarrMatrix<A, F> {
    fn normalization_stats(
        &self,
        spec: Normalization,
    ) -> Result<crate::NormalizationStats<F>, Self::Error> {
        if spec.center == Centering::Median || matches!(spec.scale, Scaling::QuantileRange(_)) {
            return crate::quantiles::fit(self, spec, |request| self.quantiles(request));
        }
        if spec == Normalization::default() {
            return Ok((None, None));
        }
        let mean = spec.center == Centering::Mean || spec.scale == Scaling::Sd;
        let extrema = spec.center == Centering::Min || spec.scale == Scaling::Range;
        let second_pass = spec.scale == Scaling::Sd
            || (spec.center != Centering::None
                && matches!(spec.scale, Scaling::L1 | Scaling::L2 | Scaling::MaxAbs));
        let stats = self.reductions(
            mean,
            extrema,
            if second_pass {
                Scaling::None
            } else {
                spec.scale
            },
        )?;
        let n = F::from_usize(self.nrows()).unwrap();
        let means = mean.then(|| stats.iter().map(|s| s.sum / n).collect::<Vec<_>>());
        let centers = match spec.center {
            Centering::None => None,
            Centering::Mean => means.clone(),
            Centering::Min => Some(stats.iter().map(Reduction::min).collect()),
            Centering::Median => unreachable!("quantile fitting is handled before reductions"),
        };
        let scales = match spec.scale {
            Scaling::None => None,
            // SD uses the mean even when the requested center is the minimum.
            Scaling::Sd => Some(self.column_norms(means.as_deref(), Scaling::Sd)?),
            Scaling::Range => Some(stats.iter().map(Reduction::range).collect()),
            scale if second_pass => Some(self.column_norms(centers.as_deref(), scale)?),
            scale => Some(
                stats
                    .iter()
                    .map(|s| finish_norm(s.norm, scale, self.nrows()))
                    .collect(),
            ),
        };
        Ok((centers, scales))
    }

    fn col_means(&self) -> Result<Vec<F>, Self::Error> {
        let n = F::from_usize(self.nrows()).unwrap();
        Ok(self
            .reductions(true, false, Scaling::None)?
            .iter()
            .map(|s| s.sum / n)
            .collect())
    }
    fn col_medians(&self) -> Result<Vec<F>, Self::Error> {
        Ok(self
            .quantiles(crate::quantiles::Request::median())?
            .0
            .unwrap())
    }
    fn col_quantile_ranges(&self, range: crate::QuantileRange) -> Result<Vec<F>, Self::Error> {
        Ok(self
            .quantiles(crate::quantiles::Request::range(range))?
            .1
            .unwrap())
    }
    fn col_sds(&self) -> Result<Vec<F>, Self::Error> {
        self.column_norms(Some(&self.col_means()?), Scaling::Sd)
    }
    fn col_mins(&self) -> Result<Vec<F>, Self::Error> {
        Ok(self
            .reductions(false, true, Scaling::None)?
            .iter()
            .map(Reduction::min)
            .collect())
    }
    fn col_ranges(&self) -> Result<Vec<F>, Self::Error> {
        Ok(self
            .reductions(false, true, Scaling::None)?
            .iter()
            .map(Reduction::range)
            .collect())
    }
    fn col_maxabs(&self) -> Result<Vec<F>, Self::Error> {
        self.column_norms(None, Scaling::MaxAbs)
    }
    fn col_l1(&self) -> Result<Vec<F>, Self::Error> {
        self.column_norms(None, Scaling::L1)
    }
    fn col_l2(&self) -> Result<Vec<F>, Self::Error> {
        self.column_norms(None, Scaling::L2)
    }
    fn col_l1_centered(&self, centers: &[F]) -> Result<Vec<F>, Self::Error> {
        self.column_norms(Some(centers), Scaling::L1)
    }
    fn col_l2_centered(&self, centers: &[F]) -> Result<Vec<F>, Self::Error> {
        self.column_norms(Some(centers), Scaling::L2)
    }
    fn col_maxabs_centered(&self, centers: &[F]) -> Result<Vec<F>, Self::Error> {
        self.column_norms(Some(centers), Scaling::MaxAbs)
    }
}
