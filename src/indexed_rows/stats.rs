use super::IndexedRows;
use crate::{
    Centering, ColumnNormalization, ColumnNormalizationStats, ColumnStats, IndexedRowSource,
    MatrixShape, QuantileRange, Scalar, Scaling,
};

#[derive(Clone, Copy)]
struct Summary<F> {
    sum: F,
    l1: F,
    squared: F,
    maxabs: F,
    min: F,
    max: F,
    count: usize,
}

impl<F: Scalar> Summary<F> {
    fn new() -> Self {
        Self {
            sum: F::zero(),
            l1: F::zero(),
            squared: F::zero(),
            maxabs: F::zero(),
            min: F::infinity(),
            max: F::neg_infinity(),
            count: 0,
        }
    }
    fn add(&mut self, value: F, count: usize) {
        if count == 0 {
            return;
        }
        let weight = F::from_usize(count).unwrap();
        self.sum = self.sum + weight * value;
        self.l1 = self.l1 + weight * value.abs();
        self.squared = self.squared + (weight * value) * value;
        if value.is_nan() {
            self.min = F::nan();
            self.max = F::nan();
            self.maxabs = F::nan();
        } else {
            self.min = if self.min.is_nan() {
                self.min
            } else {
                self.min.min(value)
            };
            self.max = if self.max.is_nan() {
                self.max
            } else {
                self.max.max(value)
            };
            self.maxabs = if self.maxabs.is_nan() {
                self.maxabs
            } else {
                self.maxabs.max(value.abs())
            };
        }
        self.count += count;
    }
}

impl<M: IndexedRowSource<F>, F: Scalar> IndexedRows<'_, M, F> {
    fn summaries(&self, centers: Option<&[F]>) -> Result<Vec<Summary<F>>, M::Error> {
        if let Some(c) = centers {
            assert_eq!(c.len(), self.ncols(), "center length mismatch");
        }
        let mut states = vec![Summary::new(); self.ncols()];
        let centered = |value: F, j| centers.map_or(value, |c| value - c[j]);
        if self.nrows() != 0 && self.ncols() != 0 {
            self.source
                .for_each_indexed_entry(self.rows, |_, j, value| {
                    states[j].add(centered(value, j), 1);
                })?;
        }
        for (j, state) in states.iter_mut().enumerate() {
            state.add(centered(F::zero(), j), self.nrows() - state.count);
        }
        Ok(states)
    }

    fn quantiles(
        &self,
        request: crate::quantiles::Request,
    ) -> Result<ColumnNormalizationStats<F>, M::Error> {
        let mut counts = vec![0; self.ncols()];
        crate::quantiles::stream(self.nrows(), self.ncols(), request, |visit| {
            counts.fill(0);
            self.source
                .for_each_indexed_entry(self.rows, |_, j, value| {
                    counts[j] += 1;
                    visit(j, value, 1);
                })?;
            for (j, &count) in counts.iter().enumerate() {
                visit(j, F::zero(), self.nrows() - count);
            }
            Ok(())
        })
    }

    fn sds_with_means(&self, df: usize, means: &[F]) -> Result<Vec<F>, M::Error> {
        let divisor = crate::normalization::sd_divisor::<F>(self.nrows(), df);
        Ok(self
            .summaries(Some(means))?
            .into_iter()
            .map(|s| (s.squared / divisor).sqrt())
            .collect())
    }
}

impl<M: IndexedRowSource<F>, F: Scalar> ColumnStats<F> for IndexedRows<'_, M, F> {
    fn normalization_stats(
        &self,
        spec: ColumnNormalization,
    ) -> Result<ColumnNormalizationStats<F>, Self::Error> {
        if spec.center == Centering::Mean && matches!(spec.scale, Scaling::Sd { .. }) {
            let centers = self.col_means()?;
            let Scaling::Sd { df } = spec.scale else {
                unreachable!()
            };
            let scales = self.sds_with_means(df, &centers)?;
            return Ok((Some(centers), Some(scales)));
        }
        crate::quantiles::fit(self, spec, |request| self.quantiles(request))
    }
    fn col_medians(&self) -> Result<Vec<F>, Self::Error> {
        Ok(self
            .quantiles(crate::quantiles::Request::median())?
            .0
            .unwrap())
    }
    fn col_quantile_ranges(&self, range: QuantileRange) -> Result<Vec<F>, Self::Error> {
        Ok(self
            .quantiles(crate::quantiles::Request::range(range))?
            .1
            .unwrap())
    }
    fn col_means(&self) -> Result<Vec<F>, Self::Error> {
        let n = F::from_usize(self.nrows()).unwrap();
        Ok(self
            .summaries(None)?
            .into_iter()
            .map(|s| s.sum / n)
            .collect())
    }
    fn col_sds(&self, df: usize) -> Result<Vec<F>, Self::Error> {
        self.sds_with_means(df, &self.col_means()?)
    }
    fn col_mins(&self) -> Result<Vec<F>, Self::Error> {
        Ok(self
            .summaries(None)?
            .into_iter()
            .map(|s| if s.count == 0 { F::nan() } else { s.min })
            .collect())
    }
    fn col_ranges(&self) -> Result<Vec<F>, Self::Error> {
        Ok(self
            .summaries(None)?
            .into_iter()
            .map(|s| {
                if s.count == 0 {
                    F::nan()
                } else {
                    s.max - s.min
                }
            })
            .collect())
    }
    fn col_maxabs(&self) -> Result<Vec<F>, Self::Error> {
        Ok(self
            .summaries(None)?
            .into_iter()
            .map(|s| s.maxabs)
            .collect())
    }
    fn col_l1(&self) -> Result<Vec<F>, Self::Error> {
        Ok(self.summaries(None)?.into_iter().map(|s| s.l1).collect())
    }
    fn col_l2(&self) -> Result<Vec<F>, Self::Error> {
        Ok(self
            .summaries(None)?
            .into_iter()
            .map(|s| s.squared.sqrt())
            .collect())
    }
    fn col_l1_centered(&self, centers: &[F]) -> Result<Vec<F>, Self::Error> {
        Ok(self
            .summaries(Some(centers))?
            .into_iter()
            .map(|s| s.l1)
            .collect())
    }
    fn col_l2_centered(&self, centers: &[F]) -> Result<Vec<F>, Self::Error> {
        Ok(self
            .summaries(Some(centers))?
            .into_iter()
            .map(|s| s.squared.sqrt())
            .collect())
    }
    fn col_maxabs_centered(&self, centers: &[F]) -> Result<Vec<F>, Self::Error> {
        Ok(self
            .summaries(Some(centers))?
            .into_iter()
            .map(|s| s.maxabs)
            .collect())
    }
}
