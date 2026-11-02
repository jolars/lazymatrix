//! Statistics of logical columns, including transformed sparse backgrounds.

use super::{
    ColumnNormalization, Combine, NormalizationChain, NormalizationChainParams, Prefix, RecipeType,
    Source, Stage,
};
use crate::{Centering, ColumnStats, MatrixErrorType, MatrixShape, QuantileRange, Scalar, Scaling};
use num_traits::{Float, FromPrimitive, Zero};

pub(super) struct Logical<'a, M, P> {
    data: &'a M,
    prefix: &'a P,
}
impl<'a, M, P> Logical<'a, M, P> {
    pub fn new(data: &'a M, prefix: &'a P) -> Self {
        Self { data, prefix }
    }
}
impl<M: MatrixShape, P> MatrixShape for Logical<'_, M, P> {
    fn nrows(&self) -> usize {
        self.data.nrows()
    }
    fn ncols(&self) -> usize {
        self.data.ncols()
    }
}
impl<M: MatrixErrorType, P> MatrixErrorType for Logical<'_, M, P> {
    type Error = M::Error;
}

#[derive(Clone, Copy)]
struct Summary<F> {
    sum: F,
    squares: F,
    l1: F,
    maxabs: F,
    min: F,
    max: F,
    count: usize,
}
impl<F: Scalar> Summary<F> {
    fn new() -> Self {
        Self {
            sum: F::zero(),
            squares: F::zero(),
            l1: F::zero(),
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
        self.squares = self.squares + weight * (value * value);
        self.l1 = self.l1 + weight * value.abs();
        if value.is_nan() {
            self.min = F::nan();
            self.max = F::nan();
            self.maxabs = F::nan();
        } else {
            if !self.min.is_nan() {
                self.min = self.min.min(value);
            }
            if !self.max.is_nan() {
                self.max = self.max.max(value);
            }
            if !self.maxabs.is_nan() {
                self.maxabs = self.maxabs.max(value.abs());
            }
        }
        self.count += count;
    }
}

impl<M, P> Logical<'_, M, P>
where
    P: Stage,
    M: Source<P::Scalar>,
{
    fn summaries(
        &self,
        centers: Option<&[P::Scalar]>,
    ) -> Result<Vec<Summary<P::Scalar>>, M::Error> {
        if let Some(c) = centers {
            assert_eq!(c.len(), self.ncols(), "pipeline centers length mismatch");
        }
        let mut states = vec![Summary::new(); self.ncols()];
        let centered = |value, j| centers.map_or(value, |c| value - c[j]);
        if self.nrows() != 0 && self.ncols() != 0 {
            self.data.visit(|i, j, value| {
                states[j].add(centered(self.prefix.entry(value, i, j), j), 1);
            })?;
        }
        for (j, state) in states.iter_mut().enumerate() {
            let missing = self.nrows() - state.count;
            if missing != 0 {
                state.add(
                    centered(self.prefix.entry(P::Scalar::zero(), 0, j), j),
                    missing,
                );
            }
        }
        Ok(states)
    }

    fn quantiles(
        &self,
        request: crate::quantiles::Request,
    ) -> Result<super::super::ColumnNormalizationStats<P::Scalar>, M::Error> {
        let mut counts = vec![0; self.ncols()];
        crate::quantiles::stream(self.nrows(), self.ncols(), request, |visit| {
            counts.fill(0);
            if self.nrows() != 0 && self.ncols() != 0 {
                self.data.visit(|i, j, value| {
                    counts[j] += 1;
                    visit(j, self.prefix.entry(value, i, j), 1);
                })?;
            }
            for (j, &count) in counts.iter().enumerate() {
                let missing = self.nrows() - count;
                if missing != 0 {
                    visit(j, self.prefix.entry(P::Scalar::zero(), 0, j), missing);
                }
            }
            Ok(())
        })
    }

    fn sds(&self, df: usize, means: &[P::Scalar]) -> Result<Vec<P::Scalar>, M::Error> {
        let divisor = super::super::sd_divisor::<P::Scalar>(self.nrows(), df);
        Ok(self
            .summaries(Some(means))?
            .into_iter()
            .map(|s| (s.squares / divisor).sqrt())
            .collect())
    }
}

impl<M, P> ColumnStats<P::Scalar> for Logical<'_, M, P>
where
    P: Stage,
    M: Source<P::Scalar>,
{
    fn normalization_stats(
        &self,
        spec: ColumnNormalization,
    ) -> Result<super::super::ColumnNormalizationStats<P::Scalar>, M::Error> {
        if spec.center == Centering::Mean && matches!(spec.scale, Scaling::Sd { .. }) {
            let means = self.col_means()?;
            let Scaling::Sd { df } = spec.scale else {
                unreachable!()
            };
            let scales = self.sds(df, &means)?;
            return Ok((Some(means), Some(scales)));
        }
        crate::quantiles::fit(self, spec, |request| self.quantiles(request))
    }
    fn col_medians(&self) -> Result<Vec<P::Scalar>, M::Error> {
        Ok(self
            .quantiles(crate::quantiles::Request::median())?
            .0
            .unwrap())
    }
    fn col_quantile_ranges(&self, range: QuantileRange) -> Result<Vec<P::Scalar>, M::Error> {
        Ok(self
            .quantiles(crate::quantiles::Request::range(range))?
            .1
            .unwrap())
    }
    fn col_means(&self) -> Result<Vec<P::Scalar>, M::Error> {
        let n = P::Scalar::from_usize(self.nrows()).unwrap();
        Ok(self
            .summaries(None)?
            .into_iter()
            .map(|s| s.sum / n)
            .collect())
    }
    fn col_sds(&self, df: usize) -> Result<Vec<P::Scalar>, M::Error> {
        self.sds(df, &self.col_means()?)
    }
    fn col_mins(&self) -> Result<Vec<P::Scalar>, M::Error> {
        Ok(self
            .summaries(None)?
            .into_iter()
            .map(|s| {
                if s.count == 0 {
                    P::Scalar::nan()
                } else {
                    s.min
                }
            })
            .collect())
    }
    fn col_ranges(&self) -> Result<Vec<P::Scalar>, M::Error> {
        Ok(self
            .summaries(None)?
            .into_iter()
            .map(|s| {
                if s.count == 0 {
                    P::Scalar::nan()
                } else {
                    s.max - s.min
                }
            })
            .collect())
    }
    fn col_maxabs(&self) -> Result<Vec<P::Scalar>, M::Error> {
        Ok(self
            .summaries(None)?
            .into_iter()
            .map(|s| s.maxabs)
            .collect())
    }
    fn col_l1(&self) -> Result<Vec<P::Scalar>, M::Error> {
        Ok(self.summaries(None)?.into_iter().map(|s| s.l1).collect())
    }
    fn col_l2(&self) -> Result<Vec<P::Scalar>, M::Error> {
        Ok(self
            .summaries(None)?
            .into_iter()
            .map(|s| s.squares.sqrt())
            .collect())
    }
    fn col_l2_centered(&self, centers: &[P::Scalar]) -> Result<Vec<P::Scalar>, M::Error> {
        Ok(self
            .summaries(Some(centers))?
            .into_iter()
            .map(|s| s.squares.sqrt())
            .collect())
    }
    fn col_l1_centered(&self, centers: &[P::Scalar]) -> Result<Vec<P::Scalar>, M::Error> {
        Ok(self
            .summaries(Some(centers))?
            .into_iter()
            .map(|s| s.l1)
            .collect())
    }
    fn col_maxabs_centered(&self, centers: &[P::Scalar]) -> Result<Vec<P::Scalar>, M::Error> {
        Ok(self
            .summaries(Some(centers))?
            .into_iter()
            .map(|s| s.maxabs)
            .collect())
    }
}

pub trait FitColumns<M: MatrixErrorType, F: Scalar>: RecipeType<F> {
    fn fit_columns<P: Stage<Scalar = F>>(
        self,
        data: &M,
        prefix: &P,
    ) -> Result<Self::Params, M::Error>;
}
impl<M: Source<F>, F: Scalar> FitColumns<M, F> for ColumnNormalization {
    fn fit_columns<P: Stage<Scalar = F>>(
        self,
        data: &M,
        prefix: &P,
    ) -> Result<Self::Params, M::Error> {
        if prefix.is_identity() {
            self.fit(data)
        } else {
            self.fit(&Logical::new(data, prefix))
        }
    }
}
impl<M, F, A, B> FitColumns<M, F> for NormalizationChain<A, B>
where
    M: Source<F>,
    F: Scalar,
    A: FitColumns<M, F>,
    B: FitColumns<M, F>,
    <A::Params as Stage>::Kind: Combine<<B::Params as Stage>::Kind>,
{
    fn fit_columns<P: Stage<Scalar = F>>(
        self,
        data: &M,
        prefix: &P,
    ) -> Result<Self::Params, M::Error> {
        let first = self.first.fit_columns(data, prefix)?;
        let next = self.next.fit_columns(data, &Prefix(prefix, &first))?;
        Ok(NormalizationChainParams { first, next })
    }
}
