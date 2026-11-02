//! Dense logical views and stage-by-stage fitting.

use super::{
    ColumnNormalization, Execute, FittedRowNormalization, NormalizationChain,
    NormalizationChainParams, Prefix, RecipeType, RowNormalization, Rows, Stage, source,
};
use crate::{
    ColumnStats, MatrixErrorType, MatrixRead, MatrixShape, Scalar, VectorView, VectorViewMut,
};

pub(super) struct Logical<'a, M, P> {
    pub data: &'a M,
    pub prefix: &'a P,
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
impl<M, P> MatrixRead<P::Scalar> for Logical<'_, M, P>
where
    P: Stage,
    M: MatrixRead<P::Scalar>,
{
    fn get(&self, row: usize, column: usize) -> P::Scalar {
        self.prefix.entry(self.data.get(row, column), row, column)
    }
}

// The source adapter lets dense and streaming fitting share the numerical
// policies without requiring a public storage capability on logical views.
impl<M, P> source::Source<P::Scalar> for Logical<'_, M, P>
where
    P: Stage,
    M: MatrixRead<P::Scalar> + MatrixErrorType,
{
    fn visit(&self, visit: impl FnMut(usize, usize, P::Scalar)) -> Result<(), Self::Error> {
        source::dense(self, visit);
        Ok(())
    }
}

macro_rules! stats {
    ($($method:ident $(($arg:ident: $argtype:ty))?),* $(,)?) => {
        impl<M, P> ColumnStats<P::Scalar> for Logical<'_, M, P>
        where P: Stage, M: MatrixRead<P::Scalar> + MatrixErrorType {
            $(
                fn $method(&self, $($arg: $argtype)?) -> Result<Vec<P::Scalar>, Self::Error> {
                    super::stats::Logical::new(self, &super::Identity(std::marker::PhantomData))
                        .$method($($arg)?)
                }
            )*
        }
    };
}
stats!(
    col_medians,
    col_quantile_ranges(range: crate::QuantileRange),
    col_means,
    col_sds(df: usize),
    col_mins,
    col_ranges,
    col_maxabs,
    col_l1,
    col_l2,
    col_l2_centered(centers: &[P::Scalar]),
    col_l1_centered(centers: &[P::Scalar]),
    col_maxabs_centered(centers: &[P::Scalar]),
);

pub trait FitDense<M: MatrixErrorType, F: Scalar>: RecipeType<F> {
    fn fit_dense<P: Stage<Scalar = F>>(
        self,
        data: &M,
        prefix: &P,
    ) -> Result<Self::Params, M::Error>;
}
impl<M: MatrixRead<F> + MatrixErrorType, F: Scalar> FitDense<M, F> for ColumnNormalization {
    fn fit_dense<P: Stage<Scalar = F>>(
        self,
        data: &M,
        prefix: &P,
    ) -> Result<Self::Params, M::Error> {
        self.fit(&Logical { data, prefix })
    }
}
impl<M: MatrixRead<F> + MatrixErrorType, F: Scalar> FitDense<M, F> for RowNormalization {
    fn fit_dense<P: Stage<Scalar = F>>(
        self,
        data: &M,
        prefix: &P,
    ) -> Result<Self::Params, M::Error> {
        Ok(FittedRowNormalization::from_parts(
            self,
            self.fit(&Logical { data, prefix })?,
        ))
    }
}
impl<M, F, A, B> FitDense<M, F> for NormalizationChain<A, B>
where
    M: MatrixRead<F> + MatrixErrorType,
    F: Scalar,
    A: FitDense<M, F>,
    B: FitDense<M, F>,
    <A::Params as Stage>::Kind: super::Combine<<B::Params as Stage>::Kind>,
{
    fn fit_dense<P: Stage<Scalar = F>>(
        self,
        data: &M,
        prefix: &P,
    ) -> Result<Self::Params, M::Error> {
        let first = self.first.fit_dense(data, prefix)?;
        let next = self.next.fit_dense(data, &Prefix(prefix, &first))?;
        Ok(NormalizationChainParams { first, next })
    }
}

impl<M, P> Execute<M, P> for Rows
where
    P: Stage,
    M: MatrixRead<P::Scalar> + MatrixErrorType,
{
    fn product<X: VectorView<P::Scalar>, Y: VectorViewMut<P::Scalar>>(
        data: &M,
        params: &P,
        transpose: bool,
        x: &X,
        out: &mut Y,
    ) -> Result<(), M::Error> {
        let (input, output) = if transpose {
            (data.nrows(), data.ncols())
        } else {
            (data.ncols(), data.nrows())
        };
        assert_eq!(x.len(), input, "pipeline input dimension mismatch");
        assert_eq!(out.len(), output, "pipeline output dimension mismatch");
        for o in 0..output {
            let value = (0..input)
                .map(|k| {
                    let (i, j) = if transpose { (k, o) } else { (o, k) };
                    params.entry(data.get(i, j), i, j) * x.get(k)
                })
                .sum();
            out.set(o, value);
        }
        Ok(())
    }
}
