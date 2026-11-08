//! Weighted operations normalize entries before accumulating logical products.

use std::ops::Range;

use super::{Columns, NormalizationChainParams, Rows, Source, Stage};
use crate::normalization::private::Params;
use crate::{
    LazyMatrix, MatrixErrorType, MatrixRead, MatrixShape, MatrixWrite, RowNormalizationParams,
    Scalar, TraversalLayout, VectorView, VectorViewMut, WeightedColumnSumsInto, WeightedGramInto,
};

fn read_panel<M, F>(
    data: &M,
    entry: &impl Fn(F, usize, usize) -> F,
    rows: Range<usize>,
    columns: Range<usize>,
    panel: &mut crate::weighted_panels::Panel<F>,
) -> Result<(), M::Error>
where
    M: Source<F>,
    F: Scalar,
{
    // Missing cells must undergo every stage too. Raw traversal has already
    // coalesced duplicates, so each callback replaces a complete raw cell.
    for i in 0..rows.len() {
        for j in 0..columns.len() {
            panel.set(i, j, F::zero());
        }
    }
    data.for_each_entry(rows.clone(), columns.clone(), |i, j, value| {
        panel.set(i - rows.start, j - columns.start, value);
    })?;
    for i in 0..rows.len() {
        for j in 0..columns.len() {
            panel.set(
                i,
                j,
                entry(panel.get(i, j), rows.start + i, columns.start + j),
            );
        }
    }
    Ok(())
}

fn sums<M, F, W, O>(
    data: &M,
    entry: impl Fn(F, usize, usize) -> F,
    weights: &W,
    out: &mut O,
) -> Result<(), M::Error>
where
    M: Source<F>,
    F: Scalar,
    W: VectorView<F> + ?Sized,
    O: VectorViewMut<F> + ?Sized,
{
    crate::weighted_sums::validate(data.nrows(), data.ncols(), weights, None, None, out);
    if data.traversal_layout() == TraversalLayout::Full {
        for j in 0..out.len() {
            out.set(j, F::zero());
        }
        if data.nrows() == 0 || data.ncols() == 0 {
            return Ok(());
        }
        // Full storage, including Zarr chunks, needs only one scan for sums.
        return data.for_each_entry(0..data.nrows(), 0..data.ncols(), |i, j, value| {
            out.set(j, out.get(j) + entry(value, i, j) * weights.get(i));
        });
    }
    crate::weighted_panels::sums(
        (data.nrows(), data.ncols()),
        weights,
        out,
        |rows, columns, panel| read_panel(data, &entry, rows, columns, panel),
    )
}

fn gram<M, F, W, O>(
    data: &M,
    entry: impl Fn(F, usize, usize) -> F,
    weights: &W,
    out: &mut O,
) -> Result<(), M::Error>
where
    M: Source<F>,
    F: Scalar,
    W: VectorView<F> + ?Sized,
    O: MatrixWrite<F> + ?Sized,
{
    crate::weighted_panels::gram(
        (data.nrows(), data.ncols()),
        weights,
        out,
        |rows, columns, panel| read_panel(data, &entry, rows, columns, panel),
    )
}

pub trait ExecuteWeighted<M: MatrixErrorType, P: Stage> {
    fn sums<W: VectorView<P::Scalar> + ?Sized, O: VectorViewMut<P::Scalar> + ?Sized>(
        data: &M,
        params: &P,
        weights: &W,
        out: &mut O,
    ) -> Result<(), M::Error>;

    fn gram<W: VectorView<P::Scalar> + ?Sized, O: MatrixWrite<P::Scalar> + ?Sized>(
        data: &M,
        params: &P,
        weights: &W,
        out: &mut O,
    ) -> Result<(), M::Error>;
}

impl<M, P> ExecuteWeighted<M, P> for Columns
where
    P: Stage<Kind = Columns>,
    M: Source<P::Scalar>,
{
    fn sums<W: VectorView<P::Scalar> + ?Sized, O: VectorViewMut<P::Scalar> + ?Sized>(
        data: &M,
        params: &P,
        weights: &W,
        out: &mut O,
    ) -> Result<(), M::Error> {
        sums(data, |v, i, j| params.entry(v, i, j), weights, out)
    }

    fn gram<W: VectorView<P::Scalar> + ?Sized, O: MatrixWrite<P::Scalar> + ?Sized>(
        data: &M,
        params: &P,
        weights: &W,
        out: &mut O,
    ) -> Result<(), M::Error> {
        gram(data, |v, i, j| params.entry(v, i, j), weights, out)
    }
}

impl<M, P> ExecuteWeighted<M, P> for Rows
where
    P: Stage,
    M: MatrixRead<P::Scalar> + MatrixErrorType,
{
    fn sums<W: VectorView<P::Scalar> + ?Sized, O: VectorViewMut<P::Scalar> + ?Sized>(
        data: &M,
        params: &P,
        weights: &W,
        out: &mut O,
    ) -> Result<(), M::Error> {
        sums(
            &super::dense::Logical {
                data,
                prefix: params,
            },
            |v, _, _| v,
            weights,
            out,
        )
    }

    fn gram<W: VectorView<P::Scalar> + ?Sized, O: MatrixWrite<P::Scalar> + ?Sized>(
        data: &M,
        params: &P,
        weights: &W,
        out: &mut O,
    ) -> Result<(), M::Error> {
        gram(
            &super::dense::Logical {
                data,
                prefix: params,
            },
            |v, _, _| v,
            weights,
            out,
        )
    }
}

macro_rules! weighted {
    ($trait:ident, $method:ident, $output:ident, $kernel:ident) => {
        impl<M, A, B> $trait<<NormalizationChainParams<A, B> as crate::NormalizationParams>::Scalar>
            for LazyMatrix<M, NormalizationChainParams<A, B>>
        where
            M: MatrixShape + MatrixErrorType,
            NormalizationChainParams<A, B>: Stage,
            <NormalizationChainParams<A, B> as Stage>::Kind:
                ExecuteWeighted<M, NormalizationChainParams<A, B>>,
        {
            fn $method<W, O>(&self, weights: &W, out: &mut O) -> Result<(), Self::Error>
            where
                W: VectorView<
                        <NormalizationChainParams<A, B> as crate::NormalizationParams>::Scalar,
                    > + ?Sized,
                O: $output<<NormalizationChainParams<A, B> as crate::NormalizationParams>::Scalar>
                    + ?Sized,
            {
                <NormalizationChainParams<A, B> as Stage>::Kind::$kernel(
                    self.data(),
                    self.normalization(),
                    weights,
                    out,
                )
            }
        }

        impl<M, F, S> $trait<F> for LazyMatrix<M, RowNormalizationParams<F, S>>
        where
            M: Source<F>,
            F: Scalar,
            S: AsRef<[F]>,
        {
            fn $method<W, O>(&self, weights: &W, out: &mut O) -> Result<(), Self::Error>
            where
                W: VectorView<F> + ?Sized,
                O: $output<F> + ?Sized,
            {
                $kernel(
                    self.data(),
                    |v, i, j| self.normalization().logical_entry(v, i, j),
                    weights,
                    out,
                )
            }
        }
    };
}
weighted!(
    WeightedColumnSumsInto,
    weighted_column_sums_into,
    VectorViewMut,
    sums
);
weighted!(WeightedGramInto, weighted_gram_into, MatrixWrite, gram);
