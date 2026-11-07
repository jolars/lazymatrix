//! Batched execution preserves fitted stage order and sparse missing ranges.

use super::{Columns, Combine, Identity, NormalizationChainParams, Rows, Source, Stage};
use crate::{
    ColumnNormalizationParams, LazyMatrix, MatMatInto, MatTransposeMatInto, MatrixErrorType,
    MatrixRead, MatrixShape, MatrixWrite, RowNormalizationParams, Scalar, TraversalLayout,
};
use num_traits::{One, Zero};
use std::marker::PhantomData;

pub trait ExecuteBatched<M: MatrixErrorType, P: Stage> {
    fn batched<R, O>(
        data: &M,
        params: &P,
        transpose: bool,
        rhs: &R,
        out: &mut O,
    ) -> Result<(), M::Error>
    where
        R: MatrixRead<P::Scalar> + ?Sized,
        O: MatrixRead<P::Scalar> + MatrixWrite<P::Scalar> + ?Sized;
}

impl<M, P> ExecuteBatched<M, P> for Columns
where
    P: Stage<Kind = Columns>,
    M: Source<P::Scalar>,
{
    fn batched<R, O>(
        data: &M,
        params: &P,
        transpose: bool,
        rhs: &R,
        out: &mut O,
    ) -> Result<(), M::Error>
    where
        R: MatrixRead<P::Scalar> + ?Sized,
        O: MatrixRead<P::Scalar> + MatrixWrite<P::Scalar> + ?Sized,
    {
        product(data, params, transpose, rhs, out)
    }
}

impl<M, P> ExecuteBatched<M, P> for Rows
where
    P: Stage,
    M: MatrixRead<P::Scalar> + MatrixErrorType,
{
    fn batched<R, O>(
        data: &M,
        params: &P,
        transpose: bool,
        rhs: &R,
        out: &mut O,
    ) -> Result<(), M::Error>
    where
        R: MatrixRead<P::Scalar> + ?Sized,
        O: MatrixRead<P::Scalar> + MatrixWrite<P::Scalar> + ?Sized,
    {
        product(
            &super::dense::Logical {
                data,
                prefix: params,
            },
            &Identity(PhantomData),
            transpose,
            rhs,
            out,
        )
    }
}

fn product<M, P, R, O>(
    data: &M,
    params: &P,
    transpose: bool,
    rhs: &R,
    out: &mut O,
) -> Result<(), M::Error>
where
    P: Stage<Kind = Columns>,
    M: Source<P::Scalar>,
    R: MatrixRead<P::Scalar> + ?Sized,
    O: MatrixRead<P::Scalar> + MatrixWrite<P::Scalar> + ?Sized,
{
    crate::matmat::validate(data, rhs, out, transpose);
    crate::matmat::zero(out);
    if rhs.ncols() == 0 || rhs.nrows() == 0 || out.nrows() == 0 {
        return Ok(());
    }
    if data.traversal_layout() == TraversalLayout::Full {
        // Decode each chunk only once, independent of the response count.
        return data.for_each_entry(0..data.nrows(), 0..data.ncols(), |i, j, value| {
            let value = params.entry(value, i, j);
            let (to, from) = if transpose { (j, i) } else { (i, j) };
            for k in 0..rhs.ncols() {
                out.set(to, k, out.get(to, k) + value * rhs.get(from, k));
            }
        });
    }
    // Bound range-sum workspace independently of the total response count.
    const RESPONSES: usize = 16;
    let background: Vec<_> = (0..data.ncols())
        .map(|j| params.entry(P::Scalar::zero(), 0, j))
        .collect();
    for start in (0..rhs.ncols()).step_by(RESPONSES) {
        let end = rhs.ncols().min(start + RESPONSES);
        let sums: Vec<_> = (start..end)
            .map(|k| {
                super::products::RangeSums::new(
                    (0..rhs.nrows())
                        .map(|i| {
                            if transpose {
                                rhs.get(i, k)
                            } else {
                                background[i] * rhs.get(i, k)
                            }
                        })
                        .collect(),
                )
            })
            .collect();
        let gap = |to: usize, from, end, response: usize| {
            sums[response].sum(
                from,
                end,
                if transpose {
                    background[to]
                } else {
                    P::Scalar::one()
                },
            )
        };
        let outer_is_output = match data.traversal_layout() {
            TraversalLayout::SparseRows => !transpose,
            TraversalLayout::SparseColumns => transpose,
            TraversalLayout::Full => unreachable!(),
        };
        // Native outer-output scans need only one cursor. The other orientation
        // tracks missing ranges independently for every output coordinate.
        let mut last = vec![0; if outer_is_output { 1 } else { out.nrows() }];
        let mut current = 0;
        data.for_each_entry(0..data.nrows(), 0..data.ncols(), |i, j, value| {
            let (to, from) = if transpose { (j, i) } else { (i, j) };
            if outer_is_output {
                while current < to {
                    for k in start..end {
                        out.set(
                            current,
                            k,
                            out.get(current, k) + gap(current, last[0], rhs.nrows(), k - start),
                        );
                    }
                    current += 1;
                    last[0] = 0;
                }
            }
            let cursor = if outer_is_output { 0 } else { to };
            let value = params.entry(value, i, j);
            for k in start..end {
                out.set(
                    to,
                    k,
                    out.get(to, k)
                        + gap(to, last[cursor], from, k - start)
                        + value * rhs.get(from, k),
                );
            }
            last[cursor] = from + 1;
        })?;
        // Complete implicit backgrounds only after the backend succeeds.
        for to in if outer_is_output {
            current..out.nrows()
        } else {
            0..out.nrows()
        } {
            let cursor = if outer_is_output { 0 } else { to };
            for k in start..end {
                out.set(
                    to,
                    k,
                    out.get(to, k) + gap(to, last[cursor], rhs.nrows(), k - start),
                );
            }
            last[cursor] = 0;
        }
    }
    Ok(())
}

macro_rules! products {
    ($trait:ident, $method:ident, $transpose:expr) => {
        impl<M, A, B> $trait<A::Scalar> for LazyMatrix<M, NormalizationChainParams<A, B>>
        where
            M: MatrixShape + MatrixErrorType,
            A: Stage,
            B: Stage<Scalar = A::Scalar>,
            A::Kind: Combine<B::Kind>,
            <NormalizationChainParams<A, B> as Stage>::Kind:
                ExecuteBatched<M, NormalizationChainParams<A, B>>,
        {
            fn $method<R, O>(&self, rhs: &R, out: &mut O) -> Result<(), Self::Error>
            where
                R: MatrixRead<A::Scalar> + ?Sized,
                O: MatrixRead<A::Scalar> + MatrixWrite<A::Scalar> + ?Sized,
            {
                <NormalizationChainParams<A, B> as Stage>::Kind::batched(
                    self.data(),
                    self.normalization(),
                    $transpose,
                    rhs,
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
            fn $method<R, O>(&self, rhs: &R, out: &mut O) -> Result<(), Self::Error>
            where
                R: MatrixRead<F> + ?Sized,
                O: MatrixRead<F> + MatrixWrite<F> + ?Sized,
            {
                let params = ColumnNormalizationParams {
                    ncols: self.nrows(),
                    centers: self.normalization().centers(),
                    scales: self.normalization().scales(),
                    scalar: PhantomData,
                };
                product(
                    &crate::normalization::row::Transposed(self.data()),
                    &params,
                    !$transpose,
                    rhs,
                    out,
                )
            }
        }
    };
}
products!(MatMatInto, matmat_into, false);
products!(MatTransposeMatInto, mat_transpose_mat_into, true);
