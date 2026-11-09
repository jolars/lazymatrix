//! Dispatch explicit conversion by fitted metadata and storage capability.

use num_traits::Zero;

use crate::normalization::pipeline::{Combine, Stage};
use crate::{
    ColumnNormalizationParams, DenseNormalize, FittedRowNormalization, MaterializeDense,
    MatrixErrorType, MatrixRead, MatrixTraversal, MatrixWrite, NormalizationChainParams,
    NormalizationParams, RowNormalizationParams, Scalar, TraversalLayout,
};

pub trait CopyStage: Stage {
    type OwnedStage: Stage<Scalar = Self::Scalar, Kind = Self::Kind>;
    fn copy_stage(&self) -> Self::OwnedStage;
}

impl<F: Scalar, S: AsRef<[F]>> CopyStage for ColumnNormalizationParams<F, S> {
    type OwnedStage = ColumnNormalizationParams<F>;
    fn copy_stage(&self) -> Self::OwnedStage {
        self.copy_owned()
    }
}

impl<F: Scalar, S: AsRef<[F]>> CopyStage for FittedRowNormalization<RowNormalizationParams<F, S>> {
    type OwnedStage = FittedRowNormalization<RowNormalizationParams<F>>;
    fn copy_stage(&self) -> Self::OwnedStage {
        FittedRowNormalization::from_parts(self.recipe(), self.parameters().copy_owned())
    }
}

impl<A, B> CopyStage for NormalizationChainParams<A, B>
where
    A: CopyStage,
    B: CopyStage<Scalar = A::Scalar>,
    A::Kind: Combine<B::Kind>,
{
    type OwnedStage = NormalizationChainParams<A::OwnedStage, B::OwnedStage>;
    fn copy_stage(&self) -> Self::OwnedStage {
        NormalizationChainParams::from_parts(self.first().copy_stage(), self.next().copy_stage())
    }
}

pub trait Materialize<M: MatrixErrorType>: NormalizationParams {
    type Owned: NormalizationParams<Scalar = Self::Scalar>;
    fn copy_params(&self) -> Self::Owned;
    fn materialize<O: MatrixWrite<Self::Scalar> + ?Sized>(
        &self,
        data: &M,
        out: &mut O,
    ) -> Result<(), M::Error>;
}

impl<M: MaterializeDense<F>, F: Scalar, S: AsRef<[F]>> Materialize<M>
    for ColumnNormalizationParams<F, S>
{
    type Owned = ColumnNormalizationParams<F>;
    fn copy_params(&self) -> Self::Owned {
        self.copy_owned()
    }
    fn materialize<O: MatrixWrite<F> + ?Sized>(
        &self,
        data: &M,
        out: &mut O,
    ) -> Result<(), M::Error> {
        data.materialize_normalized_into(self.centers(), self.scales(), out)
    }
}

fn traverse<M, P, O>(data: &M, params: &P, out: &mut O) -> Result<(), M::Error>
where
    P: NormalizationParams,
    M: MatrixTraversal<P::Scalar>,
    O: MatrixWrite<P::Scalar> + ?Sized,
{
    if data.nrows() == 0 || data.ncols() == 0 {
        return Ok(());
    }
    if data.traversal_layout() != TraversalLayout::Full {
        // Implicit zeros undergo every stage. Replacing stored cells afterward
        // avoids subtracting backgrounds, which can lose signed zeros or NaNs.
        for j in 0..data.ncols() {
            for i in 0..data.nrows() {
                out.set(i, j, params.logical_entry(P::Scalar::zero(), i, j));
            }
        }
    }
    data.for_each_entry(0..data.nrows(), 0..data.ncols(), |i, j, value| {
        out.set(i, j, params.logical_entry(value, i, j));
    })
}

impl<M: MatrixTraversal<F>, F: Scalar, S: AsRef<[F]>> Materialize<M>
    for RowNormalizationParams<F, S>
{
    type Owned = RowNormalizationParams<F>;
    fn copy_params(&self) -> Self::Owned {
        self.copy_owned()
    }
    fn materialize<O: MatrixWrite<F> + ?Sized>(
        &self,
        data: &M,
        out: &mut O,
    ) -> Result<(), M::Error> {
        traverse(data, self, out)
    }
}

impl<M, A, B> Materialize<M> for NormalizationChainParams<A, B>
where
    Self: CopyStage,
    M: MatrixTraversal<Self::Scalar>,
{
    type Owned = <Self as CopyStage>::OwnedStage;
    fn copy_params(&self) -> Self::Owned {
        self.copy_stage()
    }
    fn materialize<O: MatrixWrite<Self::Scalar> + ?Sized>(
        &self,
        data: &M,
        out: &mut O,
    ) -> Result<(), M::Error> {
        traverse(data, self, out)
    }
}

pub trait InPlace<M>: NormalizationParams {
    fn normalize(&self, data: &mut M);
}

impl<M: DenseNormalize<F>, F: Scalar, S: AsRef<[F]>> InPlace<M>
    for ColumnNormalizationParams<F, S>
{
    fn normalize(&self, data: &mut M) {
        data.normalize_in_place(self.centers(), self.scales());
    }
}

fn normalize<M, P>(data: &mut M, params: &P)
where
    P: NormalizationParams,
    M: MatrixRead<P::Scalar> + DenseNormalize<P::Scalar>,
{
    for j in 0..data.ncols() {
        for i in 0..data.nrows() {
            data.set(i, j, params.logical_entry(data.get(i, j), i, j));
        }
    }
}

impl<M, F: Scalar, S: AsRef<[F]>> InPlace<M> for RowNormalizationParams<F, S>
where
    M: MatrixRead<F> + DenseNormalize<F>,
{
    fn normalize(&self, data: &mut M) {
        normalize(data, self);
    }
}

impl<M, A, B> InPlace<M> for NormalizationChainParams<A, B>
where
    Self: NormalizationParams,
    M: MatrixRead<Self::Scalar> + DenseNormalize<Self::Scalar>,
{
    fn normalize(&self, data: &mut M) {
        normalize(data, self);
    }
}
