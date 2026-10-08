use super::{SprsCsc, SprsCsr};
use crate::{MatrixWrite, Scalar, VectorView, WeightedGramInto, WeightedGramKernel};
use crate::{VectorViewMut, WeightedColumnSumsInto, WeightedColumnSumsKernel};
use sprs::{CsMatBase, SpIndex};
use std::ops::Deref;

impl<F, IP, IS, DS, Iptr> WeightedGramInto<F> for SprsCsr<CsMatBase<F, usize, IP, IS, DS, Iptr>>
where
    F: Scalar,
    Iptr: SpIndex,
    IP: Deref<Target = [Iptr]>,
    IS: Deref<Target = [usize]>,
    DS: Deref<Target = [F]>,
{
    fn weighted_gram_into<W, O>(&self, weights: &W, out: &mut O) -> Result<(), Self::Error>
    where
        W: VectorView<F> + ?Sized,
        O: MatrixWrite<F> + ?Sized,
    {
        self.weighted_gram_normalized_into(weights, None, None, out)
    }
}

impl<F, IP, IS, DS, Iptr> WeightedGramKernel<F> for SprsCsr<CsMatBase<F, usize, IP, IS, DS, Iptr>>
where
    F: Scalar,
    Iptr: SpIndex,
    IP: Deref<Target = [Iptr]>,
    IS: Deref<Target = [usize]>,
    DS: Deref<Target = [F]>,
{
    fn weighted_gram_normalized_into<W, O>(
        &self,
        weights: &W,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut O,
    ) -> Result<(), Self::Error>
    where
        W: VectorView<F> + ?Sized,
        O: MatrixWrite<F> + ?Sized,
    {
        crate::gram::sparse_row_gram(self, weights, centers, scales, out);
        Ok(())
    }
}

impl<F, IP, IS, DS, Iptr> WeightedGramInto<F> for SprsCsc<CsMatBase<F, usize, IP, IS, DS, Iptr>>
where
    F: Scalar,
    Iptr: SpIndex,
    IP: Deref<Target = [Iptr]>,
    IS: Deref<Target = [usize]>,
    DS: Deref<Target = [F]>,
{
    fn weighted_gram_into<W, O>(&self, weights: &W, out: &mut O) -> Result<(), Self::Error>
    where
        W: VectorView<F> + ?Sized,
        O: MatrixWrite<F> + ?Sized,
    {
        self.weighted_gram_normalized_into(weights, None, None, out)
    }
}

impl<F, IP, IS, DS, Iptr> WeightedGramKernel<F> for SprsCsc<CsMatBase<F, usize, IP, IS, DS, Iptr>>
where
    F: Scalar,
    Iptr: SpIndex,
    IP: Deref<Target = [Iptr]>,
    IS: Deref<Target = [usize]>,
    DS: Deref<Target = [F]>,
{
    fn weighted_gram_normalized_into<W, O>(
        &self,
        weights: &W,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut O,
    ) -> Result<(), Self::Error>
    where
        W: VectorView<F> + ?Sized,
        O: MatrixWrite<F> + ?Sized,
    {
        crate::gram::sparse_gram(self, weights, centers, scales, out);
        Ok(())
    }
}

impl<F, IP, IS, DS, Iptr> WeightedColumnSumsInto<F>
    for SprsCsc<CsMatBase<F, usize, IP, IS, DS, Iptr>>
where
    F: Scalar,
    Iptr: SpIndex,
    IP: Deref<Target = [Iptr]>,
    IS: Deref<Target = [usize]>,
    DS: Deref<Target = [F]>,
{
    fn weighted_column_sums_into<W, O>(&self, weights: &W, out: &mut O) -> Result<(), Self::Error>
    where
        W: VectorView<F> + ?Sized,
        O: VectorViewMut<F> + ?Sized,
    {
        self.weighted_column_sums_normalized_into(weights, None, None, out)
    }
}

impl<F, IP, IS, DS, Iptr> WeightedColumnSumsKernel<F>
    for SprsCsc<CsMatBase<F, usize, IP, IS, DS, Iptr>>
where
    F: Scalar,
    Iptr: SpIndex,
    IP: Deref<Target = [Iptr]>,
    IS: Deref<Target = [usize]>,
    DS: Deref<Target = [F]>,
{
    fn weighted_column_sums_normalized_into<W, O>(
        &self,
        weights: &W,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut O,
    ) -> Result<(), Self::Error>
    where
        W: VectorView<F> + ?Sized,
        O: VectorViewMut<F> + ?Sized,
    {
        crate::weighted_sums::sparse_sums(self, weights, centers, scales, out);
        Ok(())
    }
}
