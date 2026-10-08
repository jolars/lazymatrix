use shrinkage::ProximalVector;
use shrinkage::lazymatrix::{
    ColumnStats, DotSlice, ElemDivAssign, MatTransposeVecInto, MatVecInto, MatrixErrorType,
    MatrixShape, Normalization, NormalizationStats, RawColumns, ScaledSubSlice, SubScalarAssign,
    SumEntries,
};

use crate::metrics;

pub struct Vector<V>(pub V);

impl<V: Clone> Clone for Vector<V> {
    fn clone(&self) -> Self {
        Self(metrics::clone_vector(|| self.0.clone()))
    }
}

impl<V: ElemDivAssign<f64>> ElemDivAssign<f64> for Vector<V> {
    fn elem_div_assign(&mut self, coefficients: &[f64]) {
        self.0.elem_div_assign(coefficients);
    }
}

impl<V: DotSlice<f64>> DotSlice<f64> for Vector<V> {
    fn dot_slice(&self, coefficients: &[f64]) -> f64 {
        self.0.dot_slice(coefficients)
    }
}

impl<V: SubScalarAssign<f64>> SubScalarAssign<f64> for Vector<V> {
    fn sub_scalar_assign(&mut self, value: f64) {
        self.0.sub_scalar_assign(value);
    }
}

impl<V: SumEntries<f64>> SumEntries<f64> for Vector<V> {
    fn sum_entries(&self) -> f64 {
        self.0.sum_entries()
    }
}

impl<V: ScaledSubSlice<f64>> ScaledSubSlice<f64> for Vector<V> {
    fn scaled_sub_slice(&mut self, value: f64, coefficients: &[f64]) {
        self.0.scaled_sub_slice(value, coefficients);
    }
}

impl<V: ProximalVector> ProximalVector for Vector<V> {
    fn zeros(length: usize) -> Self {
        Self(V::zeros(length))
    }

    fn get(&self, index: usize) -> f64 {
        self.0.get(index)
    }

    fn set(&mut self, index: usize, value: f64) {
        self.0.set(index, value);
    }
}

pub struct Matrix<M>(pub M);

impl<M: MatrixShape> MatrixShape for Matrix<M> {
    fn nrows(&self) -> usize {
        self.0.nrows()
    }

    fn ncols(&self) -> usize {
        self.0.ncols()
    }
}

impl<M: MatrixErrorType> MatrixErrorType for Matrix<M> {
    type Error = M::Error;
}

impl<M: RawColumns<f64>> RawColumns<f64> for Matrix<M> {
    type Column<'a>
        = M::Column<'a>
    where
        Self: 'a;

    fn raw_column(&self, column: usize) -> Self::Column<'_> {
        self.0.raw_column(column)
    }
}

macro_rules! forward_statistics {
    ($($name:ident($($argument:ident: $ty:ty),*)),* $(,)?) => {
        $(
            fn $name(&self, $($argument: $ty),*) -> Result<Vec<f64>, Self::Error> {
                self.0.$name($($argument),*)
            }
        )*
    };
}

impl<M: ColumnStats<f64>> ColumnStats<f64> for Matrix<M> {
    fn normalization_stats(
        &self,
        spec: Normalization,
    ) -> Result<NormalizationStats<f64>, Self::Error> {
        metrics::statistics(|| self.0.normalization_stats(spec))
    }

    forward_statistics!(
        col_means(),
        col_sds(),
        col_mins(),
        col_ranges(),
        col_maxabs(),
        col_l1(),
        col_l2(),
        col_l2_centered(centers: &[f64]),
        col_l1_centered(centers: &[f64]),
        col_maxabs_centered(centers: &[f64]),
    );
}

impl<M: MatVecInto<V>, V> MatVecInto<Vector<V>> for Matrix<M> {
    fn matvec_into(&self, input: &Vector<V>, output: &mut Vector<V>) -> Result<(), Self::Error> {
        metrics::forward();
        self.0.matvec_into(&input.0, &mut output.0)
    }
}

impl<M: MatTransposeVecInto<V>, V> MatTransposeVecInto<Vector<V>> for Matrix<M> {
    fn mat_transpose_vec_into(
        &self,
        input: &Vector<V>,
        output: &mut Vector<V>,
    ) -> Result<(), Self::Error> {
        metrics::transpose();
        self.0.mat_transpose_vec_into(&input.0, &mut output.0)
    }
}

impl<M: shrinkage::lazymatrix::MaterializeDense<f64>> shrinkage::lazymatrix::MaterializeDense<f64>
    for Matrix<M>
{
    fn materialize_normalized_into<O: shrinkage::lazymatrix::MatrixWrite<f64> + ?Sized>(
        &self,
        centers: Option<&[f64]>,
        scales: Option<&[f64]>,
        out: &mut O,
    ) -> Result<(), Self::Error> {
        self.0.materialize_normalized_into(centers, scales, out)
    }
}
