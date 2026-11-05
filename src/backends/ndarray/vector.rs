//! Vector operations over owned ndarray arrays and borrowed, strided views.

use super::ndarray;

use ndarray::{ArrayBase, Data, DataMut, Ix1};

use crate::traits::{
    DotProduct, DotSlice, ElemDivAssign, L2Norm, RawColumn, Scalar, ScaleAssign, ScaledAddAssign,
    ScaledSubSlice, SubScalarAssign, SumEntries, VectorView, VectorViewMut,
};

impl<F, S> VectorView<F> for ArrayBase<S, Ix1>
where
    F: Scalar,
    S: Data<Elem = F>,
{
    fn len(&self) -> usize {
        self.shape()[0]
    }

    fn get(&self, index: usize) -> F {
        self[index]
    }

    fn contiguous_values(&self) -> Option<&[F]> {
        self.as_slice()
    }
}

impl<F, S> VectorViewMut<F> for ArrayBase<S, Ix1>
where
    F: Scalar,
    S: DataMut<Elem = F>,
{
    fn set(&mut self, index: usize, value: F) {
        self[index] = value;
    }
}

fn column_reduction_values<F, S>(column: &ArrayBase<S, Ix1>) -> Option<&[F]>
where
    S: Data<Elem = F>,
{
    // Older releases recognize some singleton layouts only as logical slices.
    column.as_slice().or_else(|| column.as_slice_memory_order())
}

impl<F, S> RawColumn<F> for ArrayBase<S, Ix1>
where
    F: Scalar,
    S: Data<Elem = F>,
{
    fn len(&self) -> usize {
        self.shape()[0]
    }

    fn dense_values(&self) -> Option<&[F]> {
        self.as_slice()
    }

    fn raw_sum(&self) -> F {
        if let Some(values) = column_reduction_values(self) {
            return crate::reductions::sum(values);
        }
        self.iter().fold(F::zero(), |sum, &value| sum + value)
    }

    fn centered_sum_squares(&self, center: F) -> Option<F> {
        column_reduction_values(self)
            .map(|values| crate::reductions::centered_sum_squares(values, center))
    }

    fn stored_len(&self) -> usize {
        self.shape()[0]
    }

    fn for_each_stored(&self, mut f: impl FnMut(usize, F)) {
        for (row, &value) in self.iter().enumerate() {
            f(row, value);
        }
    }

    fn affine_add_to<V>(&self, raw_multiplier: F, offset: F, destination: &mut V)
    where
        V: VectorViewMut<F> + ?Sized,
    {
        assert_eq!(
            destination.len(),
            self.shape()[0],
            "destination length must equal column length"
        );
        // A slice iterator enables vectorization without requiring a contiguous destination.
        if let Some(source) = self.as_slice() {
            for (row, &value) in source.iter().enumerate() {
                destination.set(row, destination.get(row) + raw_multiplier * value + offset);
            }
            return;
        }
        for (row, &value) in self.iter().enumerate() {
            destination.set(row, destination.get(row) + raw_multiplier * value + offset);
        }
    }
}

impl<F, S> DotProduct<F> for ArrayBase<S, Ix1>
where
    F: Scalar,
    S: Data<Elem = F>,
{
    fn dot(&self, other: &Self) -> F {
        assert_eq!(self.shape(), other.shape(), "dot: length mismatch");
        if let (Some(left), Some(right)) = (self.as_slice(), other.as_slice()) {
            return crate::reductions::dot(left, right);
        }
        self.iter().zip(other.iter()).map(|(&a, &b)| a * b).sum()
    }
}

impl<F, S> L2Norm<F> for ArrayBase<S, Ix1>
where
    F: Scalar,
    S: Data<Elem = F>,
{
    fn norm_l2(&self) -> F {
        if let Some(values) = self.as_slice() {
            return crate::reductions::sum_map(values, |value| value * value).sqrt();
        }
        self.iter().map(|&value| value * value).sum::<F>().sqrt()
    }
}

impl<F, S> ScaledAddAssign<F> for ArrayBase<S, Ix1>
where
    F: Scalar,
    S: DataMut<Elem = F>,
{
    fn scaled_add_assign(&mut self, alpha: F, other: &Self) {
        assert_eq!(
            self.shape(),
            other.shape(),
            "scaled_add_assign: length mismatch"
        );
        for (value, &other) in self.iter_mut().zip(other.iter()) {
            *value = *value + alpha * other;
        }
    }
}

impl<F, S> ScaleAssign<F> for ArrayBase<S, Ix1>
where
    F: Scalar,
    S: DataMut<Elem = F>,
{
    fn scale_assign(&mut self, alpha: F) {
        for value in self.iter_mut() {
            *value = *value * alpha;
        }
    }
}

impl<F, S> ElemDivAssign<F> for ArrayBase<S, Ix1>
where
    F: Scalar,
    S: DataMut<Elem = F>,
{
    fn elem_div_assign(&mut self, coeffs: &[F]) {
        assert_eq!(
            self.shape()[0],
            coeffs.len(),
            "elem_div_assign: length mismatch"
        );
        for (value, &coefficient) in self.iter_mut().zip(coeffs) {
            *value = *value / coefficient;
        }
    }
}

impl<F, S> DotSlice<F> for ArrayBase<S, Ix1>
where
    F: Scalar,
    S: Data<Elem = F>,
{
    fn dot_slice(&self, coeffs: &[F]) -> F {
        assert_eq!(self.shape()[0], coeffs.len(), "dot_slice: length mismatch");
        if let Some(values) = self.as_slice() {
            return crate::reductions::dot(values, coeffs);
        }
        self.iter()
            .zip(coeffs)
            .map(|(&value, &coefficient)| value * coefficient)
            .sum()
    }
}

impl<F, S> SubScalarAssign<F> for ArrayBase<S, Ix1>
where
    F: Scalar,
    S: DataMut<Elem = F>,
{
    fn sub_scalar_assign(&mut self, k: F) {
        for value in self.iter_mut() {
            *value = *value - k;
        }
    }
}

impl<F, S> SumEntries<F> for ArrayBase<S, Ix1>
where
    F: Scalar,
    S: Data<Elem = F>,
{
    fn sum_entries(&self) -> F {
        if let Some(values) = self.as_slice() {
            return crate::reductions::sum(values);
        }
        self.iter().copied().sum()
    }
}

impl<F, S> ScaledSubSlice<F> for ArrayBase<S, Ix1>
where
    F: Scalar,
    S: DataMut<Elem = F>,
{
    fn scaled_sub_slice(&mut self, k: F, coeffs: &[F]) {
        assert_eq!(
            self.shape()[0],
            coeffs.len(),
            "scaled_sub_slice: length mismatch"
        );
        for (value, &coefficient) in self.iter_mut().zip(coeffs) {
            *value = *value - k * coefficient;
        }
    }
}

impl<F, S> crate::VectorOwned<F> for ArrayBase<S, Ix1>
where
    F: Scalar,
    S: Data<Elem = F>,
{
    type Owned = super::ndarray::Array1<F>;
    fn owned_from_fn(len: usize, value: impl FnMut(usize) -> F) -> Self::Owned {
        super::ndarray::Array1::from_shape_fn(len, value)
    }
}
