//! Vector implementations for the faer backend.

use super::{faer, faer_traits};

use faer::Col;

use crate::traits::{
    DotProduct, DotSlice, ElemDivAssign, L2Norm, Scalar, ScaleAssign, ScaledAddAssign,
    ScaledSubSlice, SubScalarAssign, SumEntries, VectorView,
};

// --- vector traits on Col<F> (scalar arithmetic only: bound F: Scalar) -------

impl<F: Scalar> DotProduct<F> for Col<F> {
    fn dot(&self, other: &Self) -> F {
        assert_eq!(self.nrows(), other.nrows(), "dot: length mismatch");
        if let (Some(left), Some(right)) = (self.contiguous_values(), other.contiguous_values()) {
            return crate::reductions::dot(left, right);
        }
        (0..self.nrows()).map(|i| self[i] * other[i]).sum()
    }
}

impl<F> L2Norm<F> for Col<F>
where
    F: Scalar + faer_traits::RealField,
{
    fn norm_l2(&self) -> F {
        self.as_ref().norm_l2()
    }
}

impl<F: Scalar> ScaledAddAssign<F> for Col<F> {
    fn scaled_add_assign(&mut self, alpha: F, other: &Self) {
        assert_eq!(
            self.nrows(),
            other.nrows(),
            "scaled_add_assign: length mismatch"
        );
        for i in 0..self.nrows() {
            self[i] = self[i] + alpha * other[i];
        }
    }
}

impl<F: Scalar> ScaleAssign<F> for Col<F> {
    fn scale_assign(&mut self, alpha: F) {
        for i in 0..self.nrows() {
            self[i] = self[i] * alpha;
        }
    }
}

impl<F: Scalar> ElemDivAssign<F> for Col<F> {
    fn elem_div_assign(&mut self, coeffs: &[F]) {
        assert_eq!(
            self.nrows(),
            coeffs.len(),
            "elem_div_assign: length mismatch"
        );
        for j in 0..self.nrows() {
            self[j] = self[j] / coeffs[j];
        }
    }
}

impl<F: Scalar> DotSlice<F> for Col<F> {
    fn dot_slice(&self, coeffs: &[F]) -> F {
        assert_eq!(self.nrows(), coeffs.len(), "dot_slice: length mismatch");
        if let Some(values) = self.contiguous_values() {
            return crate::reductions::dot(values, coeffs);
        }
        (0..self.nrows()).map(|j| self[j] * coeffs[j]).sum()
    }
}

impl<F: Scalar> SubScalarAssign<F> for Col<F> {
    fn sub_scalar_assign(&mut self, k: F) {
        for j in 0..self.nrows() {
            self[j] = self[j] - k;
        }
    }
}

impl<F: Scalar> SumEntries<F> for Col<F> {
    fn sum_entries(&self) -> F {
        VectorView::sum(self)
    }
}

impl<F: Scalar> ScaledSubSlice<F> for Col<F> {
    fn scaled_sub_slice(&mut self, k: F, coeffs: &[F]) {
        assert_eq!(
            self.nrows(),
            coeffs.len(),
            "scaled_sub_slice: length mismatch"
        );
        for j in 0..self.nrows() {
            self[j] = self[j] - k * coeffs[j];
        }
    }
}

macro_rules! impl_vector_owned {
    ($vector:ty) => {
        impl<F: Scalar> crate::VectorOwned<F> for $vector {
            type Owned = Col<F>;
            fn owned_from_fn(len: usize, value: impl FnMut(usize) -> F) -> Col<F> {
                Col::from_fn(len, value)
            }
        }
    };
}
impl_vector_owned!(Col<F>);
impl_vector_owned!(faer::ColRef<'_, F>);
impl_vector_owned!(faer::ColMut<'_, F>);
