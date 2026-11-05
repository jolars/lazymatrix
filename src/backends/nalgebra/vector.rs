//! Vector implementations for the nalgebra backend.

use super::nalgebra;

use nalgebra::DVector;

use crate::traits::{
    DotProduct, DotSlice, ElemDivAssign, L2Norm, Scalar, ScaleAssign, ScaledAddAssign,
    ScaledSubSlice, SubScalarAssign, SumEntries,
};

// --- vector traits on DVector<F> ---------------------------------------------

impl<F> DotProduct<F> for DVector<F>
where
    F: Scalar + nalgebra::RealField,
{
    fn dot(&self, other: &Self) -> F {
        DVector::dot(self, other)
    }
}

impl<F> L2Norm<F> for DVector<F>
where
    F: Scalar + nalgebra::RealField,
{
    fn norm_l2(&self) -> F {
        self.norm()
    }
}

impl<F> ScaledAddAssign<F> for DVector<F>
where
    F: Scalar + nalgebra::RealField,
{
    fn scaled_add_assign(&mut self, alpha: F, other: &Self) {
        self.axpy(alpha, other, F::one());
    }
}

impl<F> ScaleAssign<F> for DVector<F>
where
    F: Scalar + nalgebra::RealField,
{
    fn scale_assign(&mut self, alpha: F) {
        self.scale_mut(alpha);
    }
}

impl<F: Scalar + nalgebra::Scalar> ElemDivAssign<F> for DVector<F> {
    fn elem_div_assign(&mut self, coeffs: &[F]) {
        let s = self.as_mut_slice();
        assert_eq!(s.len(), coeffs.len(), "elem_div_assign: length mismatch");
        for (a, &c) in s.iter_mut().zip(coeffs) {
            *a = *a / c;
        }
    }
}

impl<F: Scalar + nalgebra::Scalar> DotSlice<F> for DVector<F> {
    fn dot_slice(&self, coeffs: &[F]) -> F {
        let s = self.as_slice();
        assert_eq!(s.len(), coeffs.len(), "dot_slice: length mismatch");
        crate::reductions::dot(s, coeffs)
    }
}

impl<F: Scalar + nalgebra::Scalar> SubScalarAssign<F> for DVector<F> {
    fn sub_scalar_assign(&mut self, k: F) {
        for a in self.as_mut_slice() {
            *a = *a - k;
        }
    }
}

impl<F: Scalar + nalgebra::Scalar> SumEntries<F> for DVector<F> {
    fn sum_entries(&self) -> F {
        crate::reductions::sum(self.as_slice())
    }
}

impl<F: Scalar + nalgebra::Scalar> ScaledSubSlice<F> for DVector<F> {
    fn scaled_sub_slice(&mut self, k: F, coeffs: &[F]) {
        let s = self.as_mut_slice();
        assert_eq!(s.len(), coeffs.len(), "scaled_sub_slice: length mismatch");
        for (a, &c) in s.iter_mut().zip(coeffs) {
            *a = *a - k * c;
        }
    }
}

impl<F, R, S> crate::VectorOwned<F> for nalgebra::Matrix<F, R, nalgebra::U1, S>
where
    F: Scalar + nalgebra::Scalar,
    R: nalgebra::Dim,
    S: nalgebra::RawStorage<F, R, nalgebra::U1>,
{
    type Owned = DVector<F>;
    fn owned_from_fn(len: usize, mut value: impl FnMut(usize) -> F) -> Self::Owned {
        DVector::from_fn(len, |i, _| value(i))
    }
}
