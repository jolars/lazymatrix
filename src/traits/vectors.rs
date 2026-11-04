use super::Scalar;

/// Read-only indexed access to a dense logical vector.
///
/// Implementations may be contiguous or strided. Indexing must take O(1)
/// time; algorithms must not assume that the entries form a slice.
pub trait VectorView<F: Scalar> {
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> F;

    /// Borrow contiguous entries in logical order, without copying or gathering.
    ///
    /// A returned slice has length [`Self::len`] and agrees with [`Self::get`]
    /// at every index. This takes O(1) time. The default returns `None`, which
    /// supports strided or otherwise noncontiguous implementations.
    fn contiguous_values(&self) -> Option<&[F]> {
        None
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Sum entries without allocating.
    ///
    /// Reduction order can depend on storage. Rounding and intermediate
    /// overflow may differ between contiguous and indexed implementations.
    fn sum(&self) -> F {
        if let Some(values) = self.contiguous_values() {
            return crate::reductions::sum(values);
        }
        (0..self.len()).map(|i| self.get(i)).sum()
    }
}

/// Mutable indexed access to a dense logical vector.
pub trait VectorViewMut<F: Scalar>: VectorView<F> {
    fn set(&mut self, index: usize, value: F);
}

/// Construct owned vector storage compatible with a vector or borrowed view.
///
/// Lazy operator components use this capability for coefficient scratch and
/// allocating results. Construction takes O(len) time and storage, without
/// requiring contiguous access to the source view.
pub trait VectorOwned<F: Scalar>: VectorView<F> {
    /// The corresponding resizable, owned backend vector type.
    type Owned: VectorOwned<F, Owned = Self::Owned> + VectorViewMut<F>;

    /// Construct a vector whose entry at each index is supplied by `value`.
    fn owned_from_fn(len: usize, value: impl FnMut(usize) -> F) -> Self::Owned;
}

impl<F: Scalar> VectorOwned<F> for Vec<F> {
    type Owned = Self;

    fn owned_from_fn(len: usize, value: impl FnMut(usize) -> F) -> Self {
        (0..len).map(value).collect()
    }
}

impl<F: Scalar> VectorOwned<F> for [F] {
    type Owned = Vec<F>;

    fn owned_from_fn(len: usize, value: impl FnMut(usize) -> F) -> Vec<F> {
        Vec::owned_from_fn(len, value)
    }
}

impl<F: Scalar, const N: usize> VectorOwned<F> for [F; N] {
    type Owned = Vec<F>;

    fn owned_from_fn(len: usize, value: impl FnMut(usize) -> F) -> Vec<F> {
        Vec::owned_from_fn(len, value)
    }
}

impl<F: Scalar> VectorView<F> for [F] {
    fn len(&self) -> usize {
        <[F]>::len(self)
    }

    fn get(&self, index: usize) -> F {
        self[index]
    }

    fn contiguous_values(&self) -> Option<&[F]> {
        Some(self)
    }
}

impl<F: Scalar> VectorViewMut<F> for [F] {
    fn set(&mut self, index: usize, value: F) {
        self[index] = value;
    }
}

impl<F: Scalar> VectorView<F> for Vec<F> {
    fn len(&self) -> usize {
        Vec::len(self)
    }

    fn get(&self, index: usize) -> F {
        self[index]
    }

    fn contiguous_values(&self) -> Option<&[F]> {
        Some(self.as_slice())
    }
}

impl<F: Scalar> VectorViewMut<F> for Vec<F> {
    fn set(&mut self, index: usize, value: F) {
        self[index] = value;
    }
}

impl<F: Scalar, const N: usize> VectorView<F> for [F; N] {
    fn len(&self) -> usize {
        N
    }

    fn get(&self, index: usize) -> F {
        self[index]
    }

    fn contiguous_values(&self) -> Option<&[F]> {
        Some(self)
    }
}

impl<F: Scalar, const N: usize> VectorViewMut<F> for [F; N] {
    fn set(&mut self, index: usize, value: F) {
        self[index] = value;
    }
}

/// Dot product of two backend vectors: `Σ self[i] · other[i]`.
///
/// # Panics
///
/// Panics if the vectors have different lengths.
pub trait DotProduct<F: Scalar> {
    fn dot(&self, other: &Self) -> F;
}

/// Euclidean norm of a backend vector.
pub trait L2Norm<F: Scalar> {
    fn norm_l2(&self) -> F;
}

/// In-place scaled vector addition: `self += alpha · other`.
///
/// # Panics
///
/// Panics if the vectors have different lengths.
pub trait ScaledAddAssign<F: Scalar> {
    fn scaled_add_assign(&mut self, alpha: F, other: &Self);
}

/// In-place vector scaling: `self *= alpha`.
pub trait ScaleAssign<F: Scalar> {
    fn scale_assign(&mut self, alpha: F);
}

/// In-place elementwise division by a coefficient slice: `self[i] /= coeffs[i]`.
///
/// Used to apply `S⁻¹` (scaling). Callers guarantee `coeffs[i] != 0`; the
/// `normalized` constructor floors zero scales to `1` so a constant column never
/// triggers a division by zero.
pub trait ElemDivAssign<F: Scalar> {
    fn elem_div_assign(&mut self, coeffs: &[F]);
}

/// Dot product of `self` with a coefficient slice: `Σ self[i] · coeffs[i]`.
///
/// Used to form the scalar `cᵀw` in the forward operator.
pub trait DotSlice<F: Scalar> {
    fn dot_slice(&self, coeffs: &[F]) -> F;
}

/// In-place broadcast subtraction of a scalar: `self[i] -= k`.
///
/// Used to apply the `− 1·(cᵀw)` centering correction in the forward operator.
pub trait SubScalarAssign<F: Scalar> {
    fn sub_scalar_assign(&mut self, k: F);
}

/// Sum of all entries: `Σ self[i]`.
///
/// Used to form `Σu` in the transpose operator.
pub trait SumEntries<F: Scalar> {
    fn sum_entries(&self) -> F;
}

/// In-place scaled subtraction against a coefficient slice: `self[i] -= k · coeffs[i]`.
///
/// Used to apply the `− Σu · c` centering correction in the transpose operator.
pub trait ScaledSubSlice<F: Scalar> {
    fn scaled_sub_slice(&mut self, k: F, coeffs: &[F]);
}

impl<F: Scalar> DotProduct<F> for Vec<F> {
    fn dot(&self, other: &Self) -> F {
        self.dot_slice(other)
    }
}

impl<F: Scalar> L2Norm<F> for Vec<F> {
    fn norm_l2(&self) -> F {
        self.iter().map(|&value| value * value).sum::<F>().sqrt()
    }
}

impl<F: Scalar> ScaledAddAssign<F> for Vec<F> {
    fn scaled_add_assign(&mut self, alpha: F, other: &Self) {
        assert_eq!(
            self.len(),
            other.len(),
            "scaled_add_assign: length mismatch"
        );
        for (value, &other) in self.iter_mut().zip(other) {
            *value = *value + alpha * other;
        }
    }
}

impl<F: Scalar> ScaleAssign<F> for Vec<F> {
    fn scale_assign(&mut self, alpha: F) {
        for value in self.iter_mut() {
            *value = *value * alpha;
        }
    }
}

impl<F: Scalar> ElemDivAssign<F> for Vec<F> {
    fn elem_div_assign(&mut self, coeffs: &[F]) {
        assert_eq!(self.len(), coeffs.len(), "elem_div_assign: length mismatch");
        for (value, &coefficient) in self.iter_mut().zip(coeffs) {
            *value = *value / coefficient;
        }
    }
}

impl<F: Scalar> DotSlice<F> for Vec<F> {
    fn dot_slice(&self, coeffs: &[F]) -> F {
        assert_eq!(self.len(), coeffs.len(), "dot_slice: length mismatch");
        self.iter().zip(coeffs).map(|(&a, &b)| a * b).sum()
    }
}

impl<F: Scalar> SubScalarAssign<F> for Vec<F> {
    fn sub_scalar_assign(&mut self, k: F) {
        for value in self.iter_mut() {
            *value = *value - k;
        }
    }
}

impl<F: Scalar> SumEntries<F> for Vec<F> {
    fn sum_entries(&self) -> F {
        self.iter().copied().sum()
    }
}

impl<F: Scalar> ScaledSubSlice<F> for Vec<F> {
    fn scaled_sub_slice(&mut self, k: F, coeffs: &[F]) {
        assert_eq!(
            self.len(),
            coeffs.len(),
            "scaled_sub_slice: length mismatch"
        );
        for (value, &coefficient) in self.iter_mut().zip(coeffs) {
            *value = *value - k * coefficient;
        }
    }
}
