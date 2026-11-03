//! Contiguous reductions shared by storage-independent column operations.

use crate::Scalar;

#[inline]
pub(crate) fn sum<F: Scalar>(values: &[F]) -> F {
    sum_map(values, |value| value)
}

#[inline]
pub(crate) fn sum_map<F: Scalar>(values: &[F], term: impl Fn(F) -> F) -> F {
    // Independent accumulators permit packed arithmetic without fast-math.
    let mut sums = [F::zero(); 8];
    let mut chunks = values.chunks_exact(8);
    for chunk in chunks.by_ref() {
        for lane in 0..8 {
            sums[lane] = sums[lane] + term(chunk[lane]);
        }
    }
    let mut sum = combine(sums);
    for &value in chunks.remainder() {
        sum = sum + term(value);
    }
    sum
}

#[inline]
pub(crate) fn dot<F: Scalar>(left: &[F], right: &[F]) -> F {
    assert_eq!(left.len(), right.len(), "dot: length mismatch");
    let mut sums = [F::zero(); 8];
    let mut left = left.chunks_exact(8);
    let mut right = right.chunks_exact(8);
    for (x, y) in left.by_ref().zip(right.by_ref()) {
        for lane in 0..8 {
            sums[lane] = sums[lane] + x[lane] * y[lane];
        }
    }
    let mut sum = combine(sums);
    for (&x, &y) in left.remainder().iter().zip(right.remainder()) {
        sum = sum + x * y;
    }
    sum
}

#[inline]
fn combine<F: Scalar>(sums: [F; 8]) -> F {
    let mut sum = F::zero();
    for lane in 0..4 {
        sum = sum + (sums[lane] + sums[lane + 4]);
    }
    sum
}
