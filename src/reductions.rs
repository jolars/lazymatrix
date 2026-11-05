//! Contiguous reductions shared by vector and column operations.

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
    sum_map2(left, right, |x, y| x * y)
}

#[inline]
pub(crate) fn sum_map2<F: Scalar>(left: &[F], right: &[F], term: impl Fn(F, F) -> F) -> F {
    assert_eq!(left.len(), right.len(), "reduction: length mismatch");
    let mut sums = [F::zero(); 8];
    let mut left = left.chunks_exact(8);
    let mut right = right.chunks_exact(8);
    for (x, y) in left.by_ref().zip(right.by_ref()) {
        for lane in 0..8 {
            sums[lane] = sums[lane] + term(x[lane], y[lane]);
        }
    }
    let mut sum = combine(sums);
    for (&x, &y) in left.remainder().iter().zip(right.remainder()) {
        sum = sum + term(x, y);
    }
    sum
}

#[inline]
pub(crate) fn weighted_dot<F: Scalar>(column: &[F], weights: &[F], vector: &[F]) -> F {
    assert_eq!(column.len(), weights.len(), "reduction: length mismatch");
    assert_eq!(column.len(), vector.len(), "reduction: length mismatch");
    let mut sums = [F::zero(); 8];
    let mut column = column.chunks_exact(8);
    let mut weights = weights.chunks_exact(8);
    let mut vector = vector.chunks_exact(8);
    for ((x, w), v) in column.by_ref().zip(weights.by_ref()).zip(vector.by_ref()) {
        for lane in 0..8 {
            sums[lane] = sums[lane] + x[lane] * w[lane] * v[lane];
        }
    }
    let mut sum = combine(sums);
    for ((&x, &w), &v) in column
        .remainder()
        .iter()
        .zip(weights.remainder())
        .zip(vector.remainder())
    {
        sum = sum + x * w * v;
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
