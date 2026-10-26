//! Center dense entries before summation without materializing a design matrix.

use crate::{Scalar, VectorView, VectorViewMut};

fn parameters<F: Scalar, N: Scalar>(
    j: usize,
    centers: &[N],
    scales: Option<&[N]>,
) -> (F, Option<F>) {
    (
        F::from(centers[j]).expect("center must convert to the matrix scalar"),
        scales.map(|scales| F::from(scales[j]).expect("scale must convert to the matrix scalar")),
    )
}

fn normalized<F: Scalar>(value: F, (center, scale): (F, Option<F>)) -> F {
    let centered = value - center;
    // Dividing entries first preserves overflow and nonfinite arithmetic even
    // when a coefficient is zero or its prescaled value would underflow.
    scale.map_or(centered, |scale| centered / scale)
}

fn write<F: Scalar, Y: VectorViewMut<F> + ?Sized>(
    out: &mut Y,
    i: usize,
    value: F,
    alpha: F,
    beta: F,
) {
    let product = alpha * value;
    out.set(
        i,
        if beta == F::zero() {
            product
        } else {
            product + beta * out.get(i)
        },
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn matvec<F, N, X, Y>(
    shape: (usize, usize),
    at: impl Fn(usize, usize) -> F,
    centers: &[N],
    scales: Option<&[N]>,
    alpha: F,
    x: &X,
    beta: F,
    out: &mut Y,
) where
    F: Scalar,
    N: Scalar,
    X: VectorView<F> + ?Sized,
    Y: VectorViewMut<F> + ?Sized,
{
    let (rows, columns) = shape;
    assert_eq!(x.len(), columns, "matvec: dimension mismatch");
    assert_eq!(out.len(), rows, "matvec: output dimension mismatch");
    crate::gram::validate_normalization(columns, Some(centers), scales);
    if alpha == F::zero() {
        crate::traits::scale_output(beta, out);
        return;
    }
    for i in 0..rows {
        let sum = (0..columns)
            .map(|j| normalized(at(i, j), parameters(j, centers, scales)) * x.get(j))
            .sum();
        write(out, i, sum, alpha, beta);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn transpose<F, N, X, Y>(
    shape: (usize, usize),
    at: impl Fn(usize, usize) -> F,
    centers: &[N],
    scales: Option<&[N]>,
    alpha: F,
    x: &X,
    beta: F,
    out: &mut Y,
) where
    F: Scalar,
    N: Scalar,
    X: VectorView<F> + ?Sized,
    Y: VectorViewMut<F> + ?Sized,
{
    let (rows, columns) = shape;
    assert_eq!(x.len(), rows, "transpose product: dimension mismatch");
    assert_eq!(
        out.len(),
        columns,
        "transpose product: output dimension mismatch"
    );
    crate::gram::validate_normalization(columns, Some(centers), scales);
    if alpha == F::zero() {
        crate::traits::scale_output(beta, out);
        return;
    }
    for j in 0..columns {
        let parameters = parameters(j, centers, scales);
        let sum = (0..rows)
            .map(|i| normalized(at(i, j), parameters) * x.get(i))
            .sum();
        write(out, j, sum, alpha, beta);
    }
}
