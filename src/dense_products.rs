//! Center dense entries before summation without materializing a design matrix.

use crate::{Scalar, VectorView, VectorViewMut};

const PANEL: usize = 128;

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
    column_major: bool,
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
    if column_major {
        match scales {
            Some(scales) => panel_product::<_, _, _, _, _, true, false>(
                shape, at, centers, scales, alpha, x, beta, out,
            ),
            None => panel_product::<_, _, _, _, _, false, false>(
                shape,
                at,
                centers,
                &[],
                alpha,
                x,
                beta,
                out,
            ),
        }
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
    column_major: bool,
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
    if !column_major {
        match scales {
            Some(scales) => panel_product::<_, _, _, _, _, true, true>(
                shape, at, centers, scales, alpha, x, beta, out,
            ),
            None => panel_product::<_, _, _, _, _, false, true>(
                shape,
                at,
                centers,
                &[],
                alpha,
                x,
                beta,
                out,
            ),
        }
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

fn panel_parameters<F: Scalar, N: Scalar, const SCALED: bool>(
    j: usize,
    centers: &[N],
    scales: &[N],
) -> (F, F) {
    (
        F::from(centers[j]).expect("center must convert to the matrix scalar"),
        if SCALED {
            F::from(scales[j]).expect("scale must convert to the matrix scalar")
        } else {
            F::one()
        },
    )
}

fn panel_normalized<F: Scalar, const SCALED: bool>(value: F, (center, scale): (F, F)) -> F {
    let centered = value - center;
    if SCALED { centered / scale } else { centered }
}

#[allow(clippy::too_many_arguments)]
fn panel_product<F, N, X, Y, At, const SCALED: bool, const TRANSPOSE: bool>(
    (rows, columns): (usize, usize),
    at: At,
    centers: &[N],
    scales: &[N],
    alpha: F,
    x: &X,
    beta: F,
    out: &mut Y,
) where
    F: Scalar,
    N: Scalar,
    X: VectorView<F> + ?Sized,
    Y: VectorViewMut<F> + ?Sized,
    At: Fn(usize, usize) -> F,
{
    // Independent outputs let panels follow storage without reassociating any
    // reduction. Scratch stays bounded, and alpha and beta are applied once.
    // Seed with Sum's identity to preserve the scalar path's signed zeros.
    if !TRANSPOSE {
        for start in (0..rows).step_by(PANEL) {
            let height = PANEL.min(rows - start);
            let mut sums = [std::iter::empty::<F>().sum::<F>(); PANEL];
            for j in 0..columns {
                let parameters = panel_parameters::<F, N, SCALED>(j, centers, scales);
                let coefficient = x.get(j);
                for (offset, sum) in sums[..height].iter_mut().enumerate() {
                    *sum = *sum
                        + panel_normalized::<F, SCALED>(at(start + offset, j), parameters)
                            * coefficient;
                }
            }
            for (offset, sum) in sums[..height].iter().enumerate() {
                write(out, start + offset, *sum, alpha, beta);
            }
        }
    } else {
        for start in (0..columns).step_by(PANEL) {
            let width = PANEL.min(columns - start);
            let mut sums = [std::iter::empty::<F>().sum::<F>(); PANEL];
            let mut params = [(F::zero(), F::one()); PANEL];
            for (offset, param) in params[..width].iter_mut().enumerate() {
                *param = panel_parameters::<F, N, SCALED>(start + offset, centers, scales);
            }
            for i in 0..rows {
                let coefficient = x.get(i);
                for (offset, sum) in sums[..width].iter_mut().enumerate() {
                    *sum = *sum
                        + panel_normalized::<F, SCALED>(at(i, start + offset), params[offset])
                            * coefficient;
                }
            }
            for (offset, sum) in sums[..width].iter().enumerate() {
                write(out, start + offset, *sum, alpha, beta);
            }
        }
    }
}
