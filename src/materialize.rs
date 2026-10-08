use crate::{MatrixShape, MatrixWrite, Scalar};

pub(crate) fn validate_allocation<F>(nrows: usize, ncols: usize) {
    assert!(
        nrows
            .checked_mul(ncols)
            .and_then(|len| len.checked_mul(std::mem::size_of::<F>()))
            .is_some_and(|bytes| bytes <= isize::MAX as usize),
        "dense allocation size overflow"
    );
}

pub(crate) fn validate_output<F: Scalar>(
    source: &(impl MatrixShape + ?Sized),
    out: &(impl MatrixWrite<F> + ?Sized),
    centers: Option<&[F]>,
    scales: Option<&[F]>,
) {
    assert_eq!(
        (out.nrows(), out.ncols()),
        (source.nrows(), source.ncols()),
        "materialization output shape must match input"
    );
    crate::gram::validate_normalization(source.ncols(), centers, scales);
}

#[cfg(any(
    feature = "faer_all",
    feature = "nalgebra_all",
    feature = "ndarray_all",
    feature = "sprs_all",
    feature = "zarrs_all"
))]
pub(crate) fn normalized<F: Scalar>(
    mut value: F,
    column: usize,
    centers: Option<&[F]>,
    scales: Option<&[F]>,
) -> F {
    if let Some(centers) = centers {
        value = value - centers[column];
    }
    if let Some(scales) = scales {
        value = value / scales[column];
    }
    value
}

#[cfg(any(feature = "faer_all", feature = "nalgebra_all", feature = "sprs_all"))]
pub(crate) fn sparse_outer<F: Scalar, O: MatrixWrite<F> + ?Sized>(
    shape: &impl MatrixShape,
    csc: bool,
    centers: Option<&[F]>,
    scales: Option<&[F]>,
    out: &mut O,
    mut visit: impl FnMut(usize, &mut dyn FnMut(usize, F)),
) {
    validate_output(shape, out, centers, scales);
    let (outer_len, inner_len) = if csc {
        (shape.ncols(), shape.nrows())
    } else {
        (shape.nrows(), shape.ncols())
    };
    if inner_len == 0 || outer_len == 0 {
        return;
    }
    let mut values = vec![F::zero(); inner_len];
    let mut seen = vec![false; inner_len];
    for outer in 0..outer_len {
        values.fill(F::zero());
        seen.fill(false);
        visit(outer, &mut |inner, value| {
            // Assign the first value directly to preserve explicitly stored negative zero.
            values[inner] = if seen[inner] {
                values[inner] + value
            } else {
                value
            };
            seen[inner] = true;
        });
        for (inner, &value) in values.iter().enumerate() {
            let (row, col) = if csc { (inner, outer) } else { (outer, inner) };
            out.set(row, col, normalized(value, col, centers, scales));
        }
    }
}
