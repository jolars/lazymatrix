//! Shared rectangular traversal and arithmetic for in-memory backends.
#![allow(clippy::too_many_arguments)]

use crate::{MatrixShape, MatrixWrite, Scalar, VectorView, VectorViewMut};
use std::ops::Range;

struct Shape(usize, usize);
impl MatrixShape for Shape {
    fn nrows(&self) -> usize {
        self.0
    }
    fn ncols(&self) -> usize {
        self.1
    }
}

fn validate<F: Scalar>(
    source: &(impl MatrixShape + ?Sized),
    rows: &Range<usize>,
    columns: &Range<usize>,
    centers: Option<&[F]>,
    scales: Option<&[F]>,
) {
    crate::submatrix::validate_ranges(source, rows, columns);
    crate::gram::validate_normalization(columns.len(), centers, scales);
}

fn write<F: Scalar, Y: VectorViewMut<F> + ?Sized>(
    out: &mut Y,
    i: usize,
    product: F,
    alpha: F,
    beta: F,
) {
    let value = alpha * product;
    out.set(
        i,
        if beta == F::zero() {
            value
        } else {
            value + beta * out.get(i)
        },
    );
}

#[cfg(any(
    feature = "faer_all",
    feature = "nalgebra_all",
    feature = "ndarray_all"
))]
pub(crate) fn dense_product<F: Scalar, X: VectorView<F> + ?Sized, Y: VectorViewMut<F> + ?Sized>(
    source: &(impl MatrixShape + ?Sized),
    rows: Range<usize>,
    columns: Range<usize>,
    at: impl Fn(usize, usize) -> F,
    centers: Option<&[F]>,
    scales: Option<&[F]>,
    transpose: bool,
    alpha: F,
    x: &X,
    beta: F,
    out: &mut Y,
) {
    validate(source, &rows, &columns, centers, scales);
    let (input, output) = if transpose {
        (rows.len(), columns.len())
    } else {
        (columns.len(), rows.len())
    };
    assert_eq!(x.len(), input, "submatrix input dimension mismatch");
    assert_eq!(out.len(), output, "submatrix output dimension mismatch");
    if alpha == F::zero() {
        crate::traits::scale_output(beta, out);
        return;
    }
    if let Some(centers) = centers {
        let shape = (rows.len(), columns.len());
        let selected = |i, j| at(rows.start + i, columns.start + j);
        // The rectangular capability does not carry strides, so retain scalar traversal.
        if transpose {
            crate::dense_products::transpose(
                shape, selected, true, centers, scales, alpha, x, beta, out,
            );
        } else {
            crate::dense_products::matvec(
                shape, selected, false, centers, scales, alpha, x, beta, out,
            );
        }
        return;
    }
    for outer in 0..output {
        let product = (0..input)
            .map(|inner| {
                let (i, j) = if transpose {
                    (inner, outer)
                } else {
                    (outer, inner)
                };
                crate::materialize::normalized(
                    at(rows.start + i, columns.start + j),
                    j,
                    centers,
                    scales,
                ) * x.get(inner)
            })
            .sum();
        write(out, outer, product, alpha, beta);
    }
}

#[cfg(any(
    feature = "faer_all",
    feature = "nalgebra_all",
    feature = "ndarray_all"
))]
pub(crate) fn dense_materialize<F: Scalar, O: MatrixWrite<F> + ?Sized>(
    source: &(impl MatrixShape + ?Sized),
    rows: Range<usize>,
    columns: Range<usize>,
    at: impl Fn(usize, usize) -> F,
    centers: Option<&[F]>,
    scales: Option<&[F]>,
    out: &mut O,
) {
    validate(source, &rows, &columns, centers, scales);
    crate::materialize::validate_output(&Shape(rows.len(), columns.len()), out, centers, scales);
    for (i, row) in rows.enumerate() {
        for (j, col) in columns.clone().enumerate() {
            out.set(
                i,
                j,
                crate::materialize::normalized(at(row, col), j, centers, scales),
            );
        }
    }
}

/// Combine duplicates with O(inner dimension) workspace and clear only touched cells.
#[cfg(any(feature = "faer_all", feature = "nalgebra_all", feature = "sprs_all"))]
struct Accumulator<F> {
    values: Vec<F>,
    seen: Vec<bool>,
    touched: Vec<usize>,
}
#[cfg(any(feature = "faer_all", feature = "nalgebra_all", feature = "sprs_all"))]
impl<F: Scalar> Accumulator<F> {
    fn new(len: usize) -> Self {
        Self {
            values: vec![F::zero(); len],
            seen: vec![false; len],
            touched: Vec::new(),
        }
    }
    fn add(&mut self, i: usize, value: F) {
        if self.seen[i] {
            self.values[i] = self.values[i] + value;
        } else {
            self.values[i] = value;
            self.seen[i] = true;
            self.touched.push(i);
        }
    }
    fn clear(&mut self) {
        for &i in &self.touched {
            self.seen[i] = false;
        }
        self.touched.clear();
    }
}

#[cfg(any(feature = "faer_all", feature = "nalgebra_all", feature = "sprs_all"))]
pub(crate) fn sparse_product<F: Scalar, X: VectorView<F> + ?Sized, Y: VectorViewMut<F> + ?Sized>(
    source: &(impl MatrixShape + ?Sized),
    rows: Range<usize>,
    columns: Range<usize>,
    csc: bool,
    mut visit: impl FnMut(usize, &mut dyn FnMut(usize, F)),
    centers: Option<&[F]>,
    scales: Option<&[F]>,
    transpose: bool,
    alpha: F,
    x: &X,
    beta: F,
    out: &mut Y,
) {
    validate(source, &rows, &columns, centers, scales);
    let (input, output) = if transpose {
        (rows.len(), columns.len())
    } else {
        (columns.len(), rows.len())
    };
    assert_eq!(x.len(), input, "submatrix input dimension mismatch");
    assert_eq!(out.len(), output, "submatrix output dimension mismatch");
    if alpha == F::zero() {
        crate::traits::scale_output(beta, out);
        return;
    }
    if rows.is_empty() || columns.is_empty() {
        for i in 0..output {
            write(out, i, F::zero(), alpha, beta);
        }
        return;
    }
    let (outer_range, inner_range) = if csc {
        (&columns, &rows)
    } else {
        (&rows, &columns)
    };
    let mut cell = Accumulator::new(inner_range.len());
    let mut products = vec![F::zero(); output];
    for (outer, global) in outer_range.clone().enumerate() {
        visit(global, &mut |inner, value| {
            if inner_range.contains(&inner) {
                cell.add(inner - inner_range.start, value);
            }
        });
        for &inner in &cell.touched {
            let (i, j) = if csc { (inner, outer) } else { (outer, inner) };
            if transpose {
                products[j] = products[j] + cell.values[inner] * x.get(i);
            } else {
                let coefficient = scales.map_or_else(|| x.get(j), |s| x.get(j) / s[j]);
                products[i] = products[i] + cell.values[inner] * coefficient;
            }
        }
        cell.clear();
    }
    if transpose {
        let sum = centers.map(|_| x.sum());
        for (j, mut value) in products.into_iter().enumerate() {
            if let Some(centers) = centers {
                value = value - centers[j] * sum.unwrap();
            }
            if let Some(scales) = scales {
                value = value / scales[j];
            }
            write(out, j, value, alpha, beta);
        }
    } else {
        let correction = centers.map(|c| {
            (0..columns.len())
                .map(|j| {
                    let coefficient = scales.map_or_else(|| x.get(j), |s| x.get(j) / s[j]);
                    c[j] * coefficient
                })
                .sum::<F>()
        });
        for (i, mut value) in products.into_iter().enumerate() {
            if let Some(correction) = correction {
                value = value - correction;
            }
            write(out, i, value, alpha, beta);
        }
    }
}

#[cfg(any(feature = "faer_all", feature = "nalgebra_all", feature = "sprs_all"))]
pub(crate) fn sparse_materialize<F: Scalar, O: MatrixWrite<F> + ?Sized>(
    source: &(impl MatrixShape + ?Sized),
    rows: Range<usize>,
    columns: Range<usize>,
    csc: bool,
    mut visit: impl FnMut(usize, &mut dyn FnMut(usize, F)),
    centers: Option<&[F]>,
    scales: Option<&[F]>,
    out: &mut O,
) {
    validate(source, &rows, &columns, centers, scales);
    let shape = Shape(rows.len(), columns.len());
    let (outer_range, inner_range) = if csc {
        (&columns, &rows)
    } else {
        (&rows, &columns)
    };
    crate::materialize::sparse_outer(&shape, csc, centers, scales, out, |outer, f| {
        visit(outer + outer_range.start, &mut |inner, value| {
            if inner_range.contains(&inner) {
                f(inner - inner_range.start, value);
            }
        });
    });
}
