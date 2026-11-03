//! Column-only products with explicit contributions from implicit cells.

use super::{Columns, Execute, Source, Stage};
use crate::{VectorView, VectorViewMut};
use num_traits::{One, Zero};

impl<M, P> Execute<M, P> for Columns
where
    P: Stage<Kind = Columns>,
    M: Source<P::Scalar>,
{
    fn product<X: VectorView<P::Scalar> + ?Sized, Y: VectorViewMut<P::Scalar> + ?Sized>(
        data: &M,
        params: &P,
        transpose: bool,
        x: &X,
        out: &mut Y,
    ) -> Result<(), M::Error> {
        let (input, output) = if transpose {
            (data.nrows(), data.ncols())
        } else {
            (data.ncols(), data.nrows())
        };
        assert_eq!(x.len(), input, "pipeline input dimension mismatch");
        assert_eq!(out.len(), output, "pipeline output dimension mismatch");
        if input == 0 || output == 0 {
            crate::traits::scale_output(P::Scalar::zero(), out);
            return Ok(());
        }
        // Keep the full operator's constant coordinates visible without
        // introducing runtime fused dispatch or its temporary-output path.
        rectangle_into(
            data,
            params,
            0..data.nrows(),
            0..data.ncols(),
            transpose,
            x,
            out,
        )
    }
}

/// Restrict native traversal while retaining the parent's fitted coordinates.
#[allow(clippy::too_many_arguments)]
pub(crate) fn product_rectangle<M, P, X, Y>(
    data: &M,
    params: &P,
    rows: std::ops::Range<usize>,
    columns: std::ops::Range<usize>,
    transpose: bool,
    alpha: P::Scalar,
    x: &X,
    beta: P::Scalar,
    out: &mut Y,
) -> Result<(), M::Error>
where
    P: Stage<Kind = Columns>,
    M: Source<P::Scalar>,
    X: VectorView<P::Scalar> + ?Sized,
    Y: VectorViewMut<P::Scalar> + ?Sized,
{
    crate::traits::validate_traversal(data, &rows, &columns);
    let (input, output) = if transpose {
        (rows.len(), columns.len())
    } else {
        (columns.len(), rows.len())
    };
    assert_eq!(x.len(), input, "pipeline input dimension mismatch");
    assert_eq!(out.len(), output, "pipeline output dimension mismatch");
    if alpha == P::Scalar::zero() || output == 0 {
        crate::traits::scale_output(beta, out);
        return Ok(());
    }
    if input == 0 {
        let product = alpha * P::Scalar::zero();
        for i in 0..out.len() {
            let previous = if beta == P::Scalar::zero() {
                P::Scalar::zero()
            } else {
                beta * out.get(i)
            };
            out.set(i, product + previous);
        }
        return Ok(());
    }
    if alpha == P::Scalar::one() && beta == P::Scalar::zero() {
        return rectangle_into(data, params, rows, columns, transpose, x, out);
    }
    // Keep beta's output separate from the product so alpha scales the final
    // sum, and no normalization correction follows a failed backend read.
    let mut product = vec![P::Scalar::zero(); output];
    rectangle_into(data, params, rows, columns, transpose, x, &mut product)?;
    crate::traits::scale_output(beta, out);
    for (i, value) in product.into_iter().enumerate() {
        out.set(i, out.get(i) + alpha * value);
    }
    Ok(())
}

// Full products need constant origins to eliminate per-entry coordinate offsets.
#[inline(always)]
fn rectangle_into<M, P, X, Y>(
    data: &M,
    params: &P,
    rows: std::ops::Range<usize>,
    columns: std::ops::Range<usize>,
    transpose: bool,
    x: &X,
    out: &mut Y,
) -> Result<(), M::Error>
where
    P: Stage<Kind = Columns>,
    M: Source<P::Scalar>,
    X: VectorView<P::Scalar> + ?Sized,
    Y: VectorViewMut<P::Scalar> + ?Sized,
{
    for i in 0..out.len() {
        out.set(i, P::Scalar::zero());
    }
    match data.traversal_layout() {
        super::source::Layout::Full => {
            let (row_start, column_start) = (rows.start, columns.start);
            data.for_each_entry(rows, columns, |i, j, value| {
                let (to, from) = if transpose {
                    (j - column_start, i - row_start)
                } else {
                    (i - row_start, j - column_start)
                };
                out.set(to, out.get(to) + params.entry(value, i, j) * x.get(from));
            })
        }
        _ => sparse_product(data, params, rows, columns, transpose, x, out),
    }
}

fn sparse_product<M, P, X, Y>(
    data: &M,
    params: &P,
    rows: std::ops::Range<usize>,
    columns: std::ops::Range<usize>,
    transpose: bool,
    x: &X,
    out: &mut Y,
) -> Result<(), M::Error>
where
    P: Stage<Kind = Columns>,
    M: Source<P::Scalar>,
    X: VectorView<P::Scalar> + ?Sized,
    Y: VectorViewMut<P::Scalar> + ?Sized,
{
    let background: Vec<_> = columns
        .clone()
        .map(|j| params.entry(P::Scalar::zero(), rows.start, j))
        .collect();
    let sums = RangeSums::new(
        (0..x.len())
            .map(|i| {
                if transpose {
                    x.get(i)
                } else {
                    background[i] * x.get(i)
                }
            })
            .collect(),
    );
    let gap = |to: usize, start, end| {
        sums.sum(
            start,
            end,
            if transpose {
                background[to]
            } else {
                P::Scalar::one()
            },
        )
    };
    let outer_is_output = match data.traversal_layout() {
        super::source::Layout::SparseRows => !transpose,
        super::source::Layout::SparseColumns => transpose,
        super::source::Layout::Full => unreachable!(),
    };
    let local = |i, j| {
        if transpose {
            (j - columns.start, i - rows.start)
        } else {
            (i - rows.start, j - columns.start)
        }
    };
    if outer_is_output {
        let mut current = 0;
        let mut last = 0;
        data.for_each_entry(rows.clone(), columns.clone(), |i, j, value| {
            let (to, from) = local(i, j);
            while current < to {
                out.set(current, out.get(current) + gap(current, last, x.len()));
                current += 1;
                last = 0;
            }
            out.set(
                to,
                out.get(to) + gap(to, last, from) + params.entry(value, i, j) * x.get(from),
            );
            last = from + 1;
        })?;
        while current < out.len() {
            out.set(current, out.get(current) + gap(current, last, x.len()));
            current += 1;
            last = 0;
        }
        return Ok(());
    }
    let mut last = vec![0; out.len()];
    data.for_each_entry(rows.clone(), columns.clone(), |i, j, value| {
        let (to, from) = local(i, j);
        let implicit = gap(to, last[to], from);
        out.set(
            to,
            out.get(to) + implicit + params.entry(value, i, j) * x.get(from),
        );
        last[to] = from + 1;
    })?;
    for (to, &start) in last.iter().enumerate() {
        out.set(to, out.get(to) + gap(to, start, x.len()));
    }
    Ok(())
}

// Summing missing ranges directly avoids subtracting almost equal background
// totals when most entries of a row or column are stored.
struct RangeSums<F> {
    values: Vec<F>,
    size: usize,
}
impl<F: crate::Scalar> RangeSums<F> {
    fn new(values: Vec<F>) -> Self {
        let size = values.len().next_power_of_two();
        let mut tree = vec![F::zero(); 2 * size];
        tree[size..size + values.len()].copy_from_slice(&values);
        for i in (1..size).rev() {
            tree[i] = tree[2 * i] + tree[2 * i + 1];
        }
        Self { values: tree, size }
    }

    fn sum(&self, start: usize, end: usize, factor: F) -> F {
        self.range(1, 0, self.size, start, end, factor)
    }

    fn range(&self, node: usize, lo: usize, hi: usize, start: usize, end: usize, factor: F) -> F {
        if start >= hi || end <= lo || start == end {
            return F::zero();
        }
        if start <= lo && hi <= end {
            let product = self.values[node] * factor;
            if hi - lo == 1
                || (factor.is_finite() && self.values[node].is_finite() && product.is_finite())
            {
                return product;
            }
        }
        // IEEE exceptional arithmetic must operate on the actual cells. In
        // particular, infinity times a range sum can hide zero coefficients.
        let mid = lo + (hi - lo) / 2;
        self.range(2 * node, lo, mid, start, end, factor)
            + self.range(2 * node + 1, mid, hi, start, end, factor)
    }
}
