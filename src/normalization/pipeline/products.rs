//! Column-only products with explicit contributions from implicit cells.

use super::{Columns, Execute, Source, Stage};
use crate::{VectorView, VectorViewMut};
use num_traits::Zero;

impl<M, P> Execute<M, P> for Columns
where
    P: Stage<Kind = Columns>,
    M: Source<P::Scalar>,
{
    fn product<X: VectorView<P::Scalar>, Y: VectorViewMut<P::Scalar>>(
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
        for i in 0..output {
            out.set(i, P::Scalar::zero());
        }
        if input == 0 || output == 0 {
            return Ok(());
        }
        match data.layout() {
            super::source::Layout::Full => data.visit(|i, j, value| {
                let (to, from) = if transpose { (j, i) } else { (i, j) };
                out.set(to, out.get(to) + params.entry(value, i, j) * x.get(from));
            }),
            #[cfg(any(feature = "faer_all", feature = "nalgebra_all", feature = "sprs_all"))]
            _ => sparse_product(data, params, transpose, x, out),
        }
    }
}

#[cfg(any(feature = "faer_all", feature = "nalgebra_all", feature = "sprs_all"))]
fn sparse_product<M, P, X, Y>(
    data: &M,
    params: &P,
    transpose: bool,
    x: &X,
    out: &mut Y,
) -> Result<(), M::Error>
where
    P: Stage<Kind = Columns>,
    M: Source<P::Scalar>,
    X: VectorView<P::Scalar>,
    Y: VectorViewMut<P::Scalar>,
{
    use num_traits::One;
    let background: Vec<_> = (0..data.ncols())
        .map(|j| params.entry(P::Scalar::zero(), 0, j))
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
    let outer_is_output = match data.layout() {
        super::source::Layout::SparseRows => !transpose,
        super::source::Layout::SparseColumns => transpose,
        super::source::Layout::Full => unreachable!(),
    };
    if outer_is_output {
        let mut current = 0;
        let mut last = 0;
        data.visit(|i, j, value| {
            let (to, from) = if transpose { (j, i) } else { (i, j) };
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
    data.visit(|i, j, value| {
        let (to, from) = if transpose { (j, i) } else { (i, j) };
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
#[cfg(any(feature = "faer_all", feature = "nalgebra_all", feature = "sprs_all"))]
struct RangeSums<F> {
    values: Vec<F>,
    size: usize,
}
#[cfg(any(feature = "faer_all", feature = "nalgebra_all", feature = "sprs_all"))]
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
