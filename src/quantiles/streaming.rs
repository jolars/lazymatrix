use super::{Request, interpolate};
use crate::{NormalizationStats, Scalar};

#[derive(Clone, Copy)]
struct Bounds<F> {
    lower: F,
    upper: F,
    finite: usize,
    negative_infinite: usize,
    nan: bool,
}

impl<F: Scalar> Bounds<F> {
    fn new() -> Self {
        Self {
            lower: F::infinity(),
            upper: F::neg_infinity(),
            finite: 0,
            negative_infinite: 0,
            nan: false,
        }
    }

    fn visit(&mut self, value: F, count: usize) {
        if count == 0 {
            return;
        }
        if value.is_nan() {
            self.nan = true;
        } else if value.is_finite() {
            self.finite += count;
            self.lower = self.lower.min(value);
            self.upper = self.upper.max(value);
        } else if value < F::zero() {
            self.negative_infinite += count;
        }
    }
}

#[derive(Clone, Copy)]
struct Selection<F> {
    rank: usize,
    lower: F,
    upper: F,
    pivot: F,
    before: F,
    after: F,
    below: usize,
    equal: usize,
    result: Option<F>,
}

impl<F: Scalar> Selection<F> {
    fn new(bounds: Bounds<F>, rank: usize, nrows: usize) -> Self {
        let result = if nrows == 0 || bounds.nan {
            Some(F::nan())
        } else if rank < bounds.negative_infinite {
            Some(F::neg_infinity())
        } else if rank - bounds.negative_infinite >= bounds.finite {
            Some(F::infinity())
        } else if bounds.lower == bounds.upper {
            Some(bounds.lower)
        } else {
            None
        };
        Self {
            rank: rank.saturating_sub(bounds.negative_infinite),
            lower: bounds.lower,
            upper: bounds.upper,
            pivot: F::zero(),
            before: F::neg_infinity(),
            after: F::infinity(),
            below: 0,
            equal: 0,
            result,
        }
    }

    fn prepare(&mut self) {
        self.pivot = interpolate(self.lower, self.upper, 0.5);
        self.before = F::neg_infinity();
        self.after = F::infinity();
        self.below = 0;
        self.equal = 0;
    }

    fn visit(&mut self, value: F, count: usize) {
        if value < self.pivot {
            self.below += count;
            self.before = self.before.max(value);
        } else if value > self.pivot {
            self.after = self.after.min(value);
        } else {
            self.equal += count;
        }
    }

    fn finish(&mut self) {
        if self.rank < self.below {
            self.upper = self.before;
        } else if self.rank - self.below < self.equal {
            self.result = Some(self.pivot);
        } else {
            self.lower = self.after;
        }
        // Observed neighbors ensure progress even when a midpoint rounds to an endpoint.
        if self.lower == self.upper {
            self.result = Some(self.lower);
        }
    }
}

pub(crate) fn stream<F: Scalar, E>(
    nrows: usize,
    ncols: usize,
    request: Request,
    mut scan: impl FnMut(&mut dyn FnMut(usize, F, usize)) -> Result<(), E>,
) -> Result<NormalizationStats<F>, E> {
    let positions = request.positions(nrows);
    let mut output = request.output(ncols);
    let mut bounds = vec![Bounds::new(); ncols];
    if nrows != 0 && ncols != 0 {
        scan(&mut |j, value, count| bounds[j].visit(value, count))?;
    }
    let ranks: Vec<_> = positions.iter().flat_map(|p| [p.lower, p.upper]).collect();
    let mut columns: Vec<Vec<_>> = bounds
        .into_iter()
        .map(|bounds| {
            ranks
                .iter()
                .map(|&rank| Selection::new(bounds, rank, nrows))
                .collect()
        })
        .collect();
    while columns.iter().flatten().any(|state| state.result.is_none()) {
        for state in columns
            .iter_mut()
            .flatten()
            .filter(|state| state.result.is_none())
        {
            state.prepare();
        }
        scan(&mut |j, value, count| {
            if count != 0 && value.is_finite() {
                for state in columns[j].iter_mut().filter(|state| state.result.is_none()) {
                    state.visit(value, count);
                }
            }
        })?;
        for state in columns
            .iter_mut()
            .flatten()
            .filter(|state| state.result.is_none())
        {
            state.finish();
        }
    }
    for column in columns {
        let mut values = [F::nan(); 6];
        for (i, state) in column.into_iter().enumerate() {
            values[i] = state.result.unwrap();
        }
        request.push(&mut output, &positions, &values);
    }
    Ok(output)
}

#[cfg(any(feature = "faer_all", feature = "nalgebra_all", feature = "sprs_all"))]
pub(crate) fn sparse_rows<F: Scalar>(
    nrows: usize,
    ncols: usize,
    request: Request,
    mut row: impl FnMut(usize, &mut dyn FnMut(usize, F)),
) -> NormalizationStats<F> {
    let mut sums = vec![F::zero(); ncols];
    let mut seen = vec![false; ncols];
    let mut touched = Vec::new();
    let mut counts = vec![0; ncols];
    stream(nrows, ncols, request, |visit| {
        counts.fill(0);
        for i in 0..nrows {
            row(i, &mut |j, value| {
                if seen[j] {
                    sums[j] = sums[j] + value;
                } else {
                    sums[j] = value;
                    seen[j] = true;
                    touched.push(j);
                }
            });
            for j in touched.drain(..) {
                visit(j, sums[j], 1);
                counts[j] += 1;
                seen[j] = false;
            }
        }
        for (j, &stored) in counts.iter().enumerate() {
            visit(j, F::zero(), nrows - stored);
        }
        Ok::<_, std::convert::Infallible>(())
    })
    .unwrap()
}
