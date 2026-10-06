//! Backend-independent validation and sparse Gram accumulation.

use crate::{MatrixWrite, Scalar, VectorView};

pub(crate) fn validate<F, W, O>(
    nrows: usize,
    ncols: usize,
    weights: &W,
    centers: Option<&[F]>,
    scales: Option<&[F]>,
    out: &O,
) where
    F: Scalar,
    W: VectorView<F> + ?Sized,
    O: MatrixWrite<F> + ?Sized,
{
    assert_eq!(
        weights.len(),
        nrows,
        "weighted_gram_into: weights length mismatch"
    );
    assert_eq!(
        (out.nrows(), out.ncols()),
        (ncols, ncols),
        "weighted_gram_into: output shape mismatch"
    );
    validate_normalization(ncols, centers, scales);
}

pub(crate) fn validate_normalization<F: Scalar>(
    ncols: usize,
    centers: Option<&[F]>,
    scales: Option<&[F]>,
) {
    if let Some(centers) = centers {
        assert_eq!(centers.len(), ncols, "centers length must equal ncols");
    }
    if let Some(scales) = scales {
        assert_eq!(scales.len(), ncols, "scales length must equal ncols");
        for (column, &scale) in scales.iter().enumerate() {
            assert!(
                scale != F::zero(),
                "scale at column {column} must be nonzero"
            );
        }
    }
}

#[cfg(any(
    feature = "ndarray_all",
    feature = "faer_all",
    feature = "nalgebra_all",
    feature = "sprs_all"
))]
pub(crate) fn normalize<F: Scalar>(
    mut value: F,
    column: usize,
    centers: Option<&[F]>,
    scales: Option<&[F]>,
) -> F {
    if let Some(c) = centers {
        value = value - c[column];
    }
    if let Some(s) = scales {
        value = value / s[column];
    }
    value
}

#[cfg(any(feature = "faer_all", feature = "nalgebra_all", feature = "sprs_all"))]
mod sparse {
    use super::{normalize, validate};
    use crate::{MatrixWrite, Scalar, SparseColumns, VectorView};

    const TILE: usize = 32;

    struct CachedColumn<'a, F> {
        rows: &'a [usize],
        raw: &'a [F],
        normalized: Vec<F>,
        background: F,
        safe: bool,
    }

    impl<F: Scalar> CachedColumn<'_, F> {
        fn value(
            &self,
            index: usize,
            column: usize,
            centers: Option<&[F]>,
            scales: Option<&[F]>,
        ) -> F {
            self.normalized
                .as_slice()
                .get(index)
                .copied()
                .unwrap_or_else(|| normalize(self.raw[index], column, centers, scales))
        }
    }

    struct Panel<F> {
        values: Vec<F>,
        cursors: [usize; TILE],
        first: usize,
        count: usize,
    }

    impl<F: Scalar> Panel<F> {
        fn new() -> Self {
            Self {
                values: vec![F::zero(); TILE * TILE],
                cursors: [0; TILE],
                first: 0,
                count: 0,
            }
        }

        fn reset(&mut self, first: usize, count: usize) {
            self.first = first;
            self.count = count;
            self.cursors.fill(0);
        }

        fn fill<M: SparseColumns<F> + ?Sized>(
            &mut self,
            matrix: &M,
            row: usize,
            rows: usize,
            centers: Option<&[F]>,
            scales: Option<&[F]>,
        ) {
            for a in 0..self.count {
                let column = self.first + a;
                let background = normalize(F::zero(), column, centers, scales);
                for i in 0..rows {
                    self.values[i * TILE + a] = background;
                }
                let (indices, values) = matrix.sparse_column(column);
                let cursor = &mut self.cursors[a];
                while *cursor < indices.len() && indices[*cursor] < row + rows {
                    self.values[(indices[*cursor] - row) * TILE + a] =
                        normalize(values[*cursor], column, centers, scales);
                    *cursor += 1;
                }
            }
        }
    }

    fn tiled_gram<F, M, W, O>(
        matrix: &M,
        weights: &W,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut O,
    ) where
        F: Scalar,
        M: SparseColumns<F> + ?Sized,
        W: VectorView<F> + ?Sized,
        O: MatrixWrite<F> + ?Sized,
    {
        let (n, p) = (matrix.nrows(), matrix.ncols());
        let mut left = Panel::new();
        let mut right = Panel::new();
        let mut block = [F::zero(); TILE * TILE];
        for j in (0..p).step_by(TILE) {
            for k in (j..p).step_by(TILE) {
                left.reset(j, (p - j).min(TILE));
                right.reset(k, (p - k).min(TILE));
                block.fill(F::zero());
                for row in (0..n).step_by(TILE) {
                    let rows = (n - row).min(TILE);
                    left.fill(matrix, row, rows, centers, scales);
                    right.fill(matrix, row, rows, centers, scales);
                    for i in 0..rows {
                        let weight = weights.get(row + i);
                        let rhs = &right.values[i * TILE..][..right.count];
                        for a in 0..left.count {
                            let lhs = left.values[i * TILE + a] * weight;
                            let destination = &mut block[a * TILE..][..right.count];
                            // Independent coefficient accumulators allow SIMD
                            // without changing the summation order within a cell.
                            for (value, &y) in destination.iter_mut().zip(rhs) {
                                *value = *value + lhs * y;
                            }
                        }
                    }
                }
                for a in 0..left.count {
                    for b in 0..right.count {
                        if j + a <= k + b {
                            let value = block[a * TILE + b];
                            out.set(j + a, k + b, value);
                            if j + a != k + b {
                                out.set(k + b, j + a, value);
                            }
                        }
                    }
                }
            }
        }
    }

    // Range sums use additions only. Subtracting a nearly equal stored-weight
    // sum from the total could discard the entire implicit-zero contribution.
    pub(crate) struct WeightSums<F> {
        nodes: Vec<F>,
        len: usize,
        ranges: Option<Vec<Vec<F>>>,
    }

    impl<F: Scalar> WeightSums<F> {
        pub(crate) fn new<W: VectorView<F> + ?Sized>(weights: &W) -> Self {
            let len = weights.len();
            let mut nodes = vec![F::zero(); 2 * len];
            for i in 0..len {
                nodes[len + i] = weights.get(i);
            }
            for i in (1..len).rev() {
                nodes[i] = nodes[2 * i] + nodes[2 * i + 1];
            }
            Self {
                nodes,
                len,
                ranges: None,
            }
        }

        pub(crate) fn new_fast<W: VectorView<F> + ?Sized>(weights: &W) -> Self {
            let mut sums = Self::new(weights);
            let n = sums.len;
            if n <= 1 {
                return sums;
            }
            let levels = (usize::BITS - (n - 1).leading_zeros()) as usize;
            // The disjoint table removes per-gap tree walks when sparse column
            // pairs make many range queries. Keep its workspace bounded.
            if n.checked_mul(levels)
                .and_then(|entries| entries.checked_mul(std::mem::size_of::<F>()))
                .is_none_or(|bytes| bytes > 64 * 1024 * 1024)
            {
                return sums;
            }
            let mut ranges = Vec::with_capacity(levels);
            for level in 0..levels {
                let half = 1usize << level;
                let mut values = vec![F::zero(); n];
                for base in (0..n).step_by(2 * half) {
                    let middle = (base + half).min(n);
                    let end = (base + 2 * half).min(n);
                    let mut total = F::zero();
                    for i in (base..middle).rev() {
                        total = weights.get(i) + total;
                        values[i] = total;
                    }
                    total = F::zero();
                    for (i, value) in values.iter_mut().enumerate().take(end).skip(middle) {
                        total = total + weights.get(i);
                        *value = total;
                    }
                }
                ranges.push(values);
            }
            sums.ranges = Some(ranges);
            sums
        }

        pub(crate) fn is_finite(&self) -> bool {
            self.nodes.iter().all(|x| x.is_finite())
                && self
                    .ranges
                    .as_ref()
                    .is_none_or(|ranges| ranges.iter().flatten().all(|x| x.is_finite()))
        }

        pub(crate) fn sum(&self, start: usize, end: usize) -> F {
            if let Some(ranges) = &self.ranges {
                if start == end {
                    return F::zero();
                }
                if start + 1 == end {
                    return self.nodes[self.len + start];
                }
                let level = (usize::BITS - 1 - (start ^ (end - 1)).leading_zeros()) as usize;
                return ranges[level][start] + ranges[level][end - 1];
            }
            let (mut l, mut r) = (start + self.len, end + self.len);
            let mut sum = F::zero();
            while l < r {
                if l % 2 == 1 {
                    sum = sum + self.nodes[l];
                    l += 1;
                }
                if r % 2 == 1 {
                    r -= 1;
                    sum = sum + self.nodes[r];
                }
                l /= 2;
                r /= 2;
            }
            sum
        }
    }

    pub(crate) fn sparse_gram<F, M, W, O>(
        matrix: &M,
        weights: &W,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut O,
    ) where
        F: Scalar,
        M: SparseColumns<F> + ?Sized,
        W: VectorView<F> + ?Sized,
        O: MatrixWrite<F> + ?Sized,
    {
        let (n, p) = (matrix.nrows(), matrix.ncols());
        validate(n, p, weights, centers, scales, out);
        if p == 0 {
            return;
        }
        let finite_weights = (0..n).all(|i| weights.get(i).is_finite());
        let max_weight = (0..n).fold(F::zero(), |m, i| m.max(weights.get(i).abs()));
        let nnz: usize = (0..p).map(|j| matrix.sparse_column(j).0.len()).sum();
        // Limit the normalized-value cache to sparse input. Dense or nearly
        // dense fallback cases must not duplicate the entire design matrix.
        let cache_normalized = nnz <= n.saturating_mul(p) / 4;
        let mut columns: Vec<_> = (0..p)
            .map(|j| {
                let (rows, values) = matrix.sparse_column(j);
                let background = normalize(F::zero(), j, centers, scales);
                let normalized: Vec<_> = if cache_normalized {
                    values
                        .iter()
                        .map(|&x| normalize(x, j, centers, scales))
                        .collect()
                } else {
                    Vec::new()
                };
                let safe = finite_weights
                    && (background * max_weight).is_finite()
                    && rows.windows(2).all(|pair| pair[0] < pair[1])
                    && rows.iter().enumerate().all(|(a, &i)| {
                        (normalized
                            .as_slice()
                            .get(a)
                            .copied()
                            .unwrap_or_else(|| normalize(values[a], j, centers, scales))
                            * weights.get(i))
                        .is_finite()
                    });
                CachedColumn {
                    rows,
                    raw: values,
                    normalized,
                    background,
                    safe,
                }
            })
            .collect();
        let centered = columns.iter().any(|column| column.background != F::zero());
        // At moderate densities, bounded panels avoid a range-sum query for
        // nearly every stored entry. Very sparse matrices retain sparse work.
        if centered
            && n > 0
            && p >= 16
            && columns.iter().all(|column| column.safe)
            && nnz as f64 / (n as f64 * p as f64) >= 0.02
        {
            tiled_gram(matrix, weights, centers, scales, out);
            return;
        }
        let weight_sums = centered.then(|| {
            if finite_weights && p >= 16 && n >= 256 && nnz as f64 / (n as f64 * p as f64) < 0.02 {
                WeightSums::new_fast(weights)
            } else {
                WeightSums::new(weights)
            }
        });
        if weight_sums.as_ref().is_some_and(|tree| !tree.is_finite()) {
            for column in &mut columns {
                column.safe = false;
            }
        }
        let mut scratch: Option<(Vec<F>, Vec<F>)> = None;
        for j in 0..p {
            for k in j..p {
                let left = &columns[j];
                let right = &columns[k];
                let (jr, kr) = (left.rows, right.rows);
                let (bj, bk) = (left.background, right.background);
                let mut sum = F::zero();
                let mut fast = left.safe && right.safe && ((bj * max_weight) * bk).is_finite();
                if fast && bj == F::zero() && bk == F::zero() {
                    let (mut a, mut b) = (0, 0);
                    while a < jr.len() && b < kr.len() {
                        match jr[a].cmp(&kr[b]) {
                            std::cmp::Ordering::Less => a += 1,
                            std::cmp::Ordering::Greater => b += 1,
                            std::cmp::Ordering::Equal => {
                                let x = left.value(a, j, centers, scales);
                                let y = right.value(b, k, centers, scales);
                                sum = sum + (x * weights.get(jr[a])) * y;
                                a += 1;
                                b += 1;
                            }
                        }
                    }
                    fast = sum.is_finite();
                } else if fast {
                    let (mut a, mut b, mut next) = (0, 0, 0);
                    while a < jr.len() || b < kr.len() {
                        let row = jr
                            .get(a)
                            .copied()
                            .unwrap_or(n)
                            .min(kr.get(b).copied().unwrap_or(n));
                        if next < row && bj != F::zero() && bk != F::zero() {
                            sum = sum + (bj * weight_sums.as_ref().unwrap().sum(next, row)) * bk;
                        }
                        let x = if jr.get(a) == Some(&row) {
                            let x = left.value(a, j, centers, scales);
                            a += 1;
                            x
                        } else {
                            bj
                        };
                        let y = if kr.get(b) == Some(&row) {
                            let y = right.value(b, k, centers, scales);
                            b += 1;
                            y
                        } else {
                            bk
                        };
                        sum = sum + (x * weights.get(row)) * y;
                        next = row + 1;
                    }
                    if next < n && bj != F::zero() && bk != F::zero() {
                        sum = sum + (bj * weight_sums.as_ref().unwrap().sum(next, n)) * bk;
                    }
                    fast = sum.is_finite();
                }
                if !fast {
                    // Two working columns retain IEEE operations on implicit
                    // zeros and combine duplicate raw entries before centering.
                    let (x, y) =
                        scratch.get_or_insert_with(|| (vec![F::zero(); n], vec![F::zero(); n]));
                    x.fill(F::zero());
                    y.fill(F::zero());
                    for (&i, &v) in jr.iter().zip(left.raw) {
                        x[i] = x[i] + v;
                    }
                    for (&i, &v) in kr.iter().zip(right.raw) {
                        y[i] = y[i] + v;
                    }
                    sum = (0..n)
                        .map(|i| {
                            (normalize(x[i], j, centers, scales) * weights.get(i))
                                * normalize(y[i], k, centers, scales)
                        })
                        .sum();
                }
                out.set(j, k, sum);
                if j != k {
                    out.set(k, j, sum);
                }
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::WeightSums;

        #[test]
        fn fast_weight_ranges_match_direct_sums() {
            for n in [1, 2, 3, 7, 16, 35, 100] {
                let weights: Vec<_> = (0..n).map(|i| (i % 7) as f64 - 2.0).collect();
                let sums = WeightSums::new_fast(&weights);
                for start in 0..=n {
                    for end in start..=n {
                        let expected: f64 = weights[start..end].iter().sum();
                        assert_eq!(sums.sum(start, end), expected);
                    }
                }
            }
            let mut weights = vec![1.0_f64; 1_000];
            weights[0] = 1e16;
            let sums = WeightSums::new_fast(&weights);
            assert_eq!(sums.sum(1, 1_000), 999.0);

            let mut weights = vec![1.0_f32; 1_000];
            weights[0] = 1e8;
            let sums = WeightSums::new_fast(&weights);
            assert_eq!(sums.sum(1, 1_000), 999.0);
        }
    }
}

#[cfg(any(feature = "faer_all", feature = "nalgebra_all", feature = "sprs_all"))]
pub(crate) use sparse::{WeightSums, sparse_gram};
