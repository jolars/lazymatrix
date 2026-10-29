use super::Request;
use crate::{ColumnNormalizationStats, Scalar};

#[cfg(any(
    feature = "faer_all",
    feature = "nalgebra_all",
    feature = "ndarray_all"
))]
pub(crate) fn dense_columns<M: crate::MatrixRead<F> + ?Sized, F: Scalar>(
    matrix: &M,
    request: Request,
) -> ColumnNormalizationStats<F> {
    let positions = request.positions(matrix.nrows());
    let mut output = request.output(matrix.ncols());
    let mut column = Vec::new();
    for j in 0..matrix.ncols() {
        column.clear();
        column.extend((0..matrix.nrows()).map(|i| matrix.get(i, j)));
        let mut selected = [F::nan(); 6];
        if !column.is_empty() && !column.iter().any(|value| value.is_nan()) {
            column.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            for (i, position) in positions.iter().enumerate() {
                selected[2 * i] = column[position.lower];
                selected[2 * i + 1] = column[position.upper];
            }
        }
        request.push(&mut output, &positions, &selected);
    }
    output
}

#[cfg(any(feature = "faer_all", feature = "nalgebra_all"))]
pub(crate) fn borrowed_columns<M, F>(matrix: &M, request: Request) -> ColumnNormalizationStats<F>
where
    M: crate::RawColumns<F> + ?Sized,
    F: Scalar,
{
    use crate::RawColumn;
    columns(matrix.nrows(), matrix.ncols(), request, |j, entries| {
        matrix
            .raw_column(j)
            .for_each_stored(|i, value| entries.push((i, value)));
    })
}

#[cfg(any(
    test,
    feature = "faer_all",
    feature = "nalgebra_all",
    feature = "sprs_all"
))]
pub(crate) fn columns<F: Scalar>(
    nrows: usize,
    ncols: usize,
    request: Request,
    mut collect: impl FnMut(usize, &mut Vec<(usize, F)>),
) -> ColumnNormalizationStats<F> {
    let positions = request.positions(nrows);
    let mut output = request.output(ncols);
    let mut entries: Vec<(usize, F)> = Vec::new();
    for j in 0..ncols {
        entries.clear();
        collect(j, &mut entries);
        let mut selected = [F::nan(); 6];
        if nrows != 0 {
            if entries.windows(2).any(|pair| pair[0].0 >= pair[1].0) {
                // Stable row sorting preserves each duplicate cell's summation order.
                entries.sort_by_key(|&(row, _)| row);
                let mut length = 0;
                for i in 0..entries.len() {
                    let (row, value) = entries[i];
                    if length != 0 && entries[length - 1].0 == row {
                        entries[length - 1].1 = entries[length - 1].1 + value;
                    } else {
                        entries[length] = (row, value);
                        length += 1;
                    }
                }
                entries.truncate(length);
            }
            if !entries.iter().any(|&(_, value)| value.is_nan()) {
                let missing = nrows - entries.len();
                entries.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                let negative = entries.partition_point(|&(_, value)| value < F::zero());
                let at = |rank: usize| {
                    if rank < negative {
                        entries[rank].1
                    } else if rank < negative + missing {
                        F::zero()
                    } else {
                        entries[rank - missing].1
                    }
                };
                for (i, position) in positions.iter().enumerate() {
                    selected[2 * i] = at(position.lower);
                    selected[2 * i + 1] = at(position.upper);
                }
            }
        }
        request.push(&mut output, &positions, &selected);
    }
    output
}
