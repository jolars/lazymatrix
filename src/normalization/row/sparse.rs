//! Reuse column kernels through a transposed traversal, without changing storage.

use crate::normalization::pipeline::{self, Columns, Execute, Source, source::Layout};
use crate::{
    Centering, ColumnNormalization, MatrixErrorType, MatrixShape, RowNormalization,
    RowNormalizationStats, Scalar, Scaling, VectorView, VectorViewMut,
};

struct Transposed<'a, M>(&'a M);

impl<M: MatrixShape> MatrixShape for Transposed<'_, M> {
    fn nrows(&self) -> usize {
        self.0.ncols()
    }
    fn ncols(&self) -> usize {
        self.0.nrows()
    }
}
impl<M: MatrixErrorType> MatrixErrorType for Transposed<'_, M> {
    type Error = M::Error;
}
impl<M: Source<F>, F: Scalar> Source<F> for Transposed<'_, M> {
    fn layout(&self) -> Layout {
        match self.0.layout() {
            Layout::SparseColumns => Layout::SparseRows,
            Layout::SparseRows => Layout::SparseColumns,
            Layout::Full => Layout::Full,
        }
    }
    fn visit(&self, mut visit: impl FnMut(usize, usize, F)) -> Result<(), Self::Error> {
        self.0.visit(|i, j, value| visit(j, i, value))
    }
}

pub(crate) fn stats<M: Source<F>, F: Scalar>(
    matrix: &M,
    spec: RowNormalization,
) -> Result<RowNormalizationStats<F>, M::Error> {
    if matches!(matrix.layout(), Layout::SparseRows) {
        return buffered_stats(matrix, spec);
    }
    pipeline::raw_stats(
        &Transposed(matrix),
        ColumnNormalization::new(spec.center, spec.scale),
    )
}

fn buffered_stats<M: Source<F>, F: Scalar>(
    matrix: &M,
    spec: RowNormalization,
) -> Result<RowNormalizationStats<F>, M::Error> {
    let mut centers = (spec.center != Centering::None).then(|| Vec::with_capacity(matrix.nrows()));
    let mut scales = (spec.scale != Scaling::None).then(|| Vec::with_capacity(matrix.nrows()));
    if centers.is_none() && scales.is_none() {
        return Ok((None, None));
    }
    let mut values = Vec::new();
    let mut current = 0;
    let mut finish = |values: &mut Vec<F>| {
        let (center, scale) = fit_row(values, matrix.ncols(), spec);
        if let Some(centers) = &mut centers {
            centers.push(center.unwrap());
        }
        if let Some(scales) = &mut scales {
            scales.push(scale.unwrap());
        }
        values.clear();
    };
    if matrix.nrows() != 0 && matrix.ncols() != 0 {
        matrix.visit(|i, _, value| {
            while current < i {
                finish(&mut values);
                current += 1;
            }
            values.push(value);
        })?;
    }
    while current < matrix.nrows() {
        finish(&mut values);
        current += 1;
    }
    Ok((centers, scales))
}

fn fit_row<F: Scalar>(
    values: &mut [F],
    ncols: usize,
    spec: RowNormalization,
) -> (Option<F>, Option<F>) {
    let missing = ncols - values.len();
    let mean = if spec.center == Centering::Mean || matches!(spec.scale, Scaling::Sd { .. }) {
        values.iter().copied().sum::<F>() / F::from_usize(ncols).unwrap()
    } else {
        F::zero()
    };
    let (min, max) = if spec.center == Centering::Min || spec.scale == Scaling::Range {
        super::extrema(values.iter().copied().chain((missing > 0).then(F::zero)))
    } else {
        (F::zero(), F::zero())
    };
    let (median, range) = crate::quantiles::sparse_row_quantiles(values, ncols, spec);
    let center = match spec.center {
        Centering::None => None,
        Centering::Mean => Some(mean),
        Centering::Min => Some(min),
        Centering::Median => median,
    };
    let centered = |v| center.map_or(v, |c| v - c);
    let scale = match spec.scale {
        Scaling::None => None,
        Scaling::QuantileRange(_) => range,
        Scaling::Range => Some(max - min),
        Scaling::Sd { df } => {
            let squares = sum_with_zeros(values, missing, |v| (v - mean) * (v - mean));
            Some((squares / super::super::sd_divisor::<F>(ncols, df)).sqrt())
        }
        Scaling::L1 => Some(sum_with_zeros(values, missing, |v| centered(v).abs())),
        Scaling::L2 => Some(sum_with_zeros(values, missing, |v| centered(v) * centered(v)).sqrt()),
        Scaling::MaxAbs => Some(
            values
                .iter()
                .copied()
                .chain((missing > 0).then(F::zero))
                .map(|v| centered(v).abs())
                .fold(F::zero(), |max, v| {
                    if max.is_nan() || v.is_nan() {
                        F::nan()
                    } else {
                        max.max(v)
                    }
                }),
        ),
    };
    (center, scale)
}

fn sum_with_zeros<F: Scalar>(values: &[F], missing: usize, transform: impl Fn(F) -> F) -> F {
    let sum = values.iter().copied().map(&transform).sum::<F>();
    if missing == 0 {
        sum
    } else {
        sum + F::from_usize(missing).unwrap() * transform(F::zero())
    }
}

pub(crate) fn product<M, F, X, Y>(
    matrix: &M,
    transpose: bool,
    x: &X,
    centers: Option<&[F]>,
    scales: Option<&[F]>,
    out: &mut Y,
) -> Result<(), M::Error>
where
    M: Source<F>,
    F: Scalar,
    X: VectorView<F> + ?Sized,
    Y: VectorViewMut<F> + ?Sized,
{
    crate::row_normalization::validate(matrix, transpose, x, centers, scales, out);
    let params = super::super::ColumnNormalizationParams {
        ncols: matrix.nrows(),
        centers,
        scales,
        scalar: std::marker::PhantomData,
    };
    Columns::product(&Transposed(matrix), &params, !transpose, x, out)
}
