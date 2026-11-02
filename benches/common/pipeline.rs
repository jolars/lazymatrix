//! Paired eager and lazy pipeline fixtures shared by timing and allocation runs.
#![allow(dead_code)]

use crate::backend_aliases::{
    ndarray::{Array1, Array2},
    sprs::CsMat,
};
use crate::normalization::Fixture;
use lazymatrix::{
    Centering, ColumnNormalization, DenseNormalize, LazyMatrix, MatTransposeVecInto, MatVecInto,
    RowNormalization, Scaling,
};

pub fn fixtures() -> impl Iterator<Item = Fixture> {
    let rows = if std::env::var_os("LAZYMATRIX_BENCH_SMOKE").is_some() {
        [256, 1024]
    } else {
        [2048, 8192]
    };
    rows.into_iter().flat_map(|rows| {
        [0.001, 0.01]
            .into_iter()
            .map(move |density| Fixture::new(rows, 128, density))
    })
}

pub fn columns() -> lazymatrix::NormalizationChain<ColumnNormalization, ColumnNormalization> {
    ColumnNormalization::new(Centering::Mean, Scaling::L2).then(ColumnNormalization::standardize())
}

pub fn mixed() -> lazymatrix::NormalizationChain<RowNormalization, ColumnNormalization> {
    RowNormalization::standardize().then(ColumnNormalization::standardize())
}

pub fn eager_columns(source: &CsMat<f64>) -> Array2<f64> {
    let mut data = LazyMatrix::new(source, *columns().first())
        .unwrap()
        .to_eager::<Array2<f64>>()
        .unwrap()
        .into_parts()
        .0;
    let params = columns().next().fit(&data).unwrap();
    data.normalize_in_place(params.centers(), params.scales());
    data
}

pub fn dense(source: &CsMat<f64>) -> Array2<f64> {
    Array2::from_shape_fn(source.shape(), |(i, j)| {
        source.get(i, j).copied().unwrap_or(0.)
    })
}

pub fn eager_mixed(source: &Array2<f64>) -> Array2<f64> {
    let params = mixed().first().fit(source).unwrap();
    let mut data = Array2::from_shape_fn(source.dim(), |(i, j)| {
        (source[(i, j)] - params.centers().unwrap()[i]) / params.scales().unwrap()[i]
    });
    let params = mixed().next().fit(&data).unwrap();
    data.normalize_in_place(params.centers(), params.scales());
    data
}

pub fn pair<M>(source: &M, fixture: &Fixture) -> (Array1<f64>, Array1<f64>)
where
    M: MatVecInto<Array1<f64>> + MatTransposeVecInto<Array1<f64>>,
{
    let (x, u, mut y, mut t) = fixture.vectors::<Array1<f64>>();
    source.matvec_into(&x, &mut y).unwrap();
    source.mat_transpose_vec_into(&u, &mut t).unwrap();
    (y, t)
}

pub fn check(actual: &(Array1<f64>, Array1<f64>), expected: &(Array1<f64>, Array1<f64>)) {
    for (&a, &b) in actual
        .0
        .iter()
        .chain(&actual.1)
        .zip(expected.0.iter().chain(&expected.1))
    {
        if b.is_nan() {
            assert!(a.is_nan());
        } else {
            approx::assert_relative_eq!(a, b, epsilon = 1e-8, max_relative = 1e-8);
        }
    }
}
