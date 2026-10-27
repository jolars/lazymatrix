// Timing and allocation executables use different parts of this shared harness.
#![allow(dead_code)]

use lazymatrix::{
    ColumnStats, LazyMatrix, MatTransposeVecInto, MatVecInto, MaterializeDense, MatrixOwned,
};

use crate::normalization::{BenchVector, Fixture, NORMALIZATIONS, selected_shapes};
#[path = "../../examples/common/least_squares.rs"]
pub mod solver;
use solver::{Fit, Options, solve};

pub fn fixtures() -> impl Iterator<Item = Fixture> {
    let sizes = if std::env::var_os("LAZYMATRIX_BENCH_SMOKE").is_some() {
        vec![(256, 16)]
    } else {
        vec![(2_048, 128), (8_192, 512), (16_384, 1_024)]
    };
    selected_shapes(sizes).flat_map(|(rows, cols)| {
        [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
            .into_iter()
            .map(move |density| Fixture::new(rows, cols, density))
    })
}

pub fn response<V: BenchVector>(fixture: &Fixture) -> V {
    V::owned_from_fn(fixture.rows, |i| {
        (i as f64 * 0.13).sin() + 0.5 * (i as f64 * 0.037).cos()
    })
}

pub fn fit<M, D, V>(source: &M, response: &V, eager: bool) -> Fit<V>
where
    M: ColumnStats<f64> + MaterializeDense<f64> + MatVecInto<V> + MatTransposeVecInto<V>,
    D: MatrixOwned<f64> + MatVecInto<V> + MatTransposeVecInto<V>,
    V: BenchVector,
{
    let lazy = LazyMatrix::new(source, NORMALIZATIONS[0].1).unwrap();
    if eager {
        let matrix = lazy.to_eager::<D>().unwrap();
        drop(lazy);
        solve(&matrix, response, Options::default()).unwrap()
    } else {
        solve(&lazy, response, Options::default()).unwrap()
    }
}

pub fn check<V: BenchVector>(lazy: &Fit<V>, eager: &Fit<V>) {
    for result in [lazy, eager] {
        assert!(result.iterations > 0);
        assert!(result.gradient_norm <= result.gradient_threshold);
    }
    approx::assert_relative_eq!(
        lazy.objective,
        eager.objective,
        epsilon = 1e-10,
        max_relative = 1e-9
    );
    for j in 0..lazy.coefficients.len() {
        approx::assert_relative_eq!(
            lazy.coefficients.get(j),
            eager.coefficients.get(j),
            epsilon = 1e-7,
            max_relative = 1e-6
        );
    }
}
