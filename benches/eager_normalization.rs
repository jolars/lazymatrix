//! Density sweeps with parameter fitting, materialization, and products timed separately.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use lazymatrix::{
    ColumnStats, LazyMatrix, MatTransposeVecInto, MatVecInto, MaterializeDense, MatrixOwned,
};

#[path = "../tests/common/backend_aliases.rs"]
mod backend_aliases;
#[path = "common/normalization.rs"]
mod normalization;

use backend_aliases::ndarray;
use normalization::{BenchVector, Fixture, NORMALIZATIONS};

fn bench_backend<M, D, V>(c: &mut Criterion, fixture: &Fixture, backend: &str, source: &M)
where
    M: ColumnStats<f64> + MaterializeDense<f64> + MatVecInto<V> + MatTransposeVecInto<V>,
    D: MatrixOwned<f64> + MatVecInto<V> + MatTransposeVecInto<V>,
    V: BenchVector,
{
    for (normalization, spec) in NORMALIZATIONS {
        let lazy = LazyMatrix::new(source, spec).unwrap();
        let eager = lazy.to_eager::<D>().unwrap();
        let (x, u, mut y, mut t) = fixture.vectors::<V>();
        let (_, _, mut eager_y, mut eager_t) = fixture.vectors::<V>();
        lazy.matvec_into(&x, &mut y).unwrap();
        lazy.mat_transpose_vec_into(&u, &mut t).unwrap();
        eager.matvec_into(&x, &mut eager_y).unwrap();
        eager.mat_transpose_vec_into(&u, &mut eager_t).unwrap();
        for (actual, expected) in [(&y, &eager_y), (&t, &eager_t)] {
            for i in 0..actual.len() {
                approx::assert_relative_eq!(
                    actual.get(i),
                    expected.get(i),
                    epsilon = 1e-8,
                    max_relative = 1e-9
                );
            }
        }
        let mut group = c.benchmark_group(fixture.name(backend, normalization));
        group.bench_function("fit_parameters", |b| {
            b.iter(|| black_box(LazyMatrix::new(black_box(source), spec).unwrap()));
        });
        group.bench_function("materialize", |b| {
            b.iter(|| black_box(lazy.to_eager::<D>().unwrap()));
        });
        macro_rules! products {
            ($prefix:literal, $matrix:expr) => {
                group.bench_function(concat!($prefix, "_forward"), |b| {
                    b.iter(|| {
                        $matrix.matvec_into(black_box(&x), &mut y).unwrap();
                        black_box(&y);
                    });
                });
                group.bench_function(concat!($prefix, "_transpose"), |b| {
                    b.iter(|| {
                        $matrix
                            .mat_transpose_vec_into(black_box(&u), &mut t)
                            .unwrap();
                        black_box(&t);
                    });
                });
            };
        }
        products!("raw", source);
        products!("lazy", lazy);
        products!("eager", eager);
        group.finish();
    }
}

fn benchmark(c: &mut Criterion) {
    for fixture in normalization::fixtures() {
        bench_backend::<_, ndarray::Array2<f64>, ndarray::Array1<f64>>(
            c,
            &fixture,
            "sprs_to_ndarray",
            &fixture.sparse,
        );
        #[cfg(feature = "faer_all")]
        {
            use backend_aliases::faer;
            let source = fixture.faer();
            bench_backend::<_, faer::Mat<f64>, faer::Col<f64>>(
                c,
                &fixture,
                "faer_csc_to_dense",
                &source,
            );
        }
    }
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
