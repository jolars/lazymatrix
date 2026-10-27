//! Complete ridge fits, including normalization, materialization, and solver setup.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use lazymatrix::{ColumnStats, MatTransposeVecInto, MatVecInto, MaterializeDense, MatrixOwned};

#[path = "../tests/common/backend_aliases.rs"]
mod backend_aliases;
#[path = "common/least_squares.rs"]
mod least_squares;
#[path = "common/normalization.rs"]
mod normalization;

use backend_aliases::ndarray;
use normalization::{BenchVector, Fixture};

fn bench_backend<M, D, V>(c: &mut Criterion, fixture: &Fixture, backend: &str, source: &M)
where
    M: ColumnStats<f64> + MaterializeDense<f64> + MatVecInto<V> + MatTransposeVecInto<V>,
    D: MatrixOwned<f64> + MatVecInto<V> + MatTransposeVecInto<V>,
    V: BenchVector,
{
    let response = least_squares::response::<V>(fixture);
    let lazy = least_squares::fit::<_, D, _>(source, &response, false);
    let eager = least_squares::fit::<_, D, _>(source, &response, true);
    least_squares::check(&lazy, &eager);
    drop((lazy, eager));
    let mut group = c.benchmark_group(format!(
        "least_squares/{backend}/{}x{}/{}",
        fixture.rows, fixture.cols, fixture.density
    ));
    for (strategy, eager) in [("lazy", false), ("eager", true)] {
        group.bench_function(strategy, |b| {
            b.iter(|| {
                black_box(least_squares::fit::<_, D, _>(
                    black_box(source),
                    black_box(&response),
                    eager,
                ))
            });
        });
    }
    group.finish();
}

fn benchmark(c: &mut Criterion) {
    for fixture in least_squares::fixtures() {
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
