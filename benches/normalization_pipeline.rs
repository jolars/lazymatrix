//! Fit-and-product comparisons over paired eager and lazy pipelines.
//!
//! Run with the memory companion in one compute bench reservation. See
//! benches/normalization_pipeline/README.md for commands and recorded results.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use lazymatrix::{MatTransposeVecInto, MatVecInto};
use std::hint::black_box;

#[path = "../tests/common/backend_aliases.rs"]
mod backend_aliases;
#[path = "common/normalization.rs"]
mod normalization;
#[path = "common/pipeline.rs"]
mod pipeline;

fn benchmark(c: &mut Criterion) {
    for fixture in pipeline::fixtures() {
        let mut group = c.benchmark_group(format!(
            "pipeline/{}x{}/{}",
            fixture.rows, fixture.cols, fixture.density
        ));
        for (orientation, source) in [
            ("csc", fixture.sparse.clone()),
            ("csr", fixture.sparse.to_csr()),
        ] {
            let lazy = pipeline::columns().fit_apply(&source).unwrap();
            let eager = pipeline::eager_columns(&source);
            pipeline::check(
                &pipeline::pair(&lazy, &fixture),
                &pipeline::pair(&eager, &fixture),
            );
            group.bench_function(
                BenchmarkId::new("columns_fit_pair_lazy", orientation),
                |b| {
                    b.iter(|| {
                        let op = pipeline::columns().fit_apply(black_box(&source)).unwrap();
                        black_box(pipeline::pair(&op, &fixture))
                    })
                },
            );
            group.bench_function(
                BenchmarkId::new("columns_fit_pair_eager", orientation),
                |b| {
                    b.iter(|| {
                        let op = pipeline::eager_columns(black_box(&source));
                        black_box(pipeline::pair(&op, &fixture))
                    })
                },
            );
            let (x, u, mut y, mut t) = fixture.vectors::<backend_aliases::ndarray::Array1<f64>>();
            group.bench_function(BenchmarkId::new("columns_pair_lazy", orientation), |b| {
                b.iter(|| {
                    lazy.matvec_into(black_box(&x), &mut y).unwrap();
                    lazy.mat_transpose_vec_into(black_box(&u), &mut t).unwrap();
                    black_box((&y, &t));
                })
            });
            group.bench_function(BenchmarkId::new("columns_pair_eager", orientation), |b| {
                b.iter(|| {
                    eager.matvec_into(black_box(&x), &mut y).unwrap();
                    eager.mat_transpose_vec_into(black_box(&u), &mut t).unwrap();
                    black_box((&y, &t));
                })
            });
        }
        let source = pipeline::dense(&fixture.sparse);
        let lazy = pipeline::mixed().fit_apply(&source).unwrap();
        let eager = pipeline::eager_mixed(&source);
        pipeline::check(
            &pipeline::pair(&lazy, &fixture),
            &pipeline::pair(&eager, &fixture),
        );
        group.bench_function("mixed_fit_pair_lazy", |b| {
            b.iter(|| {
                let op = pipeline::mixed().fit_apply(black_box(&source)).unwrap();
                black_box(pipeline::pair(&op, &fixture))
            })
        });
        group.bench_function("mixed_fit_pair_eager", |b| {
            b.iter(|| {
                let op = pipeline::eager_mixed(black_box(&source));
                black_box(pipeline::pair(&op, &fixture))
            })
        });
        group.finish();
    }
}
criterion_group!(benches, benchmark);
criterion_main!(benches);
