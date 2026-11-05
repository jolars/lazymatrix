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
criterion_group!(benches, benchmark, fused_products);
criterion_main!(benches);

fn fused_products(c: &mut Criterion) {
    use backend_aliases::ndarray::Array1;
    use lazymatrix::{
        MatTransposeVec, MatTransposeVecScaledInto, MatVec, MatVecScaledInto, RowNormalization,
    };
    fn compare<O>(
        group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
        name: &str,
        op: &O,
        x: &Array1<f64>,
        u: &Array1<f64>,
    ) where
        O: MatVec<Array1<f64>>
            + MatTransposeVec<Array1<f64>>
            + MatVecScaledInto<Array1<f64>, Array1<f64>, f64>
            + MatTransposeVecScaledInto<Array1<f64>, Array1<f64>, f64>,
    {
        let mut y = Array1::from_elem(u.len(), 1.);
        let mut t = Array1::from_elem(x.len(), 1.);
        let forward = op.matvec(x).unwrap();
        let transpose = op.mat_transpose_vec(u).unwrap();
        op.matvec_scaled_into(-0.75, x, 0.25, &mut y).unwrap();
        op.mat_transpose_vec_scaled_into(-0.75, u, 0.25, &mut t)
            .unwrap();
        for (&actual, &product) in y.iter().chain(&t).zip(forward.iter().chain(&transpose)) {
            approx::assert_relative_eq!(
                actual,
                -0.75 * product + 0.25,
                epsilon = 1e-9,
                max_relative = 1e-9
            );
        }
        group.bench_function(BenchmarkId::new(name, "manual_pair"), |b| {
            b.iter(|| {
                let forward = op.matvec(black_box(x)).unwrap();
                let transpose = op.mat_transpose_vec(black_box(u)).unwrap();
                for (old, value) in y.iter_mut().zip(forward.iter()) {
                    *old = -0.75 * value + 0.25 * (*old);
                }
                for (old, value) in t.iter_mut().zip(transpose.iter()) {
                    *old = -0.75 * value + 0.25 * (*old);
                }
                black_box((&y, &t));
            })
        });
        group.bench_function(BenchmarkId::new(name, "fused_pair"), |b| {
            b.iter(|| {
                op.matvec_scaled_into(-0.75, black_box(x), 0.25, &mut y)
                    .unwrap();
                op.mat_transpose_vec_scaled_into(-0.75, black_box(u), 0.25, &mut t)
                    .unwrap();
                black_box((&y, &t));
            })
        });
    }
    for fixture in pipeline::fixtures() {
        let mut group = c.benchmark_group(format!(
            "fused_pipeline/{}x{}/{}",
            fixture.rows, fixture.cols, fixture.density
        ));
        let (x, u, _, _) = fixture.vectors::<Array1<f64>>();
        let source = pipeline::dense(&fixture.sparse);
        compare(
            &mut group,
            "dense_rows",
            &RowNormalization::standardize().fit_apply(&source).unwrap(),
            &x,
            &u,
        );
        compare(
            &mut group,
            "dense_mixed",
            &pipeline::mixed().fit_apply(&source).unwrap(),
            &x,
            &u,
        );
        for (name, source) in [
            ("csc_columns", fixture.sparse.clone()),
            ("csr_columns", fixture.sparse.to_csr()),
        ] {
            compare(
                &mut group,
                name,
                &pipeline::columns().fit_apply(source).unwrap(),
                &x,
                &u,
            );
        }
        group.finish();
    }
}
