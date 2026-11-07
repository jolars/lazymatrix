//! Fit-and-product comparisons over paired eager and lazy pipelines.
//!
//! Run with the memory companion in one compute bench reservation. See
//! benches/normalization_pipeline/README.md for commands and recorded results.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use lazymatrix::{MatTransposeVecInto, MatVecInto};
use std::hint::black_box;

#[path = "common/allocations.rs"]
mod allocations;
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
criterion_group!(
    benches,
    benchmark,
    fused_products,
    workspace_products,
    batched_products
);
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

fn workspace_products(c: &mut Criterion) {
    use backend_aliases::ndarray::Array1;
    use lazymatrix::{MatTransposeVecScaledInto, MatVecScaledInto, RowNormalization};

    fn compare(
        group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
        label: &str,
        name: &str,
        shape: (usize, usize),
        mut fused: impl FnMut(&mut Array1<f64>, &mut Array1<f64>),
        mut workspace: impl FnMut(&mut Array1<f64>, &mut Array1<f64>),
    ) {
        let mut y = Array1::from_elem(shape.0, 1.);
        let mut t = Array1::from_elem(shape.1, 1.);
        let mut expected = y.clone();
        let mut expected_t = t.clone();
        fused(&mut expected, &mut expected_t);
        workspace(&mut y, &mut t);
        pipeline::check(&(y.clone(), t.clone()), &(expected, expected_t));
        let baseline = allocations::measure(|| fused(&mut y, &mut t));
        let candidate = allocations::measure(|| workspace(&mut y, &mut t));
        if name.starts_with("dense") {
            assert_eq!(candidate, 0, "dense workspace calls must not allocate");
        }
        eprintln!("workspace_peak_bytes,{label}/{name},{baseline},{candidate}");
        group.bench_function(BenchmarkId::new(name, "fused_pair"), |b| {
            b.iter(|| {
                fused(&mut y, &mut t);
                black_box((&y, &t));
            })
        });
        group.bench_function(BenchmarkId::new(name, "workspace_pair"), |b| {
            b.iter(|| {
                workspace(&mut y, &mut t);
                black_box((&y, &t));
            })
        });
    }

    for fixture in pipeline::fixtures() {
        let label = format!(
            "workspace_pipeline/{}x{}/{}",
            fixture.rows, fixture.cols, fixture.density
        );
        let mut group = c.benchmark_group(&label);
        let (x, u, _, _) = fixture.vectors::<Array1<f64>>();
        macro_rules! compare {
            ($name:expr, $op:expr) => {{
                let op = $op;
                let mut scratch = Array1::zeros(fixture.rows);
                let mut scratch_t = Array1::zeros(fixture.cols);
                compare(
                    &mut group,
                    &label,
                    $name,
                    (fixture.rows, fixture.cols),
                    |y, t| {
                        op.matvec_scaled_into(-0.75, black_box(&x), 0.25, y)
                            .unwrap();
                        op.mat_transpose_vec_scaled_into(-0.75, black_box(&u), 0.25, t)
                            .unwrap();
                    },
                    |y, t| {
                        op.matvec_scaled_with_workspace(
                            -0.75,
                            black_box(&x),
                            0.25,
                            y,
                            &mut scratch,
                        )
                        .unwrap();
                        op.mat_transpose_vec_scaled_with_workspace(
                            -0.75,
                            black_box(&u),
                            0.25,
                            t,
                            &mut scratch_t,
                        )
                        .unwrap();
                    },
                );
            }};
        }
        let dense = pipeline::dense(&fixture.sparse);
        compare!(
            "dense_rows",
            RowNormalization::standardize().fit_apply(&dense).unwrap()
        );
        compare!("dense_mixed", pipeline::mixed().fit_apply(&dense).unwrap());
        for (orientation, source) in [
            ("csc", fixture.sparse.clone()),
            ("csr", fixture.sparse.to_csr()),
        ] {
            compare!(
                &format!("{orientation}_rows"),
                RowNormalization::standardize().fit_apply(&source).unwrap()
            );
            compare!(
                &format!("{orientation}_columns"),
                pipeline::columns().fit_apply(&source).unwrap()
            );
        }
        group.finish();
    }
}

fn batched_products(c: &mut Criterion) {
    use backend_aliases::ndarray::{Array2, ArrayView1, ArrayViewMut1};
    use lazymatrix::{MatMatInto, MatTransposeMatInto, MatrixShape, RowNormalization};

    fn compare<M>(
        group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
        label: &str,
        name: &str,
        op: &M,
    ) where
        M: MatrixShape + MatMatInto<f64> + MatTransposeMatInto<f64>,
        for<'a, 'b> M: MatVecInto<ArrayView1<'a, f64>, ArrayViewMut1<'b, f64>>
            + MatTransposeVecInto<ArrayView1<'a, f64>, ArrayViewMut1<'b, f64>>,
    {
        let k = 17;
        let rhs = Array2::from_shape_fn((op.ncols(), k), |(j, r)| (j % 7) as f64 - r as f64 / 4.);
        let u = Array2::from_shape_fn((op.nrows(), k), |(i, r)| (i % 5) as f64 / 3. - r as f64);
        let mut y = Array2::zeros((op.nrows(), k));
        let mut t = Array2::zeros((op.ncols(), k));
        let mut expected = y.clone();
        let mut expected_t = t.clone();
        let vectors = |y: &mut Array2<f64>, t: &mut Array2<f64>| {
            for r in 0..k {
                op.matvec_into(&rhs.column(r), &mut y.column_mut(r))
                    .unwrap();
                op.mat_transpose_vec_into(&u.column(r), &mut t.column_mut(r))
                    .unwrap();
            }
        };
        let batched = |y: &mut Array2<f64>, t: &mut Array2<f64>| {
            op.matmat_into(&rhs, y).unwrap();
            op.mat_transpose_mat_into(&u, t).unwrap();
        };
        vectors(&mut expected, &mut expected_t);
        batched(&mut y, &mut t);
        for (&actual, &expected) in y.iter().chain(&t).zip(expected.iter().chain(&expected_t)) {
            approx::assert_relative_eq!(actual, expected, epsilon = 1e-8, max_relative = 1e-8);
        }
        let baseline = allocations::measure(|| vectors(&mut y, &mut t));
        let candidate = allocations::measure(|| batched(&mut y, &mut t));
        eprintln!("batched_peak_bytes,{label}/{name},{baseline},{candidate}");
        group.bench_function(BenchmarkId::new(name, "vector_pair"), |b| {
            b.iter(|| {
                vectors(&mut y, &mut t);
                black_box((&y, &t));
            })
        });
        group.bench_function(BenchmarkId::new(name, "batched_pair"), |b| {
            b.iter(|| {
                batched(&mut y, &mut t);
                black_box((&y, &t));
            })
        });
    }
    for fixture in pipeline::fixtures() {
        let label = format!(
            "batched_pipeline/{}x{}/{}/17",
            fixture.rows, fixture.cols, fixture.density
        );
        let mut group = c.benchmark_group(&label);
        let source = pipeline::dense(&fixture.sparse);
        compare(
            &mut group,
            &label,
            "dense_rows",
            &RowNormalization::standardize().fit_apply(&source).unwrap(),
        );
        compare(
            &mut group,
            &label,
            "dense_mixed",
            &pipeline::mixed().fit_apply(&source).unwrap(),
        );
        for (orientation, source) in [
            ("csc", fixture.sparse.clone()),
            ("csr", fixture.sparse.to_csr()),
        ] {
            compare(
                &mut group,
                &label,
                &format!("{orientation}_rows"),
                &RowNormalization::standardize().fit_apply(&source).unwrap(),
            );
            compare(
                &mut group,
                &label,
                &format!("{orientation}_columns"),
                &pipeline::columns().fit_apply(&source).unwrap(),
            );
        }
        group.finish();
    }
}
