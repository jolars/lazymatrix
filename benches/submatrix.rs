//! Rectangular products versus native dense views and direct sparse filtering.

use criterion::{Criterion, criterion_group, criterion_main};
use lazymatrix::{LazyMatrix, MatTransposeVecInto, MatVecInto, SprsCsc};
use std::hint::black_box;
use std::ops::Range;

#[path = "common/allocations.rs"]
mod allocations;
#[path = "../tests/common/backend_aliases.rs"]
mod backend_aliases;
use backend_aliases::{ndarray, sprs};
use ndarray::{Array1, Array2, s};

fn filter_csc(
    matrix: &sprs::CsMat<f64>,
    rows: Range<usize>,
    columns: Range<usize>,
    x: &Array1<f64>,
    out: &mut Array1<f64>,
    transpose: bool,
) {
    if transpose {
        let sum = x.sum();
        for (j, column) in columns.enumerate() {
            let raw: f64 = matrix
                .outer_view(column)
                .unwrap()
                .iter()
                .filter(|(i, _)| rows.contains(i))
                .map(|(i, &v)| v * x[i - rows.start])
                .sum();
            out[j] = (raw - sum) / 2.0;
        }
    } else {
        out.fill(-x.sum() / 2.0);
        for (j, column) in columns.enumerate() {
            for (i, &v) in matrix.outer_view(column).unwrap().iter() {
                if rows.contains(&i) {
                    out[i - rows.start] += v * (x[j] / 2.0);
                }
            }
        }
    }
}

fn benchmark(c: &mut Criterion) {
    let (n, p) = (4096, 256);
    let raw = Array2::from_shape_fn((n, p), |(i, j)| {
        if (i * 13 + j * 7) % 97 == 0 {
            (i + j) % 11
        } else {
            0
        }
    })
    .mapv(|v| v as f64);
    let mut triplets = sprs::TriMat::new((n, p));
    for ((i, j), &value) in raw.indexed_iter() {
        if value != 0.0 {
            triplets.add_triplet(i, j, value);
        }
    }
    let csc = triplets.to_csc();
    let dense = LazyMatrix::from_parts(raw.view(), Some(vec![1.0; p]), Some(vec![2.0; p]));
    let sparse = LazyMatrix::from_parts(
        SprsCsc::try_new(csc.view()).unwrap(),
        Some(vec![1.0; p]),
        Some(vec![2.0; p]),
    );
    eprintln!("rows,columns,operation,peak_workspace_bytes");
    for (nr, nc) in [(128, 32), (1024, 64)] {
        let rows = 17..17 + nr;
        let columns = 11..11 + nc;
        let selected_dense = dense.submatrix(rows.clone(), columns.clone());
        let selected_sparse = sparse.submatrix(rows.clone(), columns.clone());
        let native = LazyMatrix::from_parts(
            raw.slice(s![rows.clone(), columns.clone()]),
            Some(vec![1.0; nc]),
            Some(vec![2.0; nc]),
        );
        let x = Array1::from_shape_fn(nc, |j| j as f64 / nc as f64);
        let u = Array1::from_shape_fn(nr, |i| i as f64 / nr as f64);
        let mut y = Array1::zeros(nr);
        let mut t = Array1::zeros(nc);
        let mut expected_y = Array1::zeros(nr);
        let mut expected_t = Array1::zeros(nc);
        native.matvec_into(&x, &mut expected_y).unwrap();
        native.mat_transpose_vec_into(&u, &mut expected_t).unwrap();
        selected_sparse.matvec_into(&x, &mut y).unwrap();
        selected_sparse.mat_transpose_vec_into(&u, &mut t).unwrap();
        for (actual, expected) in y
            .iter()
            .chain(t.iter())
            .zip(expected_y.iter().chain(expected_t.iter()))
        {
            approx::assert_abs_diff_eq!(actual, expected, epsilon = 1e-10);
        }
        filter_csc(&csc, rows.clone(), columns.clone(), &x, &mut y, false);
        filter_csc(&csc, rows.clone(), columns.clone(), &u, &mut t, true);
        for (actual, expected) in y
            .iter()
            .chain(t.iter())
            .zip(expected_y.iter().chain(expected_t.iter()))
        {
            approx::assert_abs_diff_eq!(actual, expected, epsilon = 1e-10);
        }
        for (operation, bytes) in [
            (
                "construct_dense",
                allocations::measure(|| dense.submatrix(rows.clone(), columns.clone())),
            ),
            (
                "construct_sparse",
                allocations::measure(|| sparse.submatrix(rows.clone(), columns.clone())),
            ),
            (
                "dense_forward",
                allocations::measure(|| selected_dense.matvec_into(&x, &mut y).unwrap()),
            ),
            (
                "dense_transpose",
                allocations::measure(|| selected_dense.mat_transpose_vec_into(&u, &mut t).unwrap()),
            ),
            (
                "sparse_forward",
                allocations::measure(|| selected_sparse.matvec_into(&x, &mut y).unwrap()),
            ),
            (
                "sparse_transpose",
                allocations::measure(|| {
                    selected_sparse.mat_transpose_vec_into(&u, &mut t).unwrap()
                }),
            ),
        ] {
            eprintln!("{nr},{nc},{operation},{bytes}");
        }
        let mut group = c.benchmark_group(format!("submatrix/{nr}x{nc}"));
        macro_rules! products {
            ($label:literal, $matrix:expr) => {
                group.bench_function(concat!($label, "_forward"), |b| {
                    b.iter(|| {
                        $matrix.matvec_into(black_box(&x), &mut y).unwrap();
                        black_box(&y);
                    })
                });
                group.bench_function(concat!($label, "_transpose"), |b| {
                    b.iter(|| {
                        $matrix
                            .mat_transpose_vec_into(black_box(&u), &mut t)
                            .unwrap();
                        black_box(&t);
                    })
                });
            };
        }
        products!("dense", selected_dense);
        products!("native_dense_view", native);
        products!("sparse", selected_sparse);
        group.bench_function("direct_csc_forward", |b| {
            b.iter(|| {
                filter_csc(
                    &csc,
                    rows.clone(),
                    columns.clone(),
                    black_box(&x),
                    &mut y,
                    false,
                );
                black_box(&y);
            })
        });
        group.bench_function("direct_csc_transpose", |b| {
            b.iter(|| {
                filter_csc(
                    &csc,
                    rows.clone(),
                    columns.clone(),
                    black_box(&u),
                    &mut t,
                    true,
                );
                black_box(&t);
            })
        });
        group.finish();
    }
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
