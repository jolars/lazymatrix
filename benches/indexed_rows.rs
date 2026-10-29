//! Fit and apply borrowed raw selections versus explicit fold copies.
use criterion::{Criterion, criterion_group, criterion_main};
use lazymatrix::{ColumnNormalization, IndexedRows, LazyMatrix, MatVecInto};
use std::{hint::black_box, time::Duration};
#[path = "common/allocations.rs"]
mod allocations;
#[path = "../tests/common/backend_aliases.rs"]
mod backend_aliases;
use backend_aliases::{ndarray, sprs};
use ndarray::{Array1, Array2, Axis, ShapeBuilder};

fn copy_csc(matrix: &sprs::CsMat<f64>, rows: &[usize]) -> sprs::CsMat<f64> {
    let mut offsets = vec![0];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for column in matrix.outer_iterator() {
        for (i, &value) in column.iter() {
            if let Ok(local) = rows.binary_search(&i) {
                indices.push(local);
                values.push(value);
            }
        }
        offsets.push(indices.len());
    }
    sprs::CsMat::new_csc((rows.len(), matrix.cols()), offsets, indices, values)
}

fn benchmark(c: &mut Criterion) {
    let (n, p) = (4096, 64);
    let dense = Array2::from_shape_fn((n, p).f(), |(i, j)| {
        if (i * 13 + j * 7) % 23 == 0 {
            ((i + j) % 11) as f64 - 5.0
        } else {
            0.0
        }
    });
    let mut coo = sprs::TriMat::new((n, p));
    for ((i, j), &value) in dense.indexed_iter() {
        if value != 0.0 {
            coo.add_triplet(i, j, value);
        }
    }
    let csc = coo.to_csc::<usize>();
    let rows: Vec<_> = (0..n).filter(|i| i % 5 != 0).collect();
    let x = Array1::ones(p);
    let mut out = Array1::zeros(rows.len());
    eprintln!("backend,operation,representation,peak_live_bytes");
    for (name, spec) in [
        ("standardize", ColumnNormalization::standardize()),
        ("robust", ColumnNormalization::robust()),
    ] {
        let mut group = c.benchmark_group(format!("indexed_rows/{name}"));
        group
            .sample_size(10)
            .warm_up_time(Duration::from_millis(100))
            .measurement_time(Duration::from_millis(300));
        eprintln!(
            "dense,fit_{name},borrowed,{}",
            allocations::measure(
                || LazyMatrix::new(IndexedRows::new(dense.view(), &rows), spec).unwrap()
            )
        );
        eprintln!(
            "dense,fit_{name},copied,{}",
            allocations::measure(|| LazyMatrix::new(dense.select(Axis(0), &rows), spec).unwrap())
        );
        eprintln!(
            "csc,fit_{name},borrowed,{}",
            allocations::measure(
                || LazyMatrix::new(IndexedRows::new(csc.view(), &rows), spec).unwrap()
            )
        );
        eprintln!(
            "csc,fit_{name},copied,{}",
            allocations::measure(|| LazyMatrix::new(copy_csc(&csc, &rows), spec).unwrap())
        );
        group.bench_function("dense_borrowed", |b| {
            b.iter(|| {
                LazyMatrix::new(IndexedRows::new(black_box(dense.view()), &rows), spec).unwrap()
            })
        });
        group.bench_function("dense_copied", |b| {
            b.iter(|| LazyMatrix::new(black_box(&dense).select(Axis(0), &rows), spec).unwrap())
        });
        group.bench_function("csc_borrowed", |b| {
            b.iter(|| {
                LazyMatrix::new(IndexedRows::new(black_box(csc.view()), &rows), spec).unwrap()
            })
        });
        group.bench_function("csc_copied", |b| {
            b.iter(|| LazyMatrix::new(copy_csc(black_box(&csc), &rows), spec).unwrap())
        });
        group.finish();
    }
    let dense_borrowed = LazyMatrix::new(
        IndexedRows::new(dense.view(), &rows),
        ColumnNormalization::standardize(),
    )
    .unwrap();
    let dense_copy = LazyMatrix::new(
        dense.select(Axis(0), &rows),
        ColumnNormalization::standardize(),
    )
    .unwrap();
    let csc_borrowed = LazyMatrix::new(
        IndexedRows::new(csc.view(), &rows),
        ColumnNormalization::standardize(),
    )
    .unwrap();
    let csc_copy =
        LazyMatrix::new(copy_csc(&csc, &rows), ColumnNormalization::standardize()).unwrap();
    let mut group = c.benchmark_group("indexed_rows/matvec");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_millis(300));
    eprintln!(
        "dense,matvec,borrowed,{}",
        allocations::measure(|| dense_borrowed.matvec_into(&x, &mut out).unwrap())
    );
    eprintln!(
        "dense,matvec,copied,{}",
        allocations::measure(|| dense_copy.matvec_into(&x, &mut out).unwrap())
    );
    eprintln!(
        "csc,matvec,borrowed,{}",
        allocations::measure(|| csc_borrowed.matvec_into(&x, &mut out).unwrap())
    );
    eprintln!(
        "csc,matvec,copied,{}",
        allocations::measure(|| csc_copy.matvec_into(&x, &mut out).unwrap())
    );
    group.bench_function("dense_borrowed", |b| {
        b.iter(|| {
            dense_borrowed
                .matvec_into(black_box(&x), black_box(&mut out))
                .unwrap()
        })
    });
    group.bench_function("dense_copied", |b| {
        b.iter(|| {
            dense_copy
                .matvec_into(black_box(&x), black_box(&mut out))
                .unwrap()
        })
    });
    group.bench_function("csc_borrowed", |b| {
        b.iter(|| {
            csc_borrowed
                .matvec_into(black_box(&x), black_box(&mut out))
                .unwrap()
        })
    });
    group.bench_function("csc_copied", |b| {
        b.iter(|| {
            csc_copy
                .matvec_into(black_box(&x), black_box(&mut out))
                .unwrap()
        })
    });
    group.finish();
}
criterion_group!(benches, benchmark);
criterion_main!(benches);
