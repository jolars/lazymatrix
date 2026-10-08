//! Explicit materialization and repeated products, with preprocessing separated.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use lazymatrix::{Centering, LazyMatrix, MatTransposeVecInto, MatVecInto, Normalization, Scaling};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[path = "../tests/common/backend_aliases.rs"]
mod backend_aliases;
use backend_aliases::{ndarray, sprs};
use ndarray::{Array1, Array2};

fn benchmark(c: &mut Criterion) {
    for (rows, cols) in [(512, 32), (2_048, 128)] {
        let mut rng = ChaCha8Rng::seed_from_u64(412);
        let dense = Array2::from_shape_fn((rows, cols), |_| {
            if rng.random::<f64>() < 0.1 {
                rng.random_range(-2.0..2.0)
            } else {
                0.0
            }
        });
        let mut triplets = sprs::TriMat::new((rows, cols));
        for ((i, j), &value) in dense.indexed_iter() {
            if value != 0.0 {
                triplets.add_triplet(i, j, value);
            }
        }
        let sparse = triplets.to_csc::<usize>();
        let csr = sparse.to_csr();
        let spec = Normalization::new(Centering::Mean, Scaling::Sd);
        let lazy_dense = LazyMatrix::new(&dense, spec).unwrap();
        let lazy_sparse = LazyMatrix::new(&sparse, spec).unwrap();
        let lazy_csr = LazyMatrix::new(&csr, spec).unwrap();
        let eager = lazy_sparse.to_eager::<Array2<f64>>().unwrap();
        let mut group = c.benchmark_group(format!("eager/{rows}x{cols}"));
        group.bench_function("fit_parameters", |b| {
            b.iter(|| {
                black_box(LazyMatrix::new(black_box(&dense), spec).unwrap());
            })
        });
        group.bench_function("dense_materialize", |b| {
            b.iter(|| {
                black_box(lazy_dense.to_eager::<Array2<f64>>().unwrap());
            })
        });
        group.bench_function("csc_materialize", |b| {
            b.iter(|| {
                black_box(lazy_sparse.to_eager::<Array2<f64>>().unwrap());
            })
        });
        group.bench_function("csr_materialize", |b| {
            b.iter(|| {
                black_box(lazy_csr.to_eager::<Array2<f64>>().unwrap());
            })
        });
        let x = Array1::from_elem(cols, 1.0);
        let u = Array1::from_elem(rows, 1.0);
        let mut y = Array1::zeros(rows);
        let mut t = Array1::zeros(cols);
        macro_rules! products {
            ($name:literal, $operator:expr) => {
                group.bench_function($name, |b| {
                    b.iter(|| {
                        for _ in 0..10 {
                            $operator.matvec_into(black_box(&x), &mut y).unwrap();
                            $operator
                                .mat_transpose_vec_into(black_box(&u), &mut t)
                                .unwrap();
                        }
                        black_box((&y, &t));
                    })
                });
            };
        }
        products!("lazy_dense_products", lazy_dense);
        products!("lazy_sparse_products", lazy_sparse);
        products!("eager_dense_products", eager);
        group.finish();
    }
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
