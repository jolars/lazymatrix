use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use lazymatrix::{
    ColumnStats, LazyMatrix, MatTransposeVecInto, MatVecInto, SprsCsc, WeightedColumnSumsInto,
    WeightedGramInto, WithIntercept,
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[path = "../tests/common/backend_aliases.rs"]
mod backend_aliases;
#[cfg(feature = "faer_all")]
use backend_aliases::faer;
use backend_aliases::{ndarray, sprs};
use ndarray::{Array1, Array2, linalg::general_mat_mul};

fn benchmark_gram(c: &mut Criterion) {
    let mut group = c.benchmark_group("weighted_gram");
    for (n, p) in [(2_000, 32), (10_000, 128)] {
        for density in [0.001, 0.01, 0.1, 1.0] {
            let mut rng = ChaCha8Rng::seed_from_u64(517);
            let matrix = Array2::from_shape_fn((n, p), |_| {
                if rng.random::<f64>() < density {
                    rng.random_range(-2.0..2.0)
                } else {
                    0.0
                }
            });
            let mut triplets = sprs::TriMat::new((n, p));
            for ((i, j), &value) in matrix.indexed_iter() {
                if value != 0.0 {
                    triplets.add_triplet(i, j, value);
                }
            }
            let csc = triplets.to_csc::<usize>();
            let weights = Array1::from_shape_fn(n, |i| 0.5 + (i % 11) as f64 / 11.0);
            #[cfg(feature = "faer_all")]
            let faer_matrix = faer::Mat::from_fn(n, p, |i, j| matrix[(i, j)]);
            #[cfg(feature = "faer_all")]
            let faer_weights = faer::Col::from_fn(n, |i| weights[i]);
            for centered in [false, true] {
                let centers = centered.then(|| matrix.col_means().unwrap());
                let dense = LazyMatrix::from_parts(matrix.view(), centers.clone(), None);
                let sparse = LazyMatrix::from_parts(
                    SprsCsc::try_new(csc.view()).unwrap(),
                    centers.clone(),
                    None,
                );
                let case = format!("{n}x{p}/density_{density}/centered_{centered}");
                let mut out = Array2::zeros((p, p));
                group.bench_function(BenchmarkId::new("dense", &case), |b| {
                    b.iter(|| {
                        dense
                            .weighted_gram_into(black_box(&weights), black_box(&mut out))
                            .unwrap()
                    });
                });
                group.bench_function(BenchmarkId::new("csc", &case), |b| {
                    b.iter(|| {
                        sparse
                            .weighted_gram_into(black_box(&weights), black_box(&mut out))
                            .unwrap()
                    });
                });
                let mut sums = Array1::zeros(p);
                group.bench_function(BenchmarkId::new("dense_column_sums", &case), |b| {
                    b.iter(|| {
                        dense
                            .weighted_column_sums_into(black_box(&weights), black_box(&mut sums))
                            .unwrap()
                    });
                });
                #[cfg(feature = "faer_all")]
                {
                    let dense = LazyMatrix::from_parts(faer_matrix.as_ref(), centers.clone(), None);
                    let mut out = faer::Mat::zeros(p, p);
                    group.bench_function(BenchmarkId::new("faer_dense", &case), |b| {
                        b.iter(|| {
                            dense
                                .weighted_gram_into(black_box(&faer_weights), black_box(&mut out))
                                .unwrap()
                        });
                    });
                    let mut sums = faer::Col::zeros(p);
                    group.bench_function(BenchmarkId::new("faer_column_sums", &case), |b| {
                        b.iter(|| {
                            dense
                                .weighted_column_sums_into(
                                    black_box(&faer_weights),
                                    black_box(&mut sums),
                                )
                                .unwrap()
                        });
                    });
                    let augmented = WithIntercept::new(&dense);
                    let mut out = faer::Mat::zeros(p + 1, p + 1);
                    group.bench_function(BenchmarkId::new("faer_dense_intercept", &case), |b| {
                        b.iter(|| {
                            augmented
                                .weighted_gram_into(black_box(&faer_weights), black_box(&mut out))
                                .unwrap()
                        });
                    });
                }

                let dense_intercept = WithIntercept::new(&dense);
                let sparse_intercept = WithIntercept::new(&sparse);
                let mut augmented_out = Array2::zeros((p + 1, p + 1));
                group.bench_function(BenchmarkId::new("dense_intercept", &case), |b| {
                    b.iter(|| {
                        dense_intercept
                            .weighted_gram_into(black_box(&weights), black_box(&mut augmented_out))
                            .unwrap()
                    });
                });
                group.bench_function(BenchmarkId::new("csc_intercept", &case), |b| {
                    b.iter(|| {
                        sparse_intercept
                            .weighted_gram_into(black_box(&weights), black_box(&mut augmented_out))
                            .unwrap()
                    });
                });
                let augmented = Array2::from_shape_fn((n, p + 1), |(i, j)| {
                    if j == 0 {
                        1.0
                    } else {
                        matrix[(i, j - 1)] - centers.as_ref().map_or(0.0, |c| c[j - 1])
                    }
                });
                let weighted_augmented =
                    Array2::from_shape_fn((n, p + 1), |(i, j)| augmented[(i, j)] * weights[i]);
                group.bench_function(BenchmarkId::new("gemm_intercept", &case), |b| {
                    b.iter(|| {
                        general_mat_mul(
                            1.0,
                            &black_box(&augmented).t(),
                            black_box(&weighted_augmented),
                            0.0,
                            black_box(&mut augmented_out),
                        )
                    });
                });

                let mut basis = Array1::zeros(p);
                let mut column = Array1::zeros(n);
                let mut product = Array1::zeros(p);
                group.bench_function(BenchmarkId::new("dense_operators", &case), |b| {
                    b.iter(|| {
                        for j in 0..p {
                            basis[j] = 1.0;
                            dense.matvec_into(&basis, &mut column).unwrap();
                            column *= &weights;
                            dense.mat_transpose_vec_into(&column, &mut product).unwrap();
                            out.column_mut(j).assign(&product);
                            basis[j] = 0.0;
                        }
                        black_box(&out);
                    });
                });
                let mut basis = vec![0.0; p];
                let mut column = vec![0.0; n];
                let mut product = vec![0.0; p];
                group.bench_function(BenchmarkId::new("csc_operators", &case), |b| {
                    b.iter(|| {
                        for j in 0..p {
                            basis[j] = 1.0;
                            sparse.matvec_into(&basis, &mut column).unwrap();
                            for (x, &w) in column.iter_mut().zip(&weights) {
                                *x *= w;
                            }
                            sparse
                                .mat_transpose_vec_into(&column, &mut product)
                                .unwrap();
                            for k in 0..p {
                                out[(k, j)] = product[k];
                            }
                            basis[j] = 0.0;
                        }
                        black_box(&out);
                    });
                });
                let prepare = || {
                    let normalized = Array2::from_shape_fn((n, p), |(i, j)| {
                        matrix[(i, j)] - centers.as_ref().map_or(0.0, |c| c[j])
                    });
                    let weighted =
                        Array2::from_shape_fn((n, p), |(i, j)| normalized[(i, j)] * weights[i]);
                    (normalized, weighted)
                };
                let (normalized, weighted) = prepare();
                group.bench_function(BenchmarkId::new("gemm_prepared", &case), |b| {
                    b.iter(|| {
                        general_mat_mul(
                            1.0,
                            &black_box(&normalized).t(),
                            black_box(&weighted),
                            0.0,
                            black_box(&mut out),
                        )
                    });
                });
                group.bench_function(BenchmarkId::new("gemm_with_preparation", &case), |b| {
                    b.iter(|| {
                        let (normalized, weighted) = prepare();
                        general_mat_mul(1.0, &normalized.t(), &weighted, 0.0, black_box(&mut out));
                    });
                });
            }
        }
    }
    group.finish();
}

criterion_group!(benches, benchmark_gram);
criterion_main!(benches);
