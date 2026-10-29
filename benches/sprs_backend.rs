use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use lazymatrix::{
    Centering, ColumnStats, LazyMatrix, MatTransposeVecInto, MatVecInto, Normalization, Scaling,
    SprsCsc,
};

#[path = "../tests/common/backend_aliases.rs"]
mod backend_aliases;
use backend_aliases::sprs;

fn benchmark_sprs(c: &mut Criterion) {
    let mut group = c.benchmark_group("sprs");
    for (nrows, ncols) in [(1_000, 100), (10_000, 1_000)] {
        let mut triplets = sprs::TriMat::new((nrows, ncols));
        for j in 0..ncols {
            for i in (0..nrows).step_by(100) {
                triplets.add_triplet((i + 17 * j) % nrows, j, (i % 13) as f64 - 6.0);
            }
        }
        let csc = triplets.to_csc::<usize>();
        let csr = csc.to_csr();
        group.throughput(Throughput::Elements(csc.nnz() as u64));
        for (layout, matrix) in [("csc", &csc), ("csr", &csr)] {
            let case = format!("{layout}_{nrows}x{ncols}");
            group.bench_with_input(BenchmarkId::new("col_sds", &case), matrix, |b, matrix| {
                b.iter(|| black_box(matrix).col_sds(0).unwrap());
            });

            let lazy = LazyMatrix::new(
                matrix,
                Normalization::new(Centering::Mean, Scaling::Sd { df: 0 }),
            )
            .unwrap();
            let v = vec![1.0; ncols];
            let u = vec![1.0; nrows];
            let mut y = vec![0.0; nrows];
            let mut z = vec![0.0; ncols];
            group.bench_function(BenchmarkId::new("matvec_into", &case), |b| {
                b.iter(|| lazy.matvec_into(black_box(&v), black_box(&mut y)).unwrap());
            });
            group.bench_function(BenchmarkId::new("transpose_into", &case), |b| {
                b.iter(|| {
                    lazy.mat_transpose_vec_into(black_box(&u), black_box(&mut z))
                        .unwrap()
                });
            });
        }
    }
    group.finish();
}

fn benchmark_weighted_norms(c: &mut Criterion) {
    let mut group = c.benchmark_group("weighted_norm");
    let nrows = 100_000;
    let weights: Vec<_> = (0..nrows).map(|i| 0.5 + (i % 7) as f64 / 7.0).collect();
    let weight_sum = weights.iter().sum();
    for stride in [1_000, 100, 10, 1] {
        let rows: Vec<_> = (0..nrows).step_by(stride).collect();
        let values: Vec<_> = rows.iter().map(|i| (i % 13) as f64 - 6.0).collect();
        let matrix = sprs::CsMat::new_csc((nrows, 1), vec![0, rows.len()], rows, values);
        let matrix = SprsCsc::try_new(matrix.view()).unwrap();
        for center in [0.0, 0.5] {
            let lazy = LazyMatrix::from_parts(&matrix, Some(vec![center]), Some(vec![2.0]));
            let column = lazy.column(0);
            let case = format!("stride_{stride}_center_{center}");
            group.bench_function(BenchmarkId::new("direct", &case), |b| {
                b.iter(|| black_box(&column).weighted_norm_squared(black_box(&weights)));
            });
            group.bench_function(BenchmarkId::new("cached_total", &case), |b| {
                b.iter(|| {
                    black_box(&column)
                        .weighted_norm_squared_with_sum(black_box(&weights), black_box(weight_sum))
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, benchmark_sprs, benchmark_weighted_norms);
criterion_main!(benches);
