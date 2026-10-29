//! Standard-deviation reductions and faer CSC ownership comparisons.
//!
//! Run the CSC comparison in one exclusive compute window:
//! ```text
//! compute bench --reason "Compare faer CSC statistics" -- cargo bench --locked \
//!   --no-default-features --features faer_v0_24 --bench column_sds -- \
//!   faer_csc_sd --sample-size 30 --warm-up-time 1 --measurement-time 2 --noplot
//! ```
//! Each CSC fixture has 100,000 rows and 32 columns with evenly spaced stored
//! entries at 1% or 10% density. Construction and correctness checks precede
//! timing. `pointer_ranges` retains the former serial two-pass implementation
//! for canonical input; `owned` and `borrowed` include the current duplicate
//! checks. The comparison disables parallelism and measures warm in-memory data.
//!
//! `column_sds/faer_csc.csv` records medians and 95% bootstrap confidence
//! intervals in nanoseconds from October 29, 2026, using Rust 1.98.1 on an AMD
//! Ryzen 9 7900 with 24 logical CPUs. Borrowed and owned views differ by less
//! than 0.5% on these fixtures. Checking for duplicates adds about 17–20% over
//! the former implementation. Refresh the data and this description together.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

#[cfg(feature = "faer_all")]
#[path = "../tests/common/backend_aliases.rs"]
mod backend_aliases;

fn two_pass(values: &[f64], nrows: usize) -> f64 {
    let n = nrows as f64;
    let mean = values.iter().copied().sum::<f64>() / n;
    let stored = values
        .iter()
        .map(|&value| {
            let deviation = value - mean;
            deviation * deviation
        })
        .sum::<f64>();
    let implicit = (nrows - values.len()) as f64 * mean * mean;
    ((stored + implicit) / n).sqrt()
}

fn combined_variance(values: &[f64], nrows: usize) -> f64 {
    let mut stored_mean = 0.0;
    let mut stored_m2 = 0.0;

    for (index, &value) in values.iter().enumerate() {
        let count = (index + 1) as f64;
        let delta = value - stored_mean;
        stored_mean += delta / count;
        stored_m2 += delta * (value - stored_mean);
    }

    let stored_count = values.len();
    let implicit_count = nrows - stored_count;
    let cross = if stored_count == 0 || implicit_count == 0 {
        0.0
    } else {
        stored_mean * stored_mean * stored_count as f64 * implicit_count as f64 / nrows as f64
    };
    ((stored_m2 + cross) / nrows as f64).sqrt()
}

fn benchmark_column_sds(c: &mut Criterion) {
    let cases = [
        (
            "dense_large_offset",
            100_000,
            (0..100_000)
                .map(|index| 1.0e12 + (index % 17) as f64)
                .collect::<Vec<_>>(),
        ),
        (
            "ten_percent_stored",
            100_000,
            (0..10_000)
                .map(|index| (index % 17) as f64 - 8.0)
                .collect::<Vec<_>>(),
        ),
        (
            "one_percent_stored",
            100_000,
            (0..1_000)
                .map(|index| (index % 17) as f64 - 8.0)
                .collect::<Vec<_>>(),
        ),
    ];

    let mut group = c.benchmark_group("column_sd");
    for (name, nrows, values) in cases {
        group.bench_with_input(BenchmarkId::new("two_pass", name), &values, |b, values| {
            b.iter(|| two_pass(black_box(values), black_box(nrows)));
        });
        group.bench_with_input(
            BenchmarkId::new("combined_variance", name),
            &values,
            |b, values| {
                b.iter(|| combined_variance(black_box(values), black_box(nrows)));
            },
        );
    }
    group.finish();
    #[cfg(feature = "faer_all")]
    benchmark_faer_csc(c);
}

#[cfg(feature = "faer_all")]
fn benchmark_faer_csc(c: &mut Criterion) {
    use backend_aliases::faer::sparse::{SparseColMat, SymbolicSparseColMat};
    use lazymatrix::ColumnStats;

    let mut group = c.benchmark_group("faer_csc_sd");
    let (nrows, ncols) = (100_000, 32);
    for stored in [1_000, 10_000] {
        let offsets = (0..=ncols).map(|j| j * stored).collect();
        let indices = (0..ncols)
            .flat_map(|_| (0..stored).map(|i| i * (nrows / stored)))
            .collect();
        let values = (0..ncols * stored).map(|i| (i % 17) as f64 - 8.0).collect();
        let symbolic = SymbolicSparseColMat::new_checked(nrows, ncols, offsets, None, indices);
        let matrix = SparseColMat::new(symbolic, values);
        let view = matrix.as_ref();
        let expected: Vec<_> = (0..ncols)
            .map(|j| {
                two_pass(
                    &matrix.val()[matrix.col_ptr()[j]..matrix.col_ptr()[j + 1]],
                    nrows,
                )
            })
            .collect();
        assert_eq!(matrix.col_sds(0).unwrap(), expected);
        assert_eq!(view.col_sds(0).unwrap(), expected);
        group.bench_with_input(
            BenchmarkId::new("pointer_ranges", stored),
            &matrix,
            |b, m| {
                b.iter(|| {
                    let m = black_box(m);
                    // Retain the original canonical-column path as a timing baseline.
                    (0..ncols)
                        .map(|j| two_pass(&m.val()[m.col_ptr()[j]..m.col_ptr()[j + 1]], nrows))
                        .collect::<Vec<_>>()
                });
            },
        );
        group.bench_function(BenchmarkId::new("owned", stored), |b| {
            b.iter(|| black_box(&matrix).col_sds(0).unwrap());
        });
        group.bench_function(BenchmarkId::new("borrowed", stored), |b| {
            b.iter(|| black_box(&view).col_sds(0).unwrap());
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark_column_sds);
criterion_main!(benches);
