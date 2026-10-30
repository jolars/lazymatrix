//! Rectangular products versus native dense views and direct sparse filtering.

use criterion::{Criterion, criterion_group, criterion_main};
use lazymatrix::{
    LazyMatrix, MatTransposeVecInto, MatVecInto, SprsCsc, WeightedColumnSumsInto, WeightedGramInto,
};
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
    let dense = LazyMatrix::from_column_parts(raw.view(), Some(vec![1.0; p]), Some(vec![2.0; p]));
    let sparse = LazyMatrix::from_column_parts(
        SprsCsc::try_new(csc.view()).unwrap(),
        Some(vec![1.0; p]),
        Some(vec![2.0; p]),
    );
    #[cfg(feature = "zarrs_all")]
    let chunked = {
        use backend_aliases::{zarrs, zarrs_compat};
        use std::sync::Arc;
        let array = zarrs::array::ArrayBuilder::new(
            vec![n as u64, p as u64],
            vec![128, 32],
            zarrs_compat::float64(),
            0.0_f64,
        )
        .build(
            Arc::new(zarrs::storage::store::MemoryStore::new()),
            "/submatrix",
        )
        .unwrap();
        zarrs_compat::store_subset(&array, raw.as_slice().unwrap()).unwrap();
        LazyMatrix::from_column_parts(
            lazymatrix::ZarrMatrix::<_, f64>::try_new(array).unwrap(),
            Some(vec![1.0; p]),
            Some(vec![2.0; p]),
        )
    };
    eprintln!("rows,columns,operation,peak_workspace_bytes");
    for (nr, nc) in [(128, 32), (1024, 64)] {
        let rows = 17..17 + nr;
        let columns = 11..11 + nc;
        let selected_dense = dense.submatrix(rows.clone(), columns.clone());
        let selected_sparse = sparse.submatrix(rows.clone(), columns.clone());
        #[cfg(feature = "zarrs_all")]
        let selected_chunked = chunked.submatrix(rows.clone(), columns.clone());
        let native = LazyMatrix::from_column_parts(
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
        let weights = Array1::from_shape_fn(nr, |i| (i % 7) as f64 * 0.25 - 0.5);
        let mut gram = Array2::zeros((nc, nc));
        let mut sums = Array1::zeros(nc);
        let mut expected_gram = Array2::zeros((nc, nc));
        let mut expected_sums = Array1::zeros(nc);
        native
            .weighted_gram_into(&weights, &mut expected_gram)
            .unwrap();
        native
            .weighted_column_sums_into(&weights, &mut expected_sums)
            .unwrap();
        macro_rules! check_weighted {
            ($matrix:expr) => {{
                $matrix.weighted_gram_into(&weights, &mut gram).unwrap();
                $matrix
                    .weighted_column_sums_into(&weights, &mut sums)
                    .unwrap();
                for (actual, expected) in gram
                    .iter()
                    .chain(sums.iter())
                    .zip(expected_gram.iter().chain(expected_sums.iter()))
                {
                    approx::assert_abs_diff_eq!(actual, expected, epsilon = 1e-10);
                }
            }};
        }
        check_weighted!(selected_dense);
        check_weighted!(selected_sparse);
        #[cfg(feature = "zarrs_all")]
        check_weighted!(selected_chunked);
        native.matvec_into(&x, &mut expected_y).unwrap();
        native.mat_transpose_vec_into(&u, &mut expected_t).unwrap();
        macro_rules! check_products {
            ($matrix:expr) => {{
                $matrix.matvec_into(&x, &mut y).unwrap();
                $matrix.mat_transpose_vec_into(&u, &mut t).unwrap();
                for (actual, expected) in y
                    .iter()
                    .chain(t.iter())
                    .zip(expected_y.iter().chain(expected_t.iter()))
                {
                    approx::assert_abs_diff_eq!(actual, expected, epsilon = 1e-10);
                }
            }};
        }
        check_products!(selected_dense);
        check_products!(selected_sparse);
        #[cfg(feature = "zarrs_all")]
        check_products!(selected_chunked);
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
        macro_rules! measure_weighted {
            ($label:literal, $matrix:expr) => {
                for (operation, bytes) in [
                    (
                        concat!($label, "_weighted_sums"),
                        allocations::measure(|| {
                            $matrix
                                .weighted_column_sums_into(&weights, &mut sums)
                                .unwrap()
                        }),
                    ),
                    (
                        concat!($label, "_weighted_gram"),
                        allocations::measure(|| {
                            $matrix.weighted_gram_into(&weights, &mut gram).unwrap()
                        }),
                    ),
                ] {
                    eprintln!("{nr},{nc},{operation},{bytes}");
                }
            };
        }
        measure_weighted!("dense", selected_dense);
        measure_weighted!("sparse", selected_sparse);
        measure_weighted!("native_dense_view", native);
        #[cfg(feature = "zarrs_all")]
        {
            measure_weighted!("zarr", selected_chunked);
            for (operation, bytes) in [
                (
                    "zarr_forward",
                    allocations::measure(|| selected_chunked.matvec_into(&x, &mut y).unwrap()),
                ),
                (
                    "zarr_transpose",
                    allocations::measure(|| {
                        selected_chunked.mat_transpose_vec_into(&u, &mut t).unwrap()
                    }),
                ),
            ] {
                eprintln!("{nr},{nc},{operation},{bytes}");
            }
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
        #[cfg(feature = "zarrs_all")]
        products!("zarr", selected_chunked);
        macro_rules! weighted {
            ($label:literal, $matrix:expr) => {
                group.bench_function(concat!($label, "_weighted_sums"), |b| {
                    b.iter(|| {
                        $matrix
                            .weighted_column_sums_into(black_box(&weights), &mut sums)
                            .unwrap();
                        black_box(&sums);
                    })
                });
                group.bench_function(concat!($label, "_weighted_gram"), |b| {
                    b.iter(|| {
                        $matrix
                            .weighted_gram_into(black_box(&weights), &mut gram)
                            .unwrap();
                        black_box(&gram);
                    })
                });
            };
        }
        weighted!("dense", selected_dense);
        weighted!("native_dense_view", native);
        weighted!("sparse", selected_sparse);
        #[cfg(feature = "zarrs_all")]
        weighted!("zarr", selected_chunked);
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
