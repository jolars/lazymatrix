//! Product and least-squares iteration comparisons, with setup excluded from timing.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use lazymatrix::{
    Centering, ColumnStats, DotSlice, ElemDivAssign, LazyMatrix, MatTransposeVec,
    MatTransposeVecInto, MatTransposeVecScaledInto, MatVec, MatVecInto, MatVecScaledInto,
    Normalization, ScaledSubSlice, Scaling, SubScalarAssign, SumEntries, VectorOwned,
    VectorViewMut, WithIntercept, ZarrMatrix,
};

#[path = "../tests/common/backend_aliases.rs"]
mod backend_aliases;
#[path = "../tests/common/runner.rs"]
mod common;
use backend_aliases::{faer, nalgebra, nalgebra_sparse, ndarray, sprs, zarrs};

struct CountedAllocator;
static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

// SAFETY: Every allocation and deallocation delegates unchanged to System.
unsafe impl GlobalAlloc for CountedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        // SAFETY: The caller supplies the allocator's required valid layout.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        // SAFETY: The caller supplies the allocator's required valid layout.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: System allocated this pointer using the supplied layout.
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(size, Ordering::Relaxed);
        }
        // SAFETY: The caller supplies System's original pointer and valid sizes.
        unsafe { System.realloc(pointer, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountedAllocator = CountedAllocator;

const STEPS: usize = 10;

fn count_allocations(name: &str, mut run: impl FnMut()) {
    run();
    ALLOCATIONS.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    run();
    COUNTING.store(false, Ordering::Relaxed);
    eprintln!(
        "allocations {name}: {} calls, {} bytes per {STEPS} operations",
        ALLOCATIONS.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed),
    );
}

fn measure<V: VectorViewMut<f64>>(
    c: &mut Criterion,
    name: &str,
    coefficients: &mut V,
    mut batch: impl FnMut(&mut V),
    expected: Option<&[f64]>,
) -> Vec<f64> {
    let mut run = |coefficients: &mut V| {
        for j in 0..coefficients.len() {
            coefficients.set(j, 0.0);
        }
        batch(black_box(coefficients));
        black_box(&*coefficients);
    };
    count_allocations(name, || run(coefficients));
    let actual: Vec<_> = (0..coefficients.len())
        .map(|j| coefficients.get(j))
        .collect();
    if let Some(expected) = expected {
        for (&actual, &expected) in actual.iter().zip(expected) {
            approx::assert_relative_eq!(actual, expected, epsilon = 1e-10, max_relative = 1e-10);
        }
    }
    // Criterion filters still apply; allocation sampling is intentionally untimed.
    c.bench_function(name, |b| b.iter(|| run(coefficients)));
    actual
}

fn measure_product<V>(
    c: &mut Criterion,
    name: &str,
    path: &str,
    expected: &V,
    mut product: impl FnMut(&mut V),
) where
    V: VectorOwned<f64, Owned = V> + VectorViewMut<f64>,
{
    // NaNs expose accidental reads of prior output when beta is zero.
    let mut output = V::owned_from_fn(expected.len(), |_| f64::NAN);
    let mut run = || {
        for _ in 0..STEPS {
            product(black_box(&mut output));
            black_box(&output);
        }
    };
    count_allocations(&format!("{name}/{path}"), &mut run);
    for j in 0..expected.len() {
        approx::assert_relative_eq!(
            output.get(j),
            expected.get(j),
            epsilon = 1e-10,
            max_relative = 1e-10
        );
    }
    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Elements(STEPS as u64));
    group.bench_function(path, |b| {
        b.iter(|| {
            for _ in 0..STEPS {
                product(black_box(&mut output));
                black_box(&output);
            }
        });
    });
    group.finish();
}

fn bench_overwrite_operator<O, V>(
    c: &mut Criterion,
    name: &str,
    op: &O,
    response: &V,
    step: f64,
    mut forward_workspace: impl FnMut(f64, &V, f64, &mut V),
    mut transpose_workspace: impl FnMut(f64, &V, f64, &mut V),
) where
    O: MatVec<V> + MatTransposeVec<V> + MatVecInto<V> + MatTransposeVecInto<V>,
    V: VectorOwned<f64, Owned = V> + VectorViewMut<f64>,
{
    let values = common::random_vec(205, op.ncols());
    let input = V::owned_from_fn(values.len(), |j| values[j]);
    let forward_expected = op.matvec(&input).unwrap();
    let transpose_expected = op.mat_transpose_vec(response).unwrap();
    let mut coefficients = V::owned_from_fn(op.ncols(), |_| 0.0);
    let mut residual = V::owned_from_fn(op.nrows(), |_| 0.0);
    let mut gradient = V::owned_from_fn(op.ncols(), |_| 0.0);
    let mut expected_coefficients = None;
    for workspace in [false, true] {
        let path = if workspace { "workspace" } else { "overwrite" };
        measure_product(
            c,
            &format!("overwrite_products/{name}/forward"),
            path,
            &forward_expected,
            |out| {
                if workspace {
                    forward_workspace(1.0, black_box(&input), 0.0, out);
                } else {
                    op.matvec_into(black_box(&input), out).unwrap();
                }
            },
        );
        measure_product(
            c,
            &format!("overwrite_products/{name}/transpose"),
            path,
            &transpose_expected,
            |out| {
                if workspace {
                    transpose_workspace(1.0, black_box(response), 0.0, out);
                } else {
                    op.mat_transpose_vec_into(black_box(response), out).unwrap();
                }
            },
        );
        let actual = measure(
            c,
            &format!("overwrite_iterations/{name}/{path}"),
            &mut coefficients,
            |coefficients| {
                for _ in 0..STEPS {
                    if workspace {
                        forward_workspace(1.0, coefficients, 0.0, &mut residual);
                    } else {
                        op.matvec_into(coefficients, &mut residual).unwrap();
                    }
                    for i in 0..residual.len() {
                        residual.set(i, residual.get(i) - response.get(i));
                    }
                    if workspace {
                        transpose_workspace(1.0, &residual, 0.0, &mut gradient);
                    } else {
                        op.mat_transpose_vec_into(&residual, &mut gradient).unwrap();
                    }
                    for j in 0..coefficients.len() {
                        coefficients.set(j, coefficients.get(j) - step * gradient.get(j));
                    }
                }
            },
            expected_coefficients.as_deref(),
        );
        expected_coefficients = Some(actual);
    }
}

fn bench_operator<O, V>(
    c: &mut Criterion,
    name: &str,
    op: &O,
    response: &V,
    step: f64,
    mut forward_workspace: impl FnMut(f64, &V, f64, &mut V),
    mut transpose_workspace: impl FnMut(f64, &V, f64, &mut V),
) where
    O: MatVec<V>
        + MatTransposeVec<V>
        + MatVecInto<V>
        + MatTransposeVecInto<V>
        + MatVecScaledInto<V, V, f64>
        + MatTransposeVecScaledInto<V, V, f64>,
    V: VectorOwned<f64, Owned = V> + VectorViewMut<f64>,
{
    bench_overwrite_operator(
        c,
        name,
        op,
        response,
        step,
        &mut forward_workspace,
        &mut transpose_workspace,
    );
    let name = format!("consumer/{name}");
    let mut coefficients = V::owned_from_fn(op.ncols(), |_| 0.0);
    let mut residual = V::owned_from_fn(op.nrows(), |_| 0.0);
    let mut gradient = V::owned_from_fn(op.ncols(), |_| 0.0);
    let expected = measure(
        c,
        &format!("{name}/allocating"),
        &mut coefficients,
        |coefficients| {
            for _ in 0..STEPS {
                let mut residual = op.matvec(coefficients).unwrap();
                for i in 0..residual.len() {
                    residual.set(i, residual.get(i) - response.get(i));
                }
                let gradient = op.mat_transpose_vec(&residual).unwrap();
                for j in 0..coefficients.len() {
                    coefficients.set(j, coefficients.get(j) - step * gradient.get(j));
                }
            }
        },
        None,
    );
    measure(
        c,
        &format!("{name}/overwrite"),
        &mut coefficients,
        |coefficients| {
            for _ in 0..STEPS {
                op.matvec_into(coefficients, &mut residual).unwrap();
                for i in 0..residual.len() {
                    residual.set(i, residual.get(i) - response.get(i));
                }
                op.mat_transpose_vec_into(&residual, &mut gradient).unwrap();
                for j in 0..coefficients.len() {
                    coefficients.set(j, coefficients.get(j) - step * gradient.get(j));
                }
            }
        },
        Some(&expected),
    );
    measure(
        c,
        &format!("{name}/fused"),
        &mut coefficients,
        |coefficients| {
            for _ in 0..STEPS {
                for i in 0..residual.len() {
                    residual.set(i, -response.get(i));
                }
                op.matvec_scaled_into(1.0, coefficients, 1.0, &mut residual)
                    .unwrap();
                op.mat_transpose_vec_scaled_into(-step, &residual, 1.0, coefficients)
                    .unwrap();
            }
        },
        Some(&expected),
    );
    measure(
        c,
        &format!("{name}/workspace"),
        &mut coefficients,
        |coefficients| {
            for _ in 0..STEPS {
                for i in 0..residual.len() {
                    residual.set(i, -response.get(i));
                }
                forward_workspace(1.0, coefficients, 1.0, &mut residual);
                transpose_workspace(-step, &residual, 1.0, coefficients);
            }
        },
        Some(&expected),
    );
}

fn bench_matrix<M, V>(c: &mut Criterion, label: &str, matrix: &M, vector: impl Fn(&[f64]) -> V)
where
    M: MatVec<V>
        + MatTransposeVec<V>
        + MatVecInto<V>
        + MatTransposeVecInto<V>
        + ColumnStats<f64>
        + MatVecScaledInto<V, V, f64>
        + MatTransposeVecScaledInto<V, V, f64>,
    V: Clone
        + VectorOwned<f64, Owned = V>
        + VectorViewMut<f64>
        + ElemDivAssign<f64>
        + DotSlice<f64>
        + SubScalarAssign<f64>
        + SumEntries<f64>
        + ScaledSubSlice<f64>,
{
    let response = vector(&common::random_vec(204, matrix.nrows()));
    for (normalization, spec) in [
        ("raw", Normalization::default()),
        (
            "centered",
            Normalization::new(Centering::Mean, Scaling::None),
        ),
        (
            "standardized",
            Normalization::new(Centering::Mean, Scaling::Sd),
        ),
    ] {
        let lazy = LazyMatrix::new(matrix, spec).unwrap();
        // The Frobenius bound gives every representation the same stable step.
        let norms = match lazy.centers() {
            Some(centers) => matrix.col_l2_centered(centers),
            None => matrix.col_l2(),
        }
        .unwrap();
        let bound: f64 = norms
            .iter()
            .enumerate()
            .map(|(j, n)| {
                let scale = lazy.scales().map_or(1.0, |s| s[j]);
                (n / scale).powi(2)
            })
            .sum();
        let step = 0.5 / (bound + matrix.nrows() as f64).max(1.0);
        let name = format!("{label}/{normalization}");
        let mut forward_scratch = V::owned_from_fn(matrix.ncols(), |_| 0.0);
        let mut transpose_scratch = V::owned_from_fn(matrix.ncols(), |_| 0.0);
        bench_operator(
            c,
            &name,
            &lazy,
            &response,
            step,
            |alpha, x, beta, out| {
                lazy.matvec_scaled_with_workspace(alpha, x, beta, out, &mut forward_scratch)
                    .unwrap()
            },
            |alpha, x, beta, out| {
                lazy.mat_transpose_vec_scaled_with_workspace(
                    alpha,
                    x,
                    beta,
                    out,
                    &mut transpose_scratch,
                )
                .unwrap()
            },
        );
        let design = WithIntercept::new(&lazy);
        bench_operator(
            c,
            &format!("{name}/intercept"),
            &design,
            &response,
            step,
            |alpha, x, beta, out| {
                design
                    .matvec_scaled_with_workspace(alpha, x, beta, out, &mut forward_scratch)
                    .unwrap()
            },
            |alpha, x, beta, out| {
                design
                    .mat_transpose_vec_scaled_with_workspace(
                        alpha,
                        x,
                        beta,
                        out,
                        &mut transpose_scratch,
                    )
                    .unwrap()
            },
        );
    }
}

fn benchmark(c: &mut Criterion) {
    for (rows, columns) in [(2_000, 32), (10_000, 128)] {
        for density in [1.0, 0.01, 0.1] {
            let tm = common::random_matrix(203, rows, columns, density);
            let label = format!("{rows}x{columns}/density_{density}");
            if density == 1.0 {
                let matrix = faer::Mat::from_fn(rows, columns, |i, j| tm.dense[i][j]);
                bench_matrix(c, &format!("faer_dense/{label}"), &matrix, |v| {
                    faer::Col::from_fn(v.len(), |j| v[j])
                });
                let matrix = nalgebra::DMatrix::from_fn(rows, columns, |i, j| tm.dense[i][j]);
                bench_matrix(
                    c,
                    &format!("nalgebra_dense/{label}"),
                    &matrix,
                    nalgebra::DVector::from_column_slice,
                );
                let matrix =
                    ndarray::Array2::from_shape_fn((rows, columns), |(i, j)| tm.dense[i][j]);
                bench_matrix(c, &format!("ndarray/{label}"), &matrix, |v| {
                    ndarray::Array1::from_vec(v.to_vec())
                });
                for chunk in [[256, 16], [1_024, 32]] {
                    let array = zarrs::array::ArrayBuilder::new(
                        vec![rows as u64, columns as u64],
                        chunk.to_vec(),
                        zarrs::array::DataType::Float64,
                        0.0_f64,
                    )
                    .build(
                        Arc::new(zarrs::storage::store::MemoryStore::new()),
                        "/matrix",
                    )
                    .unwrap();
                    let values: Vec<_> = tm.dense.iter().flatten().copied().collect();
                    array
                        .store_array_subset_elements(&array.subset_all(), &values)
                        .unwrap();
                    let matrix = ZarrMatrix::<_, f64>::try_new(array).unwrap();
                    bench_matrix(c, &format!("zarrs_{chunk:?}/{label}"), &matrix, |v| {
                        v.to_vec()
                    });
                }
            } else {
                let triplets: Vec<_> = tm
                    .triplets
                    .iter()
                    .map(|&(i, j, v)| faer::sparse::Triplet::new(i, j, v))
                    .collect();
                let matrix = faer::sparse::SparseColMat::<usize, f64>::try_new_from_triplets(
                    rows, columns, &triplets,
                )
                .unwrap();
                bench_matrix(c, &format!("faer_csc/{label}"), &matrix, |v| {
                    faer::Col::from_fn(v.len(), |j| v[j])
                });
                let csr = faer::sparse::SparseRowMat::<usize, f64>::try_new_from_triplets(
                    rows, columns, &triplets,
                )
                .unwrap();
                bench_matrix(c, &format!("faer_csr/{label}"), &csr, |v| {
                    faer::Col::from_fn(v.len(), |j| v[j])
                });
                let mut coo = nalgebra_sparse::CooMatrix::new(rows, columns);
                let mut coo_sprs = sprs::TriMat::new((rows, columns));
                for &(i, j, value) in &tm.triplets {
                    coo.push(i, j, value);
                    coo_sprs.add_triplet(i, j, value);
                }
                bench_matrix(
                    c,
                    &format!("nalgebra_csc/{label}"),
                    &nalgebra_sparse::CscMatrix::from(&coo),
                    nalgebra::DVector::from_column_slice,
                );
                bench_matrix(
                    c,
                    &format!("nalgebra_csr/{label}"),
                    &nalgebra_sparse::CsrMatrix::from(&coo),
                    nalgebra::DVector::from_column_slice,
                );
                bench_matrix(
                    c,
                    &format!("sprs_csc/{label}"),
                    &coo_sprs.to_csc::<usize>(),
                    |v| v.to_vec(),
                );
            }
        }
    }
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
