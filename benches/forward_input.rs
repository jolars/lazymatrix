//! Compare allocating forward products with the former unconditional input clone.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};

use criterion::{Criterion, criterion_group, criterion_main};
use lazymatrix::{
    Centering, ColumnNormalization, ColumnStats, DotSlice, ElemDivAssign, LazyMatrix, MatVec,
    Scaling, SubScalarAssign, VectorView,
};

#[path = "../tests/common/backend_aliases.rs"]
mod backend_aliases;
#[path = "../tests/common/runner.rs"]
mod common;
use backend_aliases::{faer, nalgebra, nalgebra_sparse, ndarray, sprs};

struct CountedAllocator;
static COUNTING: AtomicBool = AtomicBool::new(false);
static CALLS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

fn record(bytes: usize) {
    if COUNTING.load(Relaxed) {
        CALLS.fetch_add(1, Relaxed);
        BYTES.fetch_add(bytes, Relaxed);
    }
}

// SAFETY: Every pointer and layout is forwarded unchanged to System.
unsafe impl GlobalAlloc for CountedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: The caller supplies a valid layout.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            record(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: The caller supplies a valid layout.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            record(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: System allocated this pointer using the supplied layout.
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: The caller supplies System's original pointer and valid sizes.
        let result = unsafe { System.realloc(pointer, layout, size) };
        if !result.is_null() {
            record(size);
        }
        result
    }
}

#[global_allocator]
static ALLOCATOR: CountedAllocator = CountedAllocator;

fn count<T>(run: impl FnOnce() -> T) -> (T, usize, usize) {
    CALLS.store(0, Relaxed);
    BYTES.store(0, Relaxed);
    COUNTING.store(true, Relaxed);
    let result = black_box(run());
    COUNTING.store(false, Relaxed);
    (result, CALLS.load(Relaxed), BYTES.load(Relaxed))
}

fn cloned_reference<const DEFAULT_HOOK: bool, M, V>(
    lazy: &LazyMatrix<M>,
    x: &V,
) -> Result<V, M::Error>
where
    M: MatVec<V>,
    V: Clone + ElemDivAssign<f64> + DotSlice<f64> + SubScalarAssign<f64>,
{
    // Retain the former default so the public operator's allocation saving stays
    // measurable. Dense overrides must preserve directly centered arithmetic.
    if !DEFAULT_HOOK {
        return lazy.data().matvec_normalized(
            x,
            lazy.normalization().centers(),
            lazy.normalization().scales(),
        );
    }
    let mut coefficients = x.clone();
    if let Some(scales) = lazy.normalization().scales() {
        coefficients.elem_div_assign(scales);
    }
    let mut out = lazy.data().matvec(&coefficients)?;
    if let Some(centers) = lazy.normalization().centers() {
        out.sub_scalar_assign(coefficients.dot_slice(centers));
    }
    Ok(out)
}

fn measure<V: VectorView<f64>>(
    c: &mut Criterion,
    name: &str,
    run: impl Fn() -> V,
    expected: &[f64],
) -> (usize, usize) {
    let warm = run();
    for (i, &value) in expected.iter().enumerate() {
        approx::assert_relative_eq!(warm.get(i), value, epsilon = 1e-9, max_relative = 1e-9);
    }
    let (out, calls, bytes) = count(&run);
    for i in 0..out.len() {
        assert_eq!(out.get(i), warm.get(i));
    }
    eprintln!("forward_input_allocations,{name},{calls},{bytes}");
    c.bench_function(name, |b| b.iter(&run));
    (calls, bytes)
}

fn bench_matrix<const DEFAULT_HOOK: bool, M, V>(
    c: &mut Criterion,
    label: &str,
    matrix: &M,
    fixture: &common::TestMatrix,
    vector: impl Fn(&[f64]) -> V,
) where
    M: MatVec<V> + ColumnStats<f64>,
    V: Clone + ElemDivAssign<f64> + DotSlice<f64> + SubScalarAssign<f64> + VectorView<f64>,
{
    let input = common::random_vec(643, matrix.ncols());
    let x = vector(&input);
    let reverse = std::env::var_os("FORWARD_INPUT_REVERSE").is_some();
    for (mode, normalization) in [
        ("raw", ColumnNormalization::default()),
        (
            "centered",
            ColumnNormalization::new(Centering::Mean, Scaling::None),
        ),
        (
            "standardized",
            ColumnNormalization::new(Centering::Mean, Scaling::Sd { df: 0 }),
        ),
    ] {
        let lazy = LazyMatrix::new(matrix, normalization).unwrap();
        let expected = common::dense_matvec(
            &common::materialize(
                &fixture.dense,
                lazy.normalization().centers(),
                lazy.normalization().scales(),
            ),
            &input,
        );
        let name = format!("forward_input/{label}/{mode}");
        let mut allocations = Vec::new();
        let methods = if reverse {
            ["operator", "cloned_reference"]
        } else {
            ["cloned_reference", "operator"]
        };
        for method in methods {
            let full_name = format!("{name}/{method}");
            let (calls, bytes) = if method == "operator" {
                measure(
                    c,
                    &full_name,
                    || black_box(&lazy).matvec(black_box(&x)).unwrap(),
                    &expected,
                )
            } else {
                measure(
                    c,
                    &full_name,
                    || {
                        cloned_reference::<DEFAULT_HOOK, _, _>(black_box(&lazy), black_box(&x))
                            .unwrap()
                    },
                    &expected,
                )
            };
            for (i, &value) in input.iter().enumerate() {
                assert_eq!(x.get(i), value, "input mutated: {name}/{method}");
            }
            allocations.push((method, calls, bytes));
        }
        let cloned = allocations
            .iter()
            .find(|a| a.0 == "cloned_reference")
            .unwrap();
        let borrowed = allocations.iter().find(|a| a.0 == "operator").unwrap();
        let saves_clone = usize::from(DEFAULT_HOOK && lazy.normalization().scales().is_none());
        assert_eq!(cloned.1, borrowed.1 + saves_clone, "{name}");
        assert_eq!(
            cloned.2,
            borrowed.2 + saves_clone * matrix.ncols() * size_of::<f64>(),
            "{name}"
        );
    }
}

fn bench_clone<V: Clone>(c: &mut Criterion, label: &str, x: &V) {
    drop(black_box(x).clone());
    let (_, calls, bytes) = count(|| black_box(x).clone());
    eprintln!("forward_input_allocations,input_clone/{label},{calls},{bytes}");
    c.bench_function(&format!("input_clone/{label}"), |b| {
        b.iter(|| black_box(x).clone())
    });
}

fn benchmark(c: &mut Criterion) {
    for columns in [32, 128, 16_384] {
        let values = common::random_vec(643, columns);
        bench_clone(
            c,
            &format!("faer/{columns}"),
            &faer::Col::from_fn(columns, |j| values[j]),
        );
        bench_clone(
            c,
            &format!("nalgebra/{columns}"),
            &nalgebra::DVector::from_column_slice(&values),
        );
        bench_clone(c, &format!("vec/{columns}"), &values);
    }
    for (rows, columns) in [(2_000, 32), (10_000, 128), (32, 16_384)] {
        for density in [1.0, 0.01, 0.1] {
            let tm = common::random_matrix(641, rows, columns, density);
            let label = format!("{rows}x{columns}/density_{density}");
            if density == 1.0 {
                let matrix = faer::Mat::from_fn(rows, columns, |i, j| tm.dense[i][j]);
                bench_matrix::<false, _, _>(c, &format!("faer_dense/{label}"), &matrix, &tm, |v| {
                    faer::Col::from_fn(v.len(), |j| v[j])
                });
                let matrix = nalgebra::DMatrix::from_fn(rows, columns, |i, j| tm.dense[i][j]);
                bench_matrix::<false, _, _>(
                    c,
                    &format!("nalgebra_dense/{label}"),
                    &matrix,
                    &tm,
                    nalgebra::DVector::from_column_slice,
                );
                let matrix =
                    ndarray::Array2::from_shape_fn((rows, columns), |(i, j)| tm.dense[i][j]);
                bench_matrix::<false, _, _>(c, &format!("ndarray/{label}"), &matrix, &tm, |v| {
                    ndarray::Array1::from_vec(v.to_vec())
                });
            } else {
                let triplets: Vec<_> = tm
                    .triplets
                    .iter()
                    .map(|&(i, j, v)| faer::sparse::Triplet::new(i, j, v))
                    .collect();
                let csc = faer::sparse::SparseColMat::<usize, f64>::try_new_from_triplets(
                    rows, columns, &triplets,
                )
                .unwrap();
                let csr = faer::sparse::SparseRowMat::<usize, f64>::try_new_from_triplets(
                    rows, columns, &triplets,
                )
                .unwrap();
                bench_matrix::<true, _, _>(c, &format!("faer_csc/{label}"), &csc, &tm, |v| {
                    faer::Col::from_fn(v.len(), |j| v[j])
                });
                bench_matrix::<true, _, _>(c, &format!("faer_csr/{label}"), &csr, &tm, |v| {
                    faer::Col::from_fn(v.len(), |j| v[j])
                });
                let mut coo = nalgebra_sparse::CooMatrix::new(rows, columns);
                let mut coo_sprs = sprs::TriMat::new((rows, columns));
                for &(i, j, value) in &tm.triplets {
                    coo.push(i, j, value);
                    coo_sprs.add_triplet(i, j, value);
                }
                bench_matrix::<true, _, _>(
                    c,
                    &format!("nalgebra_csc/{label}"),
                    &nalgebra_sparse::CscMatrix::from(&coo),
                    &tm,
                    nalgebra::DVector::from_column_slice,
                );
                bench_matrix::<true, _, _>(
                    c,
                    &format!("nalgebra_csr/{label}"),
                    &nalgebra_sparse::CsrMatrix::from(&coo),
                    &tm,
                    nalgebra::DVector::from_column_slice,
                );
                bench_matrix::<true, _, _>(
                    c,
                    &format!("sprs_csc/{label}"),
                    &coo_sprs.to_csc::<usize>(),
                    &tm,
                    |v| v.to_vec(),
                );
                bench_matrix::<true, _, _>(
                    c,
                    &format!("sprs_csr/{label}"),
                    &coo_sprs.to_csr::<usize>(),
                    &tm,
                    |v| v.to_vec(),
                );
            }
        }
    }
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
