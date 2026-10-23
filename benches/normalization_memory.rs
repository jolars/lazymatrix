//! Untimed allocation profiles over construction and repeated products.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};

use lazymatrix::{
    ColumnStats, LazyMatrix, MatTransposeVecInto, MatVecInto, MaterializeDense, MatrixOwned,
};

#[path = "../tests/common/backend_aliases.rs"]
mod backend_aliases;
#[path = "common/normalization.rs"]
mod normalization;

use backend_aliases::ndarray;
use normalization::{BenchVector, Fixture, NORMALIZATIONS, PAIRS};

struct Allocator;
#[global_allocator]
static ALLOCATOR: Allocator = Allocator;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static BASELINE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static ENABLED: AtomicBool = AtomicBool::new(false);

fn record(bytes: usize) {
    let live = LIVE.fetch_add(bytes, Relaxed) + bytes;
    if ENABLED.load(Relaxed) {
        PEAK.fetch_max(live.saturating_sub(BASELINE.load(Relaxed)), Relaxed);
    }
}

// SAFETY: All pointers and layouts are forwarded unchanged to System.
unsafe impl GlobalAlloc for Allocator {
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
        LIVE.fetch_sub(layout.size(), Relaxed);
        // SAFETY: System allocated this pointer using the supplied layout.
        unsafe { System.dealloc(pointer, layout) };
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: The caller supplies System's original pointer and valid sizes.
        let result = unsafe { System.realloc(pointer, layout, size) };
        if !result.is_null() {
            if size >= layout.size() {
                record(size - layout.size());
            } else {
                LIVE.fetch_sub(layout.size() - size, Relaxed);
            }
        }
        result
    }
}

fn measure<T>(run: impl FnOnce() -> T) -> usize {
    assert!(!ENABLED.load(Relaxed));
    BASELINE.store(LIVE.load(Relaxed), Relaxed);
    PEAK.store(0, Relaxed);
    ENABLED.store(true, Relaxed);
    let result = black_box(run());
    ENABLED.store(false, Relaxed);
    let peak = PEAK.load(Relaxed);
    drop(result);
    peak
}

fn products<M, V>(fixture: &Fixture, matrix: &M) -> (V, V, V, V)
where
    M: MatVecInto<V> + MatTransposeVecInto<V>,
    V: BenchVector,
{
    let (x, u, mut y, mut t) = fixture.vectors::<V>();
    for _ in 0..PAIRS {
        matrix.matvec_into(black_box(&x), &mut y).unwrap();
        matrix
            .mat_transpose_vec_into(black_box(&u), &mut t)
            .unwrap();
        black_box((&y, &t));
    }
    (x, u, y, t)
}

fn profile<M, D, V>(fixture: &Fixture, backend: &str, source: &M)
where
    M: ColumnStats<f64> + MaterializeDense<f64> + MatVecInto<V> + MatTransposeVecInto<V>,
    D: MatrixOwned<f64> + MatVecInto<V> + MatTransposeVecInto<V>,
    V: BenchVector,
{
    for (normalization, spec) in NORMALIZATIONS {
        let parameters = fixture.cols
            * size_of::<f64>()
            * if spec.center == lazymatrix::Centering::None {
                1
            } else {
                2
            };
        let dense_bytes = fixture.rows * fixture.cols * size_of::<f64>();
        let source_bytes = fixture.source_bytes();
        // Warm backend initialization before recording a complete lifecycle.
        let lazy = LazyMatrix::new(source, spec).unwrap();
        let eager = lazy.to_eager::<D>().unwrap();
        drop(products::<_, V>(fixture, &lazy));
        drop(products::<_, V>(fixture, &eager));
        drop(eager);
        drop(lazy);

        let raw_peak = measure(|| products::<_, V>(fixture, source));
        let lazy_peak = measure(|| {
            let matrix = LazyMatrix::new(source, spec).unwrap();
            let vectors = products::<_, V>(fixture, &matrix);
            (matrix, vectors)
        });
        let eager_peak = measure(|| {
            let lazy = LazyMatrix::new(source, spec).unwrap();
            let matrix = lazy.to_eager::<D>().unwrap();
            drop(lazy);
            let vectors = products::<_, V>(fixture, &matrix);
            (matrix, vectors)
        });
        for (strategy, representation, added_representation, peak) in [
            ("raw", source_bytes, 0, raw_peak),
            ("lazy", source_bytes + parameters, parameters, lazy_peak),
            (
                "eager",
                dense_bytes + parameters,
                dense_bytes + parameters,
                eager_peak,
            ),
        ] {
            assert!(peak >= added_representation);
            println!(
                "{backend},{},{},{},{},{},{normalization},{strategy},{representation},{},{},{}",
                fixture.rows,
                fixture.cols,
                fixture.density,
                fixture.sparse.nnz(),
                fixture.empty_columns(),
                peak - added_representation,
                source_bytes + peak,
                source_bytes
            );
        }
    }
}

fn main() {
    println!(
        "backend,n,p,target_density,nnz,empty_columns,normalization,strategy,representation_bytes,workspace_bytes,peak_total_bytes,source_bytes"
    );
    for fixture in normalization::fixtures() {
        profile::<_, ndarray::Array2<f64>, ndarray::Array1<f64>>(
            &fixture,
            "sprs_to_ndarray",
            &fixture.sparse,
        );
        #[cfg(feature = "faer_all")]
        {
            use backend_aliases::faer;
            let source = fixture.faer();
            profile::<_, faer::Mat<f64>, faer::Col<f64>>(&fixture, "faer_csc_to_dense", &source);
        }
    }
}
