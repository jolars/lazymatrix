//! Compare batched products and multiresponse iterations with vector consumers.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::RefCell;
use std::hint::black_box;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed},
};
use std::time::{Duration, Instant};

use lazymatrix::{
    Centering, ColumnNormalization, ColumnStats, LazyMatrix, MatMat, MatMatInto, MatTransposeMat,
    MatTransposeMatInto, MatTransposeVec, MatTransposeVecInto, MatVec, MatVecInto, MatrixErrorType,
    MatrixRead, MatrixShape, MatrixWrite, Scaling, WithIntercept, ZarrMatrix,
};

#[path = "../tests/common/backend_aliases.rs"]
mod backend_aliases;
#[path = "../examples/common/multiresponse.rs"]
mod multiresponse;
#[path = "common/normalization.rs"]
mod normalization;
use backend_aliases::{faer, nalgebra, nalgebra_sparse, ndarray, zarrs, zarrs_compat};
use ndarray::Array2;
use normalization::{BenchVector, Fixture};
use zarrs::storage::{
    MaybeBytes, MaybeBytesIterator, ReadableStorageTraits, StorageError, StoreKey,
    byte_range::ByteRangeIterator, store::MemoryStore,
};

struct Allocator;
#[global_allocator]
static ALLOCATOR: Allocator = Allocator;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static BASELINE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static CALLS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static COUNTING: AtomicBool = AtomicBool::new(false);
static READS: AtomicUsize = AtomicUsize::new(0);

fn record(bytes: usize) {
    let live = LIVE.fetch_add(bytes, Relaxed) + bytes;
    if COUNTING.load(Relaxed) {
        CALLS.fetch_add(1, Relaxed);
        BYTES.fetch_add(bytes, Relaxed);
        PEAK.fetch_max(live.saturating_sub(BASELINE.load(Relaxed)), Relaxed);
    }
}

// SAFETY: All pointers and layouts are forwarded unchanged to System.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: The caller supplies a valid layout.
        let result = unsafe { System.alloc(layout) };
        if !result.is_null() {
            record(layout.size());
        }
        result
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: The caller supplies a valid layout.
        let result = unsafe { System.alloc_zeroed(layout) };
        if !result.is_null() {
            record(layout.size());
        }
        result
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
                if COUNTING.load(Relaxed) {
                    BYTES.fetch_add(layout.size(), Relaxed);
                }
            } else {
                LIVE.fetch_sub(layout.size() - size, Relaxed);
            }
            if COUNTING.load(Relaxed) && size < layout.size() {
                CALLS.fetch_add(1, Relaxed);
                BYTES.fetch_add(size, Relaxed);
            }
        }
        result
    }
}

struct CountingStore(Arc<MemoryStore>);
impl ReadableStorageTraits for CountingStore {
    fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        READS.fetch_add(1, Relaxed);
        self.0.get(key)
    }
    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        READS.fetch_add(1, Relaxed);
        self.0.get_partial_many(key, ranges)
    }
    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        self.0.size_key(key)
    }
    fn supports_get_partial(&self) -> bool {
        self.0.supports_get_partial()
    }
}

struct Buffers<V> {
    input: V,
    output: V,
    transpose_input: V,
    transpose_output: V,
}
struct Columnwise<'a, M, V, const ALLOCATING: bool> {
    matrix: &'a M,
    buffers: RefCell<Buffers<V>>,
}
impl<M: MatrixShape, V: BenchVector, const ALLOCATING: bool> Columnwise<'_, M, V, ALLOCATING> {
    fn new(matrix: &M) -> Columnwise<'_, M, V, ALLOCATING> {
        Columnwise {
            matrix,
            buffers: RefCell::new(Buffers {
                input: V::owned_from_fn(matrix.ncols(), |_| 0.),
                output: V::owned_from_fn(matrix.nrows(), |_| 0.),
                transpose_input: V::owned_from_fn(matrix.nrows(), |_| 0.),
                transpose_output: V::owned_from_fn(matrix.ncols(), |_| 0.),
            }),
        }
    }
}
impl<M: MatrixShape, V, const A: bool> MatrixShape for Columnwise<'_, M, V, A> {
    fn nrows(&self) -> usize {
        self.matrix.nrows()
    }
    fn ncols(&self) -> usize {
        self.matrix.ncols()
    }
}
impl<M: MatrixErrorType, V, const A: bool> MatrixErrorType for Columnwise<'_, M, V, A> {
    type Error = M::Error;
}

macro_rules! columnwise {
    ($trait:ident, $product:ident, $allocating:ident, $reusable:ident, $input:ident, $output:ident, $input_rows:ident, $output_rows:ident) => {
        impl<M, V, const A: bool> $trait<f64> for Columnwise<'_, M, V, A>
        where
            M: MatVec<V> + MatTransposeVec<V> + MatVecInto<V> + MatTransposeVecInto<V>,
            V: BenchVector,
        {
            fn $product<R, O>(&self, rhs: &R, out: &mut O) -> Result<(), Self::Error>
            where
                R: MatrixRead<f64> + ?Sized,
                O: MatrixRead<f64> + MatrixWrite<f64> + ?Sized,
            {
                assert_eq!(rhs.nrows(), self.$input_rows());
                assert_eq!(
                    (out.nrows(), out.ncols()),
                    (self.$output_rows(), rhs.ncols())
                );
                if A {
                    for k in 0..rhs.ncols() {
                        let input = V::owned_from_fn(rhs.nrows(), |i| rhs.get(i, k));
                        let result = self.matrix.$allocating(&input)?;
                        for i in 0..out.nrows() {
                            out.set(i, k, result.get(i));
                        }
                    }
                } else {
                    let Buffers {
                        $input: input,
                        $output: output,
                        ..
                    } = &mut *self.buffers.borrow_mut();
                    for k in 0..rhs.ncols() {
                        for i in 0..rhs.nrows() {
                            input.set(i, rhs.get(i, k));
                        }
                        self.matrix.$reusable(input, output)?;
                        for i in 0..out.nrows() {
                            out.set(i, k, output.get(i));
                        }
                    }
                }
                Ok(())
            }
        }
    };
}
columnwise!(
    MatMatInto,
    matmat_into,
    matvec,
    matvec_into,
    input,
    output,
    ncols,
    nrows
);
columnwise!(
    MatTransposeMatInto,
    mat_transpose_mat_into,
    mat_transpose_vec,
    mat_transpose_vec_into,
    transpose_input,
    transpose_output,
    nrows,
    ncols
);

struct Context<'a> {
    backend: &'a str,
    normalization: &'a str,
    n: usize,
    p: usize,
    k: usize,
    smoke: bool,
}

impl Context<'_> {
    fn measure(&self, operation: &str, variant: &str, mut run: impl FnMut()) {
        run();
        CALLS.store(0, Relaxed);
        BYTES.store(0, Relaxed);
        READS.store(0, Relaxed);
        PEAK.store(0, Relaxed);
        BASELINE.store(LIVE.load(Relaxed), Relaxed);
        COUNTING.store(true, Relaxed);
        run();
        COUNTING.store(false, Relaxed);
        let (calls, bytes, peak, reads) = (
            CALLS.load(Relaxed),
            BYTES.load(Relaxed),
            PEAK.load(Relaxed),
            READS.load(Relaxed),
        );
        let target = Duration::from_millis(if self.smoke { 1 } else { 10 });
        let mut samples = Vec::new();
        for _ in 0..if self.smoke { 3 } else { 7 } {
            let start = Instant::now();
            let mut repeats = 0;
            while start.elapsed() < target {
                run();
                repeats += 1;
            }
            samples.push(start.elapsed().as_secs_f64() * 1e9 / repeats as f64);
        }
        samples.sort_by(f64::total_cmp);
        println!(
            "{},{},{},{},{},{operation},{variant},{:.0},{calls},{bytes},{peak},{reads}",
            self.backend,
            self.normalization,
            self.n,
            self.p,
            self.k,
            samples[samples.len() / 2]
        );
    }
}

fn check(left: &Array2<f64>, right: &Array2<f64>) {
    assert_eq!(left.dim(), right.dim());
    for (&a, &b) in left.iter().zip(right) {
        assert!((a - b).abs() <= 1e-8 * a.abs().max(b.abs()).max(1.));
    }
}

fn profile<M, V>(fixture: &Fixture, name: &str, source: &M, smoke: bool)
where
    M: ColumnStats<f64>
        + MatMatInto<f64>
        + MatTransposeMatInto<f64>
        + MatVec<V>
        + MatTransposeVec<V>
        + MatVecInto<V>
        + MatTransposeVecInto<V>,
    V: BenchVector,
{
    for (normalization, spec) in [
        (
            "raw",
            ColumnNormalization::new(Centering::None, Scaling::None),
        ),
        (
            "centered",
            ColumnNormalization::new(Centering::Mean, Scaling::None),
        ),
        (
            "scaled",
            ColumnNormalization::new(Centering::None, Scaling::Sd { df: 0 }),
        ),
        (
            "standardized",
            ColumnNormalization::new(Centering::Mean, Scaling::Sd { df: 0 }),
        ),
    ] {
        let matrix = LazyMatrix::new(source, spec).unwrap();
        for k in [1, 4, 16, 64] {
            let (n, p) = (fixture.rows, fixture.cols);
            let context = Context {
                backend: name,
                normalization,
                n,
                p,
                k,
                smoke,
            };
            let rhs = Array2::from_shape_fn((p, k), |(i, r)| ((i + r) % 7) as f64 - 3.);
            let u = Array2::from_shape_fn((n, k), |(i, r)| ((i + r) % 5) as f64 - 2.);
            let inputs: Vec<_> = (0..k)
                .map(|r| V::owned_from_fn(p, |j| rhs[(j, r)]))
                .collect();
            let transpose_inputs: Vec<_> =
                (0..k).map(|r| V::owned_from_fn(n, |i| u[(i, r)])).collect();
            let mut outputs: Vec<_> = (0..k).map(|_| V::owned_from_fn(n, |_| 0.)).collect();
            let mut transpose_outputs: Vec<_> =
                (0..k).map(|_| V::owned_from_fn(p, |_| 0.)).collect();
            let mut out = Array2::zeros((n, k));
            let mut out_t = Array2::zeros((p, k));
            let reference = Columnwise::<_, V, false>::new(&matrix);
            reference.matmat_into(&rhs, &mut out).unwrap();
            let actual: Array2<f64> = matrix.matmat(&rhs).unwrap();
            check(&actual, &out);
            reference.mat_transpose_mat_into(&u, &mut out_t).unwrap();
            let actual: Array2<f64> = matrix.mat_transpose_mat(&u).unwrap();
            check(&actual, &out_t);
            context.measure("forward", "vector_allocating", || {
                black_box(
                    inputs
                        .iter()
                        .map(|input| matrix.matvec(black_box(input)).unwrap())
                        .collect::<Vec<_>>(),
                );
            });
            context.measure("forward", "vector_into", || {
                for (input, output) in inputs.iter().zip(&mut outputs) {
                    matrix.matvec_into(black_box(input), output).unwrap();
                }
                black_box(&outputs);
            });
            context.measure("forward", "batch_allocating", || {
                black_box(matrix.matmat::<_, Array2<f64>>(black_box(&rhs)).unwrap());
            });
            context.measure("forward", "batch_into", || {
                matrix.matmat_into(black_box(&rhs), &mut out).unwrap();
                black_box(&out);
            });
            context.measure("transpose", "vector_allocating", || {
                black_box(
                    transpose_inputs
                        .iter()
                        .map(|input| matrix.mat_transpose_vec(black_box(input)).unwrap())
                        .collect::<Vec<_>>(),
                );
            });
            context.measure("transpose", "vector_into", || {
                for (input, output) in transpose_inputs.iter().zip(&mut transpose_outputs) {
                    matrix
                        .mat_transpose_vec_into(black_box(input), output)
                        .unwrap();
                }
                black_box(&transpose_outputs);
            });
            context.measure("transpose", "batch_allocating", || {
                black_box(
                    matrix
                        .mat_transpose_mat::<_, Array2<f64>>(black_box(&u))
                        .unwrap(),
                );
            });
            context.measure("transpose", "batch_into", || {
                matrix
                    .mat_transpose_mat_into(black_box(&u), &mut out_t)
                    .unwrap();
                black_box(&out_t);
            });

            if normalization == "standardized" && matches!(k, 4 | 16) {
                let design = WithIntercept::new(&matrix);
                let truth = Array2::from_shape_fn((p + 1, k), |(j, r)| ((j + r) % 11) as f64 / 11.);
                let response: Array2<f64> = design.matmat(&truth).unwrap();
                let allocating = Columnwise::<_, V, true>::new(&design);
                let reusable = Columnwise::<_, V, false>::new(&design);
                let step = 1. / (p + 1) as f64;
                let batch = multiresponse::solve(&design, &response, step, 10, 0.).unwrap();
                let columns = multiresponse::solve(&reusable, &response, step, 10, 0.).unwrap();
                check(&batch.coefficients, &columns.coefficients);
                assert_eq!(batch.iterations, 10);
                assert!(batch.gradient_norm.is_finite());
                context.measure("ten_updates", "vector_allocating", || {
                    black_box(
                        multiresponse::solve(&allocating, black_box(&response), step, 10, 0.)
                            .unwrap(),
                    );
                });
                context.measure("ten_updates", "vector_into", || {
                    black_box(
                        multiresponse::solve(&reusable, black_box(&response), step, 10, 0.)
                            .unwrap(),
                    );
                });
                context.measure("ten_updates", "batch_into", || {
                    black_box(
                        multiresponse::solve(&design, black_box(&response), step, 10, 0.).unwrap(),
                    );
                });
            }
        }
    }
}

fn main() {
    let smoke = std::env::var_os("LAZYMATRIX_BENCH_SMOKE").is_some();
    let (n, p) = if smoke { (256, 16) } else { (1024, 64) };
    let fixture = Fixture::new(n, p, 0.05);
    let mut dense = Array2::zeros((n, p));
    for (j, column) in fixture.sparse.outer_iterator().enumerate() {
        for (i, &v) in column.iter() {
            dense[(i, j)] = v;
        }
    }
    println!(
        "backend,normalization,n,p,responses,operation,variant,median_ns,allocations,allocated_bytes,peak_additional_bytes,storage_reads"
    );
    profile::<_, ndarray::Array1<f64>>(&fixture, "ndarray_dense", &dense, smoke);
    profile::<_, Vec<f64>>(&fixture, "sprs_csc", &fixture.sparse, smoke);
    profile::<_, Vec<f64>>(&fixture, "sprs_csr", &fixture.sparse.to_csr(), smoke);
    let faer_dense = faer::Mat::from_fn(n, p, |i, j| dense[(i, j)]);
    profile::<_, faer::Col<f64>>(&fixture, "faer_dense", &faer_dense, smoke);
    profile::<_, faer::Col<f64>>(&fixture, "faer_csc", &fixture.faer(), smoke);
    let triplets: Vec<_> = fixture
        .sparse
        .outer_iterator()
        .enumerate()
        .flat_map(|(j, col)| {
            col.iter()
                .map(|(i, &v)| faer::sparse::Triplet::new(i, j, v))
                .collect::<Vec<_>>()
        })
        .collect();
    let csr = faer::sparse::SparseRowMat::try_new_from_triplets(n, p, &triplets).unwrap();
    profile::<_, faer::Col<f64>>(&fixture, "faer_csr", &csr, smoke);
    let na_dense = nalgebra::DMatrix::from_fn(n, p, |i, j| dense[(i, j)]);
    profile::<_, nalgebra::DVector<f64>>(&fixture, "nalgebra_dense", &na_dense, smoke);
    let mut coo = nalgebra_sparse::CooMatrix::new(n, p);
    for (j, col) in fixture.sparse.outer_iterator().enumerate() {
        for (i, &v) in col.iter() {
            coo.push(i, j, v);
        }
    }
    profile::<_, nalgebra::DVector<f64>>(
        &fixture,
        "nalgebra_csc",
        &nalgebra_sparse::CscMatrix::from(&coo),
        smoke,
    );
    profile::<_, nalgebra::DVector<f64>>(
        &fixture,
        "nalgebra_csr",
        &nalgebra_sparse::CsrMatrix::from(&coo),
        smoke,
    );
    let store = Arc::new(MemoryStore::new());
    let array = zarrs::array::ArrayBuilder::new(
        vec![n as u64, p as u64],
        vec![128, 32],
        zarrs_compat::float64(),
        0_f64,
    )
    .build(store.clone(), "/matrix")
    .unwrap();
    array.store_metadata().unwrap();
    zarrs_compat::store_subset(&array, dense.as_slice().unwrap()).unwrap();
    let array = zarrs::array::Array::open(Arc::new(CountingStore(store)), "/matrix").unwrap();
    let matrix = ZarrMatrix::<_, f64>::try_new(array).unwrap();
    profile::<_, Vec<f64>>(&fixture, "zarrs_memory", &matrix, smoke);
}
