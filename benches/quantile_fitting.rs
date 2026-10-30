//! Compare robust fitting with an independently sorted dense reference.

use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use lazymatrix::{
    Centering, ColumnNormalization, ColumnNormalizationStats, ColumnStats, LazyMatrix, MatrixShape,
    QuantileRange, Scaling, ZarrMatrix,
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[path = "common/allocations.rs"]
mod allocations;
#[path = "../tests/common/backend_aliases.rs"]
mod backend_aliases;
use backend_aliases::{ndarray, sprs, zarrs, zarrs_compat};
use zarrs::storage::byte_range::ByteRangeIterator;
use zarrs::storage::store::MemoryStore;
use zarrs::storage::{
    MaybeBytes, MaybeBytesIterator, ReadableStorageTraits, StorageError, StoreKey,
};

struct TrackedStore {
    inner: Arc<MemoryStore>,
    reads: AtomicUsize,
}

impl ReadableStorageTraits for TrackedStore {
    fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.inner.get(key)
    }

    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.inner.get_partial_many(key, ranges)
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        self.inner.size_key(key)
    }

    fn supports_get_partial(&self) -> bool {
        self.inner.supports_get_partial()
    }
}

fn sorted_reference(matrix: &ndarray::Array2<f64>) -> ColumnNormalizationStats<f64> {
    let mut centers = Vec::with_capacity(matrix.ncols());
    let mut scales = Vec::with_capacity(matrix.ncols());
    let mut column = Vec::with_capacity(matrix.nrows());
    for j in 0..matrix.ncols() {
        column.clear();
        column.extend(matrix.column(j).iter().copied());
        column.sort_unstable_by(f64::total_cmp);
        let quantile = |q: f64| {
            let rank = q * (column.len() - 1) as f64;
            let lo = rank.floor() as usize;
            let hi = rank.ceil() as usize;
            let fraction = rank - lo as f64;
            (1.0 - fraction) * column[lo] + fraction * column[hi]
        };
        centers.push(quantile(0.5));
        let range = quantile(0.75) - quantile(0.25);
        scales.push(if range == 0.0 { 1.0 } else { range });
    }
    (Some(centers), Some(scales))
}

fn parameters<M: ColumnStats<f64> + MatrixShape>(
    matrix: &M,
    spec: ColumnNormalization,
) -> ColumnNormalizationStats<f64> {
    let (_, params) = LazyMatrix::new(matrix, spec).unwrap().into_parts();
    params.into_parts()
}

fn profile(
    case: &str,
    backend: &str,
    shape: (usize, usize),
    iterations: usize,
    storage: Option<(&TrackedStore, usize)>,
    expected: &ColumnNormalizationStats<f64>,
    mut fit: impl FnMut() -> ColumnNormalizationStats<f64>,
) {
    let actual = fit();
    for (actual, expected) in actual
        .0
        .iter()
        .flatten()
        .chain(actual.1.iter().flatten())
        .zip(
            expected
                .0
                .iter()
                .flatten()
                .chain(expected.1.iter().flatten()),
        )
    {
        approx::assert_relative_eq!(actual, expected, epsilon = 1e-12, max_relative = 1e-12);
    }
    if let Some((store, _)) = storage {
        store.reads.store(0, Ordering::Relaxed);
    }
    let peak = allocations::measure(&mut fit);
    let scans = storage.map_or(0, |(store, chunks)| {
        let reads = store.reads.load(Ordering::Relaxed);
        assert_eq!(reads % chunks, 0);
        reads / chunks
    });
    let mut times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        black_box(fit());
        times.push(start.elapsed().as_secs_f64() * 1e6);
    }
    times.sort_by(f64::total_cmp);
    println!(
        "{case},{backend},{},{},{iterations},{:.3},{peak},{scans}",
        shape.0,
        shape.1,
        times[iterations / 2]
    );
}

fn main() {
    let smoke = std::env::var_os("LAZYMATRIX_BENCH_SMOKE").is_some();
    let iterations = if smoke { 3 } else { 9 };
    let rows = if smoke { 512 } else { 4096 };
    let columns = if smoke { 4 } else { 16 };
    println!("case,backend,rows,columns,iterations,median_us,peak_fit_bytes,storage_scans");
    for (case, n, p) in [
        ("sparse", rows, columns),
        ("dense", rows, columns),
        ("wide_exponents", 256, 4),
    ] {
        let mut rng = ChaCha8Rng::seed_from_u64(591);
        let raw = ndarray::Array2::from_shape_fn((n, p), |(i, j)| {
            if case == "wide_exponents" {
                2.0_f64.powi(-512 + 4 * i as i32) * (1.0 + j as f64 / 8.0)
            } else if case == "sparse" && rng.random::<f64>() < 0.9 {
                0.0
            } else {
                rng.random_range(-10.0..10.0)
            }
        });
        let mut triplets = sprs::TriMat::new((n, p));
        for ((i, j), &value) in raw.indexed_iter() {
            if value != 0.0 {
                triplets.add_triplet(i, j, value);
            }
        }
        let csc: sprs::CsMat<f64> = triplets.to_csc();
        let csr = csc.to_csr();
        let inner = Arc::new(MemoryStore::new());
        let array = zarrs::array::ArrayBuilder::new(
            vec![n as u64, p as u64],
            vec![128, p as u64],
            zarrs_compat::float64(),
            0.0_f64,
        )
        .build(inner.clone(), "/quantiles")
        .unwrap();
        array.store_metadata().unwrap();
        zarrs_compat::store_subset(&array, raw.as_slice().unwrap()).unwrap();
        let store = Arc::new(TrackedStore {
            inner,
            reads: AtomicUsize::new(0),
        });
        let chunked =
            ZarrMatrix::try_new(zarrs::array::Array::open(store.clone(), "/quantiles").unwrap())
                .unwrap();
        let expected = sorted_reference(&raw);
        let spec = ColumnNormalization::new(
            Centering::Median,
            Scaling::QuantileRange(QuantileRange::default()),
        );
        profile(
            case,
            "sorted_reference",
            (n, p),
            iterations,
            None,
            &expected,
            || sorted_reference(&raw),
        );
        profile(case, "dense", (n, p), iterations, None, &expected, || {
            parameters(&raw, spec)
        });
        profile(case, "csc", (n, p), iterations, None, &expected, || {
            parameters(&csc, spec)
        });
        profile(case, "csr", (n, p), iterations, None, &expected, || {
            parameters(&csr, spec)
        });
        profile(
            case,
            "zarr",
            (n, p),
            iterations,
            Some((&store, n.div_ceil(128))),
            &expected,
            || parameters(&chunked, spec),
        );
    }
}
