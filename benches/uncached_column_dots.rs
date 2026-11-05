//! Compare fused uncached dots with production, separate, and cached reductions.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    hint::black_box,
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

use lazymatrix::{LazyColumn, LazyMatrix, MatrixShape, RawColumn, RawColumns, Scalar, VectorView};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

// Compile the actual production kernels rather than a copied baseline.
#[allow(dead_code)]
#[path = "../src/reductions.rs"]
mod reductions;

struct CountAllocations;
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountAllocations {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(pointer, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountAllocations = CountAllocations;

trait ProbeScalar: Scalar {
    fn same(self, other: Self) -> bool;
}

macro_rules! scalars {
    ($($scalar:ty),*) => {$(
        impl ProbeScalar for $scalar {
            fn same(self, other: Self) -> bool {
                (self.is_nan() && other.is_nan()) || self.to_bits() == other.to_bits()
            }
        }
    )*};
}
scalars!(f32, f64);

#[derive(Clone, Copy, Debug, PartialEq)]
enum Operation {
    Dot,
    WeightedDot,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Method {
    Production,
    Separate,
    Fused8,
    Fused8Split,
    Fused4,
    Cached,
}

const OPERATIONS: [Operation; 2] = [Operation::Dot, Operation::WeightedDot];
const METHODS: [Method; 6] = [
    Method::Production,
    Method::Separate,
    Method::Fused8,
    Method::Fused8Split,
    Method::Fused4,
    Method::Cached,
];

fn combine<F: Scalar, const LANES: usize>(sums: [F; LANES]) -> F {
    (0..LANES / 2).fold(F::zero(), |sum, i| sum + (sums[i] + sums[i + LANES / 2]))
}

#[inline]
fn fused<F: Scalar, const LANES: usize, const WEIGHTED: bool, const SPLIT: bool>(
    column: &[F],
    vector: &[F],
    weights: &[F],
) -> (F, F) {
    assert_eq!(column.len(), vector.len());
    assert_eq!(column.len(), weights.len());
    let mut products = [F::zero(); LANES];
    let mut corrections = [F::zero(); LANES];
    let mut column = column.chunks_exact(LANES);
    let mut vector = vector.chunks_exact(LANES);
    let mut weights = weights.chunks_exact(LANES);
    for ((x, v), w) in column.by_ref().zip(vector.by_ref()).zip(weights.by_ref()) {
        if SPLIT {
            // Group the two reductions separately to test whether LLVM packs
            // adjacent rows instead of pairing each product with a correction.
            for lane in 0..LANES {
                products[lane] = products[lane]
                    + if WEIGHTED {
                        x[lane] * w[lane] * v[lane]
                    } else {
                        x[lane] * v[lane]
                    };
            }
            for lane in 0..LANES {
                corrections[lane] =
                    corrections[lane] + if WEIGHTED { w[lane] * v[lane] } else { v[lane] };
            }
        } else {
            for lane in 0..LANES {
                // Preserve x*w*v, including intermediate overflow. Reusing w*v
                // in the raw product would change the production arithmetic.
                let (product, correction) = if WEIGHTED {
                    (x[lane] * w[lane] * v[lane], w[lane] * v[lane])
                } else {
                    (x[lane] * v[lane], v[lane])
                };
                products[lane] = products[lane] + product;
                corrections[lane] = corrections[lane] + correction;
            }
        }
    }
    let mut product = combine(products);
    let mut correction = combine(corrections);
    for ((&x, &v), &w) in column
        .remainder()
        .iter()
        .zip(vector.remainder())
        .zip(weights.remainder())
    {
        if WEIGHTED {
            product = product + x * w * v;
            correction = correction + w * v;
        } else {
            product = product + x * v;
            correction = correction + v;
        }
    }
    (product, correction)
}

// Every method crosses the same call boundary, including the public API paths.
#[inline(never)]
fn dispatch<C, F, V, W>(
    column: &LazyColumn<C, F>,
    vector: &V,
    weights: &W,
    cached_sum: F,
    operation: Operation,
    method: Method,
) -> F
where
    C: RawColumn<F>,
    F: Scalar,
    V: VectorView<F> + ?Sized,
    W: VectorView<F> + ?Sized,
{
    assert_eq!(column.len(), vector.len());
    assert_eq!(column.len(), weights.len());
    if method == Method::Cached {
        return match operation {
            Operation::Dot => column.dot_with_sum(vector, cached_sum),
            Operation::WeightedDot => column.weighted_dot_with_sum(vector, weights, cached_sum),
        };
    }
    if let (Some(x), Some(v), Some(w), false) = (
        column.raw().dense_values(),
        vector.contiguous_values(),
        weights.contiguous_values(),
        method == Method::Production,
    ) {
        let (product, correction) = match (operation, method) {
            (Operation::Dot, Method::Separate) => {
                let correction = reductions::sum(v);
                (reductions::dot(x, v), correction)
            }
            (Operation::WeightedDot, Method::Separate) => {
                let correction = reductions::dot(v, w);
                (reductions::weighted_dot(x, w, v), correction)
            }
            (Operation::Dot, Method::Fused8) => fused::<_, 8, false, false>(x, v, w),
            (Operation::WeightedDot, Method::Fused8) => fused::<_, 8, true, false>(x, v, w),
            (Operation::Dot, Method::Fused8Split) => fused::<_, 8, false, true>(x, v, w),
            (Operation::WeightedDot, Method::Fused8Split) => fused::<_, 8, true, true>(x, v, w),
            (Operation::Dot, Method::Fused4) => fused::<_, 4, false, false>(x, v, w),
            (Operation::WeightedDot, Method::Fused4) => fused::<_, 4, true, false>(x, v, w),
            _ => unreachable!(),
        };
        return (product - column.center() * correction) / column.scale();
    }
    match operation {
        Operation::Dot => column.dot(vector),
        Operation::WeightedDot => column.weighted_dot(vector, weights),
    }
}

// The adapter models the logical-order slices shared by dense backends.
// NaN padding makes accidental use of capacity or adjacent columns observable.
struct Dense<F> {
    rows: usize,
    cols: usize,
    values: Vec<F>,
}

struct DenseColumn<'a, F>(&'a [F]);

impl<F: Scalar> RawColumn<F> for DenseColumn<'_, F> {
    fn len(&self) -> usize {
        self.0.len()
    }
    fn stored_len(&self) -> usize {
        self.0.len()
    }
    fn for_each_stored(&self, mut visit: impl FnMut(usize, F)) {
        for (i, &x) in self.0.iter().enumerate() {
            visit(i, x);
        }
    }
    fn dense_values(&self) -> Option<&[F]> {
        Some(self.0)
    }
}

impl<F> MatrixShape for Dense<F> {
    fn nrows(&self) -> usize {
        self.rows
    }
    fn ncols(&self) -> usize {
        self.cols
    }
}

impl<F: Scalar> RawColumns<F> for Dense<F> {
    type Column<'a>
        = DenseColumn<'a, F>
    where
        Self: 'a;

    fn raw_column(&self, j: usize) -> Self::Column<'_> {
        DenseColumn(self.values(j))
    }
}

impl<F: Scalar> Dense<F> {
    fn new(rows: usize, cols: usize, mut value: impl FnMut() -> F) -> Self {
        let mut values = Vec::with_capacity((rows + 6) * cols + 31);
        for _ in 0..cols {
            values.extend([F::nan(); 3]);
            values.extend((0..rows).map(|_| value()));
            values.extend([F::nan(); 3]);
        }
        Self { rows, cols, values }
    }

    fn values(&self, j: usize) -> &[F] {
        let start = j * (self.rows + 6) + 3;
        &self.values[start..start + self.rows]
    }
}

fn compensated(values: impl Iterator<Item = f64>) -> f64 {
    let (mut sum, mut correction) = (0.0_f64, 0.0_f64);
    for x in values {
        let next = sum + x;
        correction += if sum.abs() >= x.abs() {
            (sum - next) + x
        } else {
            (x - next) + sum
        };
        sum = next;
    }
    sum + correction
}

fn cached_sum<F: Scalar>(v: &[F], w: &[F], operation: Operation) -> F {
    match operation {
        Operation::Dot => VectorView::sum(v),
        Operation::WeightedDot => reductions::dot(v, w),
    }
}

struct Probe {
    label: String,
    samples: usize,
    elements: usize,
    order: usize,
}

impl Probe {
    fn measure<F: ProbeScalar>(&self, rows: usize, regime: &str, bytes: usize) {
        let cols = if regime == "rotating" {
            bytes.div_ceil(3 * rows.max(1) * size_of::<F>())
        } else {
            1
        };
        let scalar = std::any::type_name::<F>();
        let mut rng = ChaCha8Rng::seed_from_u64(8421);
        let mut random = || F::from_f64(rng.random_range(-1.0..1.0)).unwrap();
        let raw = Dense::new(rows, cols, &mut random);
        let vectors = Dense::new(rows, cols, &mut random);
        let weights = Dense::new(rows, cols, random);
        let center = F::from_f64(0.375).unwrap();
        let scale = F::from_f64(-1.75).unwrap();
        let matrix =
            LazyMatrix::from_column_parts(raw, Some(vec![center; cols]), Some(vec![scale; cols]));
        let sums: Vec<_> = (0..cols)
            .map(|j| {
                OPERATIONS
                    .map(|operation| cached_sum(vectors.values(j), weights.values(j), operation))
            })
            .collect();
        for (op_index, operation) in OPERATIONS.into_iter().enumerate() {
            let mut errors = [(0.0_f64, 0.0_f64); METHODS.len()];
            for (j, sums) in sums.iter().enumerate() {
                let column = matrix.column(j);
                let v = vectors.values(j);
                let w = weights.values(j);
                let x = column.raw().dense_values().unwrap();
                let oracle = compensated((0..rows).map(|i| {
                    let value = (x[i].to_f64().unwrap() - 0.375) / -1.75;
                    let vector = v[i].to_f64().unwrap();
                    match operation {
                        Operation::Dot => value * vector,
                        Operation::WeightedDot => value * w[i].to_f64().unwrap() * vector,
                    }
                }));
                let production =
                    dispatch(&column, v, w, sums[op_index], operation, Method::Production);
                for (index, method) in METHODS.into_iter().enumerate() {
                    let allocations = ALLOCATIONS.load(Ordering::Relaxed);
                    let actual = dispatch(&column, v, w, sums[op_index], operation, method);
                    assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), allocations);
                    if method != Method::Fused4 {
                        assert!(actual.same(production), "{operation:?}/{method:?}");
                    }
                    let error = (actual.to_f64().unwrap() - oracle).abs();
                    let scaled = error / (1.0 + oracle.abs());
                    let tolerance =
                        64.0 * F::epsilon().to_f64().unwrap() * (rows as f64).sqrt().max(1.0);
                    assert!(actual.is_finite() && scaled <= tolerance);
                    errors[index].0 = errors[index].0.max(error);
                    errors[index].1 = errors[index].1.max(scaled);
                }
            }
            for (method, (absolute, scaled)) in METHODS.into_iter().zip(errors) {
                eprintln!(
                    "{},{scalar},{regime},{rows},{cols},{operation:?},{method:?},{absolute:.17e},{scaled:.17e},0",
                    self.label
                );
            }
        }
        let passes = (self.elements / (rows * cols).max(1)).clamp(1, 100_000);
        let footprint = (3 * (rows + 6) * cols + 93) * size_of::<F>() + (cols * 4) * size_of::<F>();
        for sample in 0..self.samples {
            // Rotate methods through every timing position across blocks.
            for op_position in 0..OPERATIONS.len() {
                let op_index = (op_position + sample + self.order) % OPERATIONS.len();
                let operation = OPERATIONS[op_index];
                for position in 0..METHODS.len() {
                    let index = (position + sample + self.order) % METHODS.len();
                    let method = METHODS[index];
                    let pass = || {
                        let mut checksum = F::zero();
                        for (j, sums) in sums.iter().enumerate() {
                            let column = black_box(&matrix).column(black_box(j));
                            checksum = checksum
                                + black_box(dispatch(
                                    &column,
                                    black_box(vectors.values(j)),
                                    black_box(weights.values(j)),
                                    black_box(sums[op_index]),
                                    black_box(operation),
                                    black_box(method),
                                ));
                        }
                        checksum
                    };
                    black_box(pass());
                    let allocations = ALLOCATIONS.load(Ordering::Relaxed);
                    let start = Instant::now();
                    let mut checksum = F::zero();
                    for _ in 0..passes {
                        checksum = checksum + black_box(pass());
                    }
                    let elapsed = start.elapsed().as_secs_f64() * 1e9 / (passes * cols) as f64;
                    assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), allocations);
                    println!(
                        "{},{scalar},{regime},{rows},{cols},{footprint},{operation:?},{method:?},{sample},{passes},{elapsed:.3},{checksum:?}",
                        self.label
                    );
                }
            }
        }
    }
}

fn numeric_case<F: ProbeScalar>(name: &str, x: &[F], v: &[F], w: &[F], center: F, scale: F) {
    let mut values = vec![F::nan(); 3];
    values.extend_from_slice(x);
    values.extend([F::nan(); 3]);
    let matrix = LazyMatrix::from_column_parts(
        Dense {
            rows: x.len(),
            cols: 1,
            values,
        },
        Some(vec![center]),
        Some(vec![scale]),
    );
    let column = matrix.column(0);
    for operation in OPERATIONS {
        let sum = cached_sum(v, w, operation);
        let production = dispatch(&column, v, w, sum, operation, Method::Production);
        // Direct centering is a diagnostic, not the contract of raw dots.
        let direct: f64 = (0..x.len())
            .map(|i| {
                let normalized =
                    (x[i].to_f64().unwrap() - center.to_f64().unwrap()) / scale.to_f64().unwrap();
                let v = v[i].to_f64().unwrap();
                match operation {
                    Operation::Dot => normalized * v,
                    Operation::WeightedDot => normalized * w[i].to_f64().unwrap() * v,
                }
            })
            .sum();
        let regrouped_raw: F = (0..x.len())
            .map(|i| match operation {
                Operation::Dot => x[i] * v[i],
                Operation::WeightedDot => x[i] * (w[i] * v[i]),
            })
            .sum();
        let regrouped = (regrouped_raw - center * sum) / scale;
        for method in METHODS {
            let actual = dispatch(&column, v, w, sum, operation, method);
            if method != Method::Fused4 {
                assert!(actual.same(production), "{name}/{operation:?}/{method:?}");
            }
            println!(
                "{},{name},{operation:?},{method:?},{actual:?},{production:?},{direct:.17e},{regrouped:?},{}",
                std::any::type_name::<F>(),
                actual.same(production)
            );
        }
    }
}

fn numerics<F: ProbeScalar>() {
    let f = |x| F::from_f64(x).unwrap();
    for n in [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
        let x: Vec<_> = (0..n).map(|i| f(i as f64 / 8.0)).collect();
        let v: Vec<_> = (0..n).map(|i| f((i % 5) as f64 - 2.0)).collect();
        let w: Vec<_> = (0..n).map(|i| f((i % 3) as f64 - 1.0)).collect();
        numeric_case(&format!("tail_{n}"), &x, &v, &w, f(0.375), f(-1.75));
    }
    let large = F::one() / F::epsilon();
    let maximum = F::max_value();
    let alternating: Vec<_> = (0..33)
        .map(|i| if i % 2 == 0 { large } else { -large })
        .collect();
    let offset: Vec<_> = (0..33).map(|i| large + f((i % 7) as f64)).collect();
    let one = [F::one(); 33];
    let two = [f(2.0); 33];
    let lane_cancellation: Vec<_> = (0..16)
        .map(|i| [f(4.0) * large, F::one(), -f(4.0) * large, F::one()][i % 4])
        .collect();
    assert_eq!(reductions::sum(&lane_cancellation), f(4.0));
    assert_eq!(
        fused::<_, 4, false, false>(&[F::one(); 16], &lane_cancellation, &[F::one(); 16]).0,
        f(8.0)
    );
    numeric_case(
        "lane_cancellation",
        &[F::one(); 16],
        &lane_cancellation,
        &[F::one(); 16],
        F::zero(),
        F::one(),
    );
    numeric_case("cancellation", &alternating, &one, &one, f(0.375), f(-1.75));
    numeric_case("large_offset", &offset, &one, &one, large, F::one());
    let sum_overflow: Vec<_> = (0..33)
        .map(|i| {
            if i % 16 < 8 {
                maximum / f(2.0)
            } else {
                -maximum / f(2.0)
            }
        })
        .collect();
    numeric_case(
        "sum_overflow",
        &one,
        &sum_overflow,
        &one,
        F::zero(),
        F::one(),
    );
    numeric_case(
        "correction_overflow",
        &one,
        &[maximum / f(2.0); 33],
        &one,
        F::one(),
        F::one(),
    );
    let mut product_order = [F::zero(); 33];
    product_order[8] = maximum / f(2.0);
    assert!(
        fused::<_, 8, true, false>(&product_order, &[f(0.25); 33], &[f(4.0); 33])
            .0
            .is_infinite()
    );
    let regrouped: F = product_order.iter().map(|&x| x * (f(4.0) * f(0.25))).sum();
    assert!(regrouped.is_finite());
    numeric_case(
        "weighted_product_order",
        &product_order,
        &[f(0.25); 33],
        &[f(4.0); 33],
        F::zero(),
        F::one(),
    );
    numeric_case(
        "zero_center_infinite_sum",
        &one,
        &[F::infinity(); 33],
        &one,
        F::zero(),
        F::one(),
    );
    for (name, value) in [
        ("nan", F::nan()),
        ("positive_infinity", F::infinity()),
        ("negative_infinity", F::neg_infinity()),
        ("positive_zero", F::zero()),
        ("negative_zero", -F::zero()),
    ] {
        let mut exceptional = one;
        exceptional[8] = value;
        numeric_case(
            &format!("column_{name}"),
            &exceptional,
            &one,
            &one,
            f(0.375),
            f(-1.75),
        );
        numeric_case(
            &format!("vector_{name}"),
            &two,
            &exceptional,
            &one,
            f(0.375),
            f(-1.75),
        );
        numeric_case(
            &format!("weights_{name}"),
            &two,
            &one,
            &exceptional,
            f(0.375),
            f(-1.75),
        );
        numeric_case(&format!("center_{name}"), &two, &one, &one, value, f(-1.75));
        if value != F::zero() {
            numeric_case(&format!("scale_{name}"), &two, &one, &one, f(0.375), value);
        }
    }
}

struct Sparse {
    len: usize,
    visits: Cell<usize>,
}
impl MatrixShape for Sparse {
    fn nrows(&self) -> usize {
        self.len
    }
    fn ncols(&self) -> usize {
        1
    }
}
impl RawColumn<f64> for &Sparse {
    fn len(&self) -> usize {
        self.len
    }
    fn stored_len(&self) -> usize {
        5
    }
    fn for_each_stored(&self, mut visit: impl FnMut(usize, f64)) {
        for (i, x) in [(8, 2.0), (1, -3.0), (8, 0.5), (0, 0.0), (6, 4.0)] {
            self.visits.set(self.visits.get() + 1);
            visit(i, x);
        }
    }
}
impl RawColumns<f64> for Sparse {
    type Column<'a> = &'a Sparse;
    fn raw_column(&self, _: usize) -> Self::Column<'_> {
        self
    }
}
struct CountedVector {
    len: usize,
    value: f64,
    gets: Cell<usize>,
}
impl VectorView<f64> for CountedVector {
    fn len(&self) -> usize {
        self.len
    }
    fn get(&self, i: usize) -> f64 {
        assert!(i < self.len);
        self.gets.set(self.gets.get() + 1);
        self.value
    }
    fn sum(&self) -> f64 {
        panic!("Cached sparse dots must not compute a vector sum.")
    }
}

fn sparse_checks() {
    for n in [17, 1_000_000_000] {
        let matrix = LazyMatrix::from_column_parts(
            Sparse {
                len: n,
                visits: Cell::new(0),
            },
            Some(vec![0.375]),
            Some(vec![-1.75]),
        );
        let column = matrix.column(0);
        let vector = CountedVector {
            len: n,
            value: 1.0,
            gets: Cell::new(0),
        };
        let weights = CountedVector {
            len: n,
            value: 0.5,
            gets: Cell::new(0),
        };
        for operation in OPERATIONS {
            vector.gets.set(0);
            weights.gets.set(0);
            column.raw().visits.set(0);
            let weighted = if operation == Operation::WeightedDot {
                0.5
            } else {
                1.0
            };
            let allocations = ALLOCATIONS.load(Ordering::Relaxed);
            let actual = dispatch(
                &column,
                &vector,
                &weights,
                n as f64 * weighted,
                operation,
                Method::Cached,
            );
            assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), allocations);
            assert_eq!(
                actual,
                (3.5 * weighted - 0.375 * n as f64 * weighted) / -1.75
            );
            assert_eq!(column.raw().visits.get(), 5);
            assert_eq!(vector.gets.get(), 5);
            assert_eq!(
                weights.gets.get(),
                if operation == Operation::WeightedDot {
                    5
                } else {
                    0
                }
            );
            println!(
                "{n},{operation:?},5,{},{},0,{actual:.17e}",
                vector.gets.get(),
                weights.gets.get()
            );
        }
    }
}

fn setting(name: &str, default: usize) -> usize {
    std::env::var(name).map_or(default, |value| value.parse().unwrap())
}

fn main() {
    if std::env::var_os("PROBE_NUMERICS").is_some() {
        println!(
            "scalar,case,operation,method,actual,production,direct_centering,regrouped_serial,same_as_production"
        );
        numerics::<f32>();
        numerics::<f64>();
        return;
    }
    if std::env::var_os("PROBE_SPARSE").is_some() {
        println!("rows,operation,stored_visits,vector_gets,weight_gets,allocations,actual");
        sparse_checks();
        return;
    }
    let probe = Probe {
        label: std::env::var("PROBE_LABEL").unwrap_or_else(|_| "local".into()),
        samples: setting("PROBE_SAMPLES", 2),
        elements: setting("PROBE_ELEMENTS", 8_000_000),
        order: setting("PROBE_ORDER", 0),
    };
    println!(
        "label,scalar,regime,rows,cols,working_set_bytes,operation,method,sample,passes,ns_per_dot,checksum"
    );
    eprintln!(
        "label,scalar,regime,rows,cols,operation,method,max_absolute_error,max_scaled_error,allocations"
    );
    let lengths = std::env::var("PROBE_LENGTHS")
        .unwrap_or_else(|_| "0,1,7,8,9,33,256,5000,65536,1048576,4194304".into());
    for n in lengths.split(',').map(|n| n.parse::<usize>().unwrap()) {
        for regime in ["single", "rotating"] {
            if regime == "rotating" && n < 256 {
                continue;
            }
            let bytes = setting("PROBE_BYTES", 64 * 1024 * 1024);
            probe.measure::<f32>(n, regime, bytes);
            probe.measure::<f64>(n, regime, bytes);
        }
    }
}
