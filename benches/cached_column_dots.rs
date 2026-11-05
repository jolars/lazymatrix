//! Diagnose cached column dots against native faer without changing production.

#[path = "../tests/common/backend_aliases.rs"]
mod backend_aliases;
use backend_aliases::faer;

use std::{
    alloc::{GlobalAlloc, Layout, System},
    hint::black_box,
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

use lazymatrix::{
    Centering, ColumnNormalization, ColumnNormalizationParams, LazyMatrix, RawColumn, Scalar,
    Scaling, VectorView,
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

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

#[derive(Clone, Copy, Debug, PartialEq)]
enum Method {
    Public8,
    Slice8,
    Acc16,
    Acc32,
    Avx8,
    Avx32,
    Fma32,
    Faer,
}

const METHODS: [Method; 8] = [
    Method::Public8,
    Method::Slice8,
    Method::Acc16,
    Method::Acc32,
    Method::Avx8,
    Method::Avx32,
    Method::Fma32,
    Method::Faer,
];

#[inline]
fn accumulated<F: Scalar, const N: usize>(x: &[F], y: &[F]) -> F {
    assert_eq!(x.len(), y.len());
    let mut sums = [F::zero(); N];
    let mut x = x.chunks_exact(N);
    let mut y = y.chunks_exact(N);
    for (x, y) in x.by_ref().zip(y.by_ref()) {
        for i in 0..N {
            sums[i] = sums[i] + x[i] * y[i];
        }
    }
    let mut result = (0..N / 2).fold(F::zero(), |sum, i| sum + (sums[i] + sums[i + N / 2]));
    for (&x, &y) in x.remainder().iter().zip(y.remainder()) {
        result = result + x * y;
    }
    result
}

trait ProbeScalar: Scalar {
    fn raw_dot(x: &[Self], y: &[Self], method: Method) -> Self;
    fn same(self, other: Self) -> bool;
}

macro_rules! packed_dot {
    ($x:ident, $y:ident, $scalar:ty, $width:literal, $n:literal, $fused:literal,
     $zero:ident, $load:ident, $store:ident, $mul:ident, $add:ident, $fma:ident) => {{
        use std::arch::x86_64::*;
        assert_eq!($x.len(), $y.len());
        // Unaligned loads also cover the deliberately offset borrowed inputs.
        unsafe {
            let mut sums = [$zero(); $n / $width];
            let mut x = $x.chunks_exact($n);
            let mut y = $y.chunks_exact($n);
            for (x, y) in x.by_ref().zip(y.by_ref()) {
                for (i, sum) in sums.iter_mut().enumerate() {
                    let x = $load(x.as_ptr().add(i * $width));
                    let y = $load(y.as_ptr().add(i * $width));
                    *sum = if $fused {
                        $fma(x, y, *sum)
                    } else {
                        $add(*sum, $mul(x, y))
                    };
                }
            }
            let mut lanes = [0.0 as $scalar; $n];
            for (i, sum) in sums.into_iter().enumerate() {
                $store(lanes.as_mut_ptr().add(i * $width), sum);
            }
            // Keep the portable kernel's combination order to isolate vector width.
            let mut result = (0..$n / 2).fold(0.0, |sum, i| sum + (lanes[i] + lanes[i + $n / 2]));
            for (&x, &y) in x.remainder().iter().zip(y.remainder()) {
                result = if $fused {
                    x.mul_add(y, result)
                } else {
                    result + x * y
                };
            }
            result
        }
    }};
}

macro_rules! kernels {
    ($scalar:ty, $slice:ident, $acc16:ident, $acc32:ident, $avx8:ident, $avx32:ident, $fma32:ident, $native:ident,
     $width:literal, $zero:ident, $load:ident, $store:ident, $mul:ident, $add:ident, $fma:ident) => {
        #[inline(never)]
        fn $slice(x: &[$scalar], y: &[$scalar]) -> $scalar {
            reductions::dot(x, y)
        }
        #[inline(never)]
        fn $acc16(x: &[$scalar], y: &[$scalar]) -> $scalar {
            accumulated::<_, 16>(x, y)
        }
        #[inline(never)]
        fn $acc32(x: &[$scalar], y: &[$scalar]) -> $scalar {
            accumulated::<_, 32>(x, y)
        }
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2")]
        #[inline(never)]
        unsafe fn $avx8(x: &[$scalar], y: &[$scalar]) -> $scalar {
            packed_dot!(
                x, y, $scalar, $width, 8, false, $zero, $load, $store, $mul, $add, $fma
            )
        }
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2")]
        #[inline(never)]
        unsafe fn $avx32(x: &[$scalar], y: &[$scalar]) -> $scalar {
            packed_dot!(
                x, y, $scalar, $width, 32, false, $zero, $load, $store, $mul, $add, $fma
            )
        }
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2,fma")]
        #[inline(never)]
        unsafe fn $fma32(x: &[$scalar], y: &[$scalar]) -> $scalar {
            packed_dot!(
                x, y, $scalar, $width, 32, true, $zero, $load, $store, $mul, $add, $fma
            )
        }
        #[inline(never)]
        fn $native(x: &[$scalar], y: &[$scalar]) -> $scalar {
            faer::linalg::matmul::dot::inner_prod(
                faer::ColRef::from_slice(x).transpose(),
                faer::Conj::No,
                faer::ColRef::from_slice(y),
                faer::Conj::No,
            )
        }
        impl ProbeScalar for $scalar {
            fn raw_dot(x: &[Self], y: &[Self], method: Method) -> Self {
                match method {
                    Method::Slice8 => $slice(x, y),
                    Method::Acc16 => $acc16(x, y),
                    Method::Acc32 => $acc32(x, y),
                    #[cfg(target_arch = "x86_64")]
                    Method::Avx8 if is_x86_feature_detected!("avx2") => {
                        // Runtime detection also protects generic builds on older CPUs.
                        unsafe { $avx8(x, y) }
                    }
                    #[cfg(target_arch = "x86_64")]
                    Method::Avx32 if is_x86_feature_detected!("avx2") => unsafe { $avx32(x, y) },
                    #[cfg(target_arch = "x86_64")]
                    Method::Fma32
                        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") =>
                    unsafe { $fma32(x, y) },
                    Method::Faer => $native(x, y),
                    Method::Public8 => unreachable!(),
                    _ => panic!("The selected diagnostic requires AVX2, or AVX2 and FMA."),
                }
            }
            fn same(self, other: Self) -> bool {
                (self.is_nan() && other.is_nan()) || self.to_bits() == other.to_bits()
            }
        }
    };
}

kernels!(
    f32,
    slice_f32,
    acc16_f32,
    acc32_f32,
    avx8_f32,
    avx32_f32,
    fma32_f32,
    native_f32,
    8,
    _mm256_setzero_ps,
    _mm256_loadu_ps,
    _mm256_storeu_ps,
    _mm256_mul_ps,
    _mm256_add_ps,
    _mm256_fmadd_ps
);
kernels!(
    f64,
    slice_f64,
    acc16_f64,
    acc32_f64,
    avx8_f64,
    avx32_f64,
    fma32_f64,
    native_f64,
    4,
    _mm256_setzero_pd,
    _mm256_loadu_pd,
    _mm256_storeu_pd,
    _mm256_mul_pd,
    _mm256_add_pd,
    _mm256_fmadd_pd
);

#[inline(never)]
fn dispatch<F: ProbeScalar>(
    matrix: &LazyMatrix<faer::MatRef<'_, F>, ColumnNormalizationParams<F>>,
    j: usize,
    vector: &[F],
    sum: F,
    method: Method,
) -> F {
    let column = matrix.column(j);
    if method == Method::Public8 {
        column.dot_with_sum(&faer::ColRef::from_slice(vector), sum)
    } else {
        let raw = column.raw().dense_values().unwrap();
        (F::raw_dot(raw, vector, method) - column.center() * sum) / column.scale()
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

struct Probe {
    label: String,
    samples: usize,
    elements: usize,
    order: usize,
}

impl Probe {
    fn measure<F: ProbeScalar>(&self, rows: usize, cols: usize, offset: usize, regime: &str) {
        let mut rng = ChaCha8Rng::seed_from_u64(8421);
        let f = |x| F::from_f64(x).unwrap();
        let storage = faer::Mat::from_fn(rows + offset + 5, cols + 2, |i, j| {
            if (offset..offset + rows).contains(&i) && (1..cols + 1).contains(&j) {
                f(rng.random_range(-1.0..1.0))
            } else {
                F::nan()
            }
        });
        let view = storage.as_ref().submatrix(offset, 1, rows, cols);
        let vector_storage = faer::Mat::from_fn(rows + offset + 5, 1, |i, _| {
            if (offset..offset + rows).contains(&i) {
                f(rng.random_range(-1.0..1.0))
            } else {
                F::nan()
            }
        });
        let input = vector_storage
            .as_ref()
            .col(0)
            .subrows(offset, rows)
            .try_as_col_major()
            .unwrap()
            .as_slice();
        let sum = VectorView::sum(input);
        let matrix = LazyMatrix::from_column_parts(
            view,
            Some(vec![f(0.375); cols]),
            Some(vec![f(-1.75); cols]),
        );
        let scalar = std::any::type_name::<F>();
        for method in METHODS {
            let mut max_error = 0.0_f64;
            let mut max_scaled = 0.0_f64;
            for j in 0..cols {
                let oracle = compensated((0..rows).map(|i| {
                    (view[(i, j)].to_f64().unwrap() - 0.375) / -1.75 * input[i].to_f64().unwrap()
                }));
                let allocations = ALLOCATIONS.load(Ordering::Relaxed);
                let result = dispatch(&matrix, j, input, sum, method);
                assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), allocations);
                if matches!(method, Method::Slice8 | Method::Avx8) {
                    assert!(result.same(dispatch(&matrix, j, input, sum, Method::Public8)));
                }
                if method == Method::Avx32 {
                    assert!(result.same(dispatch(&matrix, j, input, sum, Method::Acc32)));
                }
                let error = (result.to_f64().unwrap() - oracle).abs();
                let scaled = error / (1.0 + oracle.abs());
                assert!(
                    result.is_finite()
                        && scaled
                            <= 64.0
                                * F::epsilon().to_f64().unwrap()
                                * (rows as f64).sqrt().max(1.0)
                );
                max_error = max_error.max(error);
                max_scaled = max_scaled.max(scaled);
            }
            eprintln!(
                "{},{scalar},{regime},{rows},{cols},{offset},{method:?},{max_error:.17e},{max_scaled:.17e}",
                self.label
            );
        }
        let passes = (self.elements / (rows * cols).max(1)).clamp(1, 100_000);
        let footprint = (storage.as_ref().col_stride().unsigned_abs() * (cols + 2)
            + vector_storage.as_ref().col_stride().unsigned_abs()
            + 2 * cols)
            * size_of::<F>();
        for sample in 0..self.samples {
            for position in 0..METHODS.len() {
                let method = METHODS[(position + sample + self.order) % METHODS.len()];
                let pass = || {
                    let mut checksum = F::zero();
                    for j in 0..cols {
                        checksum = checksum
                            + black_box(dispatch(
                                black_box(&matrix),
                                black_box(j),
                                black_box(input),
                                black_box(sum),
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
                let ns = start.elapsed().as_secs_f64() * 1e9 / (passes * cols) as f64;
                assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), allocations);
                println!(
                    "{},{scalar},{regime},{rows},{cols},{offset},{footprint},{method:?},{sample},{passes},{ns:.3},{checksum:?}",
                    self.label
                );
            }
        }
    }
}

fn numerical<F: ProbeScalar>(name: &str, x: &[F], y: &[F], center: F, scale: F) {
    let storage = faer::Mat::from_fn(x.len(), 1, |i, _| x[i]);
    let matrix =
        LazyMatrix::from_column_parts(storage.as_ref(), Some(vec![center]), Some(vec![scale]));
    let sum = VectorView::sum(y);
    let production = dispatch(&matrix, 0, y, sum, Method::Public8);
    for method in METHODS {
        let result = dispatch(&matrix, 0, y, sum, method);
        let same = result.same(production);
        if matches!(method, Method::Slice8 | Method::Avx8) {
            assert!(same, "{name}/{method:?}");
        }
        if method == Method::Avx32 {
            assert!(result.same(dispatch(&matrix, 0, y, sum, Method::Acc32)));
        }
        println!(
            "{},{name},{method:?},{result:?},{production:?},{same}",
            std::any::type_name::<F>()
        );
    }
}

fn numerics<F: ProbeScalar>() {
    let f = |x| F::from_f64(x).unwrap();
    for n in 0..=65 {
        let x: Vec<_> = (0..n).map(|i| f(i as f64 / 8.0)).collect();
        let y: Vec<_> = (0..n).map(|i| f((i % 5) as f64 - 2.0)).collect();
        numerical(&format!("tail_{n}"), &x, &y, f(0.375), f(-1.75));
    }
    let large = F::one() / F::epsilon();
    let one = vec![F::one(); 65];
    for (name, x) in [
        (
            "cancellation",
            (0..65)
                .map(|i| [large * f(4.0), F::one(), -large * f(4.0), F::one()][i % 4])
                .collect::<Vec<_>>(),
        ),
        (
            "overflow",
            (0..65)
                .map(|i| {
                    if i % 32 < 16 {
                        F::max_value() / f(2.0)
                    } else {
                        -F::max_value() / f(2.0)
                    }
                })
                .collect(),
        ),
        (
            "large_offset",
            (0..65).map(|i| large + f((i % 7) as f64)).collect(),
        ),
    ] {
        numerical(
            name,
            &x,
            &one,
            if name == "large_offset" {
                large
            } else {
                F::zero()
            },
            F::one(),
        );
    }
    // A rounded product loses the low term; FMA can retain it when added to -1.
    let mut x = vec![F::zero(); 65];
    let mut y = x.clone();
    x[0] = -F::one();
    y[0] = F::one();
    x[32] = F::one() + F::epsilon();
    y[32] = F::one() - F::epsilon();
    numerical("fma_rounding", &x, &y, F::zero(), F::one());
    for (name, value) in [
        ("nan", F::nan()),
        ("infinity", F::infinity()),
        ("negative_infinity", F::neg_infinity()),
        ("zero", F::zero()),
        ("negative_zero", -F::zero()),
    ] {
        let mut x = one.clone();
        x[8] = value;
        numerical(&format!("column_{name}"), &x, &one, f(0.375), f(-1.75));
        numerical(&format!("vector_{name}"), &one, &x, f(0.375), f(-1.75));
        numerical(&format!("center_{name}"), &one, &one, value, f(-1.75));
        if value != F::zero() {
            numerical(&format!("scale_{name}"), &one, &one, f(0.375), value);
        }
    }
}

struct Fit {
    beta: Vec<f64>,
    intercept: f64,
    sweeps: usize,
}

fn fit(x: faer::MatRef<'_, f64>, y: &[f64], penalty: f64, method: Method) -> Fit {
    let rows = x.nrows();
    let cols = x.ncols();
    let n = rows as f64;
    let matrix =
        LazyMatrix::new(x, ColumnNormalization::new(Centering::Mean, Scaling::None)).unwrap();
    let summaries: Vec<_> = (0..cols)
        .map(|j| {
            let col = matrix.column(j);
            (col.raw().raw_sum(), col.sum(), col.norm_squared())
        })
        .collect();
    let mean = y.iter().sum::<f64>() / n;
    let mut base: Vec<_> = y.iter().map(|y| y - mean).collect();
    let mut sum = VectorView::sum(&base);
    let mut offset = 0.0;
    let mut beta = vec![0.0; cols];
    for sweep in 1..=1000 {
        for j in 0..cols {
            let (raw_sum, col_sum, norm) = summaries[j];
            let corr = dispatch(&matrix, j, &base, sum, method) + offset * col_sum;
            let partial = corr + norm * beta[j];
            let updated = partial.signum() * (partial.abs() - n * penalty).max(0.0) / norm;
            let delta = updated - beta[j];
            if delta != 0.0 {
                let column = matrix.column(j);
                column.raw().affine_add_to(-delta, 0.0, &mut base);
                sum -= delta * raw_sum;
                offset += delta * column.center();
                beta[j] = updated;
            }
        }
        let kkt = (0..cols)
            .map(|j| {
                let gradient =
                    (dispatch(&matrix, j, &base, sum, method) + offset * summaries[j].1) / n;
                if beta[j] == 0.0 {
                    (gradient.abs() - penalty).max(0.0)
                } else {
                    (gradient - penalty * beta[j].signum()).abs()
                }
            })
            .fold(0.0, f64::max);
        if kkt <= 1e-9 {
            let intercept = mean
                - (0..cols)
                    .map(|j| matrix.column(j).center() * beta[j])
                    .sum::<f64>();
            return Fit {
                beta,
                intercept,
                sweeps: sweep,
            };
        }
    }
    panic!("Coordinate descent did not converge.");
}

fn verify_fit(x: faer::MatRef<'_, f64>, y: &[f64], penalty: f64, fit: &Fit) -> (f64, f64) {
    let residual: Vec<_> = (0..y.len())
        .map(|i| {
            y[i] - fit.intercept - (0..x.ncols()).map(|j| x[(i, j)] * fit.beta[j]).sum::<f64>()
        })
        .collect();
    let n = y.len() as f64;
    let objective = residual.iter().map(|r| r * r).sum::<f64>() / (2.0 * n)
        + penalty * fit.beta.iter().map(|b| b.abs()).sum::<f64>();
    let mut kkt = (residual.iter().sum::<f64>() / n).abs();
    for j in 0..x.ncols() {
        let gradient = (0..y.len()).map(|i| x[(i, j)] * residual[i]).sum::<f64>() / n;
        let violation = if fit.beta[j] == 0.0 {
            (gradient.abs() - penalty).max(0.0)
        } else {
            (gradient - penalty * fit.beta[j].signum()).abs()
        };
        kkt = kkt.max(violation);
    }
    assert!(objective.is_finite() && kkt.is_finite() && kkt <= 2e-9);
    (objective, kkt)
}

fn consumers(probe: &Probe) {
    println!(
        "label,rows,cols,signal,penalty,method,sample,ns,sweeps,nonzero,objective,kkt,max_coefficient_difference"
    );
    for cols in [64, 1000] {
        let rows = 5000;
        let mut rng = ChaCha8Rng::seed_from_u64(8421);
        let x = faer::Mat::from_fn(rows, cols, |_, _| rng.random_range(-1.0..1.0) + 0.375);
        for (signal, penalty) in [(8, 0.01), (64, 0.005), (8, 2.0)] {
            let y: Vec<_> = (0..rows)
                .map(|i| {
                    1.25 + (0..signal)
                        .map(|j| {
                            (x[(i, j)] - 0.375)
                                * if j % 2 == 0 { 1.0 } else { -1.0 }
                                * if signal == 8 {
                                    1.0 / (j + 1) as f64
                                } else {
                                    0.125
                                }
                        })
                        .sum::<f64>()
                        + rng.random_range(-0.05..0.05)
                })
                .collect();
            let reference = fit(x.as_ref(), &y, penalty, Method::Public8);
            let (reference_objective, _) = verify_fit(x.as_ref(), &y, penalty, &reference);
            for sample in 0..probe.samples {
                for position in 0..METHODS.len() {
                    let method = METHODS[(position + sample + probe.order) % METHODS.len()];
                    black_box(fit(x.as_ref(), &y, penalty, method));
                    let start = Instant::now();
                    let result = black_box(fit(
                        black_box(x.as_ref()),
                        black_box(&y),
                        black_box(penalty),
                        black_box(method),
                    ));
                    let ns = start.elapsed().as_nanos();
                    let (objective, kkt) = verify_fit(x.as_ref(), &y, penalty, &result);
                    let difference = result
                        .beta
                        .iter()
                        .zip(&reference.beta)
                        .map(|(x, y)| (x - y).abs())
                        .fold(0.0, f64::max);
                    assert!(difference <= 1e-10);
                    assert!((objective - reference_objective).abs() <= 1e-12);
                    assert_eq!(result.sweeps, reference.sweeps);
                    let nonzero = result.beta.iter().filter(|b| **b != 0.0).count();
                    println!(
                        "{},{rows},{cols},{signal},{penalty},{method:?},{sample},{ns},{},{nonzero},{objective:.17e},{kkt:.17e},{difference:.17e}",
                        probe.label, result.sweeps
                    );
                }
            }
        }
    }
}

fn main() {
    if std::env::var_os("PROBE_ARCH").is_some() {
        println!(
            "f32: {:?}",
            <f32 as faer::traits::ComplexField>::Arch::default()
        );
        println!(
            "f64: {:?}",
            <f64 as faer::traits::ComplexField>::Arch::default()
        );
        return;
    }
    if std::env::var_os("PROBE_NUMERICS").is_some() {
        println!("scalar,case,method,result,production,same_bits_or_nan");
        numerics::<f32>();
        numerics::<f64>();
        return;
    }
    let get =
        |name: &str, default: &str| std::env::var(name).unwrap_or_else(|_| default.to_owned());
    let probe = Probe {
        label: get("PROBE_LABEL", "probe"),
        samples: get("PROBE_SAMPLES", "2").parse().unwrap(),
        elements: get("PROBE_ELEMENTS", "8000000").parse().unwrap(),
        order: get("PROBE_ORDER", "0").parse().unwrap(),
    };
    if std::env::var_os("PROBE_CONSUMER").is_some() {
        consumers(&probe);
        return;
    }
    println!(
        "label,scalar,regime,rows,cols,offset,working_set_bytes,method,sample,passes,ns_per_dot,checksum"
    );
    eprintln!("label,scalar,regime,rows,cols,offset,method,max_absolute,max_scaled");
    for shape in get(
        "PROBE_SHAPES",
        "0x1,1x1,7x1,8x1,9x1,33x1,256x1,5000x1,5000x64,5000x1000,65536x1,1048576x1",
    )
    .split(',')
    {
        let (rows, cols) = shape.split_once('x').unwrap();
        let rows: usize = rows.parse().unwrap();
        let cols: usize = cols.parse().unwrap();
        for offset in [0, 3] {
            let regime = if cols == 1 { "reused" } else { "columns" };
            probe.measure::<f32>(rows, cols, offset, regime);
            probe.measure::<f64>(rows, cols, offset, regime);
        }
    }
}
