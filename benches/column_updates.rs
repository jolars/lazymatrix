//! Compare compiled logical-column updates with a borrowed-slice control.

#[path = "../tests/common/backend_aliases.rs"]
mod backend_aliases;

use std::{
    alloc::{GlobalAlloc, Layout, System},
    hint::black_box,
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

use backend_aliases::{faer, nalgebra, ndarray};
use lazymatrix::{
    ColumnNormalizationParams, LazyMatrix, MatrixShape, RawColumn, RawColumns, Scalar, VectorView,
    VectorViewMut,
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

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

trait ProbeScalar: Scalar + 'static {
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
enum Method {
    Lazy,
    Slice,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Destination {
    Vec,
    Slice,
    Backend,
    Stride3,
    Reverse,
    Indexed,
}

impl Destination {
    fn stride(self) -> usize {
        if self == Self::Stride3 { 3 } else { 1 }
    }

    fn index(self, i: usize, n: usize) -> usize {
        if self == Self::Vec {
            i
        } else {
            3 + self.stride() * if self == Self::Reverse { n - 1 - i } else { i }
        }
    }
}

// An existing downstream implementation can omit every optional borrowing hook.
struct Indexed<'a, F>(&'a mut [F]);
impl<F: Scalar> VectorView<F> for Indexed<'_, F> {
    fn len(&self) -> usize {
        self.0.len()
    }
    fn get(&self, i: usize) -> F {
        self.0[i]
    }
}
impl<F: Scalar> VectorViewMut<F> for Indexed<'_, F> {
    fn set(&mut self, i: usize, value: F) {
        self.0[i] = value;
    }
}

#[inline]
fn slice_update<F: Scalar>(source: &[F], destination: &mut [F], multiplier: F, offset: F) {
    assert_eq!(source.len(), destination.len());
    for (&raw, value) in source.iter().zip(destination) {
        *value = *value + multiplier * raw + offset;
    }
}

// Separate monomorphizations expose the actual adapter loops in emitted assembly.
#[inline(never)]
fn update<F: Scalar, M: RawColumns<F>, V: VectorViewMut<F> + ?Sized>(
    matrix: &LazyMatrix<M, ColumnNormalizationParams<F>>,
    j: usize,
    alpha: F,
    destination: &mut V,
    method: Method,
) {
    let _ = method;
    matrix.column(j).scaled_add_to(alpha, destination);
}

#[inline(never)]
fn direct<F: Scalar, M: RawColumns<F>>(
    matrix: &LazyMatrix<M, ColumnNormalizationParams<F>>,
    j: usize,
    alpha: F,
    destination: &mut [F],
) {
    let column = matrix.column(j);
    slice_update(
        column.raw().dense_values().unwrap(),
        destination,
        alpha / column.scale(),
        -alpha * column.center() / column.scale(),
    );
}

// These labels identify the timed Vec monomorphizations in emitted assembly.
macro_rules! assembly_labels {
    ($scalar:ty, $ndarray:ident, $faer:ident, $nalgebra:ident) => {
        #[unsafe(no_mangle)]
        #[inline(never)]
        fn $ndarray(
            matrix: &LazyMatrix<
                ndarray::ArrayView2<'_, $scalar>,
                ColumnNormalizationParams<$scalar>,
            >,
            j: usize,
            alpha: $scalar,
            destination: &mut Vec<$scalar>,
        ) {
            update(matrix, j, alpha, destination, Method::Lazy);
        }

        #[unsafe(no_mangle)]
        #[inline(never)]
        fn $faer(
            matrix: &LazyMatrix<faer::MatRef<'_, $scalar>, ColumnNormalizationParams<$scalar>>,
            j: usize,
            alpha: $scalar,
            destination: &mut Vec<$scalar>,
        ) {
            update(matrix, j, alpha, destination, Method::Lazy);
        }

        #[unsafe(no_mangle)]
        #[inline(never)]
        fn $nalgebra(
            matrix: &LazyMatrix<
                nalgebra::DMatrixView<'_, $scalar, nalgebra::Dyn, nalgebra::Dyn>,
                ColumnNormalizationParams<$scalar>,
            >,
            j: usize,
            alpha: $scalar,
            destination: &mut Vec<$scalar>,
        ) {
            update(matrix, j, alpha, destination, Method::Lazy);
        }
    };
}
assembly_labels!(
    f32,
    updates_ndarray_f32,
    updates_faer_f32,
    updates_nalgebra_f32
);
assembly_labels!(
    f64,
    updates_ndarray_f64,
    updates_faer_f64,
    updates_nalgebra_f64
);

struct Probe {
    label: String,
    samples: usize,
    elements: usize,
    order: usize,
    checks_only: bool,
}

struct Case<'a> {
    backend: &'a str,
    layout: &'a str,
    rows: usize,
    cols: usize,
    destination: Destination,
    contiguous: bool,
}

impl Probe {
    fn measure<F: ProbeScalar>(
        &self,
        case: Case<'_>,
        at: impl Fn(usize, usize) -> F,
        mut run: impl FnMut(Method, usize, F, &mut Vec<F>),
    ) {
        let Case {
            backend,
            layout,
            rows,
            cols,
            destination,
            contiguous,
        } = case;
        let scalar = std::any::type_name::<F>();
        let center = F::from_f64(0.375).unwrap();
        let scale = F::from_f64(-1.75).unwrap();
        let alpha = F::from_f64(-0.125).unwrap();
        let multiplier = alpha / scale;
        let offset = -alpha * center / scale;
        let mut output = vec![F::from_f64(17.0).unwrap(); 7 + rows * destination.stride()];
        if destination == Destination::Vec {
            output.truncate(rows);
        }
        let initial = output.clone();
        let mut expected = output.clone();
        // Check every destination cell, including gaps and the surrounding range.
        // Repeated updates distinguish left-associated additions from reassociation.
        for j in 0..cols {
            for i in 0..rows {
                let index = destination.index(i, rows);
                expected[index] = expected[index] + multiplier * at(i, j) + offset;
            }
        }
        for method in [Method::Lazy, Method::Slice] {
            if method == Method::Slice && !contiguous {
                continue;
            }
            output.copy_from_slice(&initial);
            let before = ALLOCATIONS.load(Ordering::Relaxed);
            for j in 0..cols {
                run(method, j, alpha, &mut output);
            }
            assert_eq!(
                ALLOCATIONS.load(Ordering::Relaxed),
                before,
                "update allocated"
            );
            assert!(
                output.iter().zip(&expected).all(|(&a, &b)| a.same(b)),
                "{backend}/{layout}/{destination:?}/{scalar}/{method:?}"
            );
            eprintln!(
                "{},{backend},{layout},{destination:?},{scalar},{method:?},{rows},{cols},true,0",
                self.label
            );
        }
        if self.checks_only {
            return;
        }
        let passes = (self.elements / (rows * cols).max(1)).clamp(1, 10000);
        for sample in 0..self.samples {
            let methods = if (sample + self.order) % 2 == 0 {
                [Method::Lazy, Method::Slice]
            } else {
                [Method::Slice, Method::Lazy]
            };
            for method in methods {
                if method == Method::Slice && !contiguous {
                    continue;
                }
                output.copy_from_slice(&initial);
                let start = Instant::now();
                for pass in 0..passes {
                    // Alternate signs so repeated passes stay in a modest range.
                    let alpha = black_box(if pass % 2 == 0 { alpha } else { -alpha });
                    for j in 0..cols {
                        run(
                            black_box(method),
                            black_box(j),
                            alpha,
                            black_box(&mut output),
                        );
                    }
                }
                let nanos = start.elapsed().as_nanos() as f64 / passes as f64;
                let checksum = output.iter().fold(F::zero(), |sum, &value| sum + value);
                black_box(&output);
                println!(
                    "{},{backend},{layout},{destination:?},{scalar},{method:?},{sample},{rows},{cols},{passes},{nanos},{}",
                    self.label,
                    checksum.to_f64().unwrap()
                );
            }
        }
    }
}

struct Fixture<F> {
    data: Vec<F>,
    rows: usize,
    cols: usize,
    row_stride: usize,
    col_stride: usize,
}

impl<F: ProbeScalar> Fixture<F> {
    fn new(rows: usize, cols: usize, row_stride: usize, col_stride: usize, values: &[F]) -> Self {
        let mut data =
            vec![F::nan(); 7 + rows.saturating_sub(1) * row_stride + (cols - 1) * col_stride];
        for i in 0..rows {
            for j in 0..cols {
                data[3 + i * row_stride + j * col_stride] = values[i * cols + j];
            }
        }
        Self {
            data,
            rows,
            cols,
            row_stride,
            col_stride,
        }
    }

    fn run(&self, probe: &Probe, layout: &str, reverse: bool) {
        use ndarray::{ArrayView2, ArrayViewMut1, Axis, ShapeBuilder};
        let Self {
            data,
            rows,
            cols,
            row_stride,
            col_stride,
        } = self;
        let centers = || Some(vec![F::from_f64(0.375).unwrap(); *cols]);
        let scales = || Some(vec![F::from_f64(-1.75).unwrap(); *cols]);
        let at =
            |i, j| data[3 + if reverse { rows - 1 - i } else { i } * row_stride + j * col_stride];
        for destination in [
            Destination::Vec,
            Destination::Slice,
            Destination::Backend,
            Destination::Stride3,
            Destination::Reverse,
            Destination::Indexed,
        ] {
            let case = |backend| Case {
                backend,
                layout,
                rows: *rows,
                cols: *cols,
                destination,
                contiguous: *row_stride == 1
                    && !reverse
                    && matches!(
                        destination,
                        Destination::Vec | Destination::Slice | Destination::Backend
                    ),
            };
            let mut x = ArrayView2::from_shape(
                (*rows, *cols).strides((*row_stride, *col_stride)),
                &data[3..],
            )
            .unwrap();
            if reverse {
                x.invert_axis(Axis(0));
            }
            let matrix = LazyMatrix::from_column_parts(x, centers(), scales());
            probe.measure(case("ndarray"), at, |method, j, alpha, output| {
                if method == Method::Slice {
                    direct(
                        &matrix,
                        j,
                        alpha,
                        if destination == Destination::Vec {
                            output.as_mut_slice()
                        } else {
                            &mut output[3..3 + rows]
                        },
                    );
                    return;
                }
                match destination {
                    Destination::Vec => update(&matrix, j, alpha, output, method),
                    Destination::Slice => {
                        update(&matrix, j, alpha, &mut output[3..3 + rows], method)
                    }
                    Destination::Indexed => update(
                        &matrix,
                        j,
                        alpha,
                        &mut Indexed(&mut output[3..3 + rows]),
                        method,
                    ),
                    _ => {
                        let mut view = ArrayViewMut1::from_shape(
                            (*rows).strides(destination.stride()),
                            &mut output[3..],
                        )
                        .unwrap();
                        if destination == Destination::Reverse {
                            view.invert_axis(Axis(0));
                        }
                        update(&matrix, j, alpha, &mut view, method);
                    }
                }
            });
            // Every reachable cell belongs to the padded, initialized fixture.
            let mut x = unsafe {
                faer::MatRef::from_raw_parts(
                    data.as_ptr().add(3),
                    *rows,
                    *cols,
                    *row_stride as isize,
                    *col_stride as isize,
                )
            };
            if reverse {
                x = x.reverse_rows();
            }
            let matrix = LazyMatrix::from_column_parts(x, centers(), scales());
            probe.measure(case("faer"), at, |method, j, alpha, output| {
                if method == Method::Slice {
                    direct(
                        &matrix,
                        j,
                        alpha,
                        if destination == Destination::Vec {
                            output.as_mut_slice()
                        } else {
                            &mut output[3..3 + rows]
                        },
                    );
                    return;
                }
                match destination {
                    Destination::Vec => update(&matrix, j, alpha, output, method),
                    Destination::Slice => {
                        update(&matrix, j, alpha, &mut output[3..3 + rows], method)
                    }
                    Destination::Indexed => update(
                        &matrix,
                        j,
                        alpha,
                        &mut Indexed(&mut output[3..3 + rows]),
                        method,
                    ),
                    _ => {
                        // The selected mutable cells are distinct and remain in output.
                        let mut view = unsafe {
                            faer::ColMut::from_raw_parts_mut(
                                output.as_mut_ptr().add(3),
                                *rows,
                                destination.stride() as isize,
                            )
                        };
                        if destination == Destination::Reverse {
                            view = view.reverse_rows_mut();
                        }
                        update(&matrix, j, alpha, &mut view, method);
                    }
                }
            });
            if !reverse && destination != Destination::Reverse {
                let x = nalgebra::DMatrixView::from_slice_with_strides(
                    &data[3..],
                    *rows,
                    *cols,
                    *row_stride,
                    *col_stride,
                );
                let matrix = LazyMatrix::from_column_parts(x, centers(), scales());
                probe.measure(case("nalgebra"), at, |method, j, alpha, output| {
                    if method == Method::Slice {
                        direct(
                            &matrix,
                            j,
                            alpha,
                            if destination == Destination::Vec {
                                output.as_mut_slice()
                            } else {
                                &mut output[3..3 + rows]
                            },
                        );
                        return;
                    }
                    match destination {
                        Destination::Vec => update(&matrix, j, alpha, output, method),
                        Destination::Slice => {
                            update(&matrix, j, alpha, &mut output[3..3 + rows], method)
                        }
                        Destination::Indexed => update(
                            &matrix,
                            j,
                            alpha,
                            &mut Indexed(&mut output[3..3 + rows]),
                            method,
                        ),
                        _ => {
                            let mut view = nalgebra::DVectorViewMut::from_slice_with_strides_mut(
                                &mut output[3..],
                                *rows,
                                destination.stride(),
                                1,
                            );
                            update(&matrix, j, alpha, &mut view, method);
                        }
                    }
                });
            }
        }
    }
}

fn setting(name: &str, default: usize) -> usize {
    std::env::var(name).map_or(default, |value| value.parse().unwrap())
}

fn probe<F: ProbeScalar>(config: &Probe, rows: usize, cols: usize) {
    let mut rng = ChaCha8Rng::seed_from_u64(8421);
    let values: Vec<F> = (0..rows * cols)
        .map(|_| F::from_f64(rng.random_range(-1.0..1.0)).unwrap())
        .collect();
    let fixture = Fixture::new(rows, cols, 1, rows + 6, &values);
    fixture.run(config, "column_major", false);
    fixture.run(config, "reverse", true);
    Fixture::new(rows, cols, 3, rows * 3 + 6, &values).run(config, "stride3", false);
    Fixture::new(rows, cols, cols + 6, 1, &values).run(config, "row_major", false);
}

struct SparseFixture<F> {
    indices: [usize; 5],
    values: [F; 5],
}

impl<F> MatrixShape for SparseFixture<F> {
    fn nrows(&self) -> usize {
        5
    }
    fn ncols(&self) -> usize {
        1
    }
}

impl<F: Scalar> RawColumns<F> for SparseFixture<F> {
    type Column<'a>
        = &'a Self
    where
        Self: 'a;
    fn raw_column(&self, j: usize) -> Self::Column<'_> {
        assert_eq!(j, 0);
        self
    }
}

impl<F: Scalar> RawColumn<F> for &SparseFixture<F> {
    fn len(&self) -> usize {
        5
    }
    fn stored_len(&self) -> usize {
        5
    }
    fn for_each_stored(&self, mut visit: impl FnMut(usize, F)) {
        for (&i, &value) in self.indices.iter().zip(&self.values) {
            visit(i, value);
        }
    }
}

fn arithmetic_checks<F: ProbeScalar, M: RawColumns<F>>(
    config: &Probe,
    backend: &str,
    source: &M,
    values: &[F; 11],
) {
    let from = |x| F::from_f64(x).unwrap();
    for (name, center, scale, alpha, initial) in [
        ("signed_zero", F::zero(), F::one(), F::one(), -F::zero()),
        ("negative_alpha", -F::zero(), F::one(), -F::one(), F::zero()),
        ("zero_alpha", F::zero(), F::one(), F::zero(), from(17.0)),
        (
            "infinite_center",
            F::infinity(),
            F::one(),
            F::one(),
            from(17.0),
        ),
        (
            "infinite_scale",
            F::zero(),
            F::infinity(),
            F::one(),
            from(17.0),
        ),
        ("nan_scale", F::zero(), F::nan(), F::one(), from(17.0)),
        (
            "overflow_order",
            F::max_value(),
            F::one(),
            F::one(),
            F::max_value(),
        ),
    ] {
        let matrix = LazyMatrix::from_column_parts(source, Some(vec![center]), Some(vec![scale]));
        let multiplier = alpha / scale;
        let offset = -alpha * center / scale;
        let expected = values.map(|raw| initial + multiplier * raw + offset);
        let mut array = [initial; 11];
        let mut vector = Vec::with_capacity(37);
        vector.extend_from_slice(&array);
        let mut storage = [initial; 17];
        let before = ALLOCATIONS.load(Ordering::Relaxed);
        matrix.column(0).scaled_add_to(alpha, &mut array);
        matrix.column(0).scaled_add_to(alpha, &mut vector);
        matrix.column(0).scaled_add_to(alpha, &mut storage[3..14]);
        assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), before);
        for output in [array.as_slice(), vector.as_slice(), &storage[3..14]] {
            assert!(
                output.iter().zip(&expected).all(|(&a, &b)| a.same(b)),
                "{backend}/{name}"
            );
        }
        assert!(
            storage[..3]
                .iter()
                .chain(&storage[14..])
                .all(|&v| v.same(initial))
        );
        eprintln!(
            "{},{backend},{name},Array_Vec_Slice,{},Lazy,11,1,true,0",
            config.label,
            std::any::type_name::<F>()
        );
    }
}

fn special_checks<F: ProbeScalar>(config: &Probe) {
    let checks = Probe {
        checks_only: true,
        label: config.label.clone(),
        samples: 0,
        elements: 1,
        order: 0,
    };
    let values = [
        F::zero(),
        -F::zero(),
        F::one(),
        -F::one(),
        F::max_value(),
        -F::max_value(),
        F::epsilon(),
        F::min_positive_value(),
        F::infinity(),
        F::neg_infinity(),
        F::nan(),
    ];
    let fixture = Fixture::new(values.len(), 1, 1, values.len() + 6, &values);
    fixture.run(&checks, "nonfinite_and_extreme", false);
    fixture.run(&checks, "nonfinite_and_extreme_reverse", true);
    let x = ndarray::ArrayView2::from_shape((values.len(), 1), &values[..]).unwrap();
    arithmetic_checks(config, "ndarray", &x, &values);
    let x = faer::MatRef::from_column_major_slice(&values, values.len(), 1);
    arithmetic_checks(config, "faer", &x, &values);
    let x =
        nalgebra::DMatrixView::from_slice_with_strides(&values, values.len(), 1, 1, values.len());
    arithmetic_checks(config, "nalgebra", &x, &values);
    let raw = SparseFixture {
        indices: [3, 1, 3, 0, 3],
        values: [
            F::from_f64(1e20).unwrap(),
            F::from_f64(4.0).unwrap(),
            F::from_f64(-1e20).unwrap(),
            F::zero(),
            F::one(),
        ],
    };
    let center = F::from_f64(0.375).unwrap();
    let scale = F::from_f64(-1.75).unwrap();
    let matrix = LazyMatrix::from_column_parts(&raw, Some(vec![center]), Some(vec![scale]));
    for alpha in [
        F::from_f64(-0.125).unwrap(),
        F::zero(),
        -F::zero(),
        F::infinity(),
        F::nan(),
    ] {
        let multiplier = alpha / scale;
        let offset = -alpha * center / scale;
        let mut expected = [F::from_f64(17.0).unwrap(); 5];
        if offset != F::zero() {
            for value in &mut expected {
                *value = *value + offset;
            }
        }
        for (&i, &value) in raw.indices.iter().zip(&raw.values) {
            expected[i] = expected[i] + multiplier * value;
        }
        let mut array = [F::from_f64(17.0).unwrap(); 5];
        let mut vector = array.to_vec();
        let mut storage = [F::from_f64(17.0).unwrap(); 11];
        let before = ALLOCATIONS.load(Ordering::Relaxed);
        matrix.column(0).scaled_add_to(alpha, &mut array);
        matrix.column(0).scaled_add_to(alpha, &mut vector);
        matrix.column(0).scaled_add_to(alpha, &mut storage[3..8]);
        assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), before);
        for output in [array.as_slice(), vector.as_slice(), &storage[3..8]] {
            assert!(output.iter().zip(&expected).all(|(&a, &b)| a.same(b)));
        }
        assert!(
            storage[..3]
                .iter()
                .chain(&storage[8..])
                .all(|&v| v == F::from_f64(17.0).unwrap())
        );
        eprintln!(
            "{},sparse,duplicates_and_implicit_zeros,Array_Vec_Slice,{},Lazy,5,1,true,0",
            config.label,
            std::any::type_name::<F>()
        );
    }
}

fn main() {
    let config = Probe {
        label: std::env::var("PROBE_LABEL").unwrap_or_else(|_| "probe".into()),
        samples: setting("PROBE_SAMPLES", 3),
        elements: setting("PROBE_ELEMENTS", 1000000),
        order: setting("PROBE_ORDER_OFFSET", 0),
        checks_only: std::env::var_os("PROBE_CHECKS_ONLY").is_some(),
    };
    println!(
        "label,backend,layout,destination,scalar,method,sample,rows,cols,passes,ns_per_pass,checksum"
    );
    eprintln!("label,backend,layout,destination,scalar,method,rows,cols,bitwise_equal,allocations");
    special_checks::<f32>(&config);
    special_checks::<f64>(&config);
    for shape in std::env::var("PROBE_SHAPES")
        .unwrap_or_else(|_| "33x7,5000x64,5000x1000,262144x16".into())
        .split(',')
    {
        let (rows, cols) = shape.split_once('x').unwrap();
        let (rows, cols) = (rows.parse().unwrap(), cols.parse().unwrap());
        probe::<f32>(&config, rows, cols);
        probe::<f64>(&config, rows, cols);
    }
}
