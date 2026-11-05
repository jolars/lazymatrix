//! Investigate storage-order and indexed reductions without changing borrowing hooks.

#[path = "../tests/common/backend_aliases.rs"]
mod backend_aliases;

use std::{hint::black_box, time::Instant};

use backend_aliases::{faer, nalgebra, ndarray};
use lazymatrix::{ColumnNormalizationParams, LazyMatrix, RawColumn, RawColumns, Scalar};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

trait ProbeScalar: Scalar + 'static {
    fn faer_sum(column: faer::ColRef<'_, Self>) -> Self;
    fn nalgebra_sum(column: nalgebra::DVectorView<'_, Self, nalgebra::Dyn, nalgebra::Dyn>) -> Self;
}

macro_rules! native_sums {
    ($scalar:ty) => {
        impl ProbeScalar for $scalar {
            fn faer_sum(column: faer::ColRef<'_, Self>) -> Self {
                column.sum()
            }
            fn nalgebra_sum(
                column: nalgebra::DVectorView<'_, Self, nalgebra::Dyn, nalgebra::Dyn>,
            ) -> Self {
                column.sum()
            }
        }
    };
}
native_sums!(f32);
native_sums!(f64);

#[derive(Clone, Copy, Debug, PartialEq)]
enum Operation {
    Sum,
    CenteredNorm,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Method {
    Lazy,
    Serial,
    Indexed8,
    Pointer8,
    Memory8,
    Native,
}

const OPERATIONS: [Operation; 2] = [Operation::Sum, Operation::CenteredNorm];
const METHODS: [Method; 6] = [
    Method::Lazy,
    Method::Serial,
    Method::Indexed8,
    Method::Pointer8,
    Method::Memory8,
    Method::Native,
];

fn term<F: Scalar>(operation: Operation, value: F, center: F) -> F {
    match operation {
        Operation::Sum => value,
        Operation::CenteredNorm => (value - center) * (value - center),
    }
}

fn combine<F: Scalar>(sums: [F; 8]) -> F {
    (0..4).fold(F::zero(), |sum, lane| sum + (sums[lane] + sums[lane + 4]))
}

fn indexed8<F: Scalar>(len: usize, at: impl Fn(usize) -> F) -> F {
    let mut sums = [F::zero(); 8];
    let end = len / 8 * 8;
    for start in (0..end).step_by(8) {
        for (lane, sum) in sums.iter_mut().enumerate() {
            *sum = *sum + at(start + lane);
        }
    }
    (end..len).fold(combine(sums), |sum, row| sum + at(row))
}

/// Every logical row must point to a live, initialized entry for the call.
unsafe fn pointer8<F: Scalar>(
    mut pointer: *const F,
    len: usize,
    stride: isize,
    operation: Operation,
    center: F,
) -> F {
    let mut sums = [F::zero(); 8];
    for _ in 0..len / 8 {
        for sum in &mut sums {
            // The caller supplies a valid view; wrapping avoids creating an
            // out-of-bounds pointer after the last negative-stride entry.
            let value = unsafe { *pointer };
            *sum = *sum + term(operation, value, center);
            pointer = pointer.wrapping_offset(stride);
        }
    }
    let mut sum = combine(sums);
    for _ in 0..len % 8 {
        sum = sum + term(operation, unsafe { *pointer }, center);
        pointer = pointer.wrapping_offset(stride);
    }
    sum
}

fn memory8<F: Scalar>(values: &[F], operation: Operation, center: F) -> F {
    let mut sums = [F::zero(); 8];
    let mut chunks = values.chunks_exact(8);
    for chunk in chunks.by_ref() {
        for lane in 0..8 {
            sums[lane] = sums[lane] + term(operation, chunk[lane], center);
        }
    }
    chunks
        .remainder()
        .iter()
        .fold(combine(sums), |sum, &x| sum + term(operation, x, center))
}

fn compensated(values: impl Iterator<Item = f64>) -> f64 {
    let (mut sum, mut correction) = (0.0_f64, 0.0_f64);
    for value in values {
        let next = sum + value;
        correction += if sum.abs() >= value.abs() {
            (sum - next) + value
        } else {
            (value - next) + sum
        };
        sum = next;
    }
    sum + correction
}

struct Probe {
    label: String,
    samples: usize,
    elements: usize,
    order_offset: usize,
}

struct Layout<'a, F> {
    backend: &'a str,
    name: &'a str,
    rows: usize,
    cols: usize,
    center: F,
}

// Keep the same dispatch boundary for every method, including short columns.
#[inline(never)]
fn dispatch<F: Scalar>(
    run: &impl Fn(Operation, Method, usize) -> F,
    op: Operation,
    method: Method,
    j: usize,
) -> F {
    run(op, method, j)
}

impl Probe {
    fn measure<'a, F: Scalar + 'a, M: RawColumns<F>>(
        &self,
        layout: Layout<'_, F>,
        matrix: &M,
        at: impl Fn(usize, usize) -> F,
        memory: impl Fn(usize) -> Option<&'a [F]>,
        pointer: impl Fn(Operation, usize, F) -> F,
        native_sum: impl Fn(usize) -> F,
    ) {
        let Layout {
            backend,
            name,
            rows,
            cols,
            center,
        } = layout;
        let scalar = std::any::type_name::<F>();
        let scale = F::from_f64(-1.75).unwrap();
        let lazy = LazyMatrix::<_, ColumnNormalizationParams<F>>::from_column_parts(
            matrix,
            Some(vec![center; cols]),
            Some(vec![scale; cols]),
        );
        let run = |operation, method, j| {
            let result = match method {
                Method::Lazy => {
                    return match operation {
                        Operation::Sum => lazy.column(j).raw().raw_sum(),
                        Operation::CenteredNorm => lazy.column(j).norm_squared(),
                    };
                }
                Method::Serial => {
                    (0..rows).fold(F::zero(), |sum, i| sum + term(operation, at(i, j), center))
                }
                Method::Indexed8 => indexed8(rows, |i| term(operation, at(i, j), center)),
                Method::Pointer8 => pointer(operation, j, center),
                Method::Memory8 => memory8(memory(j).unwrap(), operation, center),
                Method::Native => native_sum(j),
            };
            match operation {
                Operation::Sum => result,
                Operation::CenteredNorm => result / (scale * scale),
            }
        };
        let available = |op, method| {
            !(method == Method::Memory8 && memory(0).is_none()
                || method == Method::Native && op != Operation::Sum)
        };

        // Deliberately use distinct logical vectors, independently of physical row order.
        let input: Vec<F> = (0..rows)
            .map(|i| F::from_f64((i % 13) as f64 / 8.0 - 0.75).unwrap())
            .collect();
        let weights: Vec<F> = (0..rows)
            .map(|i| F::from_f64((i % 7) as f64 / 4.0 - 0.5).unwrap())
            .collect();
        let mut alignment_errors = [[0.0_f64; 2]; 3];
        for j in 0..cols {
            let raw = matrix.raw_column(j);
            let mut visited = 0;
            raw.for_each_stored(|i, x| {
                assert_eq!(i, visited);
                assert_eq!(x, at(i, j));
                visited += 1;
            });
            assert_eq!(visited, rows);
            if let Some(values) = raw.dense_values() {
                if rows > 1 {
                    assert_eq!(name, "column_major");
                }
                assert_eq!(values.len(), rows);
                for (i, &x) in values.iter().enumerate() {
                    assert_eq!(x, at(i, j));
                }
            }
            let column = lazy.column(j);
            for (index, (actual, expected)) in [
                (
                    column.dot(&input),
                    compensated((0..rows).map(|i| {
                        (at(i, j).to_f64().unwrap() - center.to_f64().unwrap()) / -1.75
                            * input[i].to_f64().unwrap()
                    })),
                ),
                (
                    column.weighted_dot(&input, &weights),
                    compensated((0..rows).map(|i| {
                        (at(i, j).to_f64().unwrap() - center.to_f64().unwrap()) / -1.75
                            * input[i].to_f64().unwrap()
                            * weights[i].to_f64().unwrap()
                    })),
                ),
                (
                    column.weighted_norm_squared(&weights),
                    compensated((0..rows).map(|i| {
                        let x = (at(i, j).to_f64().unwrap() - center.to_f64().unwrap()) / -1.75;
                        x * x * weights[i].to_f64().unwrap()
                    })),
                ),
            ]
            .into_iter()
            .enumerate()
            {
                let (absolute, scaled) = error(actual, expected, rows);
                alignment_errors[index][0] = alignment_errors[index][0].max(absolute);
                alignment_errors[index][1] = alignment_errors[index][1].max(scaled);
            }
        }
        for (operation, [absolute, scaled]) in ["Dot", "WeightedDot", "WeightedNorm"]
            .into_iter()
            .zip(alignment_errors)
        {
            eprintln!(
                "{},{backend},{name},{scalar},{operation},alignment,{rows},{cols},{absolute:.17e},{scaled:.17e}",
                self.label
            );
        }

        for operation in OPERATIONS {
            for method in METHODS {
                if !available(operation, method) {
                    continue;
                }
                let mut max_absolute = 0.0_f64;
                let mut max_scaled = 0.0_f64;
                for j in 0..cols {
                    let expected = compensated((0..rows).map(|i| {
                        let x = at(i, j).to_f64().unwrap();
                        match operation {
                            Operation::Sum => x,
                            Operation::CenteredNorm => {
                                ((x - center.to_f64().unwrap()) / -1.75).powi(2)
                            }
                        }
                    }));
                    let actual = dispatch(&run, operation, method, j);
                    let (absolute, scaled) = error(actual, expected, rows);
                    max_absolute = max_absolute.max(absolute);
                    max_scaled = max_scaled.max(scaled);
                }
                eprintln!(
                    "{},{backend},{name},{scalar},{operation:?},{method:?},{rows},{cols},{max_absolute:.17e},{max_scaled:.17e}",
                    self.label
                );
            }
        }
        let passes = (self.elements / (rows * cols).max(1)).clamp(1, 10000);
        for sample in 0..self.samples {
            for index in 0..OPERATIONS.len() * METHODS.len() {
                let index = if (sample + self.order_offset) % 2 == 0 {
                    index
                } else {
                    OPERATIONS.len() * METHODS.len() - index - 1
                };
                let operation = OPERATIONS[index / METHODS.len()];
                let method = METHODS[index % METHODS.len()];
                if !available(operation, method) {
                    continue;
                }
                let start = Instant::now();
                let mut checksum = F::zero();
                for _ in 0..passes {
                    for j in 0..cols {
                        checksum =
                            checksum + black_box(dispatch(&run, operation, method, black_box(j)));
                    }
                }
                let nanos = start.elapsed().as_secs_f64() * 1e9 / passes as f64;
                println!(
                    "{},{backend},{name},{scalar},{operation:?},{method:?},{sample},{rows},{cols},{passes},{nanos:.3},{checksum:?}",
                    self.label
                );
            }
        }
    }
}

fn error<F: Scalar>(actual: F, expected: f64, rows: usize) -> (f64, f64) {
    let actual = actual.to_f64().unwrap();
    let absolute = (actual - expected).abs();
    let scaled = absolute / (1.0 + expected.abs());
    let tolerance = F::epsilon().to_f64().unwrap() * 32.0 * (rows as f64).sqrt().max(1.0);
    assert!(
        actual.is_finite() && expected.is_finite() && scaled <= tolerance,
        "actual={actual}, expected={expected}, scaled={scaled}, tolerance={tolerance}"
    );
    (absolute, scaled)
}

struct Fixture<F> {
    rows: usize,
    cols: usize,
    row_stride: usize,
    col_stride: usize,
    data: Vec<F>,
}

impl<F: Scalar> Fixture<F> {
    fn new(rows: usize, cols: usize, row_stride: usize, col_stride: usize, values: &[F]) -> Self {
        let mut data =
            vec![F::nan(); 7 + rows.saturating_sub(1) * row_stride + (cols - 1) * col_stride];
        for i in 0..rows {
            for j in 0..cols {
                data[3 + i * row_stride + j * col_stride] = values[i * cols + j];
            }
        }
        Self {
            rows,
            cols,
            row_stride,
            col_stride,
            data,
        }
    }

    fn run(&self, probe: &Probe, name: &str, reverse: bool, center: F)
    where
        F: ProbeScalar,
    {
        use ndarray::{ArrayView2, Axis, ShapeBuilder};
        let Self {
            rows,
            cols,
            row_stride,
            col_stride,
            data,
        } = self;
        let layout = |backend| Layout {
            backend,
            name,
            rows: *rows,
            cols: *cols,
            center,
        };
        let mut matrix = ArrayView2::from_shape(
            (*rows, *cols).strides((*row_stride, *col_stride)),
            &data[3..],
        )
        .unwrap();
        if reverse {
            matrix.invert_axis(Axis(0));
        }
        probe.measure(
            layout("ndarray"),
            &matrix,
            |i, j| matrix[(i, j)],
            |j| matrix.column(j).to_slice_memory_order(),
            |operation, j, center| {
                let column = matrix.column(j);
                // The backend view bounds every logical row of the kernel.
                unsafe {
                    pointer8(
                        column.as_ptr(),
                        *rows,
                        column.strides()[0],
                        operation,
                        center,
                    )
                }
            },
            |j| matrix.column(j).into_dyn().sum(),
        );

        // Every reachable cell is initialized in the padded fixture allocation.
        let mut matrix = unsafe {
            faer::MatRef::from_raw_parts(
                data.as_ptr().add(3),
                *rows,
                *cols,
                *row_stride as isize,
                *col_stride as isize,
            )
        };
        if reverse {
            matrix = matrix.reverse_rows();
        }
        probe.measure(
            layout("faer"),
            &matrix,
            |i, j| matrix[(i, j)],
            |j| {
                let mut column = matrix.col(j);
                if column.row_stride() < 0 {
                    column = column.reverse_rows();
                }
                column.try_as_col_major().map(|x| x.as_slice())
            },
            |operation, j, center| {
                let column = matrix.col(j);
                // The backend view bounds every logical row of the kernel.
                unsafe {
                    pointer8(
                        column.as_ptr(),
                        *rows,
                        column.row_stride(),
                        operation,
                        center,
                    )
                }
            },
            |j| F::faer_sum(matrix.col(j)),
        );

        // Nalgebra's strides are unsigned; a reversed view is not representable.
        if !reverse {
            let matrix = nalgebra::DMatrixView::from_slice_with_strides(
                &data[3..],
                *rows,
                *cols,
                *row_stride,
                *col_stride,
            );
            probe.measure(
                layout("nalgebra"),
                &matrix,
                |i, j| matrix[(i, j)],
                |j| {
                    if *row_stride == 1 {
                        Some(&data[3 + j * col_stride..3 + j * col_stride + rows])
                    } else {
                        None
                    }
                },
                |operation, j, center| {
                    let column = matrix.column(j);
                    // The backend view bounds every logical row of the kernel.
                    unsafe {
                        pointer8(
                            column.as_ptr(),
                            *rows,
                            column.strides().0 as isize,
                            operation,
                            center,
                        )
                    }
                },
                |j| F::nalgebra_sum(matrix.column(j)),
            );
        }
    }
}

fn setting(name: &str, default: usize) -> usize {
    std::env::var(name).map_or(default, |value| value.parse().unwrap())
}

fn probe<F: ProbeScalar>(probe: &Probe, rows: usize, cols: usize) {
    assert!(cols > 0);
    let mut rng = ChaCha8Rng::seed_from_u64(8421);
    let values: Vec<F> = (0..rows * cols)
        .map(|_| F::from_f64(rng.random_range(-1.0..1.0)).unwrap())
        .collect();
    let center = F::from_f64(0.375).unwrap();
    let fixture = Fixture::new(rows, cols, 1, rows + 6, &values);
    fixture.run(probe, "column_major", false, center);
    fixture.run(probe, "reverse", true, center);
    let fixture = Fixture::new(rows, cols, 3, rows * 3 + 6, &values);
    fixture.run(probe, "stride3", false, center);
    fixture.run(probe, "reverse_stride3", true, center);
    Fixture::new(rows, cols, cols + 6, 1, &values).run(probe, "row_major", false, center);
}

struct NumericalCase<'a, F> {
    name: &'a str,
    values: &'a [F],
    center: F,
    reverse: bool,
}

fn numerical_column<F: ProbeScalar, M: RawColumns<F>>(
    case: &NumericalCase<'_, F>,
    backend: &str,
    matrix: &M,
    native_sum: impl Fn() -> F,
) {
    let NumericalCase {
        name,
        values,
        center,
        reverse,
    } = *case;
    let layout = if reverse { "reverse" } else { "forward" };
    let scalar = std::any::type_name::<F>();
    let scale = F::from_f64(-1.75).unwrap();
    let lazy = LazyMatrix::from_column_parts(matrix, Some(vec![center]), Some(vec![scale]));
    let at = |i| values[if reverse { values.len() - 1 - i } else { i }];
    for operation in OPERATIONS {
        let expected = match (name, operation) {
            ("nan", _) | ("mixed_infinities", Operation::Sum) => f64::NAN,
            ("negative_infinity", Operation::Sum) => f64::NEG_INFINITY,
            ("positive_infinity" | "negative_infinity" | "mixed_infinities", _) => f64::INFINITY,
            ("intermediate_overflow", Operation::Sum) => 0.0,
            ("intermediate_overflow", Operation::CenteredNorm) => f64::INFINITY,
            _ => compensated(values.iter().map(|x| {
                let x = x.to_f64().unwrap();
                match operation {
                    Operation::Sum => x,
                    Operation::CenteredNorm => ((x - center.to_f64().unwrap()) / -1.75).powi(2),
                }
            })),
        };
        for method in METHODS {
            if operation != Operation::Sum && method == Method::Native {
                continue;
            }
            let raw = match method {
                Method::Lazy => match operation {
                    Operation::Sum => lazy.column(0).raw().raw_sum(),
                    Operation::CenteredNorm => lazy.column(0).norm_squared(),
                },
                Method::Serial => {
                    (0..values.len()).fold(F::zero(), |sum, i| sum + term(operation, at(i), center))
                }
                Method::Indexed8 => indexed8(values.len(), |i| term(operation, at(i), center)),
                Method::Pointer8 => {
                    let pointer = if reverse {
                        values.as_ptr().wrapping_add(values.len().saturating_sub(1))
                    } else {
                        values.as_ptr()
                    };
                    // Each address is a logical entry of the borrowed values.
                    unsafe {
                        pointer8(
                            pointer,
                            values.len(),
                            if reverse { -1 } else { 1 },
                            operation,
                            center,
                        )
                    }
                }
                Method::Memory8 => memory8(values, operation, center),
                Method::Native => native_sum(),
            };
            let actual = if operation == Operation::CenteredNorm && method != Method::Lazy {
                raw / (scale * scale)
            } else {
                raw
            };
            if !matches!(
                name,
                "cancellation_loss" | "cancellation_gain" | "intermediate_overflow"
            ) {
                if expected.is_nan() {
                    assert!(actual.is_nan());
                } else if expected.is_infinite() {
                    assert_eq!(actual.to_f64().unwrap(), expected);
                } else {
                    error(actual, expected, values.len());
                }
            }
            let class = if actual.is_nan() {
                "nan"
            } else if actual == F::infinity() {
                "+inf"
            } else if actual == F::neg_infinity() {
                "-inf"
            } else {
                "finite"
            };
            println!(
                "{scalar},{name},{backend},{layout},{operation:?},{method:?},{actual:?},{class},{expected:?}"
            );
        }
    }
}

fn numerics<F: ProbeScalar>() {
    use ndarray::{ArrayView2, Axis};
    let huge = F::epsilon().recip() * F::from_u32(4).unwrap();
    let mut loss = vec![F::zero(); 16];
    loss[0] = huge;
    loss[1] = -huge;
    loss[8] = F::one();
    let mut gain = vec![F::one(); 16];
    gain[0] = huge;
    gain[8] = -huge;
    let large_center = if F::epsilon().to_f64().unwrap() > 1e-10 {
        4096.0
    } else {
        1e12
    };
    for (name, values, center) in [
        ("empty", vec![], F::zero()),
        ("cancellation_loss", loss, F::zero()),
        ("cancellation_gain", gain, F::zero()),
        (
            "intermediate_overflow",
            (0..8)
                .map(|i| {
                    if i % 2 == 0 {
                        F::max_value()
                    } else {
                        -F::max_value()
                    }
                })
                .collect(),
            F::zero(),
        ),
        ("nan", vec![F::one(), F::nan(), F::one()], F::zero()),
        (
            "positive_infinity",
            vec![F::one(), F::infinity(), F::one()],
            F::zero(),
        ),
        (
            "negative_infinity",
            vec![F::one(), F::neg_infinity(), F::one()],
            F::zero(),
        ),
        (
            "mixed_infinities",
            vec![F::infinity(), F::neg_infinity()],
            F::zero(),
        ),
        (
            "large_center",
            (0..33)
                .map(|i| F::from_f64(large_center + (i % 7) as f64 / 8.0).unwrap())
                .collect(),
            F::from_f64(large_center).unwrap(),
        ),
    ] {
        for reverse in [false, true] {
            let case = NumericalCase {
                name,
                values: &values,
                center,
                reverse,
            };
            let mut matrix = ArrayView2::from_shape((values.len(), 1), values.as_slice()).unwrap();
            if reverse {
                matrix.invert_axis(Axis(0));
            }
            numerical_column(&case, "ndarray", &matrix, || {
                matrix.column(0).into_dyn().sum()
            });
            let mut matrix = faer::MatRef::from_column_major_slice(&values, values.len(), 1);
            if reverse {
                matrix = matrix.reverse_rows();
            }
            numerical_column(&case, "faer", &matrix, || F::faer_sum(matrix.col(0)));
            if !reverse {
                let matrix = nalgebra::DMatrixView::from_slice_with_strides(
                    &values,
                    values.len(),
                    1,
                    1,
                    values.len(),
                );
                numerical_column(&case, "nalgebra", &matrix, || {
                    F::nalgebra_sum(matrix.column(0))
                });
            }
        }
    }
}

fn main() {
    if std::env::var_os("PROBE_NUMERICS").is_some() {
        println!("scalar,case,backend,layout,operation,method,value,class,oracle");
        numerics::<f32>();
        numerics::<f64>();
        return;
    }
    let probe_config = Probe {
        label: std::env::var("PROBE_LABEL").unwrap_or_else(|_| "probe".into()),
        samples: setting("PROBE_SAMPLES", 5),
        elements: setting("PROBE_ELEMENTS", 10000000),
        order_offset: setting("PROBE_ORDER_OFFSET", 0),
    };
    println!(
        "label,backend,layout,scalar,operation,method,sample,rows,cols,passes,ns_per_pass,checksum"
    );
    eprintln!(
        "label,backend,layout,scalar,operation,method,rows,cols,max_absolute_error,max_scaled_error"
    );
    for shape in std::env::var("PROBE_SHAPES")
        .unwrap_or_else(|_| "33x7,5000x64,5000x1000".into())
        .split(',')
    {
        let (rows, cols) = shape.split_once('x').unwrap();
        let (rows, cols) = (rows.parse().unwrap(), cols.parse().unwrap());
        probe::<f32>(&probe_config, rows, cols);
        probe::<f64>(&probe_config, rows, cols);
    }
}
