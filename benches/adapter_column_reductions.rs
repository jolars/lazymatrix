//! Compare column adapters with native reductions on identical logical inputs.

#[path = "../tests/common/backend_aliases.rs"]
mod backend_aliases;

use std::{hint::black_box, time::Instant};

use backend_aliases::{faer, nalgebra};
use lazymatrix::{LazyMatrix, RawColumns, VectorView};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[derive(Clone, Copy, Debug)]
enum Operation {
    Sum,
    Norm,
    Dot,
}

const OPERATIONS: [Operation; 3] = [Operation::Sum, Operation::Norm, Operation::Dot];
const METHODS: [&str; 3] = ["lazy_owned", "lazy_borrowed", "native"];

struct Probe {
    label: String,
    rows: usize,
    cols: usize,
    samples: usize,
    passes: usize,
}

impl Probe {
    fn measure(
        &self,
        backend: &str,
        layout: &str,
        mut run: impl FnMut(Operation, usize, usize) -> f64,
        oracle: impl Fn(Operation, usize) -> f64,
    ) {
        for operation in OPERATIONS {
            for (method, name) in METHODS.iter().enumerate() {
                let mut max_absolute = 0.0_f64;
                let mut max_scaled = 0.0_f64;
                for column in 0..self.cols {
                    let expected = oracle(operation, column);
                    let actual = run(operation, method, column);
                    let error = (actual - expected).abs();
                    max_absolute = max_absolute.max(error);
                    max_scaled = max_scaled.max(error / (1.0 + expected.abs()));
                    assert!(actual.is_finite() && error <= 1e-10 * (1.0 + expected.abs()));
                }
                eprintln!(
                    "{},{backend},{layout},{operation:?},{name},{max_absolute:.17e},{max_scaled:.17e}",
                    self.label
                );
            }
        }
        for sample in 0..self.samples {
            // Alternate the whole operation and method order to limit drift.
            for index in 0..OPERATIONS.len() * METHODS.len() {
                let index = if sample % 2 == 0 {
                    index
                } else {
                    OPERATIONS.len() * METHODS.len() - 1 - index
                };
                let operation = OPERATIONS[index / METHODS.len()];
                let method = index % METHODS.len();
                let start = Instant::now();
                let mut checksum = 0.0;
                for _ in 0..self.passes {
                    for column in 0..self.cols {
                        checksum += black_box(run(operation, method, black_box(column)));
                    }
                }
                let nanos = start.elapsed().as_secs_f64() * 1e9 / self.passes as f64;
                println!(
                    "{},{backend},{layout},{operation:?},{},{sample},{},{},{},{nanos:.3},{checksum:.17e}",
                    self.label, METHODS[method], self.rows, self.cols, self.passes
                );
            }
        }
    }
}

fn lazy_reduction<M: RawColumns<f64>, V: VectorView<f64>>(
    matrix: &LazyMatrix<M>,
    vector: &V,
    vector_sum: f64,
    operation: Operation,
    column: usize,
) -> f64 {
    let column = matrix.column(column);
    match operation {
        Operation::Sum => column.sum(),
        Operation::Norm => column.norm_squared(),
        Operation::Dot => column.dot_with_sum(vector, vector_sum),
    }
}

fn oracle(
    operation: Operation,
    rows: usize,
    mut value: impl FnMut(usize) -> f64,
    mut input: impl FnMut(usize) -> f64,
) -> f64 {
    (0..rows)
        .map(|i| match operation {
            Operation::Sum => value(i),
            Operation::Norm => value(i).powi(2),
            Operation::Dot => value(i) * input(i),
        })
        .sum()
}

fn faer_layout(
    probe: &Probe,
    layout: &str,
    matrix: faer::MatRef<'_, f64>,
    input: faer::ColRef<'_, f64>,
) {
    let lazy = LazyMatrix::<_>::from_column_parts(matrix, None, None);
    let owned = faer::Col::from_fn(probe.rows, |i| input[i]);
    let sum = (0..probe.rows).map(|i| input[i]).sum();
    probe.measure(
        "faer",
        layout,
        |operation, method, j| match method {
            0 => lazy_reduction(&lazy, black_box(&owned), sum, operation, j),
            1 => lazy_reduction(&lazy, black_box(&input), sum, operation, j),
            _ => {
                let column = black_box(matrix).col(j);
                match operation {
                    Operation::Sum => column.sum(),
                    Operation::Norm => column.squared_norm_l2(),
                    Operation::Dot => faer::linalg::matmul::dot::inner_prod(
                        column.transpose(),
                        faer::Conj::No,
                        black_box(input),
                        faer::Conj::No,
                    ),
                }
            }
        },
        |operation, j| oracle(operation, probe.rows, |i| matrix[(i, j)], |i| input[i]),
    );
}

fn faer_probe(probe: &Probe, values: &[f64], input: &[f64]) {
    use faer::{Col, Mat, MatRef};
    let rows = probe.rows;
    let cols = probe.cols;
    let matrix = Mat::from_fn(rows, cols, |i, j| values[i * cols + j]);
    let vector = Col::from_fn(rows, |i| input[i]);
    faer_layout(probe, "column_major", matrix.as_ref(), vector.as_ref());
    let storage = Mat::from_fn(rows + 8, cols + 2, |i, j| {
        if (3..rows + 3).contains(&i) && (1..cols + 1).contains(&j) {
            values[(i - 3) * cols + j - 1]
        } else {
            f64::NAN
        }
    });
    faer_layout(
        probe,
        "borrowed_range",
        storage.as_ref().submatrix(3, 1, rows, cols),
        vector.as_ref(),
    );
    faer_layout(
        probe,
        "row_reverse",
        matrix.as_ref().reverse_rows(),
        vector.as_ref(),
    );
    faer_layout(
        probe,
        "row_major",
        MatRef::from_row_major_slice(values, rows, cols),
        vector.as_ref(),
    );
    let vector_storage = Mat::from_fn(2, rows, |i, j| if i == 1 { input[j] } else { f64::NAN });
    faer_layout(
        probe,
        "vector_strided",
        matrix.as_ref(),
        vector_storage.row(1).transpose(),
    );
}

type NalgebraMatrix<'a> = nalgebra::DMatrixView<'a, f64, nalgebra::Dyn, nalgebra::Dyn>;
type NalgebraVector<'a> = nalgebra::DVectorView<'a, f64, nalgebra::Dyn, nalgebra::Dyn>;

fn nalgebra_layout(
    probe: &Probe,
    layout: &str,
    matrix: NalgebraMatrix<'_>,
    input: NalgebraVector<'_>,
) {
    let lazy = LazyMatrix::<_>::from_column_parts(matrix, None, None);
    let owned = nalgebra::DVector::from_fn(probe.rows, |i, _| input[i]);
    let sum = input.iter().sum();
    probe.measure(
        "nalgebra",
        layout,
        |operation, method, j| match method {
            0 => lazy_reduction(&lazy, black_box(&owned), sum, operation, j),
            1 => lazy_reduction(&lazy, black_box(&input), sum, operation, j),
            _ => {
                let column = black_box(&matrix).column(j);
                match operation {
                    Operation::Sum => column.sum(),
                    Operation::Norm => column.norm_squared(),
                    Operation::Dot => column.dot(black_box(&input)),
                }
            }
        },
        |operation, j| oracle(operation, probe.rows, |i| matrix[(i, j)], |i| input[i]),
    );
}

fn nalgebra_probe(probe: &Probe, values: &[f64], input: &[f64]) {
    let rows = probe.rows;
    let cols = probe.cols;
    let matrix = nalgebra::DMatrix::from_fn(rows, cols, |i, j| values[i * cols + j]);
    let vector = NalgebraVector::from_slice_with_strides(input, rows, 1, rows);
    let view = NalgebraMatrix::from_slice_with_strides(matrix.as_slice(), rows, cols, 1, rows);
    nalgebra_layout(probe, "column_major", view, vector);
    let storage = nalgebra::DMatrix::from_fn(rows + 8, cols + 2, |i, j| {
        if (3..rows + 3).contains(&i) && (1..cols + 1).contains(&j) {
            values[(i - 3) * cols + j - 1]
        } else {
            f64::NAN
        }
    });
    let borrowed = NalgebraMatrix::from_slice_with_strides(
        &storage.as_slice()[rows + 11..],
        rows,
        cols,
        1,
        rows + 8,
    );
    nalgebra_layout(probe, "borrowed_range", borrowed, vector);
    let storage = nalgebra::DMatrix::from_fn(rows * 2, cols, |i, j| {
        if i % 2 == 0 {
            values[i / 2 * cols + j]
        } else {
            f64::NAN
        }
    });
    let strided =
        NalgebraMatrix::from_slice_with_strides(storage.as_slice(), rows, cols, 2, rows * 2);
    nalgebra_layout(probe, "row_stride2", strided, vector);
    nalgebra_layout(
        probe,
        "row_major",
        NalgebraMatrix::from_slice_with_strides(values, rows, cols, cols, 1),
        vector,
    );
    let vector_storage: Vec<_> = input.iter().flat_map(|&x| [x, f64::NAN]).collect();
    let strided_vector =
        NalgebraVector::from_slice_with_strides(&vector_storage, rows, 2, rows * 2);
    nalgebra_layout(probe, "vector_strided", view, strided_vector);
}

fn setting(name: &str, default: usize) -> usize {
    std::env::var(name).map_or(default, |value| value.parse().unwrap())
}

fn main() {
    let probe = Probe {
        label: std::env::var("PROBE_LABEL").unwrap_or_else(|_| "candidate".into()),
        rows: setting("PROBE_ROWS", 5000),
        cols: setting("PROBE_COLS", 1000),
        samples: setting("PROBE_SAMPLES", 9),
        passes: setting("PROBE_PASSES", 10),
    };
    let mut rng = ChaCha8Rng::seed_from_u64(8421);
    let values: Vec<_> = (0..probe.rows * probe.cols)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let input: Vec<_> = (0..probe.rows)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    println!("label,backend,layout,operation,method,sample,rows,cols,passes,ns_per_pass,checksum");
    eprintln!("label,backend,layout,operation,method,max_absolute_error,max_scaled_error");
    faer_probe(&probe, &values, &input);
    nalgebra_probe(&probe, &values, &input);
}
