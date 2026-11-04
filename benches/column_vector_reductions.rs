//! Measure vector sums and weighted columns against the same baseline harness.

#[path = "../tests/common/backend_aliases.rs"]
mod backend_aliases;

use std::{hint::black_box, time::Instant};

use backend_aliases::{faer, nalgebra, ndarray};
use lazymatrix::{LazyMatrix, RawColumns, VectorView};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[derive(Clone, Copy, Debug)]
enum Operation {
    VectorSum,
    Dot,
    CachedDot,
    WeightedDot,
    CachedWeightedDot,
    WeightedNorm,
}

const OPERATIONS: [Operation; 6] = [
    Operation::VectorSum,
    Operation::Dot,
    Operation::CachedDot,
    Operation::WeightedDot,
    Operation::CachedWeightedDot,
    Operation::WeightedNorm,
];

struct Probe {
    label: String,
    rows: usize,
    cols: usize,
    samples: usize,
    passes: usize,
}

#[inline(never)]
fn reduction<M: RawColumns<f64>, V: VectorView<f64>, W: VectorView<f64>>(
    matrix: &LazyMatrix<M>,
    vector: &V,
    weights: &W,
    sums: (f64, f64),
    operation: Operation,
    j: usize,
) -> f64 {
    if matches!(operation, Operation::VectorSum) {
        return VectorView::sum(vector);
    }
    let column = matrix.column(j);
    match operation {
        Operation::VectorSum => unreachable!(),
        Operation::Dot => column.dot(vector),
        Operation::CachedDot => column.dot_with_sum(vector, sums.0),
        Operation::WeightedDot => column.weighted_dot(vector, weights),
        Operation::CachedWeightedDot => column.weighted_dot_with_sum(vector, weights, sums.1),
        Operation::WeightedNorm => column.weighted_norm_squared(weights),
    }
}

impl Probe {
    fn measure<M: RawColumns<f64>, V: VectorView<f64>, W: VectorView<f64>>(
        &self,
        backend: &str,
        layout: &str,
        matrix: M,
        vector: V,
        weights: W,
    ) {
        let raw: Vec<Vec<_>> = (0..self.cols)
            .map(|j| {
                let mut values = vec![0.0; self.rows];
                lazymatrix::RawColumn::for_each_stored(&matrix.raw_column(j), |i, x| {
                    values[i] += x
                });
                values
            })
            .collect();
        let sums = (
            (0..self.rows).map(|i| vector.get(i)).sum(),
            (0..self.rows).map(|i| vector.get(i) * weights.get(i)).sum(),
        );
        let matrix = LazyMatrix::from_column_parts(
            matrix,
            Some(vec![0.375; self.cols]),
            Some(vec![-1.75; self.cols]),
        );
        for operation in OPERATIONS {
            let mut max_absolute = 0.0_f64;
            let mut max_scaled = 0.0_f64;
            for (j, values) in raw.iter().enumerate() {
                let expected: f64 = (0..self.rows)
                    .map(|i| {
                        let x = (values[i] - 0.375) / -1.75;
                        let v = vector.get(i);
                        let w = weights.get(i);
                        match operation {
                            Operation::VectorSum => v,
                            Operation::Dot | Operation::CachedDot => x * v,
                            Operation::WeightedDot | Operation::CachedWeightedDot => x * w * v,
                            Operation::WeightedNorm => w * x * x,
                        }
                    })
                    .sum();
                let actual = reduction(&matrix, &vector, &weights, sums, operation, j);
                let error = (actual - expected).abs();
                assert!(actual.is_finite() && error <= 1e-10 * (1.0 + expected.abs()));
                max_absolute = max_absolute.max(error);
                max_scaled = max_scaled.max(error / (1.0 + expected.abs()));
            }
            eprintln!(
                "{},{backend},{layout},{operation:?},lazy,{max_absolute:.17e},{max_scaled:.17e}",
                self.label
            );
        }
        // Keep materialization and independent verification outside timed passes.
        drop(raw);
        for sample in 0..self.samples {
            for index in 0..OPERATIONS.len() {
                let index = if sample % 2 == 0 {
                    index
                } else {
                    OPERATIONS.len() - 1 - index
                };
                let operation = OPERATIONS[index];
                let start = Instant::now();
                let mut checksum = 0.0;
                for _ in 0..self.passes {
                    for j in 0..self.cols {
                        checksum += black_box(reduction(
                            black_box(&matrix),
                            black_box(&vector),
                            black_box(&weights),
                            sums,
                            operation,
                            black_box(j),
                        ));
                    }
                }
                let nanos = start.elapsed().as_secs_f64() * 1e9 / self.passes as f64;
                println!(
                    "{},{backend},{layout},{operation:?},lazy,{sample},{},{},{},{nanos:.3},{checksum:.17e}",
                    self.label, self.rows, self.cols, self.passes
                );
            }
        }
    }
}

fn setting(name: &str, default: usize) -> usize {
    std::env::var(name).map_or(default, |value| value.parse().unwrap())
}

fn main() {
    use ndarray::{Array1, Array2, ShapeBuilder, s};

    let probe = Probe {
        label: std::env::var("PROBE_LABEL").unwrap_or_else(|_| "candidate".into()),
        rows: setting("PROBE_ROWS", 5000),
        cols: setting("PROBE_COLS", 1000),
        samples: setting("PROBE_SAMPLES", 3),
        passes: setting("PROBE_PASSES", 5),
    };
    let mut rng = ChaCha8Rng::seed_from_u64(8421);
    let data = Array2::from_shape_fn((probe.rows, probe.cols).f(), |_| {
        rng.random_range(-1.0..1.0)
    });
    let vector = Array1::from_shape_fn(probe.rows, |_| rng.random_range(-1.0..1.0));
    let weights = Array1::from_shape_fn(probe.rows, |_| rng.random_range(-1.0..1.0));
    let padded_vector = Array1::from_shape_fn(probe.rows * 2, |i| {
        if i % 2 == 0 { vector[i / 2] } else { f64::NAN }
    });
    let padded_weights = Array1::from_shape_fn(probe.rows * 2, |i| {
        if i % 2 == 0 { weights[i / 2] } else { f64::NAN }
    });
    println!("label,backend,layout,operation,method,sample,rows,cols,passes,ns_per_pass,checksum");
    eprintln!("label,backend,layout,operation,method,max_absolute_error,max_scaled_error");
    for layout in [
        "column_major",
        "vector_strided",
        "weights_strided",
        "row_reverse",
    ] {
        let matrix = if layout == "row_reverse" {
            data.slice(s![..;-1, ..])
        } else {
            data.view()
        };
        let input = if layout == "vector_strided" {
            padded_vector.slice(s![..;2])
        } else {
            vector.view()
        };
        let weights = if layout == "weights_strided" {
            padded_weights.slice(s![..;2])
        } else {
            weights.view()
        };
        probe.measure("ndarray", layout, matrix, input, weights);
    }
    let matrix = faer::Mat::from_fn(probe.rows, probe.cols, |i, j| data[(i, j)]);
    let input = faer::Col::from_fn(probe.rows, |i| vector[i]);
    let weight = faer::Col::from_fn(probe.rows, |i| weights[i]);
    probe.measure(
        "faer",
        "column_major",
        matrix.as_ref(),
        input.as_ref(),
        weight.as_ref(),
    );
    let matrix = nalgebra::DMatrix::from_fn(probe.rows, probe.cols, |i, j| data[(i, j)]);
    let input = nalgebra::DVector::from_fn(probe.rows, |i, _| vector[i]);
    let weight = nalgebra::DVector::from_fn(probe.rows, |i, _| weights[i]);
    probe.measure(
        "nalgebra",
        "column_major",
        matrix.view((0, 0), (probe.rows, probe.cols)),
        input.rows(0, probe.rows),
        weight.rows(0, probe.rows),
    );
}
