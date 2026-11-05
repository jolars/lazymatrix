//! Compare solver-facing vector reductions and an existing ridge consumer.

#[allow(dead_code)]
#[path = "../examples/ridge_primal_dual.rs"]
mod ridge_example;
use ridge_example::backend_aliases;

use std::{hint::black_box, time::Instant};

use backend_aliases::{faer, nalgebra, ndarray};
use lazymatrix::{DotProduct, DotSlice, L2Norm, LazyMatrix, Scalar, SumEntries};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[derive(Clone, Copy, Debug)]
enum Operation {
    Sum,
    DotSlice,
    Dot,
    Norm,
}

const OPERATIONS: [Operation; 4] = [
    Operation::Sum,
    Operation::DotSlice,
    Operation::Dot,
    Operation::Norm,
];

#[inline(never)]
fn reduction<F: Scalar, V: SumEntries<F> + DotSlice<F> + DotProduct<F> + L2Norm<F>>(
    vector: &V,
    other: &V,
    coefficients: &[F],
    operation: Operation,
) -> F {
    match operation {
        Operation::Sum => vector.sum_entries(),
        Operation::DotSlice => vector.dot_slice(coefficients),
        Operation::Dot => DotProduct::dot(vector, other),
        Operation::Norm => L2Norm::norm_l2(vector),
    }
}

#[inline(never)]
fn native<F>(operation: Operation, implementation: &impl Fn(Operation) -> F) -> F {
    implementation(operation)
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
}

impl Probe {
    fn measure<F, V>(
        &self,
        backend: &str,
        layout: &str,
        values: &[F],
        coefficients: &[F],
        vectors: (&V, &V),
        native_impl: impl Fn(Operation) -> F,
    ) where
        F: Scalar,
        V: SumEntries<F> + DotSlice<F> + DotProduct<F> + L2Norm<F>,
    {
        let len = values.len();
        let scalar = std::any::type_name::<F>();
        let expected = |operation| match operation {
            Operation::Sum => compensated(values.iter().map(|v| v.to_f64().unwrap())),
            Operation::Dot | Operation::DotSlice => compensated(
                values
                    .iter()
                    .zip(coefficients)
                    .map(|(x, y)| x.to_f64().unwrap() * y.to_f64().unwrap()),
            ),
            Operation::Norm => {
                compensated(values.iter().map(|v| v.to_f64().unwrap().powi(2))).sqrt()
            }
        };
        for operation in OPERATIONS {
            for method in ["lazy", "native"] {
                let actual = if method == "lazy" {
                    reduction(vectors.0, vectors.1, coefficients, operation)
                } else {
                    native(operation, &native_impl)
                }
                .to_f64()
                .unwrap();
                let expected = expected(operation);
                let error = (actual - expected).abs();
                let scaled = error / (1.0 + expected.abs());
                let tolerance = F::epsilon().to_f64().unwrap() * 16.0 * (len.max(1) as f64).sqrt();
                assert!(actual.is_finite() && scaled <= tolerance);
                eprintln!(
                    "{},{backend},{layout},{scalar},{operation:?},{method},{len},{error:.17e},{scaled:.17e}",
                    self.label
                );
            }
        }
        let repetitions = (self.elements / len.max(1)).clamp(8, 100_000);
        for sample in 0..self.samples {
            for index in 0..OPERATIONS.len() {
                let operation = OPERATIONS[if sample % 2 == 0 {
                    index
                } else {
                    OPERATIONS.len() - 1 - index
                }];
                for method in if sample % 2 == 0 {
                    ["lazy", "native"]
                } else {
                    ["native", "lazy"]
                } {
                    let start = Instant::now();
                    for _ in 0..repetitions {
                        if method == "lazy" {
                            black_box(reduction(
                                black_box(vectors.0),
                                black_box(vectors.1),
                                black_box(coefficients),
                                black_box(operation),
                            ));
                        } else {
                            black_box(native(black_box(operation), black_box(&native_impl)));
                        }
                    }
                    let nanos = start.elapsed().as_secs_f64() * 1e9 / repetitions as f64;
                    println!(
                        "{},{backend},{layout},{scalar},{operation:?},{method},{sample},{len},{repetitions},{nanos:.3}",
                        self.label
                    );
                }
            }
        }
    }

    fn ridge(&self, rows: usize, columns: usize) {
        use faer::sparse::{SparseColMat, Triplet};
        let mut rng = ChaCha8Rng::seed_from_u64(8421);
        let mut entries = Vec::new();
        for row in 0..rows {
            for _ in 0..4 {
                entries.push((
                    row,
                    rng.random_range(0..columns),
                    rng.random_range(-1.0..1.0),
                ));
            }
        }
        let triplets: Vec<_> = entries
            .iter()
            .map(|&(i, j, v)| Triplet::new(i, j, v))
            .collect();
        let raw =
            SparseColMat::<usize, f64>::try_new_from_triplets(rows, columns, &triplets).unwrap();
        let design = LazyMatrix::from_column_parts(
            raw,
            Some(vec![0.001; columns]),
            Some(vec![1.75; columns]),
        );
        let response = faer::Col::from_fn(rows, |i| (i % 7) as f64 - 3.0);
        let penalty = 2.0;
        let coefficients = ridge_example::ridge(&design, &response, penalty).unwrap();
        let correction = compensated((0..columns).map(|j| coefficients[j])) * 0.001 / 1.75;
        let mut residual: Vec<_> = (0..rows).map(|i| -response[i] - correction).collect();
        for &(i, j, value) in &entries {
            residual[i] += value / 1.75 * coefficients[j];
        }
        let correction = compensated(residual.iter().copied()) * 0.001 / 1.75;
        let mut gradient: Vec<_> = (0..columns)
            .map(|j| penalty * coefficients[j] - correction)
            .collect();
        for &(i, j, value) in &entries {
            gradient[j] += value / 1.75 * residual[i];
        }
        let gradient_norm = compensated(gradient.iter().map(|v| v * v)).sqrt();
        let objective = compensated(residual.iter().map(|v| v * v)) / 2.0
            + penalty * compensated((0..columns).map(|j| coefficients[j].powi(2))) / 2.0;
        assert!(objective.is_finite() && gradient_norm < 1e-6);
        eprintln!(
            "{label},ridge,{rows}x{columns},f64,Gradient,oracle,{columns},{gradient_norm:.17e},{objective:.17e}",
            label = self.label
        );
        let repetitions = setting("PROBE_FITS", 10);
        for sample in 0..self.samples {
            let start = Instant::now();
            for _ in 0..repetitions {
                black_box(
                    ridge_example::ridge(black_box(&design), black_box(&response), penalty)
                        .unwrap(),
                );
            }
            let nanos = start.elapsed().as_secs_f64() * 1e9 / repetitions as f64;
            println!(
                "{label},ridge,{rows}x{columns},f64,Fit,lazy,{sample},{columns},{repetitions},{nanos:.3}",
                label = self.label
            );
        }
    }
}

fn setting(name: &str, default: usize) -> usize {
    std::env::var(name).map_or(default, |value| value.parse().unwrap())
}

macro_rules! scalar_probes {
    ($probe:expr, $scalar:ty, $lengths:expr) => {{
        use ndarray::s;
        let mut rng = ChaCha8Rng::seed_from_u64(8421);
        for &len in $lengths {
            let values: Vec<$scalar> = (0..len).map(|_| rng.random_range(-1.0..1.0)).collect();
            let coefficients: Vec<$scalar> = (0..len).map(|_| rng.random_range(-1.0..1.0)).collect();
            $probe.measure("vec", "owned", &values, &coefficients, (&values, &coefficients), |op| {
                let x = black_box(&values);
                let y = black_box(&coefficients);
                match op {
                    Operation::Sum => x.iter().copied().sum(),
                    Operation::Dot | Operation::DotSlice => x.iter().zip(y).map(|(&x, &y)| x * y).sum(),
                    Operation::Norm => x.iter().map(|&x| x * x).sum::<$scalar>().sqrt(),
                }
            });
            let x = faer::Col::from_fn(len, |i| values[i]);
            let y = faer::Col::from_fn(len, |i| coefficients[i]);
            $probe.measure("faer", "owned", &values, &coefficients, (&x, &y), |op| {
                let x = black_box(&x);
                let y = black_box(&y);
                match op {
                    Operation::Sum => x.as_ref().sum(),
                    Operation::Dot | Operation::DotSlice => faer::linalg::matmul::dot::inner_prod(
                        x.as_ref().transpose(), faer::Conj::No, y.as_ref(), faer::Conj::No,
                    ),
                    Operation::Norm => x.as_ref().norm_l2(),
                }
            });
            let x = nalgebra::DVector::from_column_slice(&values);
            let y = nalgebra::DVector::from_column_slice(&coefficients);
            $probe.measure("nalgebra", "owned", &values, &coefficients, (&x, &y), |op| {
                let x = black_box(&x);
                let y = black_box(&y);
                match op {
                    Operation::Sum => x.sum(),
                    Operation::Dot | Operation::DotSlice => x.dot(y),
                    Operation::Norm => x.norm(),
                }
            });
            for stride in [1_isize, 2, -1, -2] {
                let mut x = ndarray::Array1::from_elem(2 * len + 5, <$scalar>::NAN);
                let mut y = x.clone();
                let end = 2 + len * stride.unsigned_abs();
                let mut x_view = x.slice_mut(s![2..end;stride]);
                let mut y_view = y.slice_mut(s![2..end;stride]);
                for i in 0..len {
                    x_view[i] = values[i];
                    y_view[i] = coefficients[i];
                }
                let x = x_view.view();
                let y = y_view.view();
                let layout = format!("borrowed_stride{stride}");
                $probe.measure("ndarray", &layout, &values, &coefficients, (&x, &y), |op| {
                    let x = black_box(&x);
                    let y = black_box(&y);
                    match op {
                        Operation::Sum => x.sum(),
                        Operation::Dot | Operation::DotSlice => ndarray::linalg::Dot::dot(x, y),
                        Operation::Norm => ndarray::linalg::Dot::dot(x, x).sqrt(),
                    }
                });
            }
        }
    }};
}

fn main() {
    let probe = Probe {
        label: std::env::var("PROBE_LABEL").unwrap_or_else(|_| "candidate".into()),
        samples: setting("PROBE_SAMPLES", 3),
        elements: setting("PROBE_ELEMENTS", 1_000_000),
    };
    let lengths: Vec<usize> = std::env::var("PROBE_LENGTHS")
        .unwrap_or_else(|_| "0,1,7,8,9,33,256,5000,262144".into())
        .split(',')
        .map(|v| v.parse().unwrap())
        .collect();
    println!("label,backend,layout,scalar,operation,method,sample,length,repetitions,nanoseconds");
    eprintln!(
        "label,backend,layout,scalar,operation,method,length,absolute_error,scaled_error_or_objective"
    );
    scalar_probes!(probe, f32, &lengths);
    scalar_probes!(probe, f64, &lengths);
    probe.ridge(512, 512);
    probe.ridge(32768, 256);
}
