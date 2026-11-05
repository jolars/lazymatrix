//! Complete hybrid fits that rebuild aggregates through logical column updates.

use std::{hint::black_box, time::Instant};

use ndarray::{Array2, ArrayView2, Axis, ShapeBuilder};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::json;
use shrinkage::{ClusteredLasso, Gaussian, Hybrid, HybridFit, Normalization, Problem, Termination};

fn verify(x: ArrayView2<'_, f64>, y: &[f64], penalty: f64, fit: &HybridFit) -> (f64, f64, f64) {
    assert_eq!(fit.termination(), Termination::Converged);
    let n = y.len() as f64;
    let beta = fit.coefficients();
    let residual: Vec<_> = (0..y.len())
        .map(|i| y[i] - fit.intercept() - (0..x.ncols()).map(|j| x[(i, j)] * beta[j]).sum::<f64>())
        .collect();
    let objective = residual.iter().map(|r| r * r).sum::<f64>() / (2.0 * n)
        + penalty * beta.iter().map(|b| b.abs()).sum::<f64>();
    let mean = residual.iter().sum::<f64>() / n;
    let mut kkt = mean.abs();
    let mut dual: Vec<_> = residual.iter().map(|r| (r - mean) / n).collect();
    let mut max_product = 0.0_f64;
    for j in 0..x.ncols() {
        let gradient = (0..y.len()).map(|i| x[(i, j)] * residual[i]).sum::<f64>() / n;
        kkt = kkt.max(if beta[j] == 0.0 {
            (gradient.abs() - penalty).max(0.0)
        } else {
            (gradient - penalty * beta[j].signum()).abs()
        });
        max_product = max_product.max((0..y.len()).map(|i| x[(i, j)] * dual[i]).sum::<f64>().abs());
    }
    let shrink = (penalty / max_product).min(1.0) * (1.0 - 1e-12);
    dual.iter_mut().for_each(|d| *d *= shrink);
    let dual_objective = y.iter().zip(&dual).map(|(y, d)| y * d).sum::<f64>()
        - n * dual.iter().map(|d| d * d).sum::<f64>() / 2.0;
    let gap = (objective - dual_objective).max(0.0);
    assert!(objective.is_finite() && kkt.is_finite() && gap.is_finite());
    assert!((objective - fit.objective()).abs() <= 1e-10 * (1.0 + objective.abs()));
    assert!(kkt <= 1e-7 && gap <= 1e-7 * (1.0 + objective));
    (objective, kkt, gap)
}

struct Case<'a> {
    label: &'a str,
    backend: &'a str,
    layout: &'a str,
    signal: usize,
    penalty: f64,
    samples: usize,
}

fn measure(case: Case<'_>, x: ArrayView2<'_, f64>, y: &[f64], fit: impl Fn() -> HybridFit) {
    let Case {
        label,
        backend,
        layout,
        signal,
        penalty,
        samples,
    } = case;
    let warmup = fit();
    verify(x, y, penalty, &warmup);
    drop(warmup);
    for sample in 0..samples {
        let start = Instant::now();
        let result = black_box(&fit)();
        let ns = start.elapsed().as_nanos() as u64;
        let (objective, kkt, gap) = verify(x, y, penalty, &result);
        println!(
            "{}",
            json!({
                "label": label, "backend": backend, "layout": layout,
                "rows": x.nrows(), "cols": x.ncols(), "signal": signal,
                "penalty": penalty, "sample": sample, "ns": ns,
                "objective": objective, "kkt": kkt, "gap": gap,
                "iterations": result.iterations(), "pg_steps": result.pg_steps(),
                "sweeps": result.coordinate_sweeps(),
                "nonzero": result.coefficients().iter().filter(|b| **b != 0.0).count(),
            })
        );
    }
}

fn main() {
    let label = std::env::var("PROBE_LABEL").unwrap();
    let samples: usize = std::env::var("PROBE_SAMPLES").unwrap().parse().unwrap();
    let order: usize = std::env::var("PROBE_ORDER_OFFSET")
        .unwrap()
        .parse()
        .unwrap();
    for cols in [64, 1000] {
        let rows = 5000;
        let mut rng = ChaCha8Rng::seed_from_u64(8421);
        let x =
            Array2::from_shape_simple_fn((rows, cols).f(), || rng.random_range(-1.0..1.0) + 0.375);
        let xf = faer::Mat::from_fn(rows, cols, |i, j| x[(i, j)]);
        let xn = nalgebra::DMatrix::from_fn(rows, cols, |i, j| x[(i, j)]);
        for (signal, penalty) in [(8, 0.01), (64, 0.005), (8, 2.0)] {
            let y: Vec<f64> = (0..rows)
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
            let backends = if order.is_multiple_of(2) {
                ["ndarray", "faer", "nalgebra"]
            } else {
                ["nalgebra", "faer", "ndarray"]
            };
            for backend in backends {
                let case = |layout| Case {
                    label: &label,
                    backend,
                    layout,
                    signal,
                    penalty,
                    samples,
                };
                macro_rules! fits {
                    ($matrix:expr, $vector:ty, $oracle:expr, $response:expr, $layout:expr) => {{
                        // Zero fusion makes the independent L1 KKT and dual
                        // checks exact while retaining the hybrid's rebuilds.
                        let problem = Problem::new(
                            $matrix,
                            Gaussian::new($response),
                            ClusteredLasso::new(penalty, 0.0),
                        )
                        .normalize(Normalization::Center);
                        let solver = Hybrid::<$vector>::new().tolerance(1e-8);
                        measure(case($layout), $oracle, $response, || {
                            problem.fit_with(&solver).unwrap()
                        });
                    }};
                }
                match backend {
                    "ndarray" => {
                        fits!(&x, ndarray::Array1<f64>, x.view(), &y, "forward");
                        let mut reversed = x.view();
                        reversed.invert_axis(Axis(0));
                        let mut response = y.clone();
                        response.reverse();
                        fits!(
                            &reversed,
                            ndarray::Array1<f64>,
                            reversed,
                            &response,
                            "reverse"
                        );
                    }
                    "faer" => fits!(&xf, faer::Col<f64>, x.view(), &y, "forward"),
                    "nalgebra" => fits!(&xn, nalgebra::DVector<f64>, x.view(), &y, "forward"),
                    _ => unreachable!(),
                }
            }
        }
    }
}
