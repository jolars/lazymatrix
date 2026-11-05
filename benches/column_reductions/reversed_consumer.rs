//! Paired complete Shrinkage fits over forward and reversed borrowed designs.

use std::{hint::black_box, time::Instant};

use ndarray::{Array2, ArrayView2, Axis, ShapeBuilder};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::json;
use shrinkage::{Lasso, LassoFit, Normalization, StoppingCriterion};

fn verify(x: ArrayView2<'_, f64>, y: &[f64], penalty: f64, fit: &LassoFit) -> (f64, f64, f64, f64) {
    let n = y.len() as f64;
    let coefficients = fit.coefficients();
    let residual: Vec<_> = (0..y.len())
        .map(|i| {
            y[i] - fit.intercept()
                - (0..x.ncols())
                    .map(|j| x[(i, j)] * coefficients[j])
                    .sum::<f64>()
        })
        .collect();
    let objective = residual.iter().map(|r| r * r).sum::<f64>() / (2.0 * n)
        + penalty * coefficients.iter().map(|x| x.abs()).sum::<f64>();
    let mut kkt = (residual.iter().sum::<f64>() / n).abs();
    let mut feasibility = fit.dual_certificate().iter().sum::<f64>().abs();
    for j in 0..x.ncols() {
        let gradient = (0..y.len()).map(|i| x[(i, j)] * residual[i]).sum::<f64>() / n;
        let violation = if coefficients[j] == 0.0 {
            (gradient.abs() - penalty).max(0.0)
        } else {
            (gradient - penalty * coefficients[j].signum()).abs()
        };
        kkt = kkt.max(violation);
        let dual_product = (0..y.len())
            .map(|i| x[(i, j)] * fit.dual_certificate()[i])
            .sum::<f64>();
        feasibility = feasibility.max((dual_product.abs() - penalty).max(0.0));
    }
    let dual_objective = y
        .iter()
        .zip(fit.dual_certificate())
        .map(|(y, d)| y * d)
        .sum::<f64>()
        - n * fit.dual_certificate().iter().map(|d| d * d).sum::<f64>() / 2.0;
    let relative_gap = (objective - dual_objective).max(0.0) / fit.reference_loss();
    assert!(
        objective.is_finite()
            && kkt.is_finite()
            && feasibility.is_finite()
            && relative_gap.is_finite()
    );
    assert!((objective - fit.objective()).abs() <= 1e-10 * (1.0 + objective.abs()));
    assert!((dual_objective - fit.dual_objective()).abs() <= 1e-10 * (1.0 + dual_objective.abs()));
    assert!((kkt - fit.kkt_violation()).abs() <= 1e-9 * (1.0 + kkt.abs()));
    assert!(relative_gap <= 1e-8 + 1e-12);
    assert!(feasibility <= 1e-10 * (1.0 + penalty));
    (objective, kkt, relative_gap, feasibility)
}

fn main() {
    let label = std::env::var("PROBE_LABEL").unwrap();
    let samples: usize = std::env::var("PROBE_SAMPLES").unwrap().parse().unwrap();
    for cols in [64, 1000] {
        let rows = 5000;
        let mut rng = ChaCha8Rng::seed_from_u64(8421);
        let x =
            Array2::from_shape_simple_fn((rows, cols).f(), || rng.random_range(-1.0..1.0) + 0.375);
        let y: Vec<f64> = (0..rows)
            .map(|i| {
                1.25 + (0..8)
                    .map(|j| {
                        (x[(i, j)] - 0.375)
                            * if j % 2 == 0 {
                                1.0 / (j + 1) as f64
                            } else {
                                -1.0 / (j + 1) as f64
                            }
                    })
                    .sum::<f64>()
                    + rng.random_range(-0.05..0.05)
            })
            .collect();
        for layout in ["forward", "reverse"] {
            let mut matrix = x.view();
            let mut response = y.clone();
            if layout == "reverse" {
                matrix.invert_axis(Axis(0));
                response.reverse();
            }
            for penalty in [0.01, 2.0] {
                let model = Lasso::new(penalty)
                    .fit_intercept(true)
                    .normalize(Normalization::Center)
                    .terminate_on(StoppingCriterion::duality_gap(1e-8))
                    .max_iterations(10_000);
                let warmup = model.fit(&matrix, &response).unwrap();
                verify(matrix, &response, penalty, &warmup);
                drop(warmup);
                for sample in 0..samples {
                    let start = Instant::now();
                    let fit = black_box(&model)
                        .fit(black_box(&matrix), black_box(&response))
                        .unwrap();
                    let nanos = start.elapsed().as_nanos();
                    let (objective, kkt, relative_gap, feasibility) =
                        verify(matrix, &response, penalty, &fit);
                    println!(
                        "{}",
                        json!({
                            "label": label, "layout": layout, "rows": rows, "cols": cols,
                            "penalty": penalty, "sample": sample, "ns": nanos as u64,
                            "objective": objective, "kkt": kkt, "relative_gap": relative_gap,
                            "feasibility": feasibility, "iterations": fit.iterations(),
                            "nonzero": fit.coefficients().iter().filter(|x| **x != 0.0).count(),
                        })
                    );
                }
            }
        }
    }
}
