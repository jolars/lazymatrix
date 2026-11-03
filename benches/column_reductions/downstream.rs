//! A disposable consumer harness; prepare-downstream.py creates its workspaces.

use std::{hint::black_box, path::Path, time::Instant};

use ndarray::{Array2, ShapeBuilder};
use serde_json::{Value, json};
use shrinkage::{Lasso, Normalization, StoppingCriterion};

fn read_values(path: &Path) -> Vec<f64> {
    let bytes = std::fs::read(path).unwrap();
    bytes
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let directory = Path::new(&args[1]);
    let label = &args[2];
    let repetitions: usize = args[3].parse().unwrap();
    let cases = [
        ("original", "original", 1.0),
        ("larger", "larger", 1.0),
        ("correlated", "correlated", 1.0),
        ("weak", "original", 0.1),
        ("strong", "original", 5.0),
        ("zero", "original", 20.0),
    ];
    for (case, source, multiplier) in cases {
        let input = directory.join(source);
        let metadata: Value =
            serde_json::from_slice(&std::fs::read(input.join("metadata.json")).unwrap()).unwrap();
        let n = metadata["n"].as_u64().unwrap() as usize;
        let p = metadata["p"].as_u64().unwrap() as usize;
        let x = Array2::from_shape_vec((n, p).f(), read_values(&input.join("x.bin"))).unwrap();
        let y = read_values(&input.join("y.bin"));
        let alpha = metadata["alpha"].as_f64().unwrap() * multiplier;
        let model = Lasso::new(alpha)
            .fit_intercept(false)
            .normalize(Normalization::None)
            .terminate_on(StoppingCriterion::duality_gap(1e-8))
            .max_iterations(10_000);
        let _ = black_box(model.fit(&x, &y).unwrap());
        for sample in 0..repetitions {
            let start = Instant::now();
            let fit = black_box(&model).fit(black_box(&x), black_box(&y)).unwrap();
            let elapsed = start.elapsed().as_nanos();
            println!(
                "{}",
                json!({
                    "build": label, "case": case, "sample": sample,
                    "ns": elapsed as u64, "alpha": alpha,
                    "iterations": fit.iterations(),
                    "termination": format!("{:?}", fit.termination()),
                    "objective": fit.objective(), "kkt": fit.kkt_violation(),
                    "duality_gap": fit.duality_gap(),
                    "dual_objective": fit.dual_objective(),
                    "reference_loss": fit.reference_loss(),
                    "coefficients": fit.coefficients(),
                    "dual_certificate": fit.dual_certificate(),
                    "feature_visits": fit.working_set_diagnostics().feature_visits,
                    "full_checks": fit.working_set_diagnostics().full_checks,
                })
            );
        }
    }
}
