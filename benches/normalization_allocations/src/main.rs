//! Allocation profiles of the downstream fitting APIs, with fixtures excluded.

mod eager;
mod fixture;
mod glm;
mod metrics;
mod proximal;
mod tracked;

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(
            actual.is_finite() && (actual - expected).abs() <= 1e-7 * expected.abs().max(1.0),
            "fit changed: {actual} != {expected}"
        );
    }
}

fn main() {
    println!(
        "consumer,backend,n,p,density,normalization,phase,iterations,forward,transpose,rejected_trials,clones,allocations,bytes,clone_allocations,clone_bytes,stats_allocations,stats_bytes,predictor_sized_calls,design_sized_calls,peak_extra_live_bytes,elapsed_ns"
    );
    for (rows, columns) in [(512, 32), (2_048, 128)] {
        for density in [1.0, 0.1] {
            let fixture = fixture::Fixture::new(rows, columns, density);
            metrics::matrix_sizes(rows, columns);
            proximal::run(&fixture);
            glm::run(&fixture);
            eager::run(&fixture);
        }
    }
}
