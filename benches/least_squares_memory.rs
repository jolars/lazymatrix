//! Allocation profiles and convergence checks for complete ridge fits.

use lazymatrix::{ColumnStats, MatTransposeVecInto, MatVecInto, MaterializeDense, MatrixOwned};

#[path = "common/allocations.rs"]
mod allocations;
#[path = "../tests/common/backend_aliases.rs"]
mod backend_aliases;
#[path = "common/least_squares.rs"]
mod least_squares;
#[path = "common/normalization.rs"]
mod normalization;

use backend_aliases::ndarray;
use normalization::{BenchVector, Fixture};

fn profile<M, D, V>(fixture: &Fixture, backend: &str, source: &M)
where
    M: ColumnStats<f64> + MaterializeDense<f64> + MatVecInto<V> + MatTransposeVecInto<V>,
    D: MatrixOwned<f64> + MatVecInto<V> + MatTransposeVecInto<V>,
    V: BenchVector,
{
    let response = least_squares::response::<V>(fixture);
    let lazy = least_squares::fit::<_, D, _>(source, &response, false);
    let eager = least_squares::fit::<_, D, _>(source, &response, true);
    least_squares::check(&lazy, &eager);
    for (strategy, materialize, result) in [("lazy", false, lazy), ("eager", true, eager)] {
        let peak =
            allocations::measure(|| least_squares::fit::<_, D, _>(source, &response, materialize));
        let source_bytes = fixture.source_bytes();
        let response_bytes = fixture.rows * size_of::<f64>();
        println!(
            "{backend},{},{},{},{},{},{strategy},{source_bytes},{response_bytes},{},{},{:.17},{:.17},{:.17}",
            fixture.rows,
            fixture.cols,
            fixture.density,
            fixture.sparse.nnz(),
            fixture.empty_columns(),
            source_bytes + response_bytes + peak,
            result.iterations,
            result.objective,
            result.gradient_norm,
            result.gradient_threshold,
        );
    }
}

fn main() {
    println!(
        "backend,n,p,target_density,nnz,empty_columns,strategy,source_bytes,response_bytes,peak_total_bytes,iterations,objective,gradient_norm,gradient_threshold"
    );
    for fixture in least_squares::fixtures() {
        profile::<_, ndarray::Array2<f64>, ndarray::Array1<f64>>(
            &fixture,
            "sprs_to_ndarray",
            &fixture.sparse,
        );
        #[cfg(feature = "faer_all")]
        {
            use backend_aliases::faer;
            let source = fixture.faer();
            profile::<_, faer::Mat<f64>, faer::Col<f64>>(&fixture, "faer_csc_to_dense", &source);
        }
    }
}
