//! Untimed allocation profiles over construction and repeated products.

use std::hint::black_box;

use lazymatrix::{
    ColumnStats, LazyMatrix, MatTransposeVecInto, MatVecInto, MaterializeDense, MatrixOwned,
};

#[path = "../tests/common/backend_aliases.rs"]
mod backend_aliases;
#[path = "common/normalization.rs"]
mod normalization;

#[path = "common/allocations.rs"]
mod allocations;
use allocations::measure;

use backend_aliases::ndarray;
use normalization::{BenchVector, Fixture, NORMALIZATIONS, PAIRS};

fn products<M, V>(fixture: &Fixture, matrix: &M) -> (V, V, V, V)
where
    M: MatVecInto<V> + MatTransposeVecInto<V>,
    V: BenchVector,
{
    let (x, u, mut y, mut t) = fixture.vectors::<V>();
    for _ in 0..PAIRS {
        matrix.matvec_into(black_box(&x), &mut y).unwrap();
        matrix
            .mat_transpose_vec_into(black_box(&u), &mut t)
            .unwrap();
        black_box((&y, &t));
    }
    (x, u, y, t)
}

fn profile<M, D, V>(fixture: &Fixture, backend: &str, source: &M)
where
    M: ColumnStats<f64> + MaterializeDense<f64> + MatVecInto<V> + MatTransposeVecInto<V>,
    D: MatrixOwned<f64> + MatVecInto<V> + MatTransposeVecInto<V>,
    V: BenchVector,
{
    for (normalization, spec) in NORMALIZATIONS {
        let parameters = fixture.cols
            * size_of::<f64>()
            * if spec.center == lazymatrix::Centering::None {
                1
            } else {
                2
            };
        let dense_bytes = fixture.rows * fixture.cols * size_of::<f64>();
        let source_bytes = fixture.source_bytes();
        // Warm backend initialization before recording a complete lifecycle.
        let lazy = LazyMatrix::new(source, spec).unwrap();
        let eager = lazy.to_eager::<D>().unwrap();
        drop(products::<_, V>(fixture, &lazy));
        drop(products::<_, V>(fixture, &eager));
        drop(eager);
        drop(lazy);

        let raw_peak = measure(|| products::<_, V>(fixture, source));
        let lazy_peak = measure(|| {
            let matrix = LazyMatrix::new(source, spec).unwrap();
            let vectors = products::<_, V>(fixture, &matrix);
            (matrix, vectors)
        });
        let eager_peak = measure(|| {
            let lazy = LazyMatrix::new(source, spec).unwrap();
            let matrix = lazy.to_eager::<D>().unwrap();
            drop(lazy);
            let vectors = products::<_, V>(fixture, &matrix);
            (matrix, vectors)
        });
        for (strategy, representation, added_representation, peak) in [
            ("raw", source_bytes, 0, raw_peak),
            ("lazy", source_bytes + parameters, parameters, lazy_peak),
            (
                "eager",
                dense_bytes + parameters,
                dense_bytes + parameters,
                eager_peak,
            ),
        ] {
            assert!(peak >= added_representation);
            println!(
                "{backend},{},{},{},{},{},{normalization},{strategy},{representation},{},{},{}",
                fixture.rows,
                fixture.cols,
                fixture.density,
                fixture.sparse.nnz(),
                fixture.empty_columns(),
                peak - added_representation,
                source_bytes + peak,
                source_bytes
            );
        }
    }
}

fn main() {
    println!(
        "backend,n,p,target_density,nnz,empty_columns,normalization,strategy,representation_bytes,workspace_bytes,peak_total_bytes,source_bytes"
    );
    for fixture in normalization::fixtures() {
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
