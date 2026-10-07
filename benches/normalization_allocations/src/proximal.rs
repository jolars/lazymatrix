use std::hint::black_box;

use faer::sparse::{SparseColMat, Triplet};
use shrinkage::lazymatrix::{ColumnStats, MatTransposeVecInto, MatVecInto, RawColumns, SprsCsc};
use shrinkage::{
    Gaussian, Normalization, Problem, ProximalGradient, ProximalVector, Ridge, Termination,
};

use crate::{assert_close, fixture::Fixture, metrics, tracked};

fn profile<M, V>(fixture: &Fixture, backend: &str, matrix: &M)
where
    M: ColumnStats<f64> + RawColumns<f64> + MatVecInto<V> + MatTransposeVecInto<V>,
    V: ProximalVector,
{
    let response = fixture.linear.as_slice().unwrap();
    for (name, normalization) in [
        ("raw", Normalization::None),
        ("centered", Normalization::Center),
        ("standardized", Normalization::Standardize),
    ] {
        let problem =
            Problem::new(matrix, Gaussian::new(response), Ridge::new(0.1)).normalize(normalization);
        let native_solver = ProximalGradient::<V>::new().initial_step(64.0);
        drop(black_box(problem.fit_with(&native_solver).unwrap()));
        let (native, native_counts) =
            metrics::measure(|| problem.fit_with(&native_solver).unwrap());
        assert_eq!(native.termination(), Termination::Converged);
        let tracked_matrix = tracked::Matrix(matrix);
        let problem = Problem::new(&tracked_matrix, Gaussian::new(response), Ridge::new(0.1))
            .normalize(normalization);
        let solver = ProximalGradient::<tracked::Vector<V>>::new().initial_step(64.0);
        // Warm all product paths before sampling the same fit from zero coefficients.
        drop(black_box(problem.fit_with(&solver).unwrap()));
        let (fit, counts) = metrics::measure(|| black_box(&problem).fit_with(&solver).unwrap());
        assert_eq!(fit.termination(), Termination::Converged);
        assert_eq!(fit.iterations(), native.iterations());
        assert_close(fit.coefficients(), native.coefficients());
        assert_close(&[fit.intercept()], &[native.intercept()]);
        assert_close(&[fit.objective()], &[native.objective()]);
        assert_eq!(counts.calls, native_counts.calls);
        assert_eq!(counts.bytes, native_counts.bytes);
        assert_eq!(counts.transpose, fit.iterations() + 1);
        assert!(
            counts.forward > 2 * counts.transpose,
            "fixture must exercise rejected trials"
        );
        assert_eq!(
            counts.clones,
            if name == "standardized" {
                counts.forward
            } else {
                0
            }
        );
        if name == "standardized" {
            assert!(counts.clone_calls >= counts.clones);
            assert!(counts.clone_bytes >= counts.clones * fixture.x.ncols() * size_of::<f64>());
        } else {
            assert_eq!(counts.clone_calls, 0);
            assert_eq!(counts.clone_bytes, 0);
        }
        counts.print("shrinkage", backend, fixture, name, "fit", fit.iterations());
    }
}

pub fn run(fixture: &Fixture) {
    let x = &fixture.x;
    let (rows, columns) = x.dim();
    let dense = faer::Mat::from_fn(rows, columns, |i, j| x[(i, j)]);
    profile::<_, faer::Col<f64>>(fixture, "faer_dense", &dense);
    profile::<_, ndarray::Array1<f64>>(fixture, "ndarray_dense", x);
    let mut triplets = Vec::new();
    let mut sprs_triplets = sprs::TriMat::new((rows, columns));
    for j in 0..columns {
        for i in 0..rows {
            let value = x[(i, j)];
            if value != 0.0 {
                triplets.push(Triplet::new(i, j, value));
                sprs_triplets.add_triplet(i, j, value);
            }
        }
    }
    let sparse =
        SparseColMat::<usize, f64>::try_new_from_triplets(rows, columns, &triplets).unwrap();
    profile::<_, faer::Col<f64>>(fixture, "faer_csc", &sparse);
    let sparse = SprsCsc::try_new(sprs_triplets.to_csc::<usize>()).unwrap();
    profile::<_, Vec<f64>>(fixture, "sprs_csc", &sparse);
}
