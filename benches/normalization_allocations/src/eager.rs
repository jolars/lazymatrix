use std::hint::black_box;

use shrinkage::lazymatrix::{
    Centering, ColumnStats, LazyMatrix, MatTransposeVecInto, MatVecInto, MaterializeDense,
    MatrixOwned, RawColumns, Scaling, SprsCsc,
};
use shrinkage::{Gaussian, Normalization, Problem, ProximalGradient, ProximalVector, Ridge};

use crate::{assert_close, fixture::Fixture, metrics, tracked};

fn profile<M, D, V>(fixture: &Fixture, name: &str, matrix: &M)
where
    M: ColumnStats<f64>
        + RawColumns<f64>
        + MaterializeDense<f64>
        + MatVecInto<V>
        + MatTransposeVecInto<V>,
    D: MatrixOwned<f64>
        + ColumnStats<f64>
        + RawColumns<f64>
        + MatVecInto<V>
        + MatTransposeVecInto<V>,
    V: ProximalVector,
{
    for (label, spec, normalization) in [
        (
            "centered",
            shrinkage::lazymatrix::Normalization::new(Centering::Mean, Scaling::None),
            Normalization::Center,
        ),
        (
            "standardized",
            shrinkage::lazymatrix::Normalization::new(Centering::Mean, Scaling::Sd),
            Normalization::Standardize,
        ),
    ] {
        let response = fixture.linear.as_slice().unwrap();
        let solver = ProximalGradient::<tracked::Vector<V>>::new().initial_step(64.0);
        let source = tracked::Matrix(matrix);
        let lazy_problem = Problem::new(&source, Gaussian::new(response), Ridge::new(0.1))
            .normalize(normalization);
        drop(black_box(lazy_problem.fit_with(&solver).unwrap()));
        let (lazy_fit, lazy_counts) = metrics::measure(|| lazy_problem.fit_with(&solver).unwrap());
        lazy_counts.print(
            "shrinkage",
            name,
            fixture,
            label,
            "lazy_fit",
            lazy_fit.iterations(),
        );

        let (eager, build_counts) = metrics::measure(|| {
            LazyMatrix::new(&source, spec)
                .unwrap()
                .to_eager::<D>()
                .unwrap()
        });
        build_counts.print("shrinkage", name, fixture, label, "eager_build", 0);
        let tracked_eager = tracked::Matrix(&eager);
        let eager_problem = Problem::new(&tracked_eager, Gaussian::new(response), Ridge::new(0.1))
            .normalize(Normalization::None);
        drop(black_box(eager_problem.fit_with(&solver).unwrap()));
        let (eager_fit, eager_counts) =
            metrics::measure(|| eager_problem.fit_with(&solver).unwrap());
        assert_eq!(eager_counts.clones, 0);
        assert_eq!(eager_counts.predictor_calls, 0);
        let coefficients: Vec<_> = eager_fit
            .coefficients()
            .iter()
            .enumerate()
            .map(|(j, &value)| eager.scales().map_or(value, |scales| value / scales[j]))
            .collect();
        let intercept = eager_fit.intercept()
            - eager.centers().map_or(0.0, |centers| {
                centers
                    .iter()
                    .zip(&coefficients)
                    .map(|(center, coefficient)| center * coefficient)
                    .sum::<f64>()
            });
        assert_close(&coefficients, lazy_fit.coefficients());
        assert_close(&[intercept], &[lazy_fit.intercept()]);
        assert_close(&[eager_fit.objective()], &[lazy_fit.objective()]);
        eager_counts.print(
            "shrinkage",
            name,
            fixture,
            label,
            "eager_fit",
            eager_fit.iterations(),
        );
    }
}

pub fn run(fixture: &Fixture) {
    let x = &fixture.x;
    profile::<_, ndarray::Array2<f64>, ndarray::Array1<f64>>(fixture, "ndarray_dense", x);
    let dense = faer::Mat::from_fn(x.nrows(), x.ncols(), |i, j| x[(i, j)]);
    profile::<_, faer::Mat<f64>, faer::Col<f64>>(fixture, "faer_dense", &dense);
    let mut triplets = sprs::TriMat::new(x.dim());
    let mut faer_triplets = Vec::new();
    for ((i, j), &value) in x.indexed_iter() {
        if value != 0.0 {
            triplets.add_triplet(i, j, value);
            faer_triplets.push(faer::sparse::Triplet::new(i, j, value));
        }
    }
    let sparse = SprsCsc::try_new(triplets.to_csc::<usize>()).unwrap();
    profile::<_, ndarray::Array2<f64>, ndarray::Array1<f64>>(fixture, "sprs_to_ndarray", &sparse);
    let sparse = faer::sparse::SparseColMat::<usize, f64>::try_new_from_triplets(
        x.nrows(),
        x.ncols(),
        &faer_triplets,
    )
    .unwrap();
    profile::<_, faer::Mat<f64>, faer::Col<f64>>(fixture, "faer_csc_to_dense", &sparse);
}
