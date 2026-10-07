use std::hint::black_box;

use ndarray_glm::{Linear, Logistic, ModelBuilder};

use crate::{assert_close, fixture::Fixture, metrics};

macro_rules! profile {
    ($fixture:expr, $response:expr, $family:ty, $name:expr) => {{
        let fixture = $fixture;
        let mut expected: Option<Vec<f64>> = None;
        for (name, standardize) in [("raw", false), ("standardized", true)] {
            let build = || {
                let builder = ModelBuilder::<$family>::data($response, &fixture.x);
                if standardize {
                    builder.build().unwrap()
                } else {
                    builder.no_standardize().build().unwrap()
                }
            };
            black_box(build().fit().unwrap());
            let (model, build_counts) = metrics::measure(build);
            let (fit, fit_counts) = metrics::measure(|| black_box(&model).fit().unwrap());
            let predictions = fit.predict(&fixture.x, None);
            if let Some(expected) = &expected {
                assert_close(predictions.as_slice().unwrap(), expected);
            } else {
                expected = Some(predictions.to_vec());
            }
            // Prediction is warmed and measured separately from construction and IRLS.
            let (predictions, predict_counts) = metrics::measure(|| fit.predict(&fixture.x, None));
            assert!(predictions.iter().all(|value| value.is_finite()));
            build_counts.print($name, "ndarray_dense", fixture, name, "build", fit.n_iter);
            fit_counts.print($name, "ndarray_dense", fixture, name, "fit", fit.n_iter);
            predict_counts.print($name, "ndarray_dense", fixture, name, "predict", fit.n_iter);
        }
    }};
}

pub fn run(fixture: &Fixture) {
    profile!(fixture, &fixture.linear, Linear, "ndarray_glm_linear");
    profile!(fixture, &fixture.logistic, Logistic, "ndarray_glm_logistic");
}
