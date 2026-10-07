use ndarray::{Array1, Array2, ShapeBuilder};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

pub struct Fixture {
    pub x: Array2<f64>,
    pub linear: Array1<f64>,
    pub logistic: Array1<bool>,
    pub density: f64,
}

impl Fixture {
    pub fn new(rows: usize, columns: usize, density: f64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(205);
        let x = Array2::from_shape_fn((rows, columns).f(), |(_, j)| {
            if rng.random::<f64>() < density {
                (0.5 + 1.5 * j as f64 / columns as f64) * rng.random_range(-1.0..1.0)
            } else {
                0.0
            }
        });
        let linear = Array1::from_shape_fn(rows, |i| {
            0.5 + 2.0 * x[(i, 0)] - 1.5 * x[(i, 2)]
                + 0.75 * x[(i, 7)]
                + 0.01 * rng.random_range(-1.0..1.0)
        });
        let logistic = Array1::from_shape_fn(rows, |i| {
            let eta = 0.2 + x[(i, 0)] - 0.75 * x[(i, 2)] + 0.5 * x[(i, 7)];
            rng.random::<f64>() < 1.0 / (1.0 + (-eta).exp())
        });
        Self {
            x,
            linear,
            logistic,
            density,
        }
    }
}
