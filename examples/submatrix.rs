//! Restrict fitted normalization without refitting a selected set of rows.

#[path = "../tests/common/backend_aliases.rs"]
mod backend_aliases;
use backend_aliases::ndarray::{Array2, array};
use lazymatrix::{Centering, LazyMatrix, MatTransposeVec, MatVec, Normalization, Scaling};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let raw = array![[1.0, 0.0, 3.0], [2.0, 3.0, 6.0], [0.0, 4.0, 9.0]];
    let lazy = LazyMatrix::new(raw.view(), Normalization::new(Centering::Mean, Scaling::Sd))?;
    let selected = lazy.submatrix(1..3, 1..3);
    assert_eq!(selected.centers(), Some(&lazy.centers().unwrap()[1..3]));
    let coefficients = array![1.0, -1.0];
    let forward = selected.matvec(&coefficients)?;
    let transpose = selected.mat_transpose_vec(&array![0.5, 1.0])?;
    let eager = selected.to_eager::<Array2<f64>>()?;
    assert_eq!(eager.matvec(&coefficients)?, forward);
    println!("Forward: {forward}\nTranspose: {transpose}");
    Ok(())
}
