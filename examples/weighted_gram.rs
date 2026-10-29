//! Borrow dense and CSC inputs and reuse an ndarray coefficient matrix.

#[path = "../tests/common/backend_aliases.rs"]
mod backend_aliases;
use backend_aliases::{ndarray, sprs};
use lazymatrix::{
    Centering, LazyMatrix, Normalization, Scaling, SprsCsc, WeightedGramInto, WithIntercept,
};
use ndarray::{Array2, array};

fn main() {
    let dense = array![[1.0, 0.0], [2.0, 3.0], [0.0, 4.0]];
    let sparse = sprs::CsMat::new_csc(
        (3, 2),
        vec![0, 2, 4],
        vec![0, 1, 1, 2],
        vec![1.0, 2.0, 3.0, 4.0],
    );
    let weights = array![1.0, 0.5, 2.0];
    let spec = Normalization::new(Centering::Mean, Scaling::Sd { df: 0 });
    let dense = LazyMatrix::new(dense.view(), spec).unwrap();
    let sparse = LazyMatrix::new(SprsCsc::try_new(sparse.view()).unwrap(), spec).unwrap();
    let mut gram = Array2::zeros((2, 2));
    dense
        .weighted_gram_into(&weights.view(), &mut gram)
        .unwrap();
    println!("Dense input:\n{gram}");
    sparse
        .weighted_gram_into(&weights, &mut gram.view_mut())
        .unwrap();
    println!("CSC input:\n{gram}");

    let dense = WithIntercept::new(&dense);
    let sparse = WithIntercept::new(&sparse);
    let mut gram = Array2::zeros((3, 3));
    dense.weighted_gram_into(&weights, &mut gram).unwrap();
    println!("Dense input with intercept:\n{gram}");
    sparse
        .weighted_gram_into(&weights, &mut gram.view_mut())
        .unwrap();
    println!("CSC input with intercept:\n{gram}");
}
