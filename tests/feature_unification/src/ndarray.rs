#[path = "oracle.rs"]
mod oracle;

/// Exercise this consumer's backend types after Cargo unifies features.
pub fn check() {
    use ndarray::{Array1, array};
    let dense = array![[1.0, 0.0], [2.0, 3.0], [0.0, 6.0]];
    {
        use lazymatrix::{Centering, LazyMatrix, MatVec, Normalization, Scaling};
        let lazy =
            LazyMatrix::new(&dense, Normalization::new(Centering::Mean, Scaling::Range)).unwrap();
        let eager = lazy.to_eager::<ndarray::Array2<f64>>().unwrap();
        assert_eq!(eager.centers(), Some(&[1.0, 3.0][..]));
        let result = eager.matvec(&array![2.0, -1.0]).unwrap();
        assert!((result[0] - 0.5).abs() < 1e-12);
        let mut destination = <ndarray::Array2<f64> as lazymatrix::MatrixOwned<f64>>::zeros(3, 2);
        let borrowed = lazy.to_eager_into(&mut destination).unwrap();
        let result = borrowed.matvec(&array![2.0, -1.0]).unwrap();
        assert!((result[2] + 1.5).abs() < 1e-12);
        let eager = LazyMatrix::from_normalization(dense.clone(), eager.normalization().clone())
            .into_eager();
        assert!((eager.matvec(&array![2.0, -1.0]).unwrap()[1] - 1.0).abs() < 1e-12);
    }
    oracle::check(
        dense.view(),
        array![2.0, -1.0],
        array![1.0, 2.0, -1.0],
        Array1::zeros(3),
        Array1::zeros(2),
    );
    oracle::check(
        dense,
        array![2.0, -1.0],
        array![1.0, 2.0, -1.0],
        Array1::zeros(3),
        Array1::zeros(2),
    );
}
