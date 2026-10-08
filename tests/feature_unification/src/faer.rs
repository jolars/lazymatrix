#[path = "oracle.rs"]
mod oracle;

/// Exercise this consumer's backend types after Cargo unifies features.
pub fn check() {
    use faer::sparse::{SparseColMat, Triplet};
    use faer::{Col, Mat};
    let values = [[1.0, 0.0], [2.0, 3.0], [0.0, 6.0]];
    let dense = Mat::from_fn(3, 2, |i, j| values[i][j]);
    let sparse = SparseColMat::try_new_from_triplets(
        3,
        2,
        &[
            Triplet::new(0, 0, 1.0),
            Triplet::new(1, 0, 2.0),
            Triplet::new(1, 1, 3.0),
            Triplet::new(2, 1, 6.0),
        ],
    )
    .unwrap();
    let input = Col::from_fn(2, |i| [2.0, -1.0][i]);
    let rows = Col::from_fn(3, |i| [1.0, 2.0, -1.0][i]);
    {
        use lazymatrix::{Centering, LazyMatrix, MatVec, Normalization, Scaling};
        let lazy =
            LazyMatrix::new(&dense, Normalization::new(Centering::Mean, Scaling::Range)).unwrap();
        let eager = lazy.to_eager::<Mat<f64>>().unwrap();
        assert_eq!(eager.centers(), Some(&[1.0, 3.0][..]));
        let result = eager.matvec(&Col::from_fn(2, |i| [2.0, -1.0][i])).unwrap();
        assert!((result[0] - 0.5).abs() < 1e-12);
        let mut destination = <Mat<f64> as lazymatrix::MatrixOwned<f64>>::zeros(3, 2);
        let borrowed = lazy.to_eager_into(&mut destination).unwrap();
        let result = borrowed
            .matvec(&Col::from_fn(2, |i| [2.0, -1.0][i]))
            .unwrap();
        assert!((result[2] + 1.5).abs() < 1e-12);
        let eager = LazyMatrix::from_normalization(dense.clone(), eager.normalization().clone())
            .into_eager();
        assert!(
            (eager.matvec(&Col::from_fn(2, |i| [2.0, -1.0][i])).unwrap()[1] - 1.0).abs() < 1e-12
        );
    }
    oracle::check(
        dense.as_ref(),
        input.clone(),
        rows.clone(),
        Col::zeros(3),
        Col::zeros(2),
    );
    oracle::check(sparse, input, rows, Col::zeros(3), Col::zeros(2));
}
