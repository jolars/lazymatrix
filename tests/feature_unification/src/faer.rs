#[path = "oracle.rs"]
mod oracle;

/// Exercise this consumer's backend types after Cargo unifies features.
pub fn check() {
    use faer::sparse::{SparseColMat, SparseRowMat, Triplet};
    use faer::{Col, Mat};
    let values = [[1.0, 0.0], [2.0, 3.0], [0.0, 6.0]];
    let mut dense = Mat::from_fn(3, 2, |i, j| values[i][j]);
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
        use lazymatrix::{LazyMatrix, WeightedColumnSumsInto, WeightedGramInto, WithIntercept};
        let lazy =
            LazyMatrix::from_parts(dense.as_ref(), Some(vec![1.0, 3.0]), Some(vec![2.0, 3.0]));
        let mut expected = Mat::full(3, 3, f64::NAN);
        WithIntercept::new(&lazy)
            .weighted_gram_into(&rows, &mut expected)
            .unwrap();
        assert_eq!(
            expected,
            Mat::from_fn(3, 3, |i, j| {
                [[2.0, 1.5, -2.0], [1.5, 0.25, 0.5], [-2.0, 0.5, 0.0]][i][j]
            })
        );
        let eager = lazy.to_eager::<Mat<f64>>().unwrap();
        let mut actual = Mat::zeros(3, 3);
        WithIntercept::new(&eager)
            .weighted_gram_into(&rows, &mut actual)
            .unwrap();
        assert_eq!(actual, expected);
        let mut sums = Col::full(2, f64::NAN);
        eager
            .weighted_column_sums_into(&rows.as_ref(), &mut sums.as_mut())
            .unwrap();
        assert_eq!(sums[0], expected[(0, 1)]);
        assert_eq!(sums[1], expected[(0, 2)]);
        let lazy =
            LazyMatrix::from_parts(dense.as_mut(), Some(vec![1.0, 3.0]), Some(vec![2.0, 3.0]));
        WithIntercept::new(&lazy)
            .weighted_gram_into(&rows, &mut actual.as_mut())
            .unwrap();
        assert_eq!(actual, expected);
    }
    {
        use lazymatrix::{Centering, ColumnNormalization, LazyMatrix, MatVec, Scaling};
        let lazy = LazyMatrix::new(
            &dense,
            ColumnNormalization::new(Centering::Mean, Scaling::Range),
        )
        .unwrap();
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
    oracle::check(
        sparse.as_ref(),
        input.clone(),
        rows.clone(),
        Col::zeros(3),
        Col::zeros(2),
    );
    oracle::check(
        sparse,
        input.clone(),
        rows.clone(),
        Col::zeros(3),
        Col::zeros(2),
    );
    let csr = SparseRowMat::try_new_from_triplets(
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
    oracle::check(
        csr.as_ref(),
        input.clone(),
        rows.clone(),
        Col::zeros(3),
        Col::zeros(2),
    );
    oracle::check(csr, input, rows, Col::zeros(3), Col::zeros(2));
}
