#[path = "oracle.rs"]
mod oracle;

/// Exercise this consumer's backend types after Cargo unifies features.
pub fn check() {
    use nalgebra::{DMatrix, DMatrixView, DVector};
    use nalgebra_sparse::{CscMatrix, CsrMatrix};
    let dense = DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 2.0, 3.0, 0.0, 6.0]);
    let sparse = CscMatrix::try_from_csc_data(
        3,
        2,
        vec![0, 2, 4],
        vec![0, 1, 1, 2],
        vec![1.0, 2.0, 3.0, 6.0],
    )
    .unwrap();
    let input = DVector::from_column_slice(&[2.0, -1.0]);
    let rows = DVector::from_column_slice(&[1.0, 2.0, -1.0]);
    let view: DMatrixView<'_, f64> = dense.as_view();
    {
        use lazymatrix::{Centering, LazyMatrix, MatVec, Normalization, Scaling};
        let lazy =
            LazyMatrix::new(&dense, Normalization::new(Centering::Mean, Scaling::Range)).unwrap();
        let eager = lazy.to_eager::<DMatrix<f64>>().unwrap();
        assert_eq!(eager.centers(), Some(&[1.0, 3.0][..]));
        let result = eager
            .matvec(&DVector::from_column_slice(&[2.0, -1.0]))
            .unwrap();
        assert!((result[0] - 0.5).abs() < 1e-12);
        let mut destination = <DMatrix<f64> as lazymatrix::MatrixOwned<f64>>::zeros(3, 2);
        let borrowed = lazy.to_eager_into(&mut destination).unwrap();
        let result = borrowed
            .matvec(&DVector::from_column_slice(&[2.0, -1.0]))
            .unwrap();
        assert!((result[2] + 1.5).abs() < 1e-12);
        let eager = LazyMatrix::from_normalization(dense.clone(), eager.normalization().clone())
            .into_eager();
        assert!(
            (eager
                .matvec(&DVector::from_column_slice(&[2.0, -1.0]))
                .unwrap()[1]
                - 1.0)
                .abs()
                < 1e-12
        );
    }
    oracle::check(
        view,
        input.clone(),
        rows.clone(),
        DVector::zeros(3),
        DVector::zeros(2),
    );
    oracle::check(
        sparse,
        input.clone(),
        rows.clone(),
        DVector::zeros(3),
        DVector::zeros(2),
    );
    let csr = CsrMatrix::try_from_csr_data(
        3,
        2,
        vec![0, 1, 3, 4],
        vec![0, 0, 1, 1],
        vec![1.0, 2.0, 3.0, 6.0],
    )
    .unwrap();
    oracle::check(csr, input, rows, DVector::zeros(3), DVector::zeros(2));
}
