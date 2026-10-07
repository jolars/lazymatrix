use lazymatrix::{
    Centering, ColumnStats, LazyMatrix, MatTransposeVec, MatTransposeVecInto,
    MatTransposeVecScaledInto, MatVec, MatVecInto, MatVecScaledInto, MatrixShape, Normalization,
    Scaling, VectorOwned, VectorView, VectorViewMut, WithIntercept,
};

pub fn check<M, V>(matrix: M, input: V, rows: V, mut out: V, mut transpose_out: V)
where
    M: MatrixShape + ColumnStats<f64>,
    LazyMatrix<M, f64>: MatVec<V>
        + MatTransposeVec<V>
        + MatVecInto<V>
        + MatTransposeVecInto<V>
        + MatVecScaledInto<V, V, f64>
        + MatTransposeVecScaledInto<V, V, f64>,
    V: VectorView<f64> + VectorViewMut<f64> + VectorOwned<f64, Owned = V>,
{
    let lazy =
        LazyMatrix::new(matrix, Normalization::new(Centering::Mean, Scaling::Range)).unwrap();
    assert_eq!(lazy.centers().unwrap(), &[1.0, 3.0]);
    assert_eq!(lazy.scales().unwrap(), &[2.0, 6.0]);
    close(&lazy.matvec(&input).unwrap(), &[0.5, 1.0, -1.5]);
    close(&lazy.mat_transpose_vec(&rows).unwrap(), &[1.5, -1.0]);
    lazy.matvec_into(&input, &mut out).unwrap();
    lazy.mat_transpose_vec_into(&rows, &mut transpose_out)
        .unwrap();
    close(&out, &[0.5, 1.0, -1.5]);
    close(&transpose_out, &[1.5, -1.0]);
    let design = WithIntercept::new(&lazy);
    let coefficients = V::owned_from_fn(3, |i| [3.0, 2.0, -1.0][i]);
    close(&design.matvec(&coefficients).unwrap(), &[3.5, 4.0, 1.5]);
    close(&design.mat_transpose_vec(&rows).unwrap(), &[2.0, 1.5, -1.0]);
    design.matvec_into(&coefficients, &mut out).unwrap();
    let mut augmented_transpose = V::owned_from_fn(3, |_| f64::NAN);
    design
        .mat_transpose_vec_into(&rows, &mut augmented_transpose)
        .unwrap();
    close(&out, &[3.5, 4.0, 1.5]);
    close(&augmented_transpose, &[2.0, 1.5, -1.0]);
    out = V::owned_from_fn(3, |_| 3.0);
    transpose_out = V::owned_from_fn(2, |_| 3.0);
    lazy.matvec_scaled_into(2.0, &input, -1.0, &mut out)
        .unwrap();
    lazy.mat_transpose_vec_scaled_into(2.0, &rows, -1.0, &mut transpose_out)
        .unwrap();
    close(&out, &[-2.0, -1.0, -6.0]);
    close(&transpose_out, &[0.0, -5.0]);
    let mut scratch = V::owned_from_fn(2, |_| f64::NAN);
    design
        .matvec_scaled_with_workspace(2.0, &coefficients, 0.0, &mut out, &mut scratch)
        .unwrap();
    design
        .mat_transpose_vec_scaled_with_workspace(
            2.0,
            &rows,
            0.0,
            &mut augmented_transpose,
            &mut scratch,
        )
        .unwrap();
    close(&out, &[7.0, 8.0, 3.0]);
    close(&augmented_transpose, &[4.0, 3.0, -2.0]);
}

fn close(vector: &impl VectorView<f64>, expected: &[f64]) {
    assert_eq!(vector.len(), expected.len());
    for (i, &value) in expected.iter().enumerate() {
        assert!((vector.get(i) - value).abs() < 1e-12);
    }
}
