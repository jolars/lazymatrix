use lazymatrix::{
    Centering, ColumnNormalization, ColumnStats, LazyMatrix, MatTransposeVec, MatTransposeVecInto,
    MatTransposeVecScaledInto, MatVec, MatVecInto, MatVecScaledInto, MatrixShape, Scaling,
    VectorOwned, VectorView, VectorViewMut, WithIntercept,
};

pub fn check<M, V>(matrix: M, input: V, rows: V, mut out: V, mut transpose_out: V)
where
    M: MatrixShape
        + ColumnStats<f64>
        + lazymatrix::MatMatInto<f64>
        + lazymatrix::MatTransposeMatInto<f64>,
    LazyMatrix<M>: MatVec<V>
        + MatTransposeVec<V>
        + MatVecInto<V>
        + MatTransposeVecInto<V>
        + MatVecScaledInto<V, V, f64>
        + MatTransposeVecScaledInto<V, V, f64>,
    V: VectorView<f64> + VectorViewMut<f64> + VectorOwned<f64, Owned = V>,
{
    let lazy = LazyMatrix::new(
        matrix,
        ColumnNormalization::new(Centering::Mean, Scaling::Range),
    )
    .unwrap();
    assert_eq!(lazy.normalization().centers().unwrap(), &[1.0, 3.0]);
    assert_eq!(lazy.normalization().scales().unwrap(), &[2.0, 6.0]);
    {
        use lazymatrix::{DenseBlock, MatMat, MatTransposeMat};
        let rhs = DenseBlock::new(&[2., 4., -1., -2.], 2, 2);
        let actual: DenseOutput = lazy.matmat(&rhs).unwrap();
        assert_eq!(actual.values, [0.5, 1., 1., 2., -1.5, -3.]);
        let u = DenseBlock::new(&[1., 2., 2., 4., -1., -2.], 3, 2);
        let actual: DenseOutput = lazy.mat_transpose_mat(&u).unwrap();
        assert_eq!(actual.values, [1.5, 3., -1., -2.]);
    }
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

struct DenseOutput {
    rows: usize,
    cols: usize,
    values: Vec<f64>,
}
impl MatrixShape for DenseOutput {
    fn nrows(&self) -> usize {
        self.rows
    }
    fn ncols(&self) -> usize {
        self.cols
    }
}
impl lazymatrix::MatrixRead<f64> for DenseOutput {
    fn get(&self, row: usize, column: usize) -> f64 {
        self.values[row * self.cols + column]
    }
}
impl lazymatrix::MatrixWrite<f64> for DenseOutput {
    fn set(&mut self, row: usize, column: usize, value: f64) {
        self.values[row * self.cols + column] = value;
    }
}
impl lazymatrix::MatrixOwned<f64> for DenseOutput {
    fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            values: vec![0.; rows * cols],
        }
    }
}

fn close(vector: &impl VectorView<f64>, expected: &[f64]) {
    assert_eq!(vector.len(), expected.len());
    for (i, &value) in expected.iter().enumerate() {
        assert!((vector.get(i) - value).abs() < 1e-12);
    }
}
