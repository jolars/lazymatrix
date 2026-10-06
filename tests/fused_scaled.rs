#[path = "common/backend_aliases.rs"]
mod backend_aliases;

#[test]
fn normalization_does_not_correct_partial_outputs_after_errors() {
    use lazymatrix::{
        LazyMatrix, MatTransposeVecScaledInto, MatVecScaledInto, MatrixErrorType, MatrixShape,
    };

    struct Failing;
    impl MatrixShape for Failing {
        fn nrows(&self) -> usize {
            2
        }
        fn ncols(&self) -> usize {
            1
        }
    }
    impl MatrixErrorType for Failing {
        type Error = std::io::Error;
    }
    impl MatVecScaledInto<Vec<f64>, Vec<f64>, f64> for Failing {
        fn matvec_scaled_into(
            &self,
            _: f64,
            _: &Vec<f64>,
            _: f64,
            out: &mut Vec<f64>,
        ) -> Result<(), Self::Error> {
            out[0] = 5.0;
            Err(std::io::Error::other("read failed"))
        }
    }
    impl MatTransposeVecScaledInto<Vec<f64>, Vec<f64>, f64> for Failing {
        fn mat_transpose_vec_scaled_into(
            &self,
            _: f64,
            _: &Vec<f64>,
            _: f64,
            out: &mut Vec<f64>,
        ) -> Result<(), Self::Error> {
            out[0] = 5.0;
            Err(std::io::Error::other("read failed"))
        }
    }

    let lazy = LazyMatrix::from_parts(Failing, Some(vec![1.0]), Some(vec![2.0]));
    let mut scratch = vec![0.0];
    let mut out = vec![11.0, 13.0];
    assert!(
        lazy.matvec_scaled_with_workspace(2.0, &vec![2.0], 1.0, &mut out, &mut scratch)
            .is_err()
    );
    assert_eq!(out, vec![5.0, 13.0]);

    let mut out = vec![11.0];
    assert!(
        lazy.mat_transpose_vec_scaled_with_workspace(
            2.0,
            &vec![1.0, 1.0],
            1.0,
            &mut out,
            &mut scratch
        )
        .is_err()
    );
    assert_eq!(out, vec![11.0]);
}

#[cfg(feature = "ndarray_all")]
macro_rules! ndarray_suite {
    ($name:ident, $backend:ident) => {
    mod $name {
    use crate::backend_aliases::$backend as ndarray;
    use lazymatrix::{LazyMatrix, MatTransposeVecScaledInto, MatVecScaledInto};
    use ndarray::{array, s};

    #[test]
    fn fused_products_handle_strides_zero_coefficients_and_normalization() {
        let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let x = array![2.0, 99.0, -1.0];
        let mut storage = array![10.0, 99.0, 20.0, 99.0, 30.0, 99.0];
        a.matvec_scaled_into(
            2.0,
            &x.slice(s![..;2]),
            -1.0,
            &mut storage.slice_mut(s![..;2]),
        )
        .unwrap();
        assert_eq!(
            storage.to_vec(),
            vec![-10.0, 99.0, -16.0, 99.0, -22.0, 99.0]
        );

        let mut out = array![f64::NAN, f64::NAN, f64::NAN];
        a.matvec_scaled_into(1.0, &array![2.0, -1.0], 0.0, &mut out)
            .unwrap();
        assert_eq!(out, array![0.0, 2.0, 4.0]);
        a.matvec_scaled_into(0.0, &array![f64::NAN, f64::NAN], 1.0, &mut out)
            .unwrap();
        assert_eq!(out, array![0.0, 2.0, 4.0]);
        a.matvec_scaled_into(0.0, &array![f64::NAN, f64::NAN], 0.0, &mut out)
            .unwrap();
        assert_eq!(out, array![0.0, 0.0, 0.0]);
        a.matvec_scaled_into(-1.0, &array![2.0, -1.0], 1.0, &mut out)
            .unwrap();
        assert_eq!(out, array![0.0, -2.0, -4.0]);

        let mut transpose = array![f64::NAN, f64::NAN];
        a.mat_transpose_vec_scaled_into(1.0, &array![1.0, 2.0, 3.0], 0.0, &mut transpose)
            .unwrap();
        assert_eq!(transpose, array![22.0, 28.0]);
        a.mat_transpose_vec_scaled_into(
            0.0,
            &array![f64::NAN, f64::NAN, f64::NAN],
            0.0,
            &mut transpose,
        )
        .unwrap();
        assert_eq!(transpose, array![0.0, 0.0]);

        let mut nonfinite = array![1.0];
        array![[2.0]]
            .matvec_scaled_into(f64::INFINITY, &array![1.0], 0.0, &mut nonfinite)
            .unwrap();
        assert!(nonfinite[0].is_infinite());
        array![[2.0]]
            .matvec_scaled_into(1.0, &array![1.0], f64::NAN, &mut nonfinite)
            .unwrap();
        assert!(nonfinite[0].is_nan());

        let lazy = LazyMatrix::from_parts(a, Some(vec![1.0, -1.0]), Some(vec![2.0, -2.0]));
        let mut scratch = array![0.0, 0.0];
        let mut out = array![10.0, 20.0, 30.0];
        lazy.matvec_scaled_with_workspace(2.0, &array![2.0, -2.0], -1.0, &mut out, &mut scratch)
            .unwrap();
        assert_eq!(scratch, array![1.0, 1.0]);
        assert_eq!(out, array![-4.0, -6.0, -8.0]);
        out.fill(10.0);
        let lazy_input = array![2.0, 99.0, -2.0];
        lazy.matvec_scaled_into(2.0, &lazy_input.slice(s![..;2]), -1.0, &mut out)
            .unwrap();
        assert_eq!(out, array![-4.0, 4.0, 12.0]);

        let mut transpose = array![7.0, 11.0];
        lazy.mat_transpose_vec_scaled_with_workspace(
            2.0,
            &array![1.0, 2.0, 3.0],
            -1.0,
            &mut transpose,
            &mut scratch,
        )
        .unwrap();
        assert_eq!(transpose, array![9.0, -45.0]);
        transpose.fill(f64::NAN);
        lazy.mat_transpose_vec_scaled_into(1.0, &array![1.0, 2.0, 3.0], 0.0, &mut transpose)
            .unwrap();
        assert_eq!(transpose, array![8.0, -17.0]);

        let empty_rows = ndarray::Array2::<f64>::zeros((0, 2));
        let mut empty = ndarray::Array1::<f64>::zeros(0);
        empty_rows
            .matvec_scaled_into(-1.0, &array![1.0, 2.0], 1.0, &mut empty)
            .unwrap();
        let mut columns = array![f64::NAN, f64::NAN];
        empty_rows
            .mat_transpose_vec_scaled_into(1.0, &empty, 0.0, &mut columns)
            .unwrap();
        assert_eq!(columns, array![0.0, 0.0]);

        let empty_columns = ndarray::Array2::<f64>::zeros((3, 0));
        let mut rows = array![f64::NAN, f64::NAN, f64::NAN];
        empty_columns
            .matvec_scaled_into(1.0, &empty, 0.0, &mut rows)
            .unwrap();
        assert_eq!(rows, array![0.0, 0.0, 0.0]);
        empty_columns
            .mat_transpose_vec_scaled_into(1.0, &rows, 0.0, &mut empty)
            .unwrap();
    }
}
    };
}

#[cfg(feature = "ndarray_v0_15")]
ndarray_suite!(ndarray_0_15_tests, ndarray_0_15);
#[cfg(feature = "ndarray_v0_16")]
ndarray_suite!(ndarray_0_16_tests, ndarray_0_16);
#[cfg(feature = "ndarray_v0_17")]
ndarray_suite!(ndarray_0_17_tests, ndarray_0_17);

#[cfg(feature = "sprs_all")]
mod sprs_tests {
    use crate::backend_aliases::sprs;
    use lazymatrix::{LazyMatrix, MatTransposeVecScaledInto, MatVecScaledInto};
    use sprs::CsMat;

    #[test]
    fn fused_products_work_for_csc_csr_and_lazy_normalization() {
        let csc = CsMat::new_csc(
            (3, 2),
            vec![0, 2, 4],
            vec![0, 2, 1, 2],
            vec![1.0, 5.0, 4.0, 6.0],
        );
        for a in [csc.clone(), csc.to_csr()] {
            let mut out = vec![10.0, 20.0, 30.0];
            a.matvec_scaled_into(2.0, &vec![2.0, -1.0], -1.0, &mut out)
                .unwrap();
            assert_eq!(out, vec![-6.0, -28.0, -22.0]);
            let mut transpose = vec![7.0, 11.0];
            a.mat_transpose_vec_scaled_into(2.0, &vec![1.0, 2.0, 3.0], -1.0, &mut transpose)
                .unwrap();
            assert_eq!(transpose, vec![25.0, 41.0]);
            a.matvec_scaled_into(0.0, &vec![f64::NAN; 2], 0.0, &mut out)
                .unwrap();
            assert_eq!(out, vec![0.0; 3]);
            a.mat_transpose_vec_scaled_into(0.0, &vec![f64::NAN; 3], 0.0, &mut transpose)
                .unwrap();
            assert_eq!(transpose, vec![0.0; 2]);

            let lazy = LazyMatrix::from_parts(a, Some(vec![1.0, -1.0]), Some(vec![2.0, -2.0]));
            let mut scratch = vec![0.0; 2];
            let mut out = vec![10.0, 20.0, 30.0];
            lazy.matvec_scaled_with_workspace(2.0, &vec![2.0, -2.0], -1.0, &mut out, &mut scratch)
                .unwrap();
            assert_eq!(out, vec![-8.0, -12.0, -8.0]);
            out.fill(f64::NAN);
            lazy.matvec_scaled_into(1.0, &vec![2.0, -2.0], 0.0, &mut out)
                .unwrap();
            assert_eq!(out, vec![1.0, 4.0, 11.0]);
            let mut transpose = vec![7.0, 11.0];
            lazy.mat_transpose_vec_scaled_with_workspace(
                2.0,
                &vec![1.0, 2.0, 3.0],
                -1.0,
                &mut transpose,
                &mut scratch,
            )
            .unwrap();
            assert_eq!(transpose, vec![3.0, -43.0]);
            transpose.fill(f64::NAN);
            lazy.mat_transpose_vec_scaled_into(1.0, &vec![1.0, 2.0, 3.0], 0.0, &mut transpose)
                .unwrap();
            assert_eq!(transpose, vec![5.0, -16.0]);
        }

        let empty_rows = CsMat::<f64>::zero((0, 2));
        let mut empty = Vec::<f64>::new();
        empty_rows
            .matvec_scaled_into(1.0, &vec![1.0, 2.0], 0.0, &mut empty)
            .unwrap();
        let mut columns = vec![f64::NAN; 2];
        empty_rows
            .mat_transpose_vec_scaled_into(1.0, &empty, 0.0, &mut columns)
            .unwrap();
        assert_eq!(columns, vec![0.0; 2]);
    }
}
