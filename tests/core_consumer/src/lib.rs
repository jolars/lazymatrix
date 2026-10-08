//! Check the dependency-light core as an external consumer on native and WASM targets.

#![forbid(unsafe_code)]

#[cfg(test)]
mod tests {
    use std::{cell::Cell, convert::Infallible};

    use lazymatrix::{
        DotSlice, ElemDivAssign, LazyMatrix, MatTransposeVecInto, MatVecInto, MatrixErrorType,
        MatrixShape, Scalar, ScaledSubSlice, SubScalarAssign, SumEntries,
    };

    struct Matrix<F> {
        values: [[F; 2]; 3],
        products: Cell<usize>,
    }

    impl<F> MatrixShape for Matrix<F> {
        fn nrows(&self) -> usize {
            3
        }
        fn ncols(&self) -> usize {
            2
        }
    }

    impl<F> MatrixErrorType for Matrix<F> {
        type Error = Infallible;
    }

    impl<F: Scalar> MatVecInto<Vec<F>> for Matrix<F> {
        fn matvec_into(&self, _: &Vec<F>, _: &mut Vec<F>) -> Result<(), Infallible> {
            panic!("normalized products must use the backend hook");
        }

        fn matvec_normalized_into<S: Scalar>(
            &self,
            x: &Vec<F>,
            centers: Option<&[S]>,
            scales: Option<&[S]>,
            out: &mut Vec<F>,
        ) -> Result<(), Infallible>
        where
            Vec<F>: ElemDivAssign<S> + DotSlice<S> + SubScalarAssign<S>,
        {
            assert_eq!(x.len(), 2);
            assert_eq!(out.len(), 3);
            self.products.set(self.products.get() + 1);
            for (row, result) in self.values.iter().zip(out) {
                *result = row
                    .iter()
                    .zip(x)
                    .enumerate()
                    .map(|(j, (&raw, &v))| {
                        let c = centers.map_or(F::zero(), |c| F::from(c[j]).unwrap());
                        let s = scales.map_or(F::one(), |s| F::from(s[j]).unwrap());
                        ((raw - c) / s) * v
                    })
                    .sum();
            }
            Ok(())
        }
    }

    impl<F: Scalar> MatTransposeVecInto<Vec<F>> for Matrix<F> {
        fn mat_transpose_vec_into(&self, _: &Vec<F>, _: &mut Vec<F>) -> Result<(), Infallible> {
            panic!("normalized transpose products must use the backend hook");
        }

        fn mat_transpose_vec_normalized_into<S: Scalar>(
            &self,
            x: &Vec<F>,
            centers: Option<&[S]>,
            scales: Option<&[S]>,
            out: &mut Vec<F>,
        ) -> Result<(), Infallible>
        where
            Vec<F>: SumEntries<S> + ScaledSubSlice<S> + ElemDivAssign<S>,
        {
            assert_eq!(x.len(), 3);
            assert_eq!(out.len(), 2);
            self.products.set(self.products.get() + 1);
            for (j, result) in out.iter_mut().enumerate() {
                let c = centers.map_or(F::zero(), |c| F::from(c[j]).unwrap());
                let s = scales.map_or(F::one(), |s| F::from(s[j]).unwrap());
                *result = self
                    .values
                    .iter()
                    .zip(x)
                    .map(|(row, &v)| ((row[j] - c) / s) * v)
                    .sum();
            }
            Ok(())
        }
    }

    fn check<F: Scalar>() {
        let value = |v| F::from_f64(v).unwrap();
        let mut matrix = Matrix {
            values: [[1., 4.], [2., 6.], [4., 10.]].map(|row| row.map(|v| value(100_000. + v))),
            products: Cell::new(0),
        };
        let centers = vec![value(100_001.), value(100_002.)];
        let scales = vec![value(2.), value(4.)];
        let mut out = vec![F::nan(); 3];
        let mut transpose_out = vec![F::nan(); 2];
        let lazy = LazyMatrix::from_parts(&matrix, Some(centers.clone()), Some(scales.clone()));
        lazy.matvec_into(&vec![value(2.), value(-1.)], &mut out)
            .unwrap();
        lazy.mat_transpose_vec_into(&vec![value(1.), value(-2.), value(3.)], &mut transpose_out)
            .unwrap();
        assert_eq!(out, vec![value(-0.5), F::zero(), F::one()]);
        assert_eq!(transpose_out, vec![value(3.5), value(4.5)]);
        assert_eq!(matrix.products.get(), 2);

        let lazy = LazyMatrix::from_parts(&mut matrix, Some(centers), Some(scales));
        lazy.matvec_into(&vec![value(2.), value(-1.)], &mut out)
            .unwrap();
        lazy.mat_transpose_vec_into(&vec![value(1.), value(-2.), value(3.)], &mut transpose_out)
            .unwrap();
        assert_eq!(out, vec![value(-0.5), F::zero(), F::one()]);
        assert_eq!(transpose_out, vec![value(3.5), value(4.5)]);
        assert_eq!(matrix.products.get(), 4);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn borrowed_f32_products_use_normalized_hooks() {
        check::<f32>();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn borrowed_f64_products_use_normalized_hooks() {
        check::<f64>();
    }
}
