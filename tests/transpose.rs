use std::convert::Infallible;

use lazymatrix::{
    MatTransposeVec, MatTransposeVecInto, MatVec, MatVecInto, MatrixErrorType, MatrixShape,
    Transpose,
};

struct TransposeOnly;

impl MatrixShape for TransposeOnly {
    fn nrows(&self) -> usize {
        3
    }

    fn ncols(&self) -> usize {
        2
    }
}

impl MatrixErrorType for TransposeOnly {
    type Error = Infallible;
}

impl MatTransposeVec<Vec<f64>> for TransposeOnly {
    fn mat_transpose_vec(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        let mut out = [0.0; 2];
        self.mat_transpose_vec_into(&[x[0], x[1], x[2]], &mut out)?;
        Ok(out.to_vec())
    }
}

impl MatTransposeVecInto<[f64; 3], [f64; 2]> for TransposeOnly {
    fn mat_transpose_vec_into(&self, x: &[f64; 3], out: &mut [f64; 2]) -> Result<(), Infallible> {
        out[0] = x.iter().sum();
        out[1] = x[0] + 2.0 * x[1] + 3.0 * x[2];
        Ok(())
    }
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn transpose_needs_only_the_opposite_product_and_preserves_vector_types() {
    let operator = TransposeOnly;
    let view = Transpose::new(&operator);
    assert_eq!(view.matvec(&vec![1.0, -2.0, 3.0]).unwrap(), [2.0, 6.0]);
    let mut out = [f64::NAN; 2];
    view.matvec_into(&[1.0, -2.0, 3.0], &mut out).unwrap();
    assert_eq!(out, [2.0, 6.0]);
    let double = Transpose::new(&view);
    double
        .mat_transpose_vec_into(&[1.0, -2.0, 3.0], &mut out)
        .unwrap();
    assert_eq!(out, [2.0, 6.0]);
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn views_copy_non_clone_operators_and_borrow_unsized_operators() {
    fn copy_view<T: Copy>(view: T) -> (T, T) {
        (view, view)
    }

    let operator = TransposeOnly;
    let (view, copy) = copy_view(Transpose::new(&operator));
    assert!(std::ptr::eq(view.as_inner(), copy.as_inner()));
    assert_eq!(
        std::mem::size_of_val(&view),
        std::mem::size_of::<&TransposeOnly>()
    );
    let shape: &dyn MatrixShape = &operator;
    let view = Transpose::new(shape);
    assert_eq!((view.nrows(), view.ncols()), (2, 3));
    assert!(std::ptr::eq(view.as_inner(), shape));
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn construction_does_not_query_the_operator() {
    struct Unreadable;
    impl MatrixShape for Unreadable {
        fn nrows(&self) -> usize {
            panic!("construction must not read dimensions");
        }
        fn ncols(&self) -> usize {
            panic!("construction must not read dimensions");
        }
    }
    let operator = Unreadable;
    let view = Transpose::new(&operator);
    assert!(std::ptr::eq(view.as_inner(), &operator));
}
