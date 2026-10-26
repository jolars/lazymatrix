#[path = "common/backend_aliases.rs"]
mod backend_aliases;

#[cfg(feature = "ndarray_all")]
#[test]
fn selection_inherits_fitted_parameters_and_nests_locally() {
    use backend_aliases::ndarray::array;
    use lazymatrix::{LazyMatrix, MatTransposeVec, MatVec};
    let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let lazy = LazyMatrix::from_parts(data, Some(vec![1.0, 2.0, 3.0]), Some(vec![2.0; 3]));
    let view = lazy.submatrix(1..3, 1..3);
    assert_eq!(view.centers(), Some([2.0, 3.0].as_slice()));
    assert_eq!(
        view.centers().unwrap().as_ptr(),
        lazy.centers().unwrap()[1..].as_ptr()
    );
    assert_eq!(view.matvec(&vec![1.0, 2.0]).unwrap(), vec![4.5, 9.0]);
    assert_eq!(
        view.mat_transpose_vec(&vec![1.0, 2.0]).unwrap(),
        vec![7.5, 7.5]
    );
    let nested = view.submatrix(1..2, 0..1);
    assert_eq!(nested.matvec(&vec![2.0]).unwrap(), vec![6.0]);
}

use lazymatrix::{
    LazyMatrix, MatrixErrorType, MatrixShape, MatrixWrite, SubmatrixKernel, VectorView,
    VectorViewMut,
};
use std::cell::Cell;
use std::ops::Range;

struct FailingSource {
    calls: Cell<usize>,
}
impl MatrixShape for FailingSource {
    fn nrows(&self) -> usize {
        4
    }
    fn ncols(&self) -> usize {
        3
    }
}
impl MatrixErrorType for FailingSource {
    type Error = std::io::Error;
}
impl SubmatrixKernel<f64> for FailingSource {
    fn submatrix_matvec_into<X: VectorView<f64> + ?Sized, Y: VectorViewMut<f64> + ?Sized>(
        &self,
        _: Range<usize>,
        _: Range<usize>,
        _: Option<&[f64]>,
        _: Option<&[f64]>,
        _: f64,
        _: &X,
        _: f64,
        out: &mut Y,
    ) -> Result<(), Self::Error> {
        self.calls.set(self.calls.get() + 1);
        out.set(0, 42.0);
        Err(std::io::Error::other("read failed"))
    }
    fn submatrix_transpose_into<X: VectorView<f64> + ?Sized, Y: VectorViewMut<f64> + ?Sized>(
        &self,
        rows: Range<usize>,
        columns: Range<usize>,
        centers: Option<&[f64]>,
        scales: Option<&[f64]>,
        alpha: f64,
        x: &X,
        beta: f64,
        out: &mut Y,
    ) -> Result<(), Self::Error> {
        self.submatrix_matvec_into(rows, columns, centers, scales, alpha, x, beta, out)
    }
    fn materialize_submatrix_into<O: MatrixWrite<f64> + ?Sized>(
        &self,
        _: Range<usize>,
        _: Range<usize>,
        _: Option<&[f64]>,
        _: Option<&[f64]>,
        out: &mut O,
    ) -> Result<(), Self::Error> {
        self.calls.set(self.calls.get() + 1);
        out.set(0, 0, 42.0);
        Err(std::io::Error::other("read failed"))
    }
}

#[test]
fn invalid_ranges_and_zero_alpha_never_evaluate_the_source() {
    use lazymatrix::{MatTransposeVecScaledInto, MatVecScaledInto};
    let source = FailingSource {
        calls: Cell::new(0),
    };
    let lazy = LazyMatrix::from_parts(source, Some(vec![f64::NAN; 3]), Some(vec![-2.0; 3]));
    let view = lazy.submatrix(1..3, 1..3);
    let mut out = vec![f64::NAN; 2];
    view.matvec_scaled_into(0.0, &vec![f64::NAN; 2], 0.0, &mut out)
        .unwrap();
    assert_eq!(out, vec![0.0; 2]);
    out.fill(3.0);
    view.mat_transpose_vec_scaled_into(0.0, &vec![f64::NAN; 2], -2.0, &mut out)
        .unwrap();
    assert_eq!(out, vec![-6.0; 2]);
    for (rows, cols) in [
        (Range { start: 2, end: 1 }, 0..1),
        (0..5, 0..1),
        (0..1, 0..4),
        (0..0, 4..4),
    ] {
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| lazy.submatrix(rows, cols)))
                .is_err()
        );
    }
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| view.submatrix(0..3, 0..1)))
            .is_err()
    );
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| view.matvec_scaled_into(
            0.0,
            &vec![0.0; 3],
            0.0,
            &mut out
        )))
        .is_err()
    );
    let mut wrong_output = vec![99.0];
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || view.mat_transpose_vec_scaled_into(0.0, &vec![0.0; 2], 0.0, &mut wrong_output)
        ))
        .is_err()
    );
    assert_eq!(wrong_output, vec![99.0]);
    assert_eq!(lazy.data().calls.get(), 0);
}

#[test]
fn product_errors_preserve_source_type_without_postprocessing() {
    use lazymatrix::{MatTransposeVecInto, MatVecInto};
    let lazy = LazyMatrix::from_parts(
        FailingSource {
            calls: Cell::new(0),
        },
        Some(vec![1.0; 3]),
        Some(vec![2.0; 3]),
    );
    let view = lazy.submatrix(1..3, 1..3);
    let mut out = vec![99.0; 2];
    let error: std::io::Error = view.matvec_into(&vec![1.0; 2], &mut out).unwrap_err();
    assert_eq!(error.to_string(), "read failed");
    assert_eq!(out, vec![42.0, 99.0]);
    out.fill(99.0);
    view.mat_transpose_vec_into(&vec![1.0; 2], &mut out)
        .unwrap_err();
    assert_eq!(out, vec![42.0, 99.0]);
    assert_eq!(lazy.data().calls.get(), 2);
}

#[cfg(feature = "ndarray_all")]
#[test]
fn selected_dense_entries_exclude_nonfinite_values_and_preserve_small_variations() {
    use backend_aliases::ndarray::{Array2, array, s};
    use lazymatrix::{MatTransposeVec, MatVec, MatVecInto};
    let data = array![
        [f64::NAN, f64::INFINITY, f64::NAN],
        [f64::NAN, 1e12 + 0.25, 1e12 + 0.5],
        [f64::NAN, 1e12 + 0.75, 1e12 + 1.0]
    ];
    let lazy = LazyMatrix::from_parts(
        data.slice(s![..;-1, ..]),
        Some(vec![f64::NAN, 1e12, 1e12]),
        None,
    );
    let view = lazy.submatrix(0..2, 1..3);
    assert_eq!(view.matvec(&vec![1.0, 2.0]).unwrap(), vec![2.75, 1.25]);
    assert_eq!(
        view.mat_transpose_vec(&vec![1.0, 2.0]).unwrap(),
        vec![1.25, 2.0]
    );
    let x = array![99.0, 2.0, 99.0, 1.0];
    let mut out = array![99.0, f64::NAN, 99.0, f64::NAN];
    view.matvec_into(&x.slice(s![..;-2]), &mut out.slice_mut(s![1..;2]))
        .unwrap();
    assert_eq!(out.to_vec(), vec![99.0, 2.75, 99.0, 1.25]);
    let eager = view.to_eager::<Array2<f64>>().unwrap();
    assert_eq!(eager.data(), &array![[0.75, 1.0], [0.25, 0.5]]);
    assert_eq!(eager.matvec(&array![1.0, 2.0]).unwrap(), array![2.75, 1.25]);
}

struct Output {
    rows: usize,
    columns: usize,
    values: Vec<f64>,
}
impl MatrixShape for Output {
    fn nrows(&self) -> usize {
        self.rows
    }
    fn ncols(&self) -> usize {
        self.columns
    }
}
impl MatrixWrite<f64> for Output {
    fn set(&mut self, i: usize, j: usize, value: f64) {
        self.values[i * self.columns + j] = value;
    }
}

#[test]
fn materialization_validates_output_before_source_reads_and_preserves_errors() {
    let lazy = LazyMatrix::from_parts(
        FailingSource {
            calls: Cell::new(0),
        },
        Some(vec![1.0; 3]),
        None,
    );
    let view = lazy.submatrix(1..3, 1..3).clone();
    let mut out = Output {
        rows: 1,
        columns: 2,
        values: vec![99.0; 2],
    };
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = view.to_eager_into(&mut out);
        }))
        .is_err()
    );
    assert_eq!(lazy.data().calls.get(), 0);
    assert_eq!(out.values, vec![99.0; 2]);
    let mut out = Output {
        rows: 2,
        columns: 2,
        values: vec![99.0; 4],
    };
    let result = view.to_eager_into(&mut out);
    assert!(matches!(result, Err(error) if error.to_string() == "read failed"));
    assert_eq!(out.values, vec![42.0, 99.0, 99.0, 99.0]);
}

#[cfg(feature = "sprs_all")]
#[test]
fn sparse_submatrices_support_alternate_indices_and_exclude_nonfinite_entries() {
    use backend_aliases::sprs::{CsMatI, TriMatI};
    use lazymatrix::{MatTransposeVec, MatVec, SprsCsc, SprsCsr};
    let mut triplets = TriMatI::<_, u32>::new((4, 3));
    for (i, j, v) in [
        (0, 1, f64::NAN),
        (1, 0, f64::NAN),
        (1, 1, 4.0),
        (2, 2, 6.0),
        (3, 2, f64::INFINITY),
    ] {
        triplets.add_triplet(i, j, v);
    }
    let csc: CsMatI<_, u32, u64> = triplets.to_csc();
    let csr = csc.to_csr();
    let wrapped_csc = SprsCsc::try_new(csc.view()).unwrap();
    let wrapped_csr = SprsCsr::try_new(csr.view()).unwrap();
    fn check<M: SubmatrixKernel<f64>>(matrix: M) {
        let lazy = LazyMatrix::from_parts(
            matrix,
            Some(vec![f64::NAN, 1.0, 2.0]),
            Some(vec![f64::NAN, 2.0, -2.0]),
        );
        let view = lazy.submatrix(1..3, 1..3);
        assert_eq!(view.matvec(&vec![1.0, 2.0]).unwrap(), vec![3.5, -4.5]);
        assert_eq!(
            view.mat_transpose_vec(&vec![1.0, 2.0]).unwrap(),
            vec![0.5, -3.0]
        );
        let nested = lazy.submatrix(0..4, 0..3).submatrix(1..3, 1..3);
        assert_eq!(nested.matvec(&vec![1.0, 2.0]).unwrap(), vec![3.5, -4.5]);
    }
    check(&csc);
    check(&csr);
    check(wrapped_csc);
    check(wrapped_csr);
}
