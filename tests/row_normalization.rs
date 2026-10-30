//! Row normalization also works for dependency-free dense consumers.

use std::{cell::Cell, convert::Infallible};

use lazymatrix::{
    Centering, LazyMatrix, MatTransposeVec, MatTransposeVecInto, MatVec, MatVecInto,
    MatrixErrorType, MatrixRead, MatrixShape, NormalizationParams, NormalizationRecipe,
    RowNormalization, RowNormalizationParams, Scalar, Scaling,
};

struct Dense<F> {
    nrows: usize,
    ncols: usize,
    values: Vec<F>,
    reads: Cell<usize>,
}

impl<F> Dense<F> {
    fn new(nrows: usize, ncols: usize, values: Vec<F>) -> Self {
        assert_eq!(nrows.checked_mul(ncols), Some(values.len()));
        Self {
            nrows,
            ncols,
            values,
            reads: Cell::new(0),
        }
    }
}

impl<F> MatrixShape for Dense<F> {
    fn nrows(&self) -> usize {
        self.nrows
    }
    fn ncols(&self) -> usize {
        self.ncols
    }
}

impl<F> MatrixErrorType for Dense<F> {
    type Error = Infallible;
}

impl<F: Scalar> MatrixRead<F> for Dense<F> {
    fn get(&self, row: usize, column: usize) -> F {
        self.reads.set(self.reads.get() + 1);
        self.values[row * self.ncols + column]
    }
}

fn fit<M, P, R>(data: M, recipe: R) -> Result<LazyMatrix<M, P>, M::Error>
where
    M: MatrixShape + MatrixErrorType,
    P: NormalizationParams,
    R: NormalizationRecipe<M, P>,
{
    LazyMatrix::new(data, recipe)
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn generic_recipe_selects_row_parameters_without_column_statistics() {
    let data = Dense::new(2, 3, vec![1.0_f32, 3.0, 5.0, 10.0, 14.0, 18.0]);
    let rows = fit(data, RowNormalization::standardize()).unwrap();
    let params: &RowNormalizationParams<f32> = rows.normalization();
    assert_eq!(params.nrows(), 2);
    assert_eq!(params.centers().unwrap(), &[3.0_f32, 14.0]);
    let (data, params) = rows.into_parts();
    let restored: LazyMatrix<Dense<f32>, RowNormalizationParams<f32>> =
        LazyMatrix::from_parts(data, params);
    assert_eq!((restored.nrows(), restored.ncols()), (2, 3));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn computed_rows_are_independent_and_keep_zero_rows() {
    let source = Dense::new(3, 2, vec![1.0, 3.0, 10.0, 14.0, 0.0, 0.0]);
    let normalized = LazyMatrix::new(&source, RowNormalization::standardize()).unwrap();
    assert_eq!(
        normalized.normalization().centers().unwrap(),
        &[2.0, 12.0, 0.0]
    );
    assert_eq!(
        normalized.normalization().scales().unwrap(),
        &[1.0, 2.0, 1.0]
    );
    assert_eq!(normalized.matvec(&vec![1.0, 2.0]).unwrap(), [1.0, 1.0, 0.0]);
    assert_eq!(
        normalized.mat_transpose_vec(&vec![1.0, 2.0, 3.0]).unwrap(),
        [-3.0, 3.0]
    );
    for scale in [Scaling::L1, Scaling::L2, Scaling::MaxAbs] {
        let normalized =
            LazyMatrix::new(&source, RowNormalization::new(Centering::None, scale)).unwrap();
        assert_eq!(normalized.normalization().scales().unwrap()[2], 1.0);
        assert_eq!(normalized.matvec(&vec![1.0, 1.0]).unwrap()[2], 0.0);
    }
    for spec in [RowNormalization::min_max(), RowNormalization::robust()] {
        let normalized = LazyMatrix::new(&source, spec).unwrap();
        assert_eq!(normalized.normalization().scales().unwrap()[2], 1.0);
    }
    let mut source = source;
    let normalized = LazyMatrix::new(&mut source, RowNormalization::default()).unwrap();
    assert!(normalized.normalization().centers().is_none());
    assert!(normalized.normalization().scales().is_none());
    assert_eq!(
        normalized.matvec(&vec![1.0, 1.0]).unwrap(),
        [4.0, 24.0, 0.0]
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn new_samples_refit_instead_of_reusing_training_rows() {
    let training = LazyMatrix::new(
        Dense::new(1, 2, vec![1.0, 3.0]),
        RowNormalization::standardize(),
    )
    .unwrap();
    let source = Dense::new(1, 2, vec![10.0, 14.0]);
    let fitted = LazyMatrix::new(&source, RowNormalization::standardize()).unwrap();
    let reused = LazyMatrix::from_parts(&source, training.normalization().clone());
    assert_eq!(fitted.matvec(&vec![1.0, 1.0]).unwrap(), [0.0]);
    assert_eq!(reused.matvec(&vec![1.0, 1.0]).unwrap(), [20.0]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn explicit_parameters_preserve_bits_and_round_trip_without_copying() {
    let centers = vec![
        f64::from_bits(0x7ff8_0000_0000_0123),
        f64::NEG_INFINITY,
        f64::INFINITY,
        -0.0,
    ];
    let scales = vec![
        f64::from_bits(0x7ff8_0000_0000_0456),
        f64::NEG_INFINITY,
        f64::INFINITY,
        -2.0,
    ];
    let bits = |values: &[f64]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    let parameters =
        RowNormalizationParams::from_parts(4, Some(centers.clone()), Some(scales.clone()));
    assert_eq!(parameters.nrows(), 4);
    let source = Dense::new(4, 1, vec![1.0; 4]);
    let normalized =
        LazyMatrix::from_row_parts(&source, Some(centers.clone()), Some(scales.clone()));
    assert_eq!((normalized.nrows(), normalized.ncols()), (4, 1));
    assert_eq!(normalized.normalization().nrows(), 4);
    assert_eq!(source.reads.get(), 0);
    assert!(std::ptr::eq(*normalized.data(), &source));
    assert_eq!(
        bits(normalized.normalization().centers().unwrap()),
        bits(&centers)
    );
    assert_eq!(
        bits(normalized.normalization().scales().unwrap()),
        bits(&scales)
    );
    let center_ptr = normalized.normalization().centers().unwrap().as_ptr();
    let scale_ptr = normalized.normalization().scales().unwrap().as_ptr();
    let (data, recovered) = normalized.into_parts();
    assert!(std::ptr::eq(data, &source));
    assert_eq!(recovered.centers().unwrap().as_ptr(), center_ptr);
    assert_eq!(recovered.scales().unwrap().as_ptr(), scale_ptr);
    let restored = LazyMatrix::from_parts(data, recovered);
    assert_eq!(
        bits(restored.normalization().centers().unwrap()),
        bits(&centers)
    );
    assert_eq!(
        bits(restored.normalization().scales().unwrap()),
        bits(&scales)
    );
    let (recovered_centers, recovered_scales) = parameters.into_parts();
    assert_eq!(bits(recovered_centers.as_ref().unwrap()), bits(&centers));
    assert_eq!(bits(recovered_scales.as_ref().unwrap()), bits(&scales));
    let centered = LazyMatrix::from_row_parts(&source, Some(centers.clone()), None);
    assert_eq!(
        bits(centered.normalization().centers().unwrap()),
        bits(&centers)
    );
    assert!(centered.normalization().scales().is_none());
    let scaled = LazyMatrix::from_row_parts(&source, None, Some(scales.clone()));
    assert_eq!(
        bits(scaled.normalization().scales().unwrap()),
        bits(&scales)
    );
    assert!(scaled.normalization().centers().is_none());
    let wider = Dense::new(4, 3, vec![1.0; 12]);
    let reused = LazyMatrix::from_parts(wider, restored.normalization().clone());
    assert_eq!(reused.ncols(), 3);
    assert_eq!(
        bits(reused.normalization().scales().unwrap()),
        bits(&scales)
    );
}

#[test]
fn explicit_parameters_validate_rows_and_report_zero_scale_index() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let source = Dense::new(2, 3, vec![1.0; 6]);
    for (centers, scales) in [(Some(vec![1.0; 3]), None), (None, Some(vec![1.0; 3]))] {
        assert!(
            catch_unwind(AssertUnwindSafe(|| LazyMatrix::from_row_parts(
                &source, centers, scales
            )))
            .is_err()
        );
    }
    for zero in [0.0, -0.0] {
        let panic = catch_unwind(AssertUnwindSafe(|| {
            LazyMatrix::from_row_parts(&source, None, Some(vec![1.0, zero]))
        }))
        .err()
        .expect("explicit zero scales must panic");
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap();
        assert!(message.contains("row 1"), "{message}");
    }
    let wrong = RowNormalizationParams::<f64>::from_parts(3, None, None);
    assert!(catch_unwind(AssertUnwindSafe(|| LazyMatrix::from_parts(&source, wrong))).is_err());
    assert_eq!(source.reads.get(), 0);
}

#[test]
fn dimension_panics_precede_reads_and_output_writes() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let source = Dense::new(2, 3, vec![1.0; 6]);
    let normalized = LazyMatrix::from_row_parts(&source, Some(vec![0.5; 2]), Some(vec![2.0; 2]));
    for (forward, input_len, output_len) in
        [(true, 2, 2), (true, 3, 3), (false, 3, 3), (false, 2, 2)]
    {
        let input = vec![1.0; input_len];
        let mut out = vec![42.0; output_len];
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                if forward {
                    normalized.matvec_into(&input, &mut out).unwrap();
                } else {
                    normalized.mat_transpose_vec_into(&input, &mut out).unwrap();
                }
            }))
            .is_err()
        );
        assert_eq!(out, vec![42.0; output_len]);
        assert_eq!(source.reads.get(), 0);
    }
    assert!(catch_unwind(AssertUnwindSafe(|| normalized.matvec(&vec![1.0; 2]))).is_err());
    assert!(
        catch_unwind(AssertUnwindSafe(
            || normalized.mat_transpose_vec(&vec![1.0; 3])
        ))
        .is_err()
    );
    assert_eq!(source.reads.get(), 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn empty_shapes_need_no_reads_or_full_design_workspace() {
    for (nrows, ncols) in [(0, 3), (4, 0), (0, 0)] {
        let source = Dense::<f64>::new(nrows, ncols, vec![]);
        let normalized = LazyMatrix::new(&source, RowNormalization::standardize()).unwrap();
        assert_eq!(normalized.normalization().centers().unwrap().len(), nrows);
        assert_eq!(
            normalized.matvec(&vec![1.0; ncols]).unwrap(),
            vec![0.0; nrows]
        );
        assert_eq!(
            normalized.mat_transpose_vec(&vec![1.0; nrows]).unwrap(),
            vec![0.0; ncols]
        );
        assert_eq!(source.reads.get(), 0);
    }
    let source = Dense::<f64>::new(0, usize::MAX, vec![]);
    let normalized = LazyMatrix::new(&source, RowNormalization::robust()).unwrap();
    assert!(normalized.normalization().centers().unwrap().is_empty());
    assert!(normalized.normalization().scales().unwrap().is_empty());
    assert_eq!(source.reads.get(), 0);
    let source = Dense::<f64>::new(usize::MAX, 0, vec![]);
    let normalized = LazyMatrix::new(&source, RowNormalization::default()).unwrap();
    assert!(normalized.normalization().centers().is_none());
    assert!(normalized.normalization().scales().is_none());
    assert_eq!(source.reads.get(), 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn f32_products_scale_entries_before_accumulation() {
    let source = Dense::new(2, 1, vec![f32::MAX; 2]);
    let normalized = LazyMatrix::from_row_parts(source, None, Some(vec![f32::MAX; 2]));
    assert_eq!(normalized.matvec(&vec![2.0]).unwrap(), [2.0; 2]);
    assert_eq!(normalized.mat_transpose_vec(&vec![2.0; 2]).unwrap(), [4.0]);
    let source = Dense::new(1, 1, vec![1e-30_f32]);
    let normalized = LazyMatrix::from_row_parts(source, None, Some(vec![1e-30]));
    assert_eq!(normalized.matvec(&vec![1e-30]).unwrap(), [1e-30]);
    assert_eq!(normalized.mat_transpose_vec(&vec![1e-30]).unwrap(), [1e-30]);
}
