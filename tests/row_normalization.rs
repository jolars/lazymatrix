//! Row normalization also works for dependency-free dense consumers.

use std::{cell::Cell, convert::Infallible};

use lazymatrix::{
    Centering, MatTransposeVec, MatTransposeVecInto, MatVec, MatVecInto, MatrixErrorType,
    MatrixRead, MatrixShape, RowNormalization, RowNormalizationParams, RowNormalizedMatrix, Scalar,
    Scaling,
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

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn computed_rows_are_independent_and_keep_zero_rows() {
    let source = Dense::new(3, 2, vec![1.0, 3.0, 10.0, 14.0, 0.0, 0.0]);
    let normalized = RowNormalizedMatrix::new(&source, RowNormalization::standardize()).unwrap();
    assert_eq!(normalized.centers().unwrap(), &[2.0, 12.0, 0.0]);
    assert_eq!(normalized.scales().unwrap(), &[1.0, 2.0, 1.0]);
    assert_eq!(normalized.matvec(&vec![1.0, 2.0]).unwrap(), [1.0, 1.0, 0.0]);
    assert_eq!(
        normalized.mat_transpose_vec(&vec![1.0, 2.0, 3.0]).unwrap(),
        [-3.0, 3.0]
    );
    for scale in [Scaling::L1, Scaling::L2, Scaling::MaxAbs] {
        let normalized =
            RowNormalizedMatrix::new(&source, RowNormalization::new(Centering::None, scale))
                .unwrap();
        assert_eq!(normalized.scales().unwrap()[2], 1.0);
        assert_eq!(normalized.matvec(&vec![1.0, 1.0]).unwrap()[2], 0.0);
    }
    for spec in [RowNormalization::min_max(), RowNormalization::robust()] {
        let normalized = RowNormalizedMatrix::new(&source, spec).unwrap();
        assert_eq!(normalized.scales().unwrap()[2], 1.0);
    }
    let mut source = source;
    let normalized = RowNormalizedMatrix::new(&mut source, RowNormalization::default()).unwrap();
    assert!(normalized.centers().is_none());
    assert!(normalized.scales().is_none());
    assert_eq!(
        normalized.matvec(&vec![1.0, 1.0]).unwrap(),
        [4.0, 24.0, 0.0]
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn new_samples_refit_instead_of_reusing_training_rows() {
    let training = RowNormalizedMatrix::new(
        Dense::new(1, 2, vec![1.0, 3.0]),
        RowNormalization::standardize(),
    )
    .unwrap();
    let source = Dense::new(1, 2, vec![10.0, 14.0]);
    let fitted = RowNormalizedMatrix::new(&source, RowNormalization::standardize()).unwrap();
    let reused = RowNormalizedMatrix::from_normalization(&source, training.normalization().clone());
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
    let normalized = RowNormalizedMatrix::from_normalization(&source, parameters.clone());
    assert_eq!((normalized.nrows(), normalized.ncols()), (4, 1));
    assert!(std::ptr::eq(*normalized.data(), &source));
    assert_eq!(bits(normalized.centers().unwrap()), bits(&centers));
    assert_eq!(bits(normalized.scales().unwrap()), bits(&scales));
    let (data, recovered_centers, recovered_scales) = normalized.into_parts();
    assert!(std::ptr::eq(data, &source));
    let restored = RowNormalizedMatrix::from_parts(data, recovered_centers, recovered_scales);
    assert_eq!(bits(restored.centers().unwrap()), bits(&centers));
    assert_eq!(bits(restored.scales().unwrap()), bits(&scales));
    let (recovered_centers, recovered_scales) = parameters.into_parts();
    assert_eq!(bits(recovered_centers.as_ref().unwrap()), bits(&centers));
    assert_eq!(bits(recovered_scales.as_ref().unwrap()), bits(&scales));
    let centered = RowNormalizedMatrix::with_centers(&source, centers.clone());
    assert_eq!(bits(centered.centers().unwrap()), bits(&centers));
    assert!(centered.scales().is_none());
    let scaled = RowNormalizedMatrix::with_scales(&source, scales.clone());
    assert_eq!(bits(scaled.scales().unwrap()), bits(&scales));
    assert!(scaled.centers().is_none());
    let wider = Dense::new(4, 3, vec![1.0; 12]);
    let reused = RowNormalizedMatrix::from_normalization(wider, restored.normalization().clone());
    assert_eq!(reused.ncols(), 3);
    assert_eq!(bits(reused.scales().unwrap()), bits(&scales));
}

#[test]
fn explicit_parameters_validate_rows_and_report_zero_scale_index() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let source = Dense::new(2, 3, vec![1.0; 6]);
    for (centers, scales) in [(Some(vec![1.0; 3]), None), (None, Some(vec![1.0; 3]))] {
        assert!(
            catch_unwind(AssertUnwindSafe(|| RowNormalizedMatrix::from_parts(
                &source, centers, scales
            )))
            .is_err()
        );
    }
    for zero in [0.0, -0.0] {
        let panic =
            catch_unwind(|| RowNormalizationParams::from_parts(2, None, Some(vec![1.0, zero])))
                .unwrap_err();
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap();
        assert!(message.contains("row 1"), "{message}");
    }
    let wrong = RowNormalizationParams::<f64>::from_parts(3, None, None);
    assert!(
        catch_unwind(AssertUnwindSafe(
            || RowNormalizedMatrix::from_normalization(&source, wrong)
        ))
        .is_err()
    );
    assert_eq!(source.reads.get(), 0);
}

#[test]
fn dimension_panics_precede_reads_and_output_writes() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let source = Dense::new(2, 3, vec![1.0; 6]);
    let normalized =
        RowNormalizedMatrix::from_parts(&source, Some(vec![0.5; 2]), Some(vec![2.0; 2]));
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
        let normalized =
            RowNormalizedMatrix::new(&source, RowNormalization::standardize()).unwrap();
        assert_eq!(normalized.centers().unwrap().len(), nrows);
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
    let normalized = RowNormalizedMatrix::new(&source, RowNormalization::robust()).unwrap();
    assert!(normalized.centers().unwrap().is_empty());
    assert!(normalized.scales().unwrap().is_empty());
    assert_eq!(source.reads.get(), 0);
    let source = Dense::<f64>::new(usize::MAX, 0, vec![]);
    let normalized = RowNormalizedMatrix::new(&source, RowNormalization::default()).unwrap();
    assert!(normalized.centers().is_none());
    assert!(normalized.scales().is_none());
    assert_eq!(source.reads.get(), 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn f32_products_scale_entries_before_accumulation() {
    let source = Dense::new(2, 1, vec![f32::MAX; 2]);
    let normalized = RowNormalizedMatrix::with_scales(source, vec![f32::MAX; 2]);
    assert_eq!(normalized.matvec(&vec![2.0]).unwrap(), [2.0; 2]);
    assert_eq!(normalized.mat_transpose_vec(&vec![2.0; 2]).unwrap(), [4.0]);
    let source = Dense::new(1, 1, vec![1e-30_f32]);
    let normalized = RowNormalizedMatrix::with_scales(source, vec![1e-30]);
    assert_eq!(normalized.matvec(&vec![1e-30]).unwrap(), [1e-30]);
    assert_eq!(normalized.mat_transpose_vec(&vec![1e-30]).unwrap(), [1e-30]);
}
