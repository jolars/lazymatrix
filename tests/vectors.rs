#[path = "common/runner.rs"]
mod common;

use lazymatrix::{
    DotSlice, ElemDivAssign, ScaledSubSlice, SubScalarAssign, SumEntries, VectorView,
};

#[test]
fn vector_reductions() {
    common::run_vector_reductions_suite::<f32, _>(|v| v.to_vec());
    common::run_vector_reductions_suite::<f64, _>(|v| v.to_vec());
}

#[test]
fn vec_algebra_and_normalization_work_without_a_backend() {
    common::vector_algebra(&|v| v.to_vec(), &Clone::clone);
    let mut values = vec![6.0, -8.0, 10.0];
    values.elem_div_assign(&[2.0, 4.0, 5.0]);
    assert_eq!(values.dot_slice(&[1.0, -2.0, 3.0]), 13.0);
    values.sub_scalar_assign(1.0);
    values.scaled_sub_slice(0.5, &[2.0, -4.0, 6.0]);
    assert_eq!(values, vec![1.0, -1.0, -2.0]);
    assert_eq!(values.sum_entries(), -2.0);
}

#[test]
fn vector_sums_respect_borrowed_ranges_without_a_backend() {
    let mut storage = [f64::NAN; 35];
    for (i, value) in storage[1..34].iter_mut().enumerate() {
        *value = (i as f64 - 16.0) / 8.0;
    }
    for len in 0..=33 {
        let values = &storage[1..len + 1];
        let expected = values.iter().sum::<f64>();
        approx::assert_relative_eq!(VectorView::sum(values), expected);
        approx::assert_relative_eq!(VectorView::sum(&values.to_vec()), expected);
    }
    assert!(VectorView::sum(&storage).is_nan());
}
