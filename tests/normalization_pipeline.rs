//! Ordered pipelines work for dependency-light dense consumers.

use lazymatrix::{
    Centering, ColumnNormalization, ColumnNormalizationParams, FittedRowNormalization, LazyMatrix,
    MatTransposeVec, MatTransposeVecInto, MatVec, MatVecInto, MatrixErrorType, MatrixRead,
    MatrixShape, NormalizationChain, NormalizationChainParams, NormalizationParams,
    NormalizationRecipe, PrepareNewSamples, RowNormalization, RowNormalizationParams, Scaling,
};
use std::{cell::Cell, convert::Infallible};

struct Dense<F = f64> {
    nrows: usize,
    ncols: usize,
    values: Vec<F>,
    reads: Cell<usize>,
}
impl<F> Dense<F> {
    fn new(nrows: usize, ncols: usize, values: Vec<F>) -> Self {
        assert_eq!(nrows * ncols, values.len());
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
impl<F: lazymatrix::Scalar> MatrixRead<F> for Dense<F> {
    fn get(&self, i: usize, j: usize) -> F {
        self.reads.set(self.reads.get() + 1);
        self.values[i * self.ncols + j]
    }
}

fn fit_chain<M, P, A, B>(recipe: NormalizationChain<A, B>, data: &M) -> Result<P, M::Error>
where
    M: MatrixShape + MatrixErrorType,
    P: NormalizationParams,
    NormalizationChain<A, B>: NormalizationRecipe<M, P>,
{
    recipe.fit(data)
}

fn fit_apply_chain<M, P, A, B>(
    recipe: NormalizationChain<A, B>,
    data: M,
) -> Result<LazyMatrix<M, P>, M::Error>
where
    M: MatrixShape + MatrixErrorType,
    P: NormalizationParams,
    NormalizationChain<A, B>: NormalizationRecipe<M, P>,
{
    recipe.fit_apply(data)
}

fn first_stage<A, B>(params: &NormalizationChainParams<A, B>) -> &A {
    params.first()
}

fn next_stage<A, B>(params: &NormalizationChainParams<A, B>) -> &B {
    params.next()
}

fn rebuild_chain<A, B>(params: NormalizationChainParams<A, B>) -> NormalizationChainParams<A, B>
where
    NormalizationChainParams<A, B>: NormalizationParams,
{
    let (first, next) = params.into_parts();
    NormalizationChainParams::from_parts(first, next)
}

fn borrow_chain<A, B>(
    params: &NormalizationChainParams<A, B>,
) -> <NormalizationChainParams<A, B> as NormalizationParams>::View<'_>
where
    NormalizationChainParams<A, B>: NormalizationParams,
{
    params.view()
}

fn apply_chain<M, A, B>(
    params: NormalizationChainParams<A, B>,
    data: M,
) -> LazyMatrix<M, NormalizationChainParams<A, B>>
where
    M: MatrixShape,
    NormalizationChainParams<A, B>: NormalizationParams,
{
    params.apply_frozen(data)
}

fn prepare_samples<M, P>(params: P, data: M) -> Result<LazyMatrix<M, P::Params>, M::Error>
where
    M: MatrixShape + MatrixErrorType,
    P: PrepareNewSamples<M>,
{
    params.for_new_samples(data)
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn generic_consumers_fit_borrow_inspect_and_prepare_with_public_bounds() {
    let training = Dense::new(2, 3, vec![1_f32, 3., 8., -2., 5., 3.]);
    let validation = Dense::new(1, 3, vec![5_f32, 1., -1.]);
    let recipe = RowNormalization::standardize().then(ColumnNormalization::standardize());
    let params = rebuild_chain(fit_chain(recipe, &training).unwrap());
    let columns_pointer = next_stage(&params).scales().unwrap().as_ptr();
    let rows_pointer = first_stage(&params)
        .parameters()
        .centers()
        .unwrap()
        .as_ptr();
    let reads = training.reads.get();
    let borrowed = borrow_chain(&params);
    let operator = apply_chain(borrowed, &training);
    assert_eq!(training.reads.get(), reads);
    assert_eq!(
        operator
            .normalization()
            .first()
            .parameters()
            .centers()
            .unwrap()
            .as_ptr(),
        rows_pointer
    );
    let prepared = prepare_samples(borrow_chain(&params), &validation).unwrap();
    assert_eq!(prepared.normalization().first().parameters().nrows(), 1);
    assert_eq!(
        prepared.normalization().next().scales().unwrap().as_ptr(),
        columns_pointer
    );
    let repeated = fit_apply_chain(recipe, &training).unwrap();
    assert_eq!(
        operator.matvec(&vec![1., -2., 3.]).unwrap(),
        repeated.matvec(&vec![1., -2., 3.]).unwrap()
    );

    let columns = next_stage(&params);
    let reads = validation.reads.get();
    let prepared = prepare_samples(NormalizationParams::view(columns), &validation).unwrap();
    assert_eq!(validation.reads.get(), reads);
    assert_eq!(
        prepared.normalization().scales().unwrap().as_ptr(),
        columns_pointer
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn f32_preparation_owns_new_rows_and_borrows_training_columns() {
    let training = Dense::new(2, 3, vec![1_f32, 3., 8., -2., 5., 3.]);
    let recipe = ColumnNormalization::standardize().then(RowNormalization::standardize());
    let fitted = recipe.fit(&training).unwrap();
    let pointer = fitted.first().scales().unwrap().as_ptr();
    let valid = fitted
        .view()
        .for_new_samples(Dense::new(1, 3, vec![5_f32, 1., -1.]))
        .unwrap();
    assert_eq!(
        valid.normalization().first().scales().unwrap().as_ptr(),
        pointer
    );
    let x = vec![1_f32, -2., 3.];
    let u = vec![0.25_f32];
    let forward = valid.matvec(&x).unwrap();
    let transpose = valid.mat_transpose_vec(&u).unwrap();
    let adjoint: f32 = x.iter().zip(transpose).map(|(a, b)| a * b).sum();
    approx::assert_relative_eq!(forward[0] * u[0], adjoint, epsilon = 1e-6);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn new_samples_refit_later_rows_after_frozen_training_columns() {
    let training = Dense::new(3, 3, vec![1., 3., 8., -2., 5., 3., 6., 4., 0.]);
    let validation = Dense::new(2, 3, vec![10., 1., 5., -8., 20., 0.]);
    let fitted = RowNormalization::min_max()
        .then(ColumnNormalization::standardize())
        .then(RowNormalization::new(Centering::Mean, Scaling::L1))
        .fit(&training)
        .unwrap();
    let op = fitted.view().for_new_samples(&validation).unwrap();
    let columns = fitted.first().next();
    let last = op.normalization().next().parameters();
    let x = vec![0.5, -2., 1.];
    let actual = op.matvec(&x).unwrap();
    for (i, raw) in validation.values.chunks(3).enumerate() {
        let min = raw.iter().copied().fold(f64::INFINITY, f64::min);
        let max = raw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let prefix: Vec<_> = raw
            .iter()
            .enumerate()
            .map(|(j, &value)| {
                (((value - min) / (max - min)) - columns.centers().unwrap()[j])
                    / columns.scales().unwrap()[j]
            })
            .collect();
        let mean = prefix.iter().sum::<f64>() / 3.;
        let l1 = prefix.iter().map(|v| (v - mean).abs()).sum::<f64>();
        approx::assert_relative_eq!(last.centers().unwrap()[i], mean, epsilon = 1e-12);
        approx::assert_relative_eq!(last.scales().unwrap()[i], l1, epsilon = 1e-12);
        let expected: f64 = prefix
            .iter()
            .zip(&x)
            .map(|(v, x)| ((v - mean) / l1) * x)
            .sum();
        approx::assert_relative_eq!(actual[i], expected, epsilon = 1e-12);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn duke_order_and_new_samples_reuse_training_columns() {
    let training = Dense::new(3, 2, vec![1., 3., 7., 3., 2., 8.]);
    let validation = Dense::new(2, 2, vec![100., 120., 20., 10.]);
    let recipe = RowNormalization::standardize().then(ColumnNormalization::standardize());
    let fitted = recipe.fit(&training).unwrap();
    assert_eq!(fitted.first().recipe(), RowNormalization::standardize());
    assert_eq!(
        fitted.first().parameters().centers().unwrap(),
        &[2., 5., 5.]
    );
    let reads = training.reads.get();
    let train = fitted.view().apply_frozen(&training);
    assert_eq!(training.reads.get(), reads);
    let valid = fitted.view().for_new_samples(&validation).unwrap();
    assert_eq!(
        valid
            .normalization()
            .first()
            .parameters()
            .centers()
            .unwrap(),
        &[110., 15.]
    );
    assert_eq!(
        valid.normalization().next().centers().unwrap().as_ptr(),
        fitted.next().centers().unwrap().as_ptr(),
    );
    let x = vec![2., -1.];
    let expected = train.matvec(&x).unwrap();
    let actual = valid.matvec(&x).unwrap();
    approx::assert_relative_eq!(actual[0], expected[0], epsilon = 1e-12);
    approx::assert_relative_eq!(actual[1], expected[1], epsilon = 1e-12);
    let reads = validation.reads.get();
    valid.matvec(&x).unwrap();
    assert_eq!(validation.reads.get() - reads, 4);
    let (data, params) = valid.into_parts();
    let reads = validation.reads.get();
    let restored = LazyMatrix::from_parts(data, params);
    assert_eq!(validation.reads.get(), reads);
    assert_eq!(restored.matvec(&x).unwrap(), actual);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn constructors_and_arbitrary_tree_shapes_agree() {
    let data = Dense::new(3, 3, vec![1., 2., 8., 4., -1., 7., 3., 8., 2.]);
    let row = RowNormalization::standardize();
    let column = ColumnNormalization::new(Centering::Mean, Scaling::L2);
    let left = LazyMatrix::new(&data, row.then(column).then(row)).unwrap();
    let right = row.then(column.then(row)).fit_apply(&data).unwrap();
    assert_eq!(
        left.matvec(&vec![1., -2., 3.]).unwrap(),
        right.matvec(&vec![1., -2., 3.]).unwrap()
    );
    let new = Dense::new(1, 3, vec![10., 1., -4.]);
    let left = left.into_parts().1.for_new_samples(&new).unwrap();
    let right = right.into_parts().1.for_new_samples(&new).unwrap();
    assert_eq!(
        left.mat_transpose_vec(&vec![2.]).unwrap(),
        right.mat_transpose_vec(&vec![2.]).unwrap()
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn explicit_stages_preserve_ieee_values_and_ownership() {
    let row =
        RowNormalizationParams::from_parts(2, Some(vec![-0., -2.]), Some(vec![-2., f64::INFINITY]));
    let pointer = row.centers().unwrap().as_ptr();
    let column = ColumnNormalizationParams::from_parts(2, None, Some(vec![-1., 2.]));
    let params = NormalizationChainParams::from_parts(
        FittedRowNormalization::from_parts(RowNormalization::standardize(), row),
        column,
    );
    assert_eq!(
        params.first().parameters().centers().unwrap()[0].to_bits(),
        (-0_f64).to_bits()
    );
    let data = Dense::new(2, 2, vec![2., 4., -2., 6.]);
    let reads = data.reads.get();
    let op = params.apply_frozen(&data);
    assert_eq!(data.reads.get(), reads);
    assert_eq!(
        op.normalization()
            .first()
            .parameters()
            .centers()
            .unwrap()
            .as_ptr(),
        pointer
    );
    assert_eq!(op.matvec(&vec![1., 1.]).unwrap(), vec![0., 0.]);
}

#[test]
fn dimensions_are_checked_before_reads_or_writes() {
    let training = Dense::new(2, 3, vec![1.; 6]);
    let params = RowNormalization::standardize()
        .then(ColumnNormalization::standardize())
        .fit(&training)
        .unwrap();
    let wrong = Dense::new(1, 2, vec![1.; 2]);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = params.view().for_new_samples(&wrong);
    }));
    assert!(result.is_err());
    assert_eq!(wrong.reads.get(), 0);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        params.view().apply_frozen(&wrong);
    }));
    assert!(result.is_err());
    assert_eq!(wrong.reads.get(), 0);
    let op = params.apply_frozen(&training);
    let reads = training.reads.get();
    let mut out = vec![99.; 2];
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        op.matvec_into(&vec![1.; 2], &mut out).unwrap();
    }));
    assert!(result.is_err());
    assert_eq!(out, vec![99.; 2]);
    assert_eq!(training.reads.get(), reads);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        op.matvec(&vec![1.; 2]).unwrap();
    }));
    assert!(result.is_err());
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        op.mat_transpose_vec(&vec![1.; 3]).unwrap();
    }));
    assert!(result.is_err());
    assert_eq!(training.reads.get(), reads);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn empty_and_repeated_row_chains_need_no_design_workspace() {
    for (nrows, ncols) in [(0, 3), (4, 0), (0, 0)] {
        let data = Dense::new(nrows, ncols, vec![]);
        let recipe = RowNormalization::robust().then(RowNormalization::standardize());
        let op = recipe.fit_apply(&data).unwrap();
        let mut forward = vec![99.; nrows];
        op.matvec_into(&vec![1.; ncols], &mut forward).unwrap();
        assert_eq!(forward, vec![0.; nrows]);
        let mut transpose = vec![99.; ncols];
        op.mat_transpose_vec_into(&vec![1.; nrows], &mut transpose)
            .unwrap();
        assert_eq!(transpose, vec![0.; ncols]);
        assert_eq!(data.reads.get(), 0);
    }
}

#[test]
fn fused_f32_pipeline_preserves_saved_stages_and_coefficients() {
    use lazymatrix::{MatTransposeVecScaledInto, MatVecScaledInto};
    let data = Dense::new(2, 3, vec![1_f32, 3., 8., -2., 5., 3.]);
    let recipe = RowNormalization::standardize().then(ColumnNormalization::standardize());
    let op = recipe.fit_apply(&data).unwrap();
    let x = vec![1_f32, -2., 3.];
    let u = vec![0.5_f32, 2.];
    let mut out = vec![3_f32; 2];
    let mut out_t = vec![3_f32; 3];
    let expected = op.matvec(&x).unwrap();
    let expected_t = op.mat_transpose_vec(&u).unwrap();
    op.matvec_scaled_into(-2., &x, 0.5, &mut out).unwrap();
    op.mat_transpose_vec_scaled_into(-2., &u, 0.5, &mut out_t)
        .unwrap();
    for (actual, expected) in out
        .iter()
        .chain(&out_t)
        .zip(expected.iter().chain(&expected_t))
    {
        approx::assert_relative_eq!(*actual, -2. * expected + 1.5, epsilon = 1e-6);
    }
}
