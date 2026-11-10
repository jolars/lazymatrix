//! Fitting and application work for external consumers without backend dependencies.

use std::{cell::Cell, convert::Infallible};

mod dense_rows {
    include!("common/dense_rows.rs");
}
dense_rows::implement!(Dense<F>);

use lazymatrix::{
    ColumnNormalization, ColumnNormalizationParams, ColumnNormalizationParamsView,
    ColumnNormalizationStats, ColumnStats, LazyMatrix, MatTransposeVecInto, MatVecInto,
    MatrixErrorType, MatrixRead, MatrixShape, NormalizationParams, NormalizationRecipe,
    QuantileRange, RowNormalization, RowNormalizationParams, RowNormalizationParamsView, Scalar,
    Scaling,
};

struct Statistics<F> {
    centers: Vec<F>,
    scales: Vec<F>,
    calls: Cell<usize>,
}

impl<F> MatrixShape for Statistics<F> {
    fn nrows(&self) -> usize {
        2
    }

    fn ncols(&self) -> usize {
        2
    }
}

impl<F> MatrixErrorType for Statistics<F> {
    type Error = Infallible;
}

macro_rules! unused_statistics {
    ($($name:ident $(($arg:ident: $ty:ty))?),* $(,)?) => {
        $(fn $name(&self $(, $arg: $ty)?) -> Result<Vec<F>, Self::Error> {
            $(let _ = $arg;)?
            panic!("fitting must use the combined statistics hook");
        })*
    };
}

impl<F: Scalar> ColumnStats<F> for Statistics<F> {
    fn normalization_stats(
        &self,
        spec: ColumnNormalization,
    ) -> Result<ColumnNormalizationStats<F>, Self::Error> {
        assert_eq!(spec, ColumnNormalization::standardize());
        self.calls.set(self.calls.get() + 1);
        Ok((Some(self.centers.clone()), Some(self.scales.clone())))
    }

    unused_statistics!(
        col_medians,
        col_quantile_ranges(range: QuantileRange),
        col_means,
        col_sds(df: usize),
        col_mins,
        col_ranges,
        col_maxabs,
        col_l1,
        col_l2,
        col_l1_centered(centers: &[F]),
        col_l2_centered(centers: &[F]),
        col_maxabs_centered(centers: &[F]),
    );
}

impl<F: Scalar> lazymatrix::RowStats<F> for Statistics<F> {
    fn row_normalization_stats(
        &self,
        spec: RowNormalization,
    ) -> Result<lazymatrix::RowNormalizationStats<F>, Self::Error> {
        assert_eq!(spec, RowNormalization::standardize());
        self.calls.set(self.calls.get() + 1);
        Ok((Some(self.centers.clone()), Some(self.scales.clone())))
    }
}

#[test]
fn calibrated_quantile_endpoints_panic_before_reading_storage() {
    let source = Statistics {
        centers: vec![0.0_f64; 2],
        scales: vec![1.0; 2],
        calls: Cell::new(0),
    };
    for range in [QuantileRange::new(0.0, 0.75), QuantileRange::new(0.25, 1.0)] {
        let scale = Scaling::QuantileRange {
            range,
            unit_variance: true,
        };
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                ColumnNormalization::new(lazymatrix::Centering::Median, scale).fit(&source)
            }))
            .is_err()
        );
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                RowNormalization::new(lazymatrix::Centering::Median, scale).fit(&source)
            }))
            .is_err()
        );
        assert_eq!(source.calls.get(), 0);
    }
}

#[test]
fn row_calibration_reuses_scans_and_preserves_zero_scale_fallback() {
    let source = Dense {
        values: [[1.0_f32, 3.0], [4.0, 4.0]],
        reads: Cell::new(0),
    };
    let raw = RowNormalization::robust().fit(&source).unwrap();
    let raw_reads = source.reads.replace(0);
    let calibrated = RowNormalization::new(
        lazymatrix::Centering::Median,
        Scaling::QuantileRange {
            range: QuantileRange::default(),
            unit_variance: true,
        },
    )
    .fit(&source)
    .unwrap();
    assert_eq!(source.reads.get(), raw_reads);
    assert_eq!(calibrated.centers(), raw.centers());
    approx::assert_relative_eq!(
        calibrated.scales().unwrap()[0],
        (1.0 / 1.3489795003921634) as f32,
        epsilon = 1e-7
    );
    assert_eq!(calibrated.scales().unwrap()[1], 1.0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn row_fitting_forwards_shared_hooks_and_replaces_only_zero_scales() {
    let mut source = Statistics {
        centers: vec![1.0, 2.0],
        scales: vec![-0.0, f64::INFINITY],
        calls: Cell::new(0),
    };
    let spec = RowNormalization::standardize();
    let params = spec.fit(&&source).unwrap();
    assert_eq!(params.centers().unwrap(), &[1.0, 2.0]);
    assert_eq!(params.scales().unwrap(), &[1.0, f64::INFINITY]);
    assert_eq!(source.calls.get(), 1);
    assert_eq!(spec.fit(&&mut source).unwrap(), params);
    assert_eq!(source.calls.get(), 2);
    let erased: &dyn lazymatrix::RowStats<f64, Error = Infallible> = &source;
    assert_eq!(spec.fit(erased).unwrap(), params);
    assert_eq!(source.calls.get(), 3);
}

struct Dense<F> {
    values: [[F; 2]; 2],
    reads: Cell<usize>,
}

impl<F> MatrixShape for Dense<F> {
    fn nrows(&self) -> usize {
        2
    }

    fn ncols(&self) -> usize {
        2
    }
}

impl<F> MatrixErrorType for Dense<F> {
    type Error = Infallible;
}

impl<F: Scalar> MatrixRead<F> for Dense<F> {
    fn get(&self, row: usize, column: usize) -> F {
        self.reads.set(self.reads.get() + 1);
        self.values[row][column]
    }
}

fn fit_recipe<M, P, R>(data: &M, recipe: R) -> Result<P, M::Error>
where
    M: MatrixShape + MatrixErrorType,
    P: NormalizationParams,
    R: NormalizationRecipe<M, P>,
{
    recipe.fit(data)
}

fn parameter_view<P: NormalizationParams>(parameters: &P) -> P::View<'_> {
    parameters.view()
}

trait StatisticsSource<F: Scalar>: ColumnStats<F> + MatrixShape {}
impl<F: Scalar> StatisticsSource<F> for Statistics<F> {}

trait DenseSource<F: Scalar>: lazymatrix::RowStats<F> {}
impl<F: Scalar> DenseSource<F> for Dense<F> {}

fn check_fitting<F: Scalar>() {
    let value = |value| F::from_f64(value).unwrap();
    let columns = {
        let source = Statistics {
            centers: vec![value(1.0), value(2.0)],
            scales: vec![-F::zero(), value(2.0)],
            calls: Cell::new(0),
        };
        let params = ColumnNormalization::standardize().fit(&source).unwrap();
        let fitted = LazyMatrix::from_parts(&source, params);
        assert_eq!(source.calls.get(), 1);

        let generic = fit_recipe(&source, ColumnNormalization::standardize()).unwrap();
        assert_eq!(fitted.normalization(), &generic);
        let erased: &dyn StatisticsSource<F, Error = Infallible> = &source;
        assert_eq!(
            ColumnNormalization::standardize().fit(erased).unwrap(),
            generic
        );
        fitted.into_parts().1
    };
    let _: &ColumnNormalizationParams<F> = &columns;
    assert_eq!(columns.ncols(), 2);
    assert_eq!(columns.centers().unwrap(), &[value(1.0), value(2.0)]);
    assert_eq!(columns.scales().unwrap(), &[F::one(), value(2.0)]);

    let rows = {
        let source = Dense {
            values: [[value(1.0), value(3.0)], [value(10.0), value(14.0)]],
            reads: Cell::new(0),
        };
        let params = RowNormalization::standardize().fit(&source).unwrap();
        let reads = source.reads.get();
        let fitted = LazyMatrix::from_parts(&source, params);
        assert_eq!(source.reads.get(), reads);

        let generic = fit_recipe(&source, RowNormalization::standardize()).unwrap();
        assert_eq!(fitted.normalization(), &generic);
        let erased: &dyn DenseSource<F, Error = Infallible> = &source;
        assert_eq!(
            RowNormalization::standardize().fit(erased).unwrap(),
            generic
        );
        fitted.into_parts().1
    };
    let _: &RowNormalizationParams<F> = &rows;
    assert_eq!(rows.nrows(), 2);
    assert_eq!(rows.centers().unwrap(), &[value(2.0), value(12.0)]);
    assert_eq!(rows.scales().unwrap(), &[F::one(), value(2.0)]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fitting_infers_f32_and_owns_parameters() {
    check_fitting::<f32>();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fitting_infers_f64_and_owns_parameters() {
    check_fitting::<f64>();
}

fn borrowed_columns<'a, M: MatrixShape, F: Scalar>(
    params: &'a ColumnNormalizationParams<F>,
    data: M,
) -> LazyMatrix<M, ColumnNormalizationParamsView<'a, F>> {
    params.view().apply(data)
}

fn borrowed_rows<'a, M: MatrixShape, F: Scalar>(
    params: &'a RowNormalizationParams<F>,
    data: M,
) -> LazyMatrix<M, RowNormalizationParamsView<'a, F>> {
    params.view().apply_frozen(data)
}

fn owned_columns<M: MatrixShape + ColumnStats<F>, F: Scalar>(
    recipe: ColumnNormalization,
    data: M,
) -> Result<LazyMatrix<M, ColumnNormalizationParams<F>>, M::Error> {
    recipe.fit_apply(data)
}

fn owned_rows<M: lazymatrix::RowStats<F>, F: Scalar>(
    recipe: RowNormalization,
    data: M,
) -> Result<LazyMatrix<M, RowNormalizationParams<F>>, M::Error> {
    recipe.fit_apply(data)
}

fn check_application<F: Scalar>() {
    let value = |v| F::from_f64(v).unwrap();
    let source = Statistics {
        centers: vec![value(1.0), value(2.0)],
        scales: vec![value(-2.0), value(3.0)],
        calls: Cell::new(0),
    };
    let params = fit_recipe(&source, ColumnNormalization::standardize()).unwrap();
    let centers = params.centers().unwrap().as_ptr();
    let scales = params.scales().unwrap().as_ptr();
    let first = borrowed_columns(&params, &source);
    let second = params.view().apply(Shape(7, 2));
    let (raw, view) = first.into_parts();
    let rebuilt = LazyMatrix::from_parts(raw, view);
    for view in [rebuilt.normalization(), second.normalization()] {
        assert_eq!(view.centers().unwrap().as_ptr(), centers);
        assert_eq!(view.scales().unwrap().as_ptr(), scales);
        assert_eq!(view.ncols(), 2);
    }
    let (ncols, c, s) = view.into_parts();
    assert_eq!(ncols, 2);
    assert_eq!(c.unwrap().as_ptr(), centers);
    assert_eq!(s.unwrap().as_ptr(), scales);
    assert_eq!(source.calls.get(), 1);
    let moved = params.apply(&source);
    assert_eq!(moved.normalization().centers().unwrap().as_ptr(), centers);
    assert_eq!(moved.normalization().scales().unwrap().as_ptr(), scales);
    let fitted = owned_columns(ColumnNormalization::standardize(), source).unwrap();
    assert_eq!(fitted.data().calls.get(), 2);

    let mut source = Dense {
        values: [[value(1.0), value(3.0)], [value(10.0), value(14.0)]],
        reads: Cell::new(0),
    };
    let params = fit_recipe(&source, RowNormalization::standardize()).unwrap();
    let reads = source.reads.get();
    let centers = params.centers().unwrap().as_ptr();
    let scales = params.scales().unwrap().as_ptr();
    let first = borrowed_rows(&params, &source);
    let second = params.view().apply_frozen(Shape(2, 7));
    assert_eq!(first.normalization().centers().unwrap().as_ptr(), centers);
    assert_eq!(second.normalization().scales().unwrap().as_ptr(), scales);
    assert_eq!(source.reads.get(), reads);
    let mut forward = vec![F::nan(); 2];
    let mut transpose = vec![F::nan(); 2];
    first
        .matvec_into(&vec![F::one(), value(2.0)], &mut forward)
        .unwrap();
    first
        .mat_transpose_vec_into(&vec![F::one(), value(2.0)], &mut transpose)
        .unwrap();
    assert_eq!(forward, vec![F::one(); 2]);
    assert_eq!(transpose, vec![value(-3.0), value(3.0)]);
    let after_products = source.reads.get();
    let (_, view) = first.into_parts();
    let (nrows, c, s) = view.into_parts();
    assert_eq!(nrows, 2);
    assert_eq!(c.unwrap().as_ptr(), centers);
    assert_eq!(s.unwrap().as_ptr(), scales);
    assert_eq!(source.reads.get(), after_products);
    let moved = params.apply_frozen(&mut source);
    assert_eq!(moved.normalization().centers().unwrap().as_ptr(), centers);
    assert_eq!(moved.normalization().scales().unwrap().as_ptr(), scales);
    assert_eq!(moved.data().reads.get(), after_products);
    let fitted = owned_rows(RowNormalization::standardize(), source).unwrap();
    assert_eq!(fitted.data().reads.get(), after_products + reads);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn application_infers_f32_and_shares_or_moves_vectors() {
    check_application::<f32>();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn application_infers_f64_and_shares_or_moves_vectors() {
    check_application::<f64>();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn parameter_parts_preserve_identity_dimensions_and_owned_vectors() {
    for dimension in [0, 7] {
        let columns = ColumnNormalizationParams::<f64>::from_parts(dimension, None, None);
        assert_eq!(
            parameter_view(&columns).into_parts(),
            (dimension, None, None)
        );
        let (ncols, centers, scales) = columns.into_parts();
        let rebuilt = ColumnNormalizationParams::from_parts(ncols, centers, scales);
        assert_eq!(rebuilt.ncols(), dimension);
        assert_eq!(rebuilt.view().apply(Shape(3, dimension)).ncols(), dimension);

        let rows = RowNormalizationParams::<f32>::from_parts(dimension, None, None);
        assert_eq!(parameter_view(&rows).into_parts(), (dimension, None, None));
        let (nrows, centers, scales) = rows.into_parts();
        let rebuilt = RowNormalizationParams::from_parts(nrows, centers, scales);
        assert_eq!(rebuilt.nrows(), dimension);
        assert_eq!(
            rebuilt.view().apply_frozen(Shape(dimension, 3)).nrows(),
            dimension
        );
    }

    let centers = vec![1., 2.];
    let scales = vec![-2., 3.];
    let centers_pointer = centers.as_ptr();
    let scales_pointer = scales.as_ptr();
    let columns = ColumnNormalizationParams::from_parts(2, Some(centers), Some(scales));
    let (ncols, centers, scales) = columns.into_parts();
    assert_eq!(ncols, 2);
    assert_eq!(centers.as_ref().unwrap().as_ptr(), centers_pointer);
    assert_eq!(scales.as_ref().unwrap().as_ptr(), scales_pointer);
    let columns = ColumnNormalizationParams::from_parts(ncols, centers, scales);
    assert_eq!(
        parameter_view(&columns).centers().unwrap().as_ptr(),
        centers_pointer
    );
    assert_eq!(
        parameter_view(&columns).scales().unwrap().as_ptr(),
        scales_pointer
    );
}

struct Shape(usize, usize);

impl MatrixShape for Shape {
    fn nrows(&self) -> usize {
        self.0
    }
    fn ncols(&self) -> usize {
        self.1
    }
}

#[test]
fn application_checks_the_fitted_axis_without_reading_data() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    for n in [0, 2] {
        for active in [false, true] {
            let centers = active.then(|| vec![-0.0; n]);
            let scales = active.then(|| vec![-2.0; n]);
            let columns =
                ColumnNormalizationParams::<f64>::from_parts(n, centers.clone(), scales.clone());
            let rows = RowNormalizationParams::<f64>::from_parts(n, centers, scales);
            let column_view = columns.view();
            let row_view = rows.view();
            assert_eq!(column_view.apply(Shape(7, n)).nrows(), 7);
            assert_eq!(row_view.apply_frozen(Shape(n, 7)).ncols(), 7);
            assert!(catch_unwind(AssertUnwindSafe(|| column_view.apply(Shape(n, n + 1)))).is_err());
            assert!(
                catch_unwind(AssertUnwindSafe(|| row_view.apply_frozen(Shape(n + 1, n)))).is_err()
            );
            assert!(catch_unwind(AssertUnwindSafe(|| columns.apply(Shape(n, n + 1)))).is_err());
            assert!(catch_unwind(AssertUnwindSafe(|| rows.apply_frozen(Shape(n + 1, n)))).is_err());
        }
    }
}

#[test]
fn views_preserve_explicit_ieee_values() {
    let centers = vec![-0.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
    let scales = vec![-2.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN];
    let columns =
        ColumnNormalizationParams::from_parts(4, Some(centers.clone()), Some(scales.clone()));
    let rows = RowNormalizationParams::from_parts(4, Some(centers.clone()), Some(scales.clone()));
    let c = columns.view().apply(Shape(3, 4));
    let r = rows.view().apply_frozen(Shape(4, 3));
    for (actual, expected) in [
        (c.normalization().centers(), &centers),
        (c.normalization().scales(), &scales),
        (r.normalization().centers(), &centers),
        (r.normalization().scales(), &scales),
    ] {
        assert_eq!(
            actual
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
}

#[derive(Debug, PartialEq)]
struct ReadError;

impl std::fmt::Display for ReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("injected read failure")
    }
}
impl std::error::Error for ReadError {}

#[derive(Debug)]
struct Failing;

impl MatrixShape for Failing {
    fn nrows(&self) -> usize {
        2
    }
    fn ncols(&self) -> usize {
        2
    }
}
impl MatrixErrorType for Failing {
    type Error = ReadError;
}
impl<F: Scalar> ColumnStats<F> for Failing {
    fn normalization_stats(
        &self,
        _: ColumnNormalization,
    ) -> Result<ColumnNormalizationStats<F>, ReadError> {
        Err(ReadError)
    }
    unused_statistics!(
        col_medians, col_quantile_ranges(range: QuantileRange), col_means,
        col_sds(df: usize), col_mins, col_ranges, col_maxabs, col_l1, col_l2,
        col_l1_centered(centers: &[F]), col_l2_centered(centers: &[F]),
        col_maxabs_centered(centers: &[F]),
    );
}
impl MatVecInto<Vec<f64>> for Failing {
    fn matvec_into(&self, _: &Vec<f64>, out: &mut Vec<f64>) -> Result<(), ReadError> {
        out[0] = 42.0;
        Err(ReadError)
    }
}
impl<F: Scalar> lazymatrix::RowStats<F> for Failing {
    fn row_normalization_stats(
        &self,
        _: RowNormalization,
    ) -> Result<lazymatrix::RowNormalizationStats<F>, ReadError> {
        Err(ReadError)
    }
}

macro_rules! failing_row_product {
    ($method:ident) => {
        fn $method<
            X: lazymatrix::VectorView<F> + ?Sized,
            Y: lazymatrix::VectorViewMut<F> + ?Sized,
        >(
            &self,
            x: &X,
            centers: Option<&[F]>,
            scales: Option<&[F]>,
            out: &mut Y,
        ) -> Result<(), ReadError> {
            assert_eq!(x.len(), 2);
            assert_eq!(out.len(), 2);
            if let Some(c) = centers {
                assert_eq!(c.len(), 2);
            }
            if let Some(s) = scales {
                assert_eq!(s.len(), 2);
                assert!(s.iter().all(|&v| v != F::zero()));
            }
            out.set(0, F::from_u32(42).unwrap());
            Err(ReadError)
        }
    };
}
impl<F: Scalar> lazymatrix::RowNormalizationKernel<F> for Failing {
    failing_row_product!(row_normalized_matvec_into);
    failing_row_product!(row_normalized_transpose_into);
}
impl MatTransposeVecInto<Vec<f64>> for Failing {
    fn mat_transpose_vec_into(&self, _: &Vec<f64>, out: &mut Vec<f64>) -> Result<(), ReadError> {
        out[0] = 42.0;
        Err(ReadError)
    }
}

#[test]
fn fitting_and_borrowed_products_preserve_backend_errors() {
    let recipe = ColumnNormalization::standardize();
    let fitted: Result<ColumnNormalizationParams<f32>, _> = recipe.fit(&Failing);
    assert_eq!(fitted.unwrap_err(), ReadError);
    let fitted: Result<LazyMatrix<_, ColumnNormalizationParams<f64>>, _> =
        recipe.fit_apply(Failing);
    assert_eq!(fitted.unwrap_err(), ReadError);
    let params = ColumnNormalizationParams::from_parts(2, Some(vec![3.0; 2]), Some(vec![2.0; 2]));
    let lazy = params.view().apply(Failing);
    let mut out = vec![99.0; 2];
    assert_eq!(lazy.matvec_into(&vec![1.0; 2], &mut out), Err(ReadError));
    assert_eq!(out, vec![42.0, 99.0]);
    out.fill(99.0);
    assert_eq!(
        lazy.mat_transpose_vec_into(&vec![1.0; 2], &mut out),
        Err(ReadError)
    );
    assert_eq!(out, vec![42.0, 99.0]);
    let recipe = RowNormalization::standardize();
    let fitted: Result<RowNormalizationParams<f32>, _> = recipe.fit(&Failing);
    assert_eq!(fitted.unwrap_err(), ReadError);
    let fitted: Result<LazyMatrix<_, RowNormalizationParams<f64>>, _> = recipe.fit_apply(Failing);
    assert_eq!(fitted.unwrap_err(), ReadError);
    let params = RowNormalizationParams::from_parts(2, Some(vec![3.0; 2]), Some(vec![2.0; 2]));
    let rows = params.view().apply_frozen(Failing);
    out.fill(99.0);
    assert_eq!(rows.matvec_into(&vec![1.0; 2], &mut out), Err(ReadError));
    assert_eq!(out, vec![42.0, 99.0]);
    out.fill(99.0);
    assert_eq!(
        rows.mat_transpose_vec_into(&vec![1.0; 2], &mut out),
        Err(ReadError)
    );
    assert_eq!(out, vec![42.0, 99.0]);
}
