//! Public fitting works without a backend dependency or retained input storage.

use std::{cell::Cell, convert::Infallible};

use lazymatrix::{
    ColumnNormalization, ColumnNormalizationParams, ColumnNormalizationStats, ColumnStats,
    LazyMatrix, MatrixErrorType, MatrixRead, MatrixShape, NormalizationParams, NormalizationRecipe,
    QuantileRange, RowNormalization, RowNormalizationParams, Scalar,
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

trait StatisticsSource<F: Scalar>: ColumnStats<F> + MatrixShape {}
impl<F: Scalar> StatisticsSource<F> for Statistics<F> {}

trait DenseSource<F: Scalar>: MatrixRead<F> + MatrixErrorType {}
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
