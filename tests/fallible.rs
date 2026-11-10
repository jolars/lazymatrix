use std::cell::Cell;
use std::fmt;

use lazymatrix::{
    Centering, ColumnNormalization, ColumnStats, DotSlice, ElemDivAssign, LazyMatrix,
    MatTransposeVec, MatTransposeVecInto, MatVec, MatVecInto, MatrixErrorType, MatrixShape,
    Scaling, SubScalarAssign, Transpose,
};

#[derive(Debug, PartialEq, Eq)]
struct ReadError;

impl fmt::Display for ReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("injected read failure")
    }
}

impl std::error::Error for ReadError {}

struct Source {
    fail: Cell<bool>,
    fused_calls: Cell<usize>,
}

impl MatrixErrorType for Source {
    type Error = ReadError;
}

impl MatrixShape for Source {
    fn nrows(&self) -> usize {
        2
    }
    fn ncols(&self) -> usize {
        2
    }
}

impl MatVecInto<Vec<f64>> for Source {
    fn matvec_into(&self, x: &Vec<f64>, out: &mut Vec<f64>) -> Result<(), ReadError> {
        out[0] = 17.0;
        if self.fail.get() {
            return Err(ReadError);
        }
        out[0] = x[0] + 2.0 * x[1];
        out[1] = 3.0 * x[0] + 4.0 * x[1];
        Ok(())
    }
}

impl MatTransposeVecInto<Vec<f64>> for Source {
    fn mat_transpose_vec_into(&self, x: &Vec<f64>, out: &mut Vec<f64>) -> Result<(), ReadError> {
        out[0] = 17.0;
        if self.fail.get() {
            return Err(ReadError);
        }
        out[0] = x[0] + 3.0 * x[1];
        out[1] = 2.0 * x[0] + 4.0 * x[1];
        Ok(())
    }
}

impl MatVec<Vec<f64>> for Source {
    fn matvec(&self, x: &Vec<f64>) -> Result<Vec<f64>, ReadError> {
        let mut out = vec![0.0; 2];
        self.matvec_into(x, &mut out)?;
        Ok(out)
    }
}

impl MatTransposeVec<Vec<f64>> for Source {
    fn mat_transpose_vec(&self, x: &Vec<f64>) -> Result<Vec<f64>, ReadError> {
        let mut out = vec![0.0; 2];
        self.mat_transpose_vec_into(x, &mut out)?;
        Ok(out)
    }
}

#[derive(Default)]
struct ProductCounts {
    clones: Cell<usize>,
    dots: Cell<usize>,
    corrections: Cell<usize>,
}

struct TrackedVector<'a> {
    values: Vec<f64>,
    counts: &'a ProductCounts,
}

impl Clone for TrackedVector<'_> {
    fn clone(&self) -> Self {
        self.counts.clones.set(self.counts.clones.get() + 1);
        Self {
            values: self.values.clone(),
            counts: self.counts,
        }
    }
}

impl ElemDivAssign<f64> for TrackedVector<'_> {
    fn elem_div_assign(&mut self, scales: &[f64]) {
        self.values.elem_div_assign(scales);
    }
}

impl DotSlice<f64> for TrackedVector<'_> {
    fn dot_slice(&self, centers: &[f64]) -> f64 {
        self.counts.dots.set(self.counts.dots.get() + 1);
        self.values.dot_slice(centers)
    }
}

impl SubScalarAssign<f64> for TrackedVector<'_> {
    fn sub_scalar_assign(&mut self, value: f64) {
        self.counts
            .corrections
            .set(self.counts.corrections.get() + 1);
        self.values.sub_scalar_assign(value);
    }
}

impl<'a> MatVec<TrackedVector<'a>> for Source {
    fn matvec(&self, x: &TrackedVector<'a>) -> Result<TrackedVector<'a>, ReadError> {
        Ok(TrackedVector {
            values: self.matvec(&x.values)?,
            counts: x.counts,
        })
    }
}

#[test]
fn allocating_forward_products_borrow_inputs_without_scaling() {
    for centers in [None, Some(vec![1.0, 2.0])] {
        for scales in [
            None,
            Some(vec![2.0, 4.0]),
            Some(vec![1.0, 1.0]),
            Some(vec![-2.0, 4.0]),
        ] {
            let source = Source {
                fail: Cell::new(false),
                fused_calls: Cell::new(0),
            };
            let counts = ProductCounts::default();
            let x = TrackedVector {
                values: vec![2.0, 4.0],
                counts: &counts,
            };
            let lazy = LazyMatrix::from_column_parts(&source, centers.clone(), scales);
            let expected = common::dense_matvec(
                &common::materialize(
                    &[vec![1.0, 2.0], vec![3.0, 4.0]],
                    lazy.normalization().centers(),
                    lazy.normalization().scales(),
                ),
                &x.values,
            );
            assert_eq!(lazy.matvec(&x).unwrap().values, expected);
            assert_eq!(x.values, [2.0, 4.0]);
            assert_eq!(
                counts.clones.get(),
                usize::from(lazy.normalization().scales().is_some())
            );
            assert_eq!(counts.dots.get(), usize::from(centers.is_some()));
            assert_eq!(counts.corrections.get(), usize::from(centers.is_some()));
        }
    }
}

#[test]
fn allocating_forward_errors_skip_centering_with_or_without_scaling() {
    for scales in [None, Some(vec![2.0, 4.0])] {
        let source = Source {
            fail: Cell::new(true),
            fused_calls: Cell::new(0),
        };
        let counts = ProductCounts::default();
        let x = TrackedVector {
            values: vec![2.0, 4.0],
            counts: &counts,
        };
        let lazy = LazyMatrix::from_column_parts(&source, Some(vec![1.0, 2.0]), scales);
        assert!(matches!(lazy.matvec(&x), Err(ReadError)));
        assert_eq!(x.values, [2.0, 4.0]);
        assert_eq!(
            counts.clones.get(),
            usize::from(lazy.normalization().scales().is_some())
        );
        assert_eq!(counts.dots.get(), 0);
        assert_eq!(counts.corrections.get(), 0);
    }
}

macro_rules! failing_stats {
    ($($name:ident $(($arg:ident: $ty:ty))?),* $(,)?) => {
        $(fn $name(&self $(, $arg: $ty)?) -> Result<Vec<f64>, ReadError> {
            $(let _ = $arg;)?
            Err(ReadError)
        })*
    };
}

impl ColumnStats<f64> for Source {
    fn col_quantile_ranges(&self, _: lazymatrix::QuantileRange) -> Result<Vec<f64>, ReadError> {
        Err(ReadError)
    }
    failing_stats!(
        col_medians,
        col_means,
        col_sds(df: usize),
        col_mins,
        col_ranges,
        col_maxabs,
        col_l1,
        col_l2,
        col_l1_centered(centers: &[f64]),
        col_l2_centered(centers: &[f64]),
        col_maxabs_centered(centers: &[f64])
    );

    fn normalization_stats(
        &self,
        _: ColumnNormalization,
    ) -> Result<lazymatrix::ColumnNormalizationStats<f64>, ReadError> {
        self.fused_calls.set(self.fused_calls.get() + 1);
        if self.fail.get() {
            return Err(ReadError);
        }
        Ok((Some(vec![1.0, 2.0]), Some(vec![-0.0, f64::INFINITY])))
    }
}

#[test]
fn normalization_uses_borrowed_fused_hook_and_preserves_scale_policy() {
    let source = Source {
        fail: Cell::new(false),
        fused_calls: Cell::new(0),
    };
    let spec = ColumnNormalization::new(Centering::Mean, Scaling::Sd { df: 0 });
    let params = spec.fit(&source).unwrap();
    assert_eq!(source.fused_calls.get(), 1);
    let lazy = LazyMatrix::from_parts(&source, params);
    assert_eq!(source.fused_calls.get(), 1);
    assert_eq!(lazy.normalization().centers(), Some([1.0, 2.0].as_slice()));
    assert_eq!(
        lazy.normalization().scales(),
        Some([1.0, f64::INFINITY].as_slice())
    );
    let combined = LazyMatrix::new(&source, spec).unwrap();
    assert_eq!(source.fused_calls.get(), 2);
    assert_eq!(combined.normalization(), lazy.normalization());
    source.fail.set(true);
    assert!(matches!(spec.fit(&source), Err(ReadError)));
    assert_eq!(source.fused_calls.get(), 3);
    assert!(matches!(LazyMatrix::new(&source, spec), Err(ReadError)));
    assert_eq!(source.fused_calls.get(), 4);
}

#[test]
fn product_errors_propagate_before_normalization_and_retry_overwrites() {
    let source = Source {
        fail: Cell::new(true),
        fused_calls: Cell::new(0),
    };
    let lazy = LazyMatrix::from_column_parts(&source, Some(vec![1.0, 2.0]), Some(vec![2.0, 4.0]));
    let x = vec![2.0, 4.0];
    let mut out = vec![99.0; 2];
    assert_eq!(lazy.matvec_into(&x, &mut out), Err(ReadError));
    assert_eq!(out, [17.0, 99.0]);
    assert_eq!(lazy.mat_transpose_vec_into(&x, &mut out), Err(ReadError));
    assert_eq!(out, [17.0, 99.0]);
    assert_eq!(lazy.matvec(&x), Err(ReadError));
    assert_eq!(lazy.mat_transpose_vec(&x), Err(ReadError));
    let view = Transpose::new(&lazy);
    assert_eq!(view.matvec_into(&x, &mut out), Err(ReadError));
    assert_eq!(out, [17.0, 99.0]);
    assert_eq!(view.mat_transpose_vec_into(&x, &mut out), Err(ReadError));
    assert_eq!(out, [17.0, 99.0]);
    assert_eq!(view.matvec(&x), Err(ReadError));
    assert_eq!(view.mat_transpose_vec(&x), Err(ReadError));
    source.fail.set(false);
    view.matvec_into(&x, &mut out).unwrap();
    assert_eq!(out, [4.0, 2.0]);
    view.mat_transpose_vec_into(&x, &mut out).unwrap();
    assert_eq!(out, [0.0, 4.0]);
    lazy.matvec_into(&x, &mut out).unwrap();
    assert_eq!(out, [0.0, 4.0]);
    lazy.mat_transpose_vec_into(&x, &mut out).unwrap();
    assert_eq!(out, [4.0, 2.0]);
}

struct DefaultStats;
impl MatrixErrorType for DefaultStats {
    type Error = ReadError;
}
impl ColumnStats<f64> for DefaultStats {
    fn col_quantile_ranges(&self, _: lazymatrix::QuantileRange) -> Result<Vec<f64>, ReadError> {
        Err(ReadError)
    }
    failing_stats!(
        col_medians,
        col_means,
        col_sds(df: usize),
        col_mins,
        col_ranges,
        col_maxabs,
        col_l1,
        col_l2,
        col_l1_centered(centers: &[f64]),
        col_l2_centered(centers: &[f64]),
        col_maxabs_centered(centers: &[f64])
    );
}

#[test]
fn default_normalization_hook_propagates_statistic_errors() {
    assert_eq!(
        DefaultStats.normalization_stats(ColumnNormalization::default()),
        Ok((None, None))
    );
    for center in [
        Centering::None,
        Centering::Mean,
        Centering::Min,
        Centering::Median,
    ] {
        for scale in [
            Scaling::None,
            Scaling::Sd { df: 0 },
            Scaling::Sd { df: 1 },
            Scaling::L1,
            Scaling::L2,
            Scaling::MaxAbs,
            Scaling::Range,
            Scaling::QuantileRange {
                range: lazymatrix::QuantileRange::default(),
                unit_variance: false,
            },
        ] {
            let spec = ColumnNormalization::new(center, scale);
            if spec != ColumnNormalization::default() {
                assert_eq!(DefaultStats.normalization_stats(spec), Err(ReadError));
            }
        }
    }
}

#[path = "common/runner.rs"]
mod common;

impl lazymatrix::MaterializeDense<f64> for Source {
    fn materialize_normalized_into<O: lazymatrix::MatrixWrite<f64> + ?Sized>(
        &self,
        centers: Option<&[f64]>,
        scales: Option<&[f64]>,
        out: &mut O,
    ) -> Result<(), ReadError> {
        out.set(0, 0, 17.0);
        if self.fail.get() {
            return Err(ReadError);
        }
        for i in 0..2 {
            for j in 0..2 {
                let mut value = (2 * i + j + 1) as f64;
                if let Some(c) = centers {
                    value -= c[j];
                }
                if let Some(s) = scales {
                    value /= s[j];
                }
                out.set(i, j, value);
            }
        }
        Ok(())
    }
}

#[test]
fn materialization_failure_returns_no_wrapper_and_retry_overwrites_output() {
    use lazymatrix::MatrixOwned;
    let source = Source {
        fail: Cell::new(false),
        fused_calls: Cell::new(0),
    };
    let lazy = LazyMatrix::new(&source, ColumnNormalization::default()).unwrap();
    assert_eq!(source.fused_calls.get(), 1);
    let mut out = common::MaterializeOutput::zeros(2, 2);
    out.values.fill(99.0);
    source.fail.set(true);
    assert!(matches!(lazy.to_eager_into(&mut out), Err(ReadError)));
    assert_eq!(out.values, [17.0, 99.0, 99.0, 99.0]);
    assert!(matches!(
        lazy.to_eager::<common::MaterializeOutput>(),
        Err(ReadError)
    ));
    source.fail.set(false);
    let eager = lazy.to_eager_into(&mut out).unwrap();
    assert_eq!(eager.data().values, [0.0, 0.0, 2.0, 0.0]);
    assert_eq!(source.fused_calls.get(), 1);
}

impl lazymatrix::MatrixTraversal<f64> for Source {
    fn traversal_layout(&self) -> lazymatrix::TraversalLayout {
        lazymatrix::TraversalLayout::Full
    }
    fn for_each_entry(
        &self,
        rows: std::ops::Range<usize>,
        columns: std::ops::Range<usize>,
        mut visit: impl FnMut(usize, usize, f64),
    ) -> Result<(), ReadError> {
        assert!(rows.start <= rows.end && rows.end <= 2);
        assert!(columns.start <= columns.end && columns.end <= 2);
        if rows.is_empty() || columns.is_empty() {
            return Ok(());
        }
        self.fused_calls.set(self.fused_calls.get() + 1);
        for j in columns {
            for i in rows.clone() {
                visit(i, j, (2 * i + j + 1) as f64);
                if self.fail.get() {
                    return Err(ReadError);
                }
            }
        }
        Ok(())
    }
}

fn row_product<
    X: lazymatrix::VectorView<f64> + ?Sized,
    Y: lazymatrix::VectorViewMut<f64> + ?Sized,
>(
    source: &Source,
    x: &X,
    centers: Option<&[f64]>,
    scales: Option<&[f64]>,
    out: &mut Y,
    transpose: bool,
) -> Result<(), ReadError> {
    use lazymatrix::MatrixTraversal;
    assert_eq!(x.len(), 2);
    assert_eq!(out.len(), 2);
    for i in 0..2 {
        out.set(i, 0.);
    }
    source.for_each_entry(0..2, 0..2, |i, j, raw| {
        let mut value = raw;
        if let Some(c) = centers {
            value -= c[i];
        }
        if let Some(s) = scales {
            value /= s[i];
        }
        let (to, from) = if transpose { (j, i) } else { (i, j) };
        out.set(to, out.get(to) + value * x.get(from));
    })
}
impl lazymatrix::RowNormalizationKernel<f64> for Source {
    fn row_normalized_matvec_into<
        X: lazymatrix::VectorView<f64> + ?Sized,
        Y: lazymatrix::VectorViewMut<f64> + ?Sized,
    >(
        &self,
        x: &X,
        centers: Option<&[f64]>,
        scales: Option<&[f64]>,
        out: &mut Y,
    ) -> Result<(), ReadError> {
        row_product(self, x, centers, scales, out, false)
    }
    fn row_normalized_transpose_into<
        X: lazymatrix::VectorView<f64> + ?Sized,
        Y: lazymatrix::VectorViewMut<f64> + ?Sized,
    >(
        &self,
        x: &X,
        centers: Option<&[f64]>,
        scales: Option<&[f64]>,
        out: &mut Y,
    ) -> Result<(), ReadError> {
        row_product(self, x, centers, scales, out, true)
    }
}

#[test]
fn fused_rows_and_chains_skip_reads_and_preserve_errors() {
    use lazymatrix::{
        ColumnNormalizationParams, MatTransposeVecScaledInto, MatVecScaledInto,
        NormalizationChainParams, RowNormalizationParams, VectorView, VectorViewMut,
    };
    struct Unreadable(Vec<f64>);
    impl VectorView<f64> for Unreadable {
        fn len(&self) -> usize {
            self.0.len()
        }
        fn get(&self, _: usize) -> f64 {
            panic!("zero coefficients must skip reads");
        }
    }
    impl VectorViewMut<f64> for Unreadable {
        fn set(&mut self, i: usize, value: f64) {
            self.0[i] = value;
        }
    }
    let source = Source {
        fail: Cell::new(true),
        fused_calls: Cell::new(0),
    };
    let rows = RowNormalizationParams::from_parts(2, Some(vec![0.5; 2]), Some(vec![-2.; 2]));
    let columns = NormalizationChainParams::from_parts(
        ColumnNormalizationParams::from_parts(2, Some(vec![0.5; 2]), Some(vec![-2.; 2])),
        ColumnNormalizationParams::from_parts(2, Some(vec![0.25; 2]), Some(vec![1.5; 2])),
    );
    macro_rules! check {
        ($op:expr) => {{
            let op = $op;
            let mut out = vec![99.; 2];
            assert_eq!(
                op.matvec_scaled_into(2., &vec![1.; 2], 3., &mut out),
                Err(ReadError)
            );
            assert_eq!(out, [99.; 2]);
            assert_eq!(
                op.mat_transpose_vec_scaled_into(2., &vec![1.; 2], 3., &mut out),
                Err(ReadError)
            );
            assert_eq!(out, [99.; 2]);
            let mut scratch = vec![f64::NAN; 2];
            assert_eq!(
                op.matvec_scaled_with_workspace(2., &vec![1.; 2], 3., &mut out, &mut scratch),
                Err(ReadError)
            );
            assert_eq!(out, [99.; 2]);
            assert!(scratch[0].is_finite());
            scratch.fill(f64::NAN);
            assert_eq!(
                op.mat_transpose_vec_scaled_with_workspace(
                    2.,
                    &vec![1.; 2],
                    3.,
                    &mut out,
                    &mut scratch,
                ),
                Err(ReadError)
            );
            assert_eq!(out, [99.; 2]);
            assert!(scratch[0].is_finite());
            let before = source.fused_calls.get();
            let x = Unreadable(vec![0.; 2]);
            let mut out = Unreadable(vec![f64::NAN; 2]);
            op.matvec_scaled_into(0., &x, 0., &mut out).unwrap();
            assert_eq!(out.0, [0.; 2]);
            op.mat_transpose_vec_scaled_into(-0., &x, -0., &mut out)
                .unwrap();
            assert_eq!(out.0, [0.; 2]);
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| op.matvec_scaled_into(
                    0.,
                    &Unreadable(vec![0.; 1]),
                    0.,
                    &mut out
                )))
                .is_err()
            );
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                    || op.mat_transpose_vec_scaled_into(0., &x, 0., &mut Unreadable(vec![0.; 1]))
                ))
                .is_err()
            );
            scratch.fill(42.);
            op.matvec_scaled_with_workspace(-0., &x, -0., &mut out, &mut scratch)
                .unwrap();
            op.mat_transpose_vec_scaled_with_workspace(0., &x, 0., &mut out, &mut scratch)
                .unwrap();
            assert_eq!(out.0, [0.; 2]);
            assert_eq!(scratch, [42.; 2]);
            for transpose in [false, true] {
                for (input_len, output_len, scratch_len) in [(1, 2, 2), (2, 1, 2), (2, 2, 1)] {
                    let input = Unreadable(vec![0.; input_len]);
                    let mut out = Unreadable(vec![99.; output_len]);
                    let mut scratch = vec![42.; scratch_len];
                    assert!(
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            if transpose {
                                op.mat_transpose_vec_scaled_with_workspace(
                                    0.,
                                    &input,
                                    0.,
                                    &mut out,
                                    &mut scratch,
                                )
                                .unwrap();
                            } else {
                                op.matvec_scaled_with_workspace(
                                    0.,
                                    &input,
                                    0.,
                                    &mut out,
                                    &mut scratch,
                                )
                                .unwrap();
                            }
                        }))
                        .is_err()
                    );
                    assert_eq!(out.0, vec![99.; output_len]);
                    assert_eq!(scratch, vec![42.; scratch_len]);
                }
            }
            assert_eq!(source.fused_calls.get(), before);
            source.fail.set(false);
            op.matvec_scaled_into(2., &vec![1.; 2], 0., &mut out)
                .unwrap();
            assert!(out.0.iter().all(|v| v.is_finite()));
            op.mat_transpose_vec_scaled_into(2., &vec![1.; 2], 0., &mut out)
                .unwrap();
            assert!(out.0.iter().all(|v| v.is_finite()));
            op.matvec_scaled_with_workspace(2., &vec![1.; 2], 0., &mut out, &mut scratch)
                .unwrap();
            assert!(out.0.iter().all(|v| v.is_finite()));
            op.mat_transpose_vec_scaled_with_workspace(
                2.,
                &vec![1.; 2],
                0.,
                &mut out,
                &mut scratch,
            )
            .unwrap();
            assert!(out.0.iter().all(|v| v.is_finite()));
            assert_eq!(source.fused_calls.get(), before + 4);
            source.fail.set(true);
        }};
    }
    check!(rows.view().apply_frozen(&source));
    check!(columns.view().apply_frozen(&source));
}
