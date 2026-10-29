use std::cell::Cell;
use std::fmt;

use lazymatrix::{
    Centering, ColumnStats, DotSlice, ElemDivAssign, LazyMatrix, MatTransposeVec,
    MatTransposeVecInto, MatVec, MatVecInto, MatrixErrorType, MatrixShape, Normalization, Scaling,
    SubScalarAssign,
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
            let lazy = LazyMatrix::from_parts(&source, centers.clone(), scales);
            let expected = common::dense_matvec(
                &common::materialize(
                    &[vec![1.0, 2.0], vec![3.0, 4.0]],
                    lazy.centers(),
                    lazy.scales(),
                ),
                &x.values,
            );
            assert_eq!(lazy.matvec(&x).unwrap().values, expected);
            assert_eq!(x.values, [2.0, 4.0]);
            assert_eq!(counts.clones.get(), usize::from(lazy.scales().is_some()));
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
        let lazy = LazyMatrix::from_parts(&source, Some(vec![1.0, 2.0]), scales);
        assert!(matches!(lazy.matvec(&x), Err(ReadError)));
        assert_eq!(x.values, [2.0, 4.0]);
        assert_eq!(counts.clones.get(), usize::from(lazy.scales().is_some()));
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
        _: Normalization,
    ) -> Result<lazymatrix::NormalizationStats<f64>, ReadError> {
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
    let spec = Normalization::new(Centering::Mean, Scaling::Sd { df: 0 });
    let lazy = LazyMatrix::new(&source, spec).unwrap();
    assert_eq!(source.fused_calls.get(), 1);
    assert_eq!(lazy.centers(), Some([1.0, 2.0].as_slice()));
    assert_eq!(lazy.scales(), Some([1.0, f64::INFINITY].as_slice()));
    source.fail.set(true);
    assert!(matches!(LazyMatrix::new(&source, spec), Err(ReadError)));
}

#[test]
fn product_errors_propagate_before_normalization_and_retry_overwrites() {
    let source = Source {
        fail: Cell::new(true),
        fused_calls: Cell::new(0),
    };
    let lazy = LazyMatrix::from_parts(&source, Some(vec![1.0, 2.0]), Some(vec![2.0, 4.0]));
    let x = vec![2.0, 4.0];
    let mut out = vec![99.0; 2];
    assert_eq!(lazy.matvec_into(&x, &mut out), Err(ReadError));
    assert_eq!(out, [17.0, 99.0]);
    assert_eq!(lazy.mat_transpose_vec_into(&x, &mut out), Err(ReadError));
    assert_eq!(out, [17.0, 99.0]);
    assert_eq!(lazy.matvec(&x), Err(ReadError));
    assert_eq!(lazy.mat_transpose_vec(&x), Err(ReadError));
    source.fail.set(false);
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
        DefaultStats.normalization_stats(Normalization::default()),
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
            Scaling::QuantileRange(lazymatrix::QuantileRange::default()),
        ] {
            let spec = Normalization::new(center, scale);
            if spec != Normalization::default() {
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
    let lazy = LazyMatrix::new(&source, Normalization::default()).unwrap();
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
