use lazymatrix::{
    LazyMatrix, MatrixErrorType, MatrixShape, MatrixWrite, VectorView, WeightedGramInto,
    WeightedGramKernel,
};
use std::{cell::Cell, fmt};

struct Output([[f64; 2]; 2]);
impl MatrixShape for Output {
    fn nrows(&self) -> usize {
        2
    }
    fn ncols(&self) -> usize {
        2
    }
}
impl MatrixWrite<f64> for Output {
    fn set(&mut self, row: usize, column: usize, value: f64) {
        self.0[row][column] = value;
    }
}

#[derive(Debug, PartialEq)]
struct ReadError;
impl fmt::Display for ReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("injected failure")
    }
}
impl std::error::Error for ReadError {}
struct Source {
    fail: Cell<bool>,
    calls: Cell<usize>,
}
impl MatrixShape for Source {
    fn nrows(&self) -> usize {
        1
    }
    fn ncols(&self) -> usize {
        2
    }
}
impl MatrixErrorType for Source {
    type Error = ReadError;
}
impl WeightedGramKernel<f64> for Source {
    fn weighted_gram_normalized_into<W, O>(
        &self,
        weights: &W,
        centers: Option<&[f64]>,
        scales: Option<&[f64]>,
        out: &mut O,
    ) -> Result<(), ReadError>
    where
        W: VectorView<f64> + ?Sized,
        O: MatrixWrite<f64> + ?Sized,
    {
        self.calls.set(self.calls.get() + 1);
        assert_eq!(centers, Some([1.0, 2.0].as_slice()));
        assert_eq!(scales, Some([2.0, -4.0].as_slice()));
        out.set(0, 0, 17.0);
        if self.fail.get() {
            return Err(ReadError);
        }
        let x = [2.0, -1.0];
        for i in 0..2 {
            for j in 0..2 {
                out.set(i, j, (x[i] * weights.get(0)) * x[j]);
            }
        }
        Ok(())
    }
}

#[test]
fn gram_forwards_normalization_and_preserves_partial_output_on_error() {
    let source = Source {
        fail: Cell::new(true),
        calls: Cell::new(0),
    };
    let borrowed = &source;
    let lazy =
        LazyMatrix::from_column_parts(&borrowed, Some(vec![1.0, 2.0]), Some(vec![2.0, -4.0]));
    let mut out = Output([[99.0; 2]; 2]);
    assert_eq!(lazy.weighted_gram_into(&[3.0], &mut out), Err(ReadError));
    assert_eq!(out.0, [[17.0, 99.0], [99.0, 99.0]]);
    source.fail.set(false);
    WeightedGramInto::weighted_gram_into(&&lazy, &[3.0], &mut out).unwrap();
    assert_eq!(out.0, [[12.0, -6.0], [-6.0, 3.0]]);
    assert_eq!(source.calls.get(), 2);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            lazy.weighted_gram_into(&[], &mut out).unwrap();
        }))
        .is_err()
    );
    assert_eq!(source.calls.get(), 2);
}
