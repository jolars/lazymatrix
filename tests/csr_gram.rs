#![cfg(feature = "sprs_all")]

use lazymatrix::{LazyMatrix, MatrixShape, MatrixWrite, SprsCsr, WeightedGramInto};
use sprs::CsMat;

struct Output {
    size: usize,
    values: Vec<f64>,
}

impl MatrixShape for Output {
    fn nrows(&self) -> usize {
        self.size
    }
    fn ncols(&self) -> usize {
        self.size
    }
}

impl MatrixWrite<f64> for Output {
    fn set(&mut self, row: usize, col: usize, value: f64) {
        self.values[row * self.size + col] = value;
    }
}

fn check(
    matrix: &CsMat<f64>,
    centers: Option<Vec<f64>>,
    scales: Option<Vec<f64>>,
    weights: &[f64],
) {
    let csr = SprsCsr::try_new(matrix.view()).unwrap();
    let lazy = LazyMatrix::from_parts(csr, centers.clone(), scales.clone());
    let p = matrix.cols();
    let mut out = Output {
        size: p,
        values: vec![99.0; p * p],
    };
    lazy.weighted_gram_into(&weights.to_vec(), &mut out)
        .unwrap();
    for j in 0..p {
        for k in 0..p {
            let mut expected = 0.0;
            for (i, &weight) in weights.iter().enumerate() {
                let normalize = |col| {
                    let raw = matrix.get(i, col).copied().unwrap_or(0.0);
                    (raw - centers.as_ref().map_or(0.0, |c| c[col]))
                        / scales.as_ref().map_or(1.0, |s| s[col])
                };
                expected += (normalize(j) * weight) * normalize(k);
            }
            let actual = out.values[j * p + k];
            if expected.is_nan() {
                assert!(actual.is_nan(), "({j}, {k}): {actual}");
            } else {
                assert_eq!(actual, expected, "({j}, {k})");
            }
        }
    }
}

#[test]
fn csr_gram_matches_direct_centered_products_across_tiles() {
    let n = 35;
    let p = 67;
    let mut indptr = vec![0];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for i in 0..n {
        if i % 5 != 0 {
            for j in 0..p {
                if (i + j) % 7 == 0 {
                    indices.push(j);
                    data.push((i + j) as f64 - 20.0);
                }
            }
        }
        indptr.push(data.len());
    }
    let matrix = CsMat::new((n, p), indptr, indices, data);
    let weights: Vec<_> = (0..n).map(|i| (i % 3) as f64 - 1.0).collect();
    for centers in [None, Some(vec![2.0; p])] {
        for scales in [None, Some(vec![-2.0; p])] {
            check(&matrix, centers.clone(), scales, &weights);
        }
    }
}

#[test]
fn csr_gram_preserves_centered_variation_and_nonfinite_values() {
    let offset = 1e12;
    let matrix = CsMat::new(
        (3, 2),
        vec![0, 2, 4, 6],
        vec![0, 1, 0, 1, 0, 1],
        vec![
            offset + 1.0,
            offset - 2.0,
            offset + 2.0,
            offset + 3.0,
            offset - 3.0,
            offset - 1.0,
        ],
    );
    check(&matrix, Some(vec![offset; 2]), None, &[1.0; 3]);
    let sparse = CsMat::new(
        (3, 2),
        vec![0, 1, 1, 2],
        vec![0, 1],
        vec![f64::INFINITY, 0.0],
    );
    check(&sparse, None, None, &[0.0, f64::INFINITY, -1.0]);
    check(
        &sparse,
        Some(vec![f64::NAN, 1.0]),
        Some(vec![1.0, -2.0]),
        &[1.0; 3],
    );
}

#[test]
fn csr_gram_accepts_sliced_views_and_validates_before_writing() {
    let matrix = CsMat::new(
        (4, 2),
        vec![0, 1, 3, 3, 4],
        vec![0, 0, 1, 1],
        vec![9.0, 2.0, 3.0, 7.0],
    );
    let csr = SprsCsr::try_new(matrix.slice_outer(1..3)).unwrap();
    let lazy = LazyMatrix::from_parts(csr, Some(vec![1.0, 2.0]), None);
    let mut out = Output {
        size: 2,
        values: vec![99.0; 4],
    };
    lazy.weighted_gram_into(&vec![1.0; 2], &mut out).unwrap();
    assert_eq!(out.values, vec![2.0, 3.0, 3.0, 5.0]);
    let before = out.values.clone();
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            lazy.weighted_gram_into(&vec![1.0], &mut out).unwrap();
        }))
        .is_err()
    );
    assert_eq!(out.values, before);
    let empty = CsMat::<f64>::new((0, 2), vec![0], vec![], vec![]);
    check(&empty, Some(vec![1.0; 2]), None, &[]);
}
