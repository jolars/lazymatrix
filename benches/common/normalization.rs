// Each harness uses a different subset of the shared fixture metadata.
#![allow(dead_code)]

use lazymatrix::{
    Centering, DotSlice, ElemDivAssign, Normalization, ScaledSubSlice, Scaling, SubScalarAssign,
    SumEntries, VectorOwned, VectorViewMut,
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::backend_aliases::sprs;

pub const SEED: u64 = 205;
pub const PAIRS: usize = 10;
pub const NORMALIZATIONS: [(&str, Normalization); 2] = [
    (
        "standardized",
        Normalization {
            center: Centering::Mean,
            scale: Scaling::Sd,
        },
    ),
    (
        "scaled",
        Normalization {
            center: Centering::None,
            scale: Scaling::Sd,
        },
    ),
];

pub trait BenchVector:
    VectorOwned<f64, Owned = Self>
    + VectorViewMut<f64>
    + Clone
    + ElemDivAssign<f64>
    + DotSlice<f64>
    + SubScalarAssign<f64>
    + SumEntries<f64>
    + ScaledSubSlice<f64>
{
}

impl<V> BenchVector for V where
    V: VectorOwned<f64, Owned = V>
        + VectorViewMut<f64>
        + Clone
        + ElemDivAssign<f64>
        + DotSlice<f64>
        + SubScalarAssign<f64>
        + SumEntries<f64>
        + ScaledSubSlice<f64>
{
}

pub struct Fixture {
    pub rows: usize,
    pub cols: usize,
    pub density: f64,
    pub sparse: sprs::CsMat<f64>,
}

pub fn fixtures() -> impl Iterator<Item = Fixture> {
    let smoke = std::env::var_os("LAZYMATRIX_BENCH_SMOKE").is_some();
    let sizes = if smoke {
        vec![(256, 16), (64, 64), (16, 256)]
    } else {
        // Match element counts to separate aspect ratio from total matrix size.
        vec![
            (2_048, 128),
            (512, 512),
            (128, 2_048),
            (8_192, 512),
            (2_048, 2_048),
            (512, 8_192),
            (16_384, 1_024),
            (4_096, 4_096),
            (1_024, 16_384),
        ]
    };
    let densities = if smoke {
        vec![0.01, 0.1, 1.0]
    } else {
        vec![0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    };
    selected_shapes(sizes).flat_map(move |(rows, cols)| {
        densities
            .clone()
            .into_iter()
            .map(move |density| Fixture::new(rows, cols, density))
    })
}

pub fn selected_shapes(sizes: Vec<(usize, usize)>) -> impl Iterator<Item = (usize, usize)> {
    let selected = std::env::var("LAZYMATRIX_BENCH_SHAPES").ok();
    sizes.into_iter().filter(move |(rows, cols)| {
        selected.as_ref().is_none_or(|selected| {
            selected
                .split(',')
                .any(|shape| shape == format!("{rows}x{cols}"))
        })
    })
}

impl Fixture {
    pub fn new(rows: usize, cols: usize, density: f64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(SEED);
        let mut offsets = Vec::with_capacity(cols + 1);
        let mut indices = Vec::new();
        let mut values = Vec::new();
        offsets.push(0);
        for col in 0..cols {
            for row in 0..rows {
                if rng.random::<f64>() < density {
                    indices.push(row);
                    // Nonzero means exercise the dense background introduced by centering.
                    values.push((0.5 + col as f64 / cols as f64) * rng.random_range(0.5..1.5));
                }
            }
            offsets.push(values.len());
        }
        indices.shrink_to_fit();
        values.shrink_to_fit();
        Self {
            rows,
            cols,
            density,
            sparse: sprs::CsMat::new_csc((rows, cols), offsets, indices, values),
        }
    }

    pub fn name(&self, backend: &str, normalization: &str) -> String {
        format!(
            "normalization/{backend}/{normalization}/{}x{}/{}",
            self.rows, self.cols, self.density
        )
    }

    pub fn source_bytes(&self) -> usize {
        self.sparse.nnz() * (size_of::<f64>() + size_of::<usize>())
            + (self.cols + 1) * size_of::<usize>()
    }

    pub fn empty_columns(&self) -> usize {
        self.sparse
            .outer_iterator()
            .filter(|col| col.nnz() == 0)
            .count()
    }

    pub fn vectors<V: BenchVector>(&self) -> (V, V, V, V) {
        let x = V::owned_from_fn(self.cols, |i| 0.5 + (i % 13) as f64 / 13.0);
        let u = V::owned_from_fn(self.rows, |i| 0.25 + (i % 17) as f64 / 17.0);
        let y = V::owned_from_fn(self.rows, |_| f64::NAN);
        let t = V::owned_from_fn(self.cols, |_| f64::NAN);
        (x, u, y, t)
    }

    #[cfg(feature = "faer_all")]
    pub fn faer(&self) -> crate::backend_aliases::faer::sparse::SparseColMat<usize, f64> {
        use crate::backend_aliases::faer::sparse::{SparseColMat, SymbolicSparseColMat};
        let symbolic = SymbolicSparseColMat::new_checked(
            self.rows,
            self.cols,
            self.sparse.indptr().raw_storage().to_vec(),
            None,
            self.sparse.indices().to_vec(),
        );
        SparseColMat::new(symbolic, self.sparse.data().to_vec())
    }
}
