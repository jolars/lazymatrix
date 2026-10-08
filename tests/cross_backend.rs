#![cfg(any(
    feature = "faer_all",
    feature = "nalgebra_all",
    feature = "ndarray_all",
    feature = "sprs_all"
))]
//! Agreement between every enabled backend release, including versions of the same backend.

#[path = "common/backend_aliases.rs"]
mod backend_aliases;
#[path = "common/runner.rs"]
mod common;

use common::{TestMatrix, assert_close, random_matrix, random_vec};
use lazymatrix::{Centering, LazyMatrix, MatTransposeVec, MatVec, Normalization, Scaling};

type Products = (Vec<f64>, Vec<f64>);
type ProductFn = fn(&TestMatrix, Normalization, &[f64], &[f64]) -> Products;

#[cfg(feature = "faer_all")]
macro_rules! faer_adapter {
    ($name:ident, $backend:ident) => {
        mod $name {
            use super::*;
            use crate::backend_aliases::$backend as faer;
            pub(super) fn materialized<O: lazymatrix::MatrixWrite<f64>>(out: &mut O) {
                let matrix = faer::sparse::SparseColMat::try_new_from_triplets(
                    3,
                    2,
                    &[
                        faer::sparse::Triplet::new(0, 0, 1.0),
                        faer::sparse::Triplet::new(1, 0, 2.0),
                        faer::sparse::Triplet::new(1, 1, 3.0),
                        faer::sparse::Triplet::new(2, 1, 6.0),
                    ],
                )
                .unwrap();
                LazyMatrix::from_parts(matrix, Some(vec![1.0, 3.0]), Some(vec![2.0, 6.0]))
                    .to_eager_into(out)
                    .unwrap();
            }
            pub(super) fn products(
                tm: &TestMatrix,
                spec: Normalization,
                v: &[f64],
                u: &[f64],
            ) -> Products {
                use faer::Col;
                use faer::sparse::{SparseColMat, Triplet};

                let t: Vec<Triplet<usize, usize, f64>> = tm
                    .triplets
                    .iter()
                    .map(|&(r, c, v)| Triplet::new(r, c, v))
                    .collect();
                let matrix = SparseColMat::try_new_from_triplets(tm.nrows, tm.ncols, &t).unwrap();
                let lazy = LazyMatrix::new(matrix, spec).unwrap();
                let y = lazy.matvec(&Col::from_fn(v.len(), |i| v[i])).unwrap();
                let z = lazy
                    .mat_transpose_vec(&Col::from_fn(u.len(), |i| u[i]))
                    .unwrap();
                (
                    (0..y.nrows()).map(|i| y[i]).collect(),
                    (0..z.nrows()).map(|i| z[i]).collect(),
                )
            }
            #[cfg(feature = "ndarray_all")]
            pub(super) fn gram<O: lazymatrix::MatrixWrite<f32>>(out: &mut O) {
                use lazymatrix::WeightedGramInto;
                let matrix = faer::sparse::SparseColMat::try_new_from_triplets(
                    2,
                    2,
                    &[
                        faer::sparse::Triplet::new(0, 0, 1.0_f32),
                        faer::sparse::Triplet::new(1, 0, 2.0),
                        faer::sparse::Triplet::new(1, 1, 3.0),
                    ],
                )
                .unwrap();
                let lazy =
                    LazyMatrix::from_parts(matrix, Some(vec![0.5, 1.0]), Some(vec![-2.0, 3.0]));
                lazy.weighted_gram_into(&[2.0_f32, -1.0], out).unwrap();
            }
        }
    };
}
#[cfg(feature = "faer_v0_22")]
faer_adapter!(faer_0_22, faer_0_22);
#[cfg(feature = "faer_v0_23")]
faer_adapter!(faer_0_23, faer_0_23);
#[cfg(feature = "faer_v0_24")]
faer_adapter!(faer_0_24, faer_0_24);

#[cfg(feature = "nalgebra_all")]
macro_rules! nalgebra_adapter {
    ($name:ident, $backend:ident, $sparse:ident) => {
        mod $name {
            use super::*;
            use crate::backend_aliases::$backend as nalgebra;
            use crate::backend_aliases::$sparse as nalgebra_sparse;
            pub(super) fn materialized<O: lazymatrix::MatrixWrite<f64>>(out: &mut O) {
                let matrix = nalgebra_sparse::CscMatrix::try_from_csc_data(
                    3,
                    2,
                    vec![0, 2, 4],
                    vec![0, 1, 1, 2],
                    vec![1.0, 2.0, 3.0, 6.0],
                )
                .unwrap();
                LazyMatrix::from_parts(matrix, Some(vec![1.0, 3.0]), Some(vec![2.0, 6.0]))
                    .to_eager_into(out)
                    .unwrap();
            }
            pub(super) fn products(
                tm: &TestMatrix,
                spec: Normalization,
                v: &[f64],
                u: &[f64],
            ) -> Products {
                use nalgebra::DVector;
                use nalgebra_sparse::{CooMatrix, CscMatrix};

                let mut coo = CooMatrix::new(tm.nrows, tm.ncols);
                for &(r, c, v) in &tm.triplets {
                    coo.push(r, c, v);
                }
                let lazy = LazyMatrix::new(CscMatrix::from(&coo), spec).unwrap();
                (
                    lazy.matvec(&DVector::from_column_slice(v))
                        .unwrap()
                        .as_slice()
                        .to_vec(),
                    lazy.mat_transpose_vec(&DVector::from_column_slice(u))
                        .unwrap()
                        .as_slice()
                        .to_vec(),
                )
            }
            #[cfg(feature = "ndarray_all")]
            pub(super) fn gram<O: lazymatrix::MatrixWrite<f32>>(out: &mut O) {
                use lazymatrix::WeightedGramInto;
                let matrix = nalgebra_sparse::CscMatrix::try_from_csc_data(
                    2,
                    2,
                    vec![0, 2, 3],
                    vec![0, 1, 1],
                    vec![1.0_f32, 2.0, 3.0],
                )
                .unwrap();
                let lazy =
                    LazyMatrix::from_parts(matrix, Some(vec![0.5, 1.0]), Some(vec![-2.0, 3.0]));
                lazy.weighted_gram_into(&[2.0_f32, -1.0], out).unwrap();
            }
        }
    };
}
#[cfg(feature = "nalgebra_v0_32")]
nalgebra_adapter!(nalgebra_0_32, nalgebra_0_32, nalgebra_sparse_0_9);
#[cfg(feature = "nalgebra_v0_33")]
nalgebra_adapter!(nalgebra_0_33, nalgebra_0_33, nalgebra_sparse_0_10);
#[cfg(feature = "nalgebra_v0_34")]
nalgebra_adapter!(nalgebra_0_34, nalgebra_0_34, nalgebra_sparse_0_11);
#[cfg(feature = "nalgebra_v0_35")]
nalgebra_adapter!(nalgebra_0_35, nalgebra_0_35, nalgebra_sparse_0_12);

#[cfg(feature = "ndarray_all")]
macro_rules! ndarray_adapter {
    ($name:ident, $backend:ident) => {
        mod $name {
            use super::*;
            use crate::backend_aliases::$backend as ndarray;
            pub(super) fn materialized<O: lazymatrix::MatrixWrite<f64>>(out: &mut O) {
                let matrix = ndarray::array![[1.0, 0.0], [2.0, 3.0], [0.0, 6.0]];
                LazyMatrix::from_parts(matrix, Some(vec![1.0, 3.0]), Some(vec![2.0, 6.0]))
                    .to_eager_into(out)
                    .unwrap();
            }
            pub(super) fn products(
                tm: &TestMatrix,
                spec: Normalization,
                v: &[f64],
                u: &[f64],
            ) -> Products {
                use ndarray::{Array1, Array2};

                let matrix = Array2::from_shape_fn((tm.nrows, tm.ncols), |(i, j)| tm.dense[i][j]);
                let lazy = LazyMatrix::new(matrix, spec).unwrap();
                (
                    lazy.matvec(&Array1::from_vec(v.to_vec())).unwrap().to_vec(),
                    lazy.mat_transpose_vec(&Array1::from_vec(u.to_vec()))
                        .unwrap()
                        .to_vec(),
                )
            }
            #[cfg(feature = "ndarray_all")]
            pub(super) fn gram<O: lazymatrix::MatrixWrite<f32>>(out: &mut O) {
                use lazymatrix::WeightedGramInto;
                let matrix = ndarray::array![[1.0_f32, 0.0], [2.0, 3.0]];
                let lazy =
                    LazyMatrix::from_parts(matrix, Some(vec![0.5, 1.0]), Some(vec![-2.0, 3.0]));
                lazy.weighted_gram_into(&[2.0_f32, -1.0], out).unwrap();
            }
        }
    };
}
#[cfg(feature = "ndarray_v0_15")]
ndarray_adapter!(ndarray_0_15, ndarray_0_15);
#[cfg(feature = "ndarray_v0_16")]
ndarray_adapter!(ndarray_0_16, ndarray_0_16);
#[cfg(feature = "ndarray_v0_17")]
ndarray_adapter!(ndarray_0_17, ndarray_0_17);

#[cfg(feature = "sprs_all")]
mod sprs {
    use super::*;
    use crate::backend_aliases::sprs;
    #[cfg(any(
        feature = "faer_all",
        feature = "nalgebra_all",
        feature = "ndarray_all"
    ))]
    pub(super) fn materialized<O: lazymatrix::MatrixWrite<f64>>(out: &mut O) {
        let matrix = sprs::CsMat::new_csc(
            (3, 2),
            vec![0, 2, 4],
            vec![0, 1, 1, 2],
            vec![1.0, 2.0, 3.0, 6.0],
        );
        for matrix in [&matrix, &matrix.to_csr()] {
            LazyMatrix::from_parts(matrix, Some(vec![1.0, 3.0]), Some(vec![2.0, 6.0]))
                .to_eager_into(out)
                .unwrap();
        }
    }
    pub(super) fn products(tm: &TestMatrix, spec: Normalization, v: &[f64], u: &[f64]) -> Products {
        let mut triplets = sprs::TriMat::new((tm.nrows, tm.ncols));
        for &(row, col, value) in &tm.triplets {
            triplets.add_triplet(row, col, value);
        }
        let lazy = LazyMatrix::new(triplets.to_csc::<usize>(), spec).unwrap();
        (
            lazy.matvec(&v.to_vec()).unwrap(),
            lazy.mat_transpose_vec(&u.to_vec()).unwrap(),
        )
    }
    #[cfg(feature = "ndarray_all")]
    pub(super) fn gram<O: lazymatrix::MatrixWrite<f32>>(out: &mut O) {
        use lazymatrix::WeightedGramInto;
        let matrix = lazymatrix::SprsCsc::try_new(sprs::CsMat::new_csc(
            (2, 2),
            vec![0, 2, 3],
            vec![0, 1, 1],
            vec![1.0_f32, 2.0, 3.0],
        ))
        .unwrap();
        let lazy = LazyMatrix::from_parts(matrix, Some(vec![0.5, 1.0]), Some(vec![-2.0, 3.0]));
        lazy.weighted_gram_into(&[2.0_f32, -1.0], out).unwrap();
    }
}

fn check_agreement(
    left: impl Fn(&TestMatrix, Normalization, &[f64], &[f64]) -> Products,
    right: impl Fn(&TestMatrix, Normalization, &[f64], &[f64]) -> Products,
) {
    let tm = random_matrix(42, 14, 9, 0.4);
    let v = random_vec(43, tm.ncols);
    let u = random_vec(44, tm.nrows);

    for center in [Centering::None, Centering::Mean, Centering::Min] {
        for scale in [
            Scaling::None,
            Scaling::Sd,
            Scaling::L1,
            Scaling::L2,
            Scaling::MaxAbs,
            Scaling::Range,
        ] {
            let spec = Normalization::new(center, scale);

            let (left_y, left_z) = left(&tm, spec, &v, &u);
            let (right_y, right_z) = right(&tm, spec, &v, &u);
            assert_close(&left_y, &right_y, 1e-10);
            assert_close(&left_z, &right_z, 1e-10);
        }
    }
}

#[test]
fn all_enabled_releases_agree() {
    let backends: &[(&str, ProductFn)] = &[
        #[cfg(feature = "faer_v0_22")]
        ("faer_0_22", faer_0_22::products),
        #[cfg(feature = "faer_v0_23")]
        ("faer_0_23", faer_0_23::products),
        #[cfg(feature = "faer_v0_24")]
        ("faer_0_24", faer_0_24::products),
        #[cfg(feature = "nalgebra_v0_32")]
        ("nalgebra_0_32", nalgebra_0_32::products),
        #[cfg(feature = "nalgebra_v0_33")]
        ("nalgebra_0_33", nalgebra_0_33::products),
        #[cfg(feature = "nalgebra_v0_34")]
        ("nalgebra_0_34", nalgebra_0_34::products),
        #[cfg(feature = "nalgebra_v0_35")]
        ("nalgebra_0_35", nalgebra_0_35::products),
        #[cfg(feature = "ndarray_v0_15")]
        ("ndarray_0_15", ndarray_0_15::products),
        #[cfg(feature = "ndarray_v0_16")]
        ("ndarray_0_16", ndarray_0_16::products),
        #[cfg(feature = "ndarray_v0_17")]
        ("ndarray_0_17", ndarray_0_17::products),
        #[cfg(feature = "sprs_v0_11")]
        ("sprs", sprs::products),
    ];
    for (i, &(left_name, left)) in backends.iter().enumerate() {
        for &(right_name, right) in &backends[i + 1..] {
            eprintln!("Comparing {left_name} and {right_name}");
            check_agreement(left, right);
        }
    }
}

#[cfg(feature = "ndarray_all")]
macro_rules! gram_output_suite {
    ($name:ident, $backend:ident) => {
        mod $name {
            use super::*;
            use crate::backend_aliases::$backend as ndarray;

            #[test]
            fn every_release_writes_grams_into_ndarray() {
                use lazymatrix::WeightedGramInto;
                let dense = ndarray::array![[1.0_f32, 0.0], [2.0, 3.0]];
                let lazy = LazyMatrix::from_parts(
                    dense.view(),
                    Some(vec![0.5, 1.0]),
                    Some(vec![-2.0, 3.0]),
                );
                let mut expected = ndarray::Array2::zeros((2, 2));
                lazy.weighted_gram_into(&[2.0_f32, -1.0], &mut expected.view_mut())
                    .unwrap();
                type GramFn = fn(&mut ndarray::Array2<f32>);
                let backends: &[(&str, GramFn)] = &[
                    #[cfg(feature = "faer_v0_22")]
                    ("faer_0_22", faer_0_22::gram),
                    #[cfg(feature = "faer_v0_23")]
                    ("faer_0_23", faer_0_23::gram),
                    #[cfg(feature = "faer_v0_24")]
                    ("faer_0_24", faer_0_24::gram),
                    #[cfg(feature = "nalgebra_v0_32")]
                    ("nalgebra_0_32", nalgebra_0_32::gram),
                    #[cfg(feature = "nalgebra_v0_33")]
                    ("nalgebra_0_33", nalgebra_0_33::gram),
                    #[cfg(feature = "nalgebra_v0_34")]
                    ("nalgebra_0_34", nalgebra_0_34::gram),
                    #[cfg(feature = "nalgebra_v0_35")]
                    ("nalgebra_0_35", nalgebra_0_35::gram),
                    #[cfg(feature = "ndarray_v0_15")]
                    ("ndarray_0_15", ndarray_0_15::gram),
                    #[cfg(feature = "ndarray_v0_16")]
                    ("ndarray_0_16", ndarray_0_16::gram),
                    #[cfg(feature = "ndarray_v0_17")]
                    ("ndarray_0_17", ndarray_0_17::gram),
                    #[cfg(feature = "sprs_v0_11")]
                    ("sprs", sprs::gram),
                ];
                for &(name, gram) in backends {
                    let mut out = ndarray::Array2::from_elem((2, 2), f32::NAN);
                    gram(&mut out);
                    for (&a, &b) in out.iter().zip(&expected) {
                        assert!((a - b).abs() < 1e-6, "{name}: {a} != {b}");
                    }
                }
            }
        }
    };
}

#[cfg(feature = "ndarray_v0_15")]
gram_output_suite!(gram_ndarray_0_15, ndarray_0_15);

#[cfg(feature = "ndarray_v0_16")]
gram_output_suite!(gram_ndarray_0_16, ndarray_0_16);

#[cfg(feature = "ndarray_v0_17")]
gram_output_suite!(gram_ndarray_0_17, ndarray_0_17);

#[cfg(any(
    feature = "faer_all",
    feature = "nalgebra_all",
    feature = "ndarray_all"
))]
fn check_materialization_destination<
    O: lazymatrix::MatrixWrite<f64> + lazymatrix::RawColumns<f64>,
>(
    out: &mut O,
) {
    use lazymatrix::RawColumn;
    fn check<O: lazymatrix::RawColumns<f64>>(out: &O) {
        for col in 0..2 {
            out.raw_column(col).for_each_stored(|row, value| {
                assert_eq!(value, [[0.0, -0.5], [0.5, 0.0], [-0.5, 0.5]][row][col]);
            });
        }
    }
    #[cfg(feature = "faer_v0_22")]
    {
        faer_0_22::materialized(out);
        check(out);
    }
    #[cfg(feature = "faer_v0_23")]
    {
        faer_0_23::materialized(out);
        check(out);
    }
    #[cfg(feature = "faer_v0_24")]
    {
        faer_0_24::materialized(out);
        check(out);
    }
    #[cfg(feature = "nalgebra_v0_32")]
    {
        nalgebra_0_32::materialized(out);
        check(out);
    }
    #[cfg(feature = "nalgebra_v0_33")]
    {
        nalgebra_0_33::materialized(out);
        check(out);
    }
    #[cfg(feature = "nalgebra_v0_34")]
    {
        nalgebra_0_34::materialized(out);
        check(out);
    }
    #[cfg(feature = "nalgebra_v0_35")]
    {
        nalgebra_0_35::materialized(out);
        check(out);
    }
    #[cfg(feature = "ndarray_v0_15")]
    {
        ndarray_0_15::materialized(out);
        check(out);
    }
    #[cfg(feature = "ndarray_v0_16")]
    {
        ndarray_0_16::materialized(out);
        check(out);
    }
    #[cfg(feature = "ndarray_v0_17")]
    {
        ndarray_0_17::materialized(out);
        check(out);
    }
    #[cfg(feature = "sprs_all")]
    {
        sprs::materialized(out);
        check(out);
    }
}

#[cfg(feature = "faer_v0_22")]
#[test]
fn materialization_to_faer_0_22() {
    let mut out =
        <backend_aliases::faer_0_22::Mat<f64> as lazymatrix::MatrixOwned<f64>>::zeros(3, 2);
    check_materialization_destination(&mut out);
}

#[cfg(feature = "faer_v0_23")]
#[test]
fn materialization_to_faer_0_23() {
    let mut out =
        <backend_aliases::faer_0_23::Mat<f64> as lazymatrix::MatrixOwned<f64>>::zeros(3, 2);
    check_materialization_destination(&mut out);
}

#[cfg(feature = "faer_v0_24")]
#[test]
fn materialization_to_faer_0_24() {
    let mut out =
        <backend_aliases::faer_0_24::Mat<f64> as lazymatrix::MatrixOwned<f64>>::zeros(3, 2);
    check_materialization_destination(&mut out);
}

#[cfg(feature = "nalgebra_v0_32")]
#[test]
fn materialization_to_nalgebra_0_32() {
    let mut out =
        <backend_aliases::nalgebra_0_32::DMatrix<f64> as lazymatrix::MatrixOwned<f64>>::zeros(3, 2);
    check_materialization_destination(&mut out);
}

#[cfg(feature = "nalgebra_v0_33")]
#[test]
fn materialization_to_nalgebra_0_33() {
    let mut out =
        <backend_aliases::nalgebra_0_33::DMatrix<f64> as lazymatrix::MatrixOwned<f64>>::zeros(3, 2);
    check_materialization_destination(&mut out);
}

#[cfg(feature = "nalgebra_v0_34")]
#[test]
fn materialization_to_nalgebra_0_34() {
    let mut out =
        <backend_aliases::nalgebra_0_34::DMatrix<f64> as lazymatrix::MatrixOwned<f64>>::zeros(3, 2);
    check_materialization_destination(&mut out);
}

#[cfg(feature = "nalgebra_v0_35")]
#[test]
fn materialization_to_nalgebra_0_35() {
    let mut out =
        <backend_aliases::nalgebra_0_35::DMatrix<f64> as lazymatrix::MatrixOwned<f64>>::zeros(3, 2);
    check_materialization_destination(&mut out);
}

#[cfg(feature = "ndarray_v0_15")]
#[test]
fn materialization_to_ndarray_0_15() {
    let mut out =
        <backend_aliases::ndarray_0_15::Array2<f64> as lazymatrix::MatrixOwned<f64>>::zeros(3, 2);
    check_materialization_destination(&mut out);
}

#[cfg(feature = "ndarray_v0_16")]
#[test]
fn materialization_to_ndarray_0_16() {
    let mut out =
        <backend_aliases::ndarray_0_16::Array2<f64> as lazymatrix::MatrixOwned<f64>>::zeros(3, 2);
    check_materialization_destination(&mut out);
}

#[cfg(feature = "ndarray_v0_17")]
#[test]
fn materialization_to_ndarray_0_17() {
    let mut out =
        <backend_aliases::ndarray_0_17::Array2<f64> as lazymatrix::MatrixOwned<f64>>::zeros(3, 2);
    check_materialization_destination(&mut out);
}
