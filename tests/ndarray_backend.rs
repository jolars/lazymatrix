#![cfg(feature = "ndarray_all")]
//! Verification of ndarray arrays and strided views against the dense oracle.

#[path = "common/backend_aliases.rs"]
mod backend_aliases;

#[path = "common/runner.rs"]
mod common;

#[cfg(any(feature = "ndarray_v0_15", feature = "ndarray_v0_16"))]
macro_rules! no_extra_tests {
    () => {};
}

#[cfg(feature = "ndarray_v0_17")]
macro_rules! mmap_tests {
    () => {
        #[test]
        fn memory_mapped_npy_views_borrow_and_match_owned_arrays() {
            use ndarray_npy::{ViewNpyExt, WriteNpyExt};
            let tm = common::random_matrix(91, 7, 5, 0.6);
            for build in [build, build_fortran] {
                let array = build(&tm);
                let mut file = tempfile::tempfile().unwrap();
                array.write_npy(&mut file).unwrap();
                // This private file is never modified while its mapping exists.
                let mapping = unsafe { memmap2::Mmap::map(&file).unwrap() };
                let view = ArrayView2::<f64>::view_npy(&mapping).unwrap();
                let start = mapping.as_ptr() as usize;
                let data = view.as_ptr() as usize;
                assert!(
                    data >= start && data + view.len() * size_of::<f64>() <= start + mapping.len()
                );
                assert_eq!(view.strides(), array.strides());
                check_matrix_view(view);
            }
        }
    };
}

macro_rules! backend_suite {
    ($name:ident, $backend:ident, $extra:ident) => {
        mod $name {
            use crate::backend_aliases::$backend as ndarray;
            use crate::common;
            use common::{TestMatrix, assert_close, dense_matvec, dense_tmatvec, materialize};
            use lazymatrix::{
                Centering, ColumnStats, DotProduct, DotSlice, ElemDivAssign, L2Norm, LazyMatrix,
                MatTransposeVec, MatTransposeVecInto, MatVec, MatVecInto, Normalization, ScaleAssign,
                ScaledAddAssign, ScaledSubSlice, Scaling, SubScalarAssign, SumEntries, VectorView,
                VectorViewMut,
            };
            use ndarray::{Array1, Array2, ArrayView2, ShapeBuilder, array, s};

            fn build(tm: &TestMatrix) -> Array2<f64> {
                Array2::from_shape_fn((tm.nrows, tm.ncols), |(i, j)| tm.dense[i][j])
            }

            fn build_fortran(tm: &TestMatrix) -> Array2<f64> {
                Array2::from_shape_fn((tm.nrows, tm.ncols).f(), |(i, j)| tm.dense[i][j])
            }

            #[test]
            fn intercept_accepts_strided_vectors_and_outputs() {
                use lazymatrix::{WithIntercept, WeightedGramInto, WeightedColumnSumsInto, MatVecScaledInto, MatTransposeVecScaledInto};
                let storage = Array2::from_shape_fn((6, 4), |(i, j)| (i + j) as f64);
                let matrix = storage.slice(s![..;2, ..;2]);
                let lazy = LazyMatrix::from_parts(matrix, Some(vec![1.0, 2.0]), Some(vec![2.0, -1.0]));
                let augmented = WithIntercept::new(&lazy);
                let input = array![2.0, 99.0, 3.0, 99.0, -1.0, 99.0];
                let x = input.slice(s![..;2]);
                let mut output = Array1::from_elem(6, 99.0);
                augmented.matvec_into(&x, &mut output.slice_mut(s![..;2])).unwrap();
                assert_close(&output.slice(s![..;2]).to_vec(), &augmented.matvec(&x.to_owned()).unwrap().to_vec(), 1e-10);
                assert!(output.slice(s![1..;2]).iter().all(|&x| x == 99.0));
                augmented.mat_transpose_vec_into(&x, &mut output.slice_mut(s![..;-2])).unwrap();
                assert_close(&output.slice(s![..;-2]).to_vec(), &augmented.mat_transpose_vec(&x.to_owned()).unwrap().to_vec(), 1e-10);
                output.fill(99.0);
                let expected = augmented.matvec(&x.to_owned()).unwrap();
                augmented.matvec_scaled_into(2.0, &x, -1.0, &mut output.slice_mut(s![..;2])).unwrap();
                assert_close(&output.slice(s![..;2]).to_vec(), &expected.mapv(|v| 2.0 * v - 99.0).to_vec(), 1e-10);
                assert!(output.slice(s![1..;2]).iter().all(|&v| v == 99.0));
                let mut scratch = Array1::zeros(2);
                output.fill(99.0);
                augmented.matvec_scaled_with_workspace(2.0, &x, -1.0, &mut output.slice_mut(s![..;2]), &mut scratch).unwrap();
                assert_close(&output.slice(s![..;2]).to_vec(), &expected.mapv(|v| 2.0 * v - 99.0).to_vec(), 1e-10);
                assert!(output.slice(s![1..;2]).iter().all(|&v| v == 99.0));
                output.fill(99.0);
                let expected = augmented.mat_transpose_vec(&x.to_owned()).unwrap();
                augmented.mat_transpose_vec_scaled_with_workspace(2.0, &x, -1.0, &mut output.slice_mut(s![..;-2]), &mut scratch).unwrap();
                assert_close(&output.slice(s![..;-2]).to_vec(), &expected.mapv(|v| 2.0 * v - 99.0).to_vec(), 1e-10);
                output.fill(99.0);
                augmented.mat_transpose_vec_scaled_into(1.0, &x, 0.0, &mut output.slice_mut(s![..;-2])).unwrap();
                let mut sums = Array1::from_elem(6, 99.0);
                augmented.weighted_column_sums_into(&x, &mut sums.slice_mut(s![..;-2])).unwrap();
                assert_close(&sums.slice(s![..;-2]).to_vec(), &output.slice(s![..;-2]).to_vec(), 1e-10);
                assert!(sums.slice(s![..;2]).iter().all(|&x| x == 99.0));
                let mut gram = Array2::from_elem((6, 6), 99.0);
                augmented.weighted_gram_into(&x, &mut gram.slice_mut(s![..;-2, ..;2])).unwrap();
                let actual = common::GramOutput(gram.slice(s![..;-2, ..;2]).rows().into_iter().map(|r| r.to_vec()).collect());
                let dense: Vec<_> = matrix.rows().into_iter().map(|r| r.to_vec()).collect();
                let normalized: Vec<Vec<_>> = materialize(&dense, lazy.centers(), lazy.scales()).iter().map(|r| std::iter::once(1.0).chain(r.iter().copied()).collect()).collect();
                common::assert_gram(&actual, &normalized, &x.to_vec());
                assert!(gram.slice(s![..;2, ..]).iter().all(|&x| x == 99.0));
                assert!(gram.slice(s![.., 1..;2]).iter().all(|&x| x == 99.0));
                let small = array![[1.0_f32], [3.0]];
                let small = WithIntercept::new(LazyMatrix::from_parts(small, Some(vec![1.0_f32]), None));
                let mut gram = Array2::zeros((2, 2));
                small.weighted_gram_into(&[2.0_f32, -1.0], &mut gram).unwrap();
                assert_eq!(gram, array![[1.0, -2.0], [-2.0, -4.0]]);
                assert_eq!(small.matvec(&array![2.0, 3.0]).unwrap(), array![2.0, 8.0]);
            }

            #[test]
            fn ndarray_backend_suite() {
                for build in [build, build_fortran] {
                    common::run_gram_suite(build);
                    common::run_fused_suite(build, |v| Array1::from_vec(v.to_vec()));
                    common::run_backend_suite(build, |v| Array1::from_vec(v.to_vec()), |v| v.to_vec());
                    common::run_logical_columns_suite(build);
                }
            }

            #[test]
            fn ndarray_gram_strides_tiles_and_f32() {
                use lazymatrix::WeightedGramInto;
                let storage = Array2::from_shape_fn((600, 74), |(i, j)| ((i + j * 3) % 13) as f64 + 1e9);
                let matrix = storage.slice(s![..;-2, ..;2]);
                let weight_storage = Array1::from_shape_fn(600, |i| (i % 7) as f64 - 3.0);
                let weights = weight_storage.slice(s![..;-2]);
                let lazy = LazyMatrix::from_parts(matrix, Some(vec![1e9; 37]), Some(vec![-2.0; 37]));
                let mut output = Array2::from_elem((74, 74), -99.0);
                lazy.weighted_gram_into(&weights, &mut output.slice_mut(s![..;-2, ..;2]))
                    .unwrap();
                let actual = common::GramOutput(
                    output
                        .slice(s![..;-2, ..;2])
                        .rows()
                        .into_iter()
                        .map(|r| r.to_vec())
                        .collect(),
                );
                let dense: Vec<_> = matrix.rows().into_iter().map(|r| r.to_vec()).collect();
                common::assert_gram(
                    &actual,
                    &materialize(&dense, lazy.centers(), lazy.scales()),
                    &weights.to_vec(),
                );
                assert!(output.slice(s![..;2, ..]).iter().all(|&x| x == -99.0));
                assert!(output.slice(s![.., 1..;2]).iter().all(|&x| x == -99.0));

                let matrix = array![[1.0_f32, 0.0], [2.0, 3.0]];
                let mut out = Array2::from_elem((2, 2).f(), f32::NAN);
                matrix
                    .weighted_gram_into(&[2.0_f32, -1.0], &mut out)
                    .unwrap();
                assert_eq!(out, array![[-2.0, -6.0], [-6.0, -9.0]]);
                let broadcast = matrix.row(0);
                let broadcast = broadcast.broadcast((3, 2)).unwrap();
                broadcast
                    .weighted_gram_into(&[1.0_f32; 3], &mut out)
                    .unwrap();
                assert_eq!(out, array![[3.0, 0.0], [0.0, 0.0]]);
            }

            fn check_matrix_view(matrix: ArrayView2<'_, f64>) {
                let dense: Vec<Vec<_>> = matrix.rows().into_iter().map(|row| row.to_vec()).collect();
                common::check_fused_operator(&matrix, &dense, matrix.ncols(), &|v| Array1::from_vec(v.to_vec()));
                let v = Array1::from_vec(common::random_vec(51, matrix.ncols()));
                let u = Array1::from_vec(common::random_vec(52, matrix.nrows()));
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
                        let lazy = LazyMatrix::new(matrix, spec).unwrap();
                        let reference = LazyMatrix::new(matrix.to_owned(), spec).unwrap();
                        assert_eq!(lazy.centers(), reference.centers());
                        assert_eq!(lazy.scales(), reference.scales());
                        let normalized = materialize(&dense, lazy.centers(), lazy.scales());
                        assert_close(
                            &lazy.matvec(&v).unwrap().to_vec(),
                            &dense_matvec(&normalized, &v.to_vec()),
                            1e-10,
                        );
                        assert_close(
                            &lazy.mat_transpose_vec(&u).unwrap().to_vec(),
                            &dense_tmatvec(&normalized, &u.to_vec()),
                            1e-10,
                        );
                        for j in 0..matrix.ncols() {
                            let column = lazy.column(j);
                            assert_eq!(column.raw().as_ptr(), &matrix[(0, j)] as *const f64);
                            assert_eq!(column.raw().strides()[0], matrix.strides()[0]);
                            let expected: f64 = normalized.iter().zip(&u).map(|(row, u)| row[j] * u).sum();
                            approx::assert_abs_diff_eq!(column.dot(&u.view()), expected, epsilon = 1e-10);
                        }
                    }
                }
            }

            #[test]
            fn ndarray_matrix_views_preserve_strides_and_borrow_storage() {
                let mut storage = Array2::from_shape_fn((8, 6), |(i, j)| (i * 6 + j) as f64 - 10.0);
                check_matrix_view(storage.view());
                check_matrix_view(storage.t());
                check_matrix_view(storage.slice(s![1..;2, ..;2]));
                check_matrix_view(storage.slice(s![..;-1, ..;-1]));
                let row = array![1.0, 2.0, 3.0];
                check_matrix_view(row.broadcast((4, 3)).unwrap());

                let lazy = LazyMatrix::new(
                    storage.view_mut(),
                    Normalization::new(Centering::Mean, Scaling::L2),
                )
                .unwrap();
                assert_eq!(lazy.nrows(), 8);
                assert_eq!(lazy.column(0).len(), 8);
            }

            #[test]
            fn ndarray_strided_inputs_and_outputs_match_dense_oracle() {
                let matrix = array![[1.0, 0.0], [2.0, 3.0], [-1.0, 4.0]];
                let dense = vec![vec![1.0, 0.0], vec![2.0, 3.0], vec![-1.0, 4.0]];
                let input_storage = array![2.0, -99.0, -1.0];
                let input = input_storage.slice(s![..;2]);
                let row_storage = array![3.0, -99.0, 2.0, -99.0, 1.0];
                let rows = row_storage.slice(s![..;-2]);
                let weights = row_storage.slice(s![..;2]);

                let mut raw_output = Array1::from_elem(6, f64::NAN);
                matrix
                    .matvec_into(&input, &mut raw_output.slice_mut(s![..;-2]))
                    .unwrap();
                assert_close(
                    &raw_output.slice(s![..;-2]).to_vec(),
                    &[2.0, 1.0, -6.0],
                    1e-12,
                );

                for center in [false, true] {
                    for scale in [false, true] {
                        let lazy = LazyMatrix::from_parts(
                            matrix.view(),
                            center.then(|| vec![0.5, -1.0]),
                            scale.then(|| vec![2.0, 4.0]),
                        );
                        let normalized = materialize(&dense, lazy.centers(), lazy.scales());
                        let mut output = Array1::from_elem(6, -99.0);
                        output.slice_mut(s![..;-2]).fill(f64::NAN);
                        let owned_input = input.to_owned();
                        lazy.matvec_into(&owned_input, &mut output.slice_mut(s![..;-2]))
                            .unwrap();
                        assert_close(
                            &output.slice(s![..;-2]).to_vec(),
                            &dense_matvec(&normalized, &input.to_vec()),
                            1e-12,
                        );
                        assert_eq!(output.slice(s![..;2]).to_vec(), vec![-99.0; 3]);
                        assert_eq!(owned_input.to_vec(), input.to_vec());

                        let mut transpose_output = array![-99.0, f64::NAN, -99.0, f64::NAN];
                        lazy.mat_transpose_vec_into(&rows, &mut transpose_output.slice_mut(s![1..;2]))
                            .unwrap();
                        assert_close(
                            &transpose_output.slice(s![1..;2]).to_vec(),
                            &dense_tmatvec(&normalized, &rows.to_vec()),
                            1e-12,
                        );
                        assert_eq!(transpose_output.slice(s![..;2]).to_vec(), vec![-99.0; 2]);

                        let column = lazy.column(1);
                        let expected_dot: f64 = (0..3)
                            .map(|i| normalized[i][1] * rows[i] * weights[i])
                            .sum();
                        approx::assert_abs_diff_eq!(
                            column.weighted_dot(&rows, &weights),
                            expected_dot,
                            epsilon = 1e-12
                        );
                        let expected_norm: f64 = (0..3).map(|i| normalized[i][1].powi(2) * weights[i]).sum();
                        approx::assert_abs_diff_eq!(
                            column.weighted_norm_squared(&weights),
                            expected_norm,
                            epsilon = 1e-12
                        );
                        let mut destination = Array1::from_elem(6, 1.0);
                        column.scaled_add_to(-0.5, &mut destination.slice_mut(s![..;-2]));
                        let expected: Vec<_> = normalized.iter().map(|row| 1.0 - 0.5 * row[1]).collect();
                        assert_close(&destination.slice(s![..;-2]).to_vec(), &expected, 1e-12);
                        assert_eq!(destination.slice(s![..;2]).to_vec(), vec![1.0; 3]);
                    }
                }
            }

            #[test]
            fn ndarray_vector_traits_follow_logical_order() {
                let storage = array![3.0, -99.0, 4.0];
                let view = storage.slice(s![..;-2]);
                assert_eq!(VectorView::len(&view), 2);
                assert_eq!(VectorView::get(&view, 0), 4.0);
                assert_eq!(view.sum_entries(), 7.0);
                assert_eq!(view.dot_slice(&[2.0, 1.0]), 11.0);
                assert_eq!(DotProduct::dot(&view, &view), 25.0);
                assert_eq!(view.norm_l2(), 5.0);

                let mut destination = array![3.0, -99.0, 4.0];
                let mut other_storage = array![2.0, -99.0, 1.0];
                {
                    let mut view = destination.slice_mut(s![..;-2]);
                    view.scaled_add_assign(2.0, &other_storage.slice_mut(s![..;-2]));
                    view.scale_assign(2.0);
                    view.elem_div_assign(&[3.0, 7.0]);
                    view.sub_scalar_assign(1.0);
                    view.scaled_sub_slice(0.5, &[2.0, 4.0]);
                    assert_eq!(view.to_vec(), vec![2.0, -1.0]);
                    VectorViewMut::set(&mut view, 1, 8.0);
                }
                assert_eq!(destination.to_vec(), vec![8.0, -99.0, 2.0]);
            }

            #[test]
            fn ndarray_f32_normalization_and_products() {
                let matrix = array![[1.0_f32, 3.0], [3.0, 3.0]];
                let lazy = LazyMatrix::new(matrix, Normalization::new(Centering::Mean, Scaling::Sd)).unwrap();
                assert_eq!(lazy.centers(), Some([2.0, 3.0].as_slice()));
                assert_eq!(lazy.scales(), Some([1.0, 1.0].as_slice()));
                assert_eq!(
                    lazy.matvec(&array![2.0, 1.0]).unwrap().to_vec(),
                    vec![-2.0, 2.0]
                );
                assert_eq!(
                    lazy.mat_transpose_vec(&array![1.0, 2.0]).unwrap().to_vec(),
                    vec![1.0, 0.0]
                );
            }

            #[test]
            fn ndarray_infinite_statistics_preserve_ieee_values() {
                let matrix = array![[f64::INFINITY, f64::NEG_INFINITY], [1.0, 1.0]];
                assert_eq!(
                    matrix.col_means().unwrap(),
                    vec![f64::INFINITY, f64::NEG_INFINITY]
                );
                assert!(matrix.col_sds().unwrap().iter().all(|x| x.is_nan()));
                assert_eq!(matrix.col_maxabs().unwrap(), vec![f64::INFINITY; 2]);
                assert_eq!(matrix.col_ranges().unwrap(), vec![f64::INFINITY; 2]);
            }

            $extra!();
        }
    };
}

#[cfg(feature = "ndarray_v0_15")]
backend_suite!(v0_15, ndarray_0_15, no_extra_tests);

#[cfg(feature = "ndarray_v0_16")]
backend_suite!(v0_16, ndarray_0_16, no_extra_tests);

#[cfg(feature = "ndarray_v0_17")]
backend_suite!(v0_17, ndarray_0_17, mmap_tests);
