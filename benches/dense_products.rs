//! Isolate traversal and division costs without changing the public product API.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use lazymatrix::{
    DotSlice, ElemDivAssign, LazyMatrix, MatTransposeVecInto, MatVecInto, ScaledSubSlice,
    SubScalarAssign, SumEntries, VectorOwned, VectorViewMut,
};

#[path = "../tests/common/backend_aliases.rs"]
mod backend_aliases;
use backend_aliases::{faer, nalgebra, ndarray};

// These experiments deliberately retain the former scalar traversal and unsafe
// scaling rearrangements so future compiler and CPU changes can be measured.
#[allow(clippy::too_many_arguments)]
fn candidate<const PANEL: usize, V: VectorViewMut<f64>>(
    rows: usize,
    columns: usize,
    at: &impl Fn(usize, usize) -> f64,
    centers: &[f64],
    scales: Option<&[f64]>,
    x: &V,
    out: &mut V,
    transpose: bool,
    move_scaling: bool,
) {
    let scaled = if move_scaling && !transpose {
        Some(
            (0..columns)
                .map(|j| scales.map_or(x.get(j), |s| x.get(j) / s[j]))
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };
    if PANEL == 0 {
        if transpose {
            for j in 0..columns {
                let sum: f64 = (0..rows)
                    .map(|i| {
                        let value = at(i, j) - centers[j];
                        if move_scaling {
                            value * x.get(i)
                        } else {
                            scales.map_or(value, |s| value / s[j]) * x.get(i)
                        }
                    })
                    .sum();
                out.set(
                    j,
                    if move_scaling {
                        scales.map_or(sum, |s| sum / s[j])
                    } else {
                        sum
                    },
                );
            }
        } else {
            for i in 0..rows {
                let sum = (0..columns)
                    .map(|j| {
                        let value = at(i, j) - centers[j];
                        if move_scaling {
                            value * scaled.as_ref().unwrap()[j]
                        } else {
                            scales.map_or(value, |s| value / s[j]) * x.get(j)
                        }
                    })
                    .sum();
                out.set(i, sum);
            }
        }
    } else if transpose {
        for start in (0..columns).step_by(PANEL) {
            let width = PANEL.min(columns - start);
            let mut sums = [0.0; PANEL];
            for i in 0..rows {
                let coefficient = x.get(i);
                for (offset, sum) in sums[..width].iter_mut().enumerate() {
                    let j = start + offset;
                    let value = at(i, j) - centers[j];
                    *sum += scales.map_or(value, |s| value / s[j]) * coefficient;
                }
            }
            for (offset, sum) in sums[..width].iter().enumerate() {
                out.set(start + offset, *sum);
            }
        }
    } else {
        for start in (0..rows).step_by(PANEL) {
            let height = PANEL.min(rows - start);
            let mut sums = [0.0; PANEL];
            for j in 0..columns {
                let center = centers[j];
                let coefficient = x.get(j);
                for (offset, sum) in sums[..height].iter_mut().enumerate() {
                    let value = at(start + offset, j) - center;
                    *sum += scales.map_or(value, |s| value / s[j]) * coefficient;
                }
            }
            for (offset, sum) in sums[..height].iter().enumerate() {
                out.set(start + offset, *sum);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn terms<V: VectorViewMut<f64>>(
    rows: usize,
    columns: usize,
    at: &impl Fn(usize, usize) -> f64,
    centers: &[f64],
    scales: Option<&[f64]>,
    x: &V,
    out: &mut V,
    transpose: bool,
) {
    let (outputs, inputs) = if transpose {
        (columns, rows)
    } else {
        (rows, columns)
    };
    for output in 0..outputs {
        let mut sum = 0.0;
        for start in (0..inputs).step_by(128) {
            let width = 128.min(inputs - start);
            let mut terms = [0.0; 128];
            for (offset, term) in terms[..width].iter_mut().enumerate() {
                let input = start + offset;
                let (i, j) = if transpose {
                    (input, output)
                } else {
                    (output, input)
                };
                let value = at(i, j) - centers[j];
                *term = scales.map_or(value, |s| value / s[j]) * x.get(input);
            }
            for term in &terms[..width] {
                sum += term;
            }
        }
        out.set(output, sum);
    }
}

fn fixture(i: usize, j: usize) -> f64 {
    1e6 + ((i * 7 + j * 3) % 97) as f64 / 16.0 - 3.0
}

fn bench_matrix<M, V>(
    c: &mut Criterion,
    label: &str,
    matrix: &M,
    at: impl Fn(usize, usize) -> f64,
    vector: impl Fn(&[f64]) -> V,
) where
    M: MatVecInto<V> + MatTransposeVecInto<V>,
    V: Clone
        + VectorOwned<f64, Owned = V>
        + VectorViewMut<f64>
        + ElemDivAssign<f64>
        + DotSlice<f64>
        + SubScalarAssign<f64>
        + SumEntries<f64>
        + ScaledSubSlice<f64>,
{
    let rows = matrix.nrows();
    let columns = matrix.ncols();
    let centers = vec![1e6; columns];
    let scales: Vec<_> = (0..columns).map(|j| [1.3, -2.7, 0.75][j % 3]).collect();
    for (normalization, scales) in [
        ("centered", None),
        ("standardized", Some(scales.as_slice())),
    ] {
        let lazy =
            LazyMatrix::from_column_parts(matrix, Some(centers.clone()), scales.map(<[_]>::to_vec));
        for transpose in [false, true] {
            let direction = if transpose { "transpose" } else { "forward" };
            let name =
                format!("dense_products/{label}/{rows}x{columns}/{normalization}/{direction}");
            let input_len = if transpose { rows } else { columns };
            let output_len = if transpose { columns } else { rows };
            let x = vector(
                &(0..input_len)
                    .map(|i| (i % 11) as f64 / 7.0 - 0.4)
                    .collect::<Vec<_>>(),
            );
            let mut expected = V::owned_from_fn(output_len, |_| f64::NAN);
            candidate::<0, _>(
                rows,
                columns,
                &at,
                &centers,
                scales,
                &x,
                &mut expected,
                transpose,
                false,
            );
            let mut out = V::owned_from_fn(output_len, |_| f64::NAN);
            let mut group = c.benchmark_group(name);
            group.throughput(Throughput::Elements((rows * columns) as u64));
            for method in [
                "scalar",
                "panel32",
                "panel128",
                "terms128",
                "moved_scaling",
                "operator",
            ] {
                let run = |out: &mut V| match method {
                    "terms128" => terms(
                        rows,
                        columns,
                        &at,
                        &centers,
                        scales,
                        black_box(&x),
                        out,
                        transpose,
                    ),
                    "operator" if transpose => {
                        lazy.mat_transpose_vec_into(black_box(&x), out).unwrap()
                    }
                    "operator" => lazy.matvec_into(black_box(&x), out).unwrap(),
                    "panel32" => candidate::<32, _>(
                        rows,
                        columns,
                        &at,
                        &centers,
                        scales,
                        black_box(&x),
                        out,
                        transpose,
                        false,
                    ),
                    "panel128" => candidate::<128, _>(
                        rows,
                        columns,
                        &at,
                        &centers,
                        scales,
                        black_box(&x),
                        out,
                        transpose,
                        false,
                    ),
                    _ => candidate::<0, _>(
                        rows,
                        columns,
                        &at,
                        &centers,
                        scales,
                        black_box(&x),
                        out,
                        transpose,
                        method == "moved_scaling",
                    ),
                };
                run(&mut out);
                for i in 0..output_len {
                    assert!(
                        (out.get(i) - expected.get(i)).abs()
                            <= 1e-10 * (1.0 + expected.get(i).abs()),
                        "{label}/{method}/{direction}: entry {i}"
                    );
                }
                group.bench_function(BenchmarkId::from_parameter(method), |b| {
                    b.iter(|| {
                        run(&mut out);
                        black_box(&out);
                    })
                });
            }
            group.finish();
        }
    }
}

fn benchmark(c: &mut Criterion) {
    use ndarray::{ShapeBuilder, s};
    for (rows, columns) in [(2_000, 32), (10_000, 128)] {
        let col = faer::Mat::from_fn(rows, columns, fixture);
        bench_matrix(
            c,
            "faer/column",
            &col,
            |i, j| col[(i, j)],
            |x| faer::Col::from_fn(x.len(), |i| x[i]),
        );
        let row = faer::Mat::from_fn(columns, rows, |j, i| fixture(i, j));
        let row = row.as_ref().transpose();
        bench_matrix(
            c,
            "faer/row",
            &row,
            |i, j| row[(i, j)],
            |x| faer::Col::from_fn(x.len(), |i| x[i]),
        );
        let strided = col.as_ref().reverse_rows();
        bench_matrix(
            c,
            "faer/strided",
            &strided,
            |i, j| strided[(i, j)],
            |x| faer::Col::from_fn(x.len(), |i| x[i]),
        );

        let col = nalgebra::DMatrix::from_fn(rows, columns, fixture);
        bench_matrix(
            c,
            "nalgebra/column",
            &col,
            |i, j| col[(i, j)],
            nalgebra::DVector::from_column_slice,
        );
        let row = nalgebra::DMatrix::from_fn(columns, rows, |j, i| fixture(i, j));
        let row = nalgebra::DMatrixView::from_slice_with_strides(
            row.as_slice(),
            rows,
            columns,
            columns,
            1,
        );
        bench_matrix(
            c,
            "nalgebra/row",
            &row,
            |i, j| row[(i, j)],
            nalgebra::DVector::from_column_slice,
        );
        let storage = nalgebra::DMatrix::from_fn(2 * rows, 2 * columns, fixture);
        let strided = storage.view_with_steps((0, 0), (rows, columns), (1, 1));
        bench_matrix(
            c,
            "nalgebra/strided",
            &strided,
            |i, j| strided[(i, j)],
            nalgebra::DVector::from_column_slice,
        );

        let col = ndarray::Array2::from_shape_fn((rows, columns).f(), |(i, j)| fixture(i, j));
        bench_matrix(
            c,
            "ndarray/column",
            &col,
            |i, j| col[(i, j)],
            |x| ndarray::Array1::from_vec(x.to_vec()),
        );
        let row = ndarray::Array2::from_shape_fn((rows, columns), |(i, j)| fixture(i, j));
        bench_matrix(
            c,
            "ndarray/row",
            &row,
            |i, j| row[(i, j)],
            |x| ndarray::Array1::from_vec(x.to_vec()),
        );
        let strided = row.slice(s![..;-1, ..;-1]);
        bench_matrix(
            c,
            "ndarray/strided",
            &strided,
            |i, j| strided[(i, j)],
            |x| ndarray::Array1::from_vec(x.to_vec()),
        );
    }
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
