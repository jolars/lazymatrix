//! Isolate borrowing, traversal, and reduction order without changing the library.

use std::{hint::black_box, path::Path, time::Instant};

use lazymatrix::{LazyMatrix, MatrixShape, RawColumn, RawColumns};
use ndarray::{Array2, ArrayView1, ArrayView2, ShapeBuilder, s};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[derive(Clone, Copy, Debug)]
enum Method {
    PublishedDot,
    CurrentDot,
    CurrentDotView,
    InlineDot,
    VisitorDot,
    SliceDot,
    Indexed8Dot,
    Chunk8DotVec,
    Slice8Dot,
    NdarrayDot,
    PublishedSum,
    CurrentSum,
    InlineSum,
    SliceSum,
    Indexed8Sum,
    Slice8Sum,
    NdarraySum,
    PublishedNorm,
    CurrentNorm,
    InlineNorm,
    SliceNorm,
    Indexed8Norm,
    Slice8Norm,
    NdarrayNorm,
    CurrentSummary,
    SeparateSliceSummary,
    FusedSliceSummary,
}

const METHODS: &[Method] = &[
    Method::PublishedDot,
    Method::CurrentDot,
    Method::CurrentDotView,
    Method::InlineDot,
    Method::VisitorDot,
    Method::SliceDot,
    Method::Indexed8Dot,
    Method::Chunk8DotVec,
    Method::Slice8Dot,
    Method::NdarrayDot,
    Method::PublishedSum,
    Method::CurrentSum,
    Method::InlineSum,
    Method::SliceSum,
    Method::Indexed8Sum,
    Method::Slice8Sum,
    Method::NdarraySum,
    Method::PublishedNorm,
    Method::CurrentNorm,
    Method::InlineNorm,
    Method::SliceNorm,
    Method::Indexed8Norm,
    Method::Slice8Norm,
    Method::NdarrayNorm,
];

const SUMMARIES: &[Method] = &[
    Method::CurrentSummary,
    Method::SeparateSliceSummary,
    Method::FusedSliceSummary,
];

// These prototypes deliberately change the addition order. Keeping them outside
// RawColumn lets the investigation measure that tradeoff before choosing an API.
#[inline]
fn reduce8(len: usize, mut term: impl FnMut(usize) -> f64) -> f64 {
    let mut sums = [0.0; 8];
    let end = len / 8 * 8;
    for start in (0..end).step_by(8) {
        for (offset, sum) in sums.iter_mut().enumerate() {
            *sum += term(start + offset);
        }
    }
    let mut sum = (sums[0] + sums[4]) + (sums[1] + sums[5]);
    sum += sums[2] + sums[6];
    sum += sums[3] + sums[7];
    for i in end..len {
        sum += term(i);
    }
    sum
}

#[inline(never)]
fn slice_dot(column: &[f64], vector: &[f64]) -> f64 {
    assert_eq!(column.len(), vector.len());
    column.iter().zip(vector).map(|(x, y)| x * y).sum()
}

#[inline(never)]
fn slice8_dot(column: &[f64], vector: &[f64]) -> f64 {
    assert_eq!(column.len(), vector.len());
    let mut sums = [0.0; 8];
    let mut left = column.chunks_exact(8);
    let mut right = vector.chunks_exact(8);
    for (x, y) in left.by_ref().zip(right.by_ref()) {
        for lane in 0..8 {
            sums[lane] += x[lane] * y[lane];
        }
    }
    let mut sum = (sums[0] + sums[4]) + (sums[1] + sums[5]);
    sum += sums[2] + sums[6];
    sum += sums[3] + sums[7];
    for (x, y) in left.remainder().iter().zip(right.remainder()) {
        sum += x * y;
    }
    sum
}

#[inline(never)]
fn indexed8_dot<V: lazymatrix::VectorView<f64> + ?Sized>(
    column: ArrayView1<'_, f64>,
    vector: &V,
) -> f64 {
    assert_eq!(column.len(), vector.len());
    reduce8(column.len(), |i| column[i] * vector.get(i))
}

#[inline(never)]
fn chunk8_dot<V: lazymatrix::VectorView<f64> + ?Sized>(column: &[f64], vector: &V) -> f64 {
    assert_eq!(column.len(), vector.len());
    let mut sums = [0.0; 8];
    let mut chunks = column.chunks_exact(8);
    for (block, chunk) in chunks.by_ref().enumerate() {
        for lane in 0..8 {
            sums[lane] += chunk[lane] * vector.get(block * 8 + lane);
        }
    }
    let mut sum = (sums[0] + sums[4]) + (sums[1] + sums[5]);
    sum += sums[2] + sums[6];
    sum += sums[3] + sums[7];
    let start = column.len() - chunks.remainder().len();
    for (i, &value) in chunks.remainder().iter().enumerate() {
        sum += value * vector.get(start + i);
    }
    sum
}

#[inline]
fn slice_reduce8(column: &[f64], mut term: impl FnMut(f64) -> f64) -> f64 {
    let mut sums = [0.0; 8];
    let mut chunks = column.chunks_exact(8);
    for chunk in chunks.by_ref() {
        for lane in 0..8 {
            sums[lane] += term(chunk[lane]);
        }
    }
    let mut sum = (sums[0] + sums[4]) + (sums[1] + sums[5]);
    sum += sums[2] + sums[6];
    sum += sums[3] + sums[7];
    for &value in chunks.remainder() {
        sum += term(value);
    }
    sum
}

struct InlineMatrix<'a>(ArrayView2<'a, f64>);
struct InlineColumn<'a>(ArrayView1<'a, f64>);

impl MatrixShape for InlineMatrix<'_> {
    fn nrows(&self) -> usize {
        self.0.nrows()
    }
    fn ncols(&self) -> usize {
        self.0.ncols()
    }
}

impl RawColumns<f64> for InlineMatrix<'_> {
    type Column<'a>
        = InlineColumn<'a>
    where
        Self: 'a;
    fn raw_column(&self, j: usize) -> Self::Column<'_> {
        InlineColumn(self.0.column(j))
    }
}

impl RawColumn<f64> for InlineColumn<'_> {
    fn len(&self) -> usize {
        self.0.len()
    }
    fn stored_len(&self) -> usize {
        self.0.len()
    }
    #[inline]
    fn for_each_stored(&self, mut f: impl FnMut(usize, f64)) {
        for (row, &value) in self.0.iter().enumerate() {
            f(row, value);
        }
    }
}

#[inline(never)]
fn indexed8_sum(column: ArrayView1<'_, f64>) -> f64 {
    reduce8(column.len(), |i| column[i])
}

#[inline(never)]
fn indexed8_norm(column: ArrayView1<'_, f64>, center: f64) -> f64 {
    reduce8(column.len(), |i| {
        let value = column[i] - center;
        value * value
    })
}

#[inline(never)]
fn slice8_summary(column: &[f64], center: f64) -> (f64, f64) {
    let mut sums = [0.0; 8];
    let mut squares = [0.0; 8];
    let mut chunks = column.chunks_exact(8);
    for chunk in chunks.by_ref() {
        for lane in 0..8 {
            let x = chunk[lane];
            sums[lane] += x;
            let deviation = x - center;
            squares[lane] += deviation * deviation;
        }
    }
    let combine = |values: [f64; 8]| {
        ((values[0] + values[4]) + (values[1] + values[5]))
            + (values[2] + values[6])
            + (values[3] + values[7])
    };
    let mut sum = combine(sums);
    let mut norm = combine(squares);
    for &x in chunks.remainder() {
        sum += x;
        let deviation = x - center;
        norm += deviation * deviation;
    }
    (sum, norm)
}

fn scan(passes: usize, ncols: usize, mut column: impl FnMut(usize) -> f64) -> f64 {
    let mut checksum = 0.0;
    for _ in 0..passes {
        for j in 0..ncols {
            checksum += column(j);
        }
    }
    black_box(checksum)
}

struct Fixture<'a> {
    matrix: ArrayView2<'a, f64>,
    vector: ArrayView1<'a, f64>,
    vector_owned: Vec<f64>,
    vector_sum: f64,
    center: f64,
    scale: f64,
    lazy: LazyMatrix<ArrayView2<'a, f64>>,
    old: published::LazyMatrix<ArrayView2<'a, f64>>,
    inline: LazyMatrix<InlineMatrix<'a>>,
}

impl<'a> Fixture<'a> {
    fn new(matrix: ArrayView2<'a, f64>, vector: ArrayView1<'a, f64>, center: f64) -> Self {
        assert_eq!(matrix.nrows(), vector.len());
        let scale = if center == 0.0 { 1.0 } else { 1.75 };
        let centers = (center != 0.0).then(|| vec![center; matrix.ncols()]);
        let scales = (scale != 1.0).then(|| vec![scale; matrix.ncols()]);
        Self {
            matrix,
            vector,
            vector_owned: vector.to_vec(),
            vector_sum: vector.iter().sum(),
            center,
            scale,
            lazy: LazyMatrix::from_column_parts(matrix, centers.clone(), scales.clone()),
            inline: LazyMatrix::from_column_parts(
                InlineMatrix(matrix),
                centers.clone(),
                scales.clone(),
            ),
            old: published::LazyMatrix::from_parts(matrix, centers, scales),
        }
    }

    fn run(&self, method: Method, passes: usize) -> f64 {
        let matrix = &self.matrix;
        let vector = &self.vector_owned;
        let vector_view = &self.vector;
        let vector_sum = self.vector_sum;
        let center = self.center;
        let scale = self.scale;
        let lazy = &self.lazy;
        let old = &self.old;
        let inline = &self.inline;
        let dot = |value: f64| (value - center * vector_sum) / scale;
        let sum = |value: f64| (value - matrix.nrows() as f64 * center) / scale;
        let norm = |value: f64| value / (scale * scale);
        match method {
            Method::PublishedDot => scan(passes, matrix.ncols(), |j| {
                black_box(&old)
                    .column(j)
                    .dot_with_sum(black_box(vector), black_box(vector_sum))
            }),
            Method::CurrentDot => scan(passes, matrix.ncols(), |j| {
                black_box(&lazy)
                    .column(j)
                    .dot_with_sum(black_box(vector), black_box(vector_sum))
            }),
            Method::CurrentDotView => scan(passes, matrix.ncols(), |j| {
                black_box(&lazy)
                    .column(j)
                    .dot_with_sum(black_box(vector_view), black_box(vector_sum))
            }),
            Method::InlineDot => scan(passes, matrix.ncols(), |j| {
                black_box(inline)
                    .column(j)
                    .dot_with_sum(black_box(vector), black_box(vector_sum))
            }),
            Method::VisitorDot => scan(passes, matrix.ncols(), |j| {
                let column = black_box(matrix).column(j);
                let vector = black_box(vector);
                let mut value = 0.0;
                column.for_each_stored(|i, x| value += x * vector[i]);
                dot(value)
            }),
            Method::SliceDot => scan(passes, matrix.ncols(), |j| {
                dot(slice_dot(
                    black_box(matrix).column(j).as_slice().unwrap(),
                    black_box(vector),
                ))
            }),
            Method::Indexed8Dot => scan(passes, matrix.ncols(), |j| {
                dot(indexed8_dot(
                    black_box(matrix).column(j),
                    black_box(vector_view),
                ))
            }),
            Method::Chunk8DotVec => scan(passes, matrix.ncols(), |j| {
                dot(chunk8_dot(
                    black_box(matrix).column(j).as_slice().unwrap(),
                    black_box(vector),
                ))
            }),
            Method::Slice8Dot => scan(passes, matrix.ncols(), |j| {
                dot(slice8_dot(
                    black_box(matrix).column(j).as_slice().unwrap(),
                    black_box(vector),
                ))
            }),
            Method::NdarrayDot => scan(passes, matrix.ncols(), |j| {
                dot(black_box(matrix).column(j).dot(black_box(vector_view)))
            }),
            Method::PublishedSum => {
                scan(passes, matrix.ncols(), |j| black_box(&old).column(j).sum())
            }
            Method::CurrentSum => {
                scan(passes, matrix.ncols(), |j| black_box(&lazy).column(j).sum())
            }
            Method::InlineSum => scan(passes, matrix.ncols(), |j| {
                black_box(inline).column(j).sum()
            }),
            Method::SliceSum => scan(passes, matrix.ncols(), |j| {
                sum(black_box(matrix).column(j).as_slice().unwrap().iter().sum())
            }),
            Method::Indexed8Sum => scan(passes, matrix.ncols(), |j| {
                sum(indexed8_sum(black_box(matrix).column(j)))
            }),
            Method::Slice8Sum => scan(passes, matrix.ncols(), |j| {
                sum(slice_reduce8(
                    black_box(matrix).column(j).as_slice().unwrap(),
                    |x| x,
                ))
            }),
            Method::NdarraySum => scan(passes, matrix.ncols(), |j| {
                sum(black_box(matrix).column(j).sum())
            }),
            Method::PublishedNorm => scan(passes, matrix.ncols(), |j| {
                black_box(&old).column(j).norm_squared()
            }),
            Method::CurrentNorm => scan(passes, matrix.ncols(), |j| {
                black_box(&lazy).column(j).norm_squared()
            }),
            Method::InlineNorm => scan(passes, matrix.ncols(), |j| {
                black_box(inline).column(j).norm_squared()
            }),
            Method::SliceNorm => scan(passes, matrix.ncols(), |j| {
                norm(
                    black_box(matrix)
                        .column(j)
                        .as_slice()
                        .unwrap()
                        .iter()
                        .map(|x| (x - center) * (x - center))
                        .sum(),
                )
            }),
            Method::Indexed8Norm => scan(passes, matrix.ncols(), |j| {
                norm(indexed8_norm(black_box(matrix).column(j), center))
            }),
            Method::Slice8Norm => scan(passes, matrix.ncols(), |j| {
                norm(slice_reduce8(
                    black_box(matrix).column(j).as_slice().unwrap(),
                    |x| (x - center) * (x - center),
                ))
            }),
            Method::NdarrayNorm => scan(passes, matrix.ncols(), |j| {
                let column = black_box(matrix).column(j);
                assert_eq!(center, 0.0);
                norm(column.dot(&column))
            }),
            Method::CurrentSummary => scan(passes, matrix.ncols(), |j| {
                let column = black_box(lazy).column(j);
                column.sum() + column.norm_squared()
            }),
            Method::SeparateSliceSummary => scan(passes, matrix.ncols(), |j| {
                let column = black_box(matrix).column(j);
                let values = column.as_slice().unwrap();
                sum(slice_reduce8(values, |x| x))
                    + norm(slice_reduce8(values, |x| (x - center) * (x - center)))
            }),
            Method::FusedSliceSummary => scan(passes, matrix.ncols(), |j| {
                let column = black_box(matrix).column(j);
                let (raw_sum, squared_deviations) =
                    slice8_summary(column.as_slice().unwrap(), center);
                sum(raw_sum) + norm(squared_deviations)
            }),
        }
    }

    fn supports(&self, method: Method) -> bool {
        match method {
            Method::SliceDot
            | Method::Chunk8DotVec
            | Method::Slice8Dot
            | Method::SliceSum
            | Method::SliceNorm
            | Method::Slice8Sum
            | Method::Slice8Norm => self.matrix.strides()[0] == 1,
            Method::CurrentSummary | Method::SeparateSliceSummary | Method::FusedSliceSummary => {
                self.matrix.strides()[0] == 1
            }
            Method::NdarrayNorm => self.center == 0.0,
            _ => true,
        }
    }

    fn validate(&self, label: &str, layout: &str, methods: &[Method]) {
        let mut errors = vec![0.0_f64; methods.len()];
        for j in 0..self.matrix.ncols() {
            let column = self.matrix.column(j);
            let expected_dot = compensated_sum(
                column
                    .iter()
                    .zip(self.vector)
                    .map(|(&x, &y)| (x - self.center) / self.scale * y),
            );
            let expected_sum =
                compensated_sum(column.iter().map(|&x| (x - self.center) / self.scale));
            let expected_norm = compensated_sum(column.iter().map(|&x| {
                let value = (x - self.center) / self.scale;
                value * value
            }));
            if self.matrix.strides()[0] == 1 {
                let (raw_sum, squared_deviations) =
                    slice8_summary(column.as_slice().unwrap(), self.center);
                let actual_sum = (raw_sum - self.matrix.nrows() as f64 * self.center) / self.scale;
                let actual_norm = squared_deviations / (self.scale * self.scale);
                assert!((actual_sum - expected_sum).abs() <= 1e-9 + 1e-12 * expected_sum.abs());
                assert!((actual_norm - expected_norm).abs() <= 1e-9 + 1e-12 * expected_norm.abs());
            }
            let single = Fixture::new(
                self.matrix.slice(s![.., j..j + 1]),
                self.vector,
                self.center,
            );
            for (i, &method) in methods
                .iter()
                .enumerate()
                .filter(|(_, &method)| self.supports(method))
            {
                let expected = match method {
                    Method::CurrentSummary
                    | Method::SeparateSliceSummary
                    | Method::FusedSliceSummary => expected_sum + expected_norm,
                    Method::PublishedSum
                    | Method::CurrentSum
                    | Method::InlineSum
                    | Method::SliceSum
                    | Method::Indexed8Sum
                    | Method::Slice8Sum
                    | Method::NdarraySum => expected_sum,
                    Method::PublishedNorm
                    | Method::CurrentNorm
                    | Method::InlineNorm
                    | Method::SliceNorm
                    | Method::Indexed8Norm
                    | Method::Slice8Norm
                    | Method::NdarrayNorm => expected_norm,
                    _ => expected_dot,
                };
                let actual = single.run(method, 1);
                let error = (actual - expected).abs();
                assert!(
                    error <= 1e-9 + 1e-12 * expected.abs(),
                    "{layout}/{method:?}/{j}: {actual} != {expected}"
                );
                errors[i] = errors[i].max(error);
            }
        }
        for (i, &method) in methods
            .iter()
            .enumerate()
            .filter(|(_, &method)| self.supports(method))
        {
            eprintln!("error,{label},{layout},{method:?},{:.17e}", errors[i]);
        }
    }
}

fn compensated_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut sum: f64 = 0.0;
    let mut correction = 0.0;
    for value in values {
        let next = sum + value;
        correction += if sum.abs() >= value.abs() {
            (sum - next) + value
        } else {
            (value - next) + sum
        };
        sum = next;
    }
    sum + correction
}

fn read_values(path: &Path) -> Vec<f64> {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(bytes.len() % 8, 0);
    bytes
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn numerical_cases() {
    println!("case,reference,serial,eight_accumulators,ndarray");
    let mut loses_unit = vec![0.0; 16];
    loses_unit[0] = 1e16;
    loses_unit[1] = -1e16;
    loses_unit[8] = 1.0;
    let cases = [
        ("cancellation_loses_unit", loses_unit, 1.0),
        (
            "cancellation_recovers_units",
            vec![1e16, 1.0, 1.0, 1.0, -1e16, 1.0, 1.0, 1.0],
            6.0,
        ),
        (
            "finite_intermediate_overflow",
            vec![
                f64::MAX,
                -f64::MAX,
                f64::MAX,
                -f64::MAX,
                f64::MAX,
                -f64::MAX,
                f64::MAX,
                -f64::MAX,
            ],
            0.0,
        ),
        ("positive_infinity", vec![f64::INFINITY; 16], f64::INFINITY),
        (
            "opposite_infinities",
            vec![f64::INFINITY, f64::NEG_INFINITY],
            f64::NAN,
        ),
        ("nan", vec![f64::NAN; 16], f64::NAN),
        ("empty", vec![], 0.0),
    ];
    for (name, values, expected) in cases {
        let ones = vec![1.0; values.len()];
        let serial = slice_dot(&values, &ones);
        let eight = slice8_dot(&values, &ones);
        let native = ArrayView1::from(&values).dot(&ArrayView1::from(&ones));
        println!("{name},{expected:?},{serial:?},{eight:?},{native:?}");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|x| x == "--numerics") {
        numerical_cases();
        return;
    }
    let arg = |name: &str| {
        args.iter()
            .position(|x| x == name)
            .map(|i| args[i + 1].as_str())
    };
    let nrows = arg("--rows").unwrap_or("5000").parse::<usize>().unwrap();
    let ncols = arg("--cols").unwrap_or("1000").parse::<usize>().unwrap();
    let samples = arg("--samples").unwrap_or("9").parse::<usize>().unwrap();
    let passes = arg("--passes").unwrap_or("30").parse::<usize>().unwrap();
    let label = arg("--label").unwrap_or("default");
    let mut rng = ChaCha8Rng::seed_from_u64(7363749798296254);
    let (matrix, vector) = if let Some(directory) = arg("--data") {
        let directory = Path::new(directory);
        (
            Array2::from_shape_vec((nrows, ncols).f(), read_values(&directory.join("x.bin")))
                .unwrap(),
            read_values(&directory.join("y.bin")),
        )
    } else {
        (
            Array2::from_shape_fn((nrows, ncols).f(), |_| rng.random_range(-1.0..1.0)),
            (0..nrows).map(|_| rng.random_range(-1.0..1.0)).collect(),
        )
    };
    let row_major = Array2::from_shape_fn((nrows, ncols), |(i, j)| matrix[(i, j)]);
    let vector_view = ArrayView1::from(&vector);
    let mut fixtures = vec![("fortran", Fixture::new(matrix.view(), vector_view, 0.0))];
    if args.iter().any(|x| x == "--layouts") {
        fixtures.extend([
            ("centered", Fixture::new(matrix.view(), vector_view, 0.375)),
            (
                "row_major",
                Fixture::new(row_major.view(), vector_view, 0.0),
            ),
            (
                "row_stride2",
                Fixture::new(
                    matrix.slice(s![..;2, ..]),
                    vector_view.slice(s![..;2]),
                    0.375,
                ),
            ),
            (
                "row_reverse",
                Fixture::new(
                    matrix.slice(s![..;-1, ..]),
                    vector_view.slice(s![..;-1]),
                    0.375,
                ),
            ),
        ]);
    }
    println!("build,layout,method,sample,passes,rows,cols,ns,checksum");
    let selected = if args.iter().any(|x| x == "--summaries") {
        SUMMARIES
    } else {
        METHODS
    };
    for (layout, fixture) in fixtures {
        fixture.validate(label, layout, selected);
        for sample in 0..samples {
            let mut methods = selected.to_vec();
            if sample % 2 == 1 {
                methods.reverse();
            }
            for method in methods
                .into_iter()
                .filter(|&method| fixture.supports(method))
            {
                if sample == 0 {
                    black_box(fixture.run(method, 1));
                }
                let start = Instant::now();
                let checksum = fixture.run(method, passes);
                let elapsed = start.elapsed().as_nanos();
                println!(
                    "{label},{layout},{method:?},{sample},{passes},{},{},{elapsed},{checksum:.17e}",
                    fixture.matrix.nrows(),
                    fixture.matrix.ncols()
                );
            }
        }
    }
}
