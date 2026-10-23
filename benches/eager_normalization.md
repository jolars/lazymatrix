# Normalization trade-offs

The website's runtime, memory, and repetition plots use
[`normalization_tradeoffs/results.csv`](normalization_tradeoffs/results.csv)
and its [run metadata](normalization_tradeoffs/metadata.json).

```sh
bash scripts/bench-normalization-tradeoffs.sh
mkdir -p benches/normalization_tradeoffs
cp target/normalization-tradeoffs/{results.csv,metadata.json} benches/normalization_tradeoffs/
task website:check
```

The collector compiles both harnesses first, profiles allocations in
`normalization_memory`, and then runs Criterion in `eager_normalization`.
Allocation tracking is absent from the timing executable. Both use the same
seeded canonical CSC fixtures, with nonzero means and seven target densities from
0.1% to 100%. Each of three sizes includes tall ($n/p = 16$), square ($n = p$),
and wide ($p/n = 16$) matrices with matching element counts:

| Dense storage | Tall | Square | Wide |
| --- | --- | --- | --- |
| 2 MiB | 2,048 × 128 | 512 × 512 | 128 × 2,048 |
| 32 MiB | 8,192 × 512 | 2,048 × 2,048 | 512 × 8,192 |
| 128 MiB | 16,384 × 1,024 | 4,096 × 4,096 | 1,024 × 16,384 |

The sparse sources store `f64` values and `usize` indices. The CSV reports
realized nonzero and empty-column counts. Wide, very sparse fixtures can have
many empty columns because each column has few opportunities for a nonzero.

Each fixture compares sprs CSC with its ndarray dense destination and faer CSC
with its faer dense destination. Both mean centering plus SD scaling and SD
scaling alone are measured. The latter preserves structural zeros; dense eager
conversion is not a specialized sparse scaling strategy. Products reuse output
buffers, include scratch from the public `*_into` methods, and use deterministic,
nonconstant input vectors whose sums are nonzero. Each timed case checks lazy
and eager forward and transpose results using the same fitted parameters.

Criterion measures parameter fitting, dense materialization, and raw, lazy,
and eager forward and transpose products separately. The collector selects 30
samples, 0.3 seconds of warmup, at least one second of measurement, and 10,000
bootstrap resamples. `LAZYMATRIX_BENCH_QUICK=1` instead selects ten samples,
0.05 seconds of warmup, at least 0.1 seconds of measurement, and 1,000 resamples.
The checked-in tall cases retain the original longer run; square and wide cases
use the quick settings. Their metadata records both collections. To collect a
subset, set `LAZYMATRIX_BENCH_SHAPES` to comma-separated shapes such as
`512x512,128x2048`.

The collector exports means and 95% confidence intervals. The total
time plots add one parameter fit and, for eager normalization, one
materialization to a fixed number of forward and transpose products. These
curves are projections, not complete solver measurements.

Memory profiling warms each backend, then measures a complete lifecycle from
parameter fitting through construction and ten product pairs. The original
sparse matrix stays live on both paths. Its storage is added to the measured
peak increase, including values, row indices, and column pointers. Normalized
representation storage includes fitted parameters; eager representation bytes
exclude the retained source, which is counted separately in the total peak.
Workspace is the peak increase minus added representation storage, including
input and output vectors and any temporary statistics, construction buffers,
or product scratch. These are requested Rust bytes, excluding allocator
overhead, native-library allocations, and unrelated process memory. Reallocation
records its resulting size. Lifecycle peaks are measured directly rather than
computed by adding phase peaks.

For a quick numerical check:

```sh
LAZYMATRIX_BENCH_SMOKE=1 cargo bench --locked --bench eager_normalization \
  --features ndarray,sprs,faer -- --test
```

Omit `faer` to measure only sprs with ndarray. The version-selection aliases used
by examples and benchmarks select the newest enabled backend release. The
website reads checked-in measurements and never invokes either harness.

## Historical consumer fitting measurements

The remaining notes describe the initial October 8, 2026, runs on a different
machine. Their single-run timings are not part of the website's product sweep.

The eager conversion API lets a caller choose between a normalized dense copy,
reusing dense output storage, and consuming writable dense storage in place.
It does not automatically choose an execution strategy.

## Reproduce

```sh
cargo bench --locked --bench eager_normalization --features ndarray,sprs
bash scripts/bench-normalization-allocations.sh > target/normalization-allocations.csv
```

The Criterion benchmark measures fitting parameters, materializing dense, CSC,
and CSR inputs, and batches of ten forward and transpose products. It separates
materialization from repeated products. The allocation harness fits the same
Gaussian ridge problem through lazy and eager operators, with an intercept and
penalty 0.1. It compares coefficients, intercepts, and objectives after converting
eager coefficients back with the retained parameters. Solver normalization is
turned off for already normalized eager inputs.

The [allocation profiles](normalization_allocations/eager_results.csv) contain
96 measurements from October 8, 2026, using Rust 1.89.0 in release mode on an
Intel Core Ultra 7 155U. Fixtures use seed 205, 512 × 32 and 2,048 × 128 matrices,
and target densities 100% and 10%. OpenBLAS and OpenMP use one thread. Fixtures
and source storage conversions happen outside measurement. Each fitting path
is warmed before measurement. Timings are single instrumented runs and should
be treated as illustrative; other builds ran concurrently.

## Fitting time and allocations

For standardized 2,048 × 128 predictors:

| Source and dense destination | Density | Lazy fit (ms) | Eager build (ms) | Eager fit (ms) | Lazy fit allocations | Eager fit allocations |
| --- | --- | --- | --- | --- | --- | --- |
| ndarray → ndarray | 100% | 6.52 | 2.53 | 6.26 | 71 | 9 |
| faer dense → faer dense | 100% | 24.89 | 1.53 | 24.10 | 71 | 9 |
| sprs CSC → ndarray | 100% | 19.56 | 2.89 | 6.35 | 71 | 9 |
| faer CSC → faer dense | 100% | 16.53 | 1.81 | 23.25 | 70 | 9 |
| ndarray → ndarray | 10% | 6.62 | 2.70 | 6.35 | 67 | 9 |
| faer dense → faer dense | 10% | 23.84 | 1.63 | 25.13 | 67 | 9 |
| sprs CSC → ndarray | 10% | 3.16 | 1.80 | 5.75 | 67 | 9 |
| faer CSC → faer dense | 10% | 2.49 | 0.80 | 20.06 | 66 | 9 |

Every eager fit made nine Rust allocation requests and no coefficient-vector
clones. Standardized lazy fits cloned the coefficient vector for every forward
product. Eager normalization eliminates that recurring allocation, but it does
not necessarily improve fitting time: sparse products can remain much cheaper
than products with the materialized dense matrix.

The allocator also reports `peak_extra_live_bytes`: the peak live requested Rust
bytes above the live bytes at the start of each measured phase. This excludes
native BLAS/LAPACK allocations, allocator overhead, and resident-memory effects.
A reallocation records the resulting live requested size, not temporary work
inside the system allocator.

For these standardized 2,048 × 128 fixtures, lazy fitting peaked at 73,728 extra
live bytes. Eager fitting peaked at 70,656 extra live bytes, with its dense matrix
already present before that phase. Eager construction peaked at 2,101,248 extra
bytes for dense sources and 2,117,632 for sparse sources. The output alone
requires 2,097,152 bytes and remains allocated during fitting. The fit-phase
figures therefore do not imply a reduction in overall memory from materializing.
Sparse construction uses one working column and duplicate bookkeeping, rather
than another full design matrix.

## Product benchmark

A short Criterion run used 200 ms warmup, 500 ms measurement, and ten samples
on the same machine while other builds were active. For 2,048 × 128 predictors
at 10% density, point estimates for ten forward/transpose pairs were 4.60 ms
for lazy ndarray, 4.62 ms for eager ndarray, and 0.37 ms for lazy sprs CSC.
These measurements support keeping conversion explicit. Run the default,
longer Criterion configuration on an otherwise idle machine when comparing
execution strategies for a downstream consumer.
