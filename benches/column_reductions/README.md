# Dense column reductions

Contiguous columns now use allocation-free reductions with eight independent
accumulators for raw sums, dots, weighted dots, and squared norms. Contiguous
vector sums use the same kernels, including the sums computed by uncached dots.
This covers ndarray 0.15–0.17, faer 0.22–0.24, and nalgebra 0.32–0.35. The core
shares the reduction implementation through optional `RawColumn::dense_values`
and `VectorView::contiguous_values` hooks. Both owned and borrowed contiguous
vectors expose the vector hook. Other storage and vector layouts retain their
default implementations.

Reduction order follows the usual backend convention: rounding, cancellation,
and intermediate overflow may differ from sequential accumulation. The dense
fast paths use the same arithmetic strategy as ndarray's non-BLAS contiguous
reductions. The regression suite covers both `f32` and `f64`, short tails,
large-offset centered norms, nonfinite values, and strided inputs.

## Remaining solver-facing vector reductions

The follow-up uses baseline revision `8750c46` and the production source
identified in [`solver_vector_metadata.json`](solver_vector_metadata.json). Core
`Vec` reductions, ndarray's contiguous vector reductions, and faer's sequential
sums and dots now reuse the shared eight-accumulator kernels. Nalgebra's
`sum_entries` and `dot_slice` use them as well. Nalgebra retains its native dot
and norm implementations. Faer retains its native scaled L2 norm, which avoids
intermediate overflow and underflow for extreme magnitudes. No new dependency,
allocation, or public capability is required. Ndarray's strided and reversed
views retain their original iterator fallbacks.

The harness compares `sum_entries`, `dot_slice`, `DotProduct::dot`, and
`norm_l2` for `f32` and `f64` at lengths 0, 1, 7, 8, 9, 33, 256, 5,000, and
262,144. Each type uses `ChaCha8Rng`, seed 8421, and values drawn uniformly from
\[-1, 1). Owned core, faer, and nalgebra vectors share the same logical inputs.
Borrowed ndarray views include an offset into NaN-padded storage and strides 1,
2, -1, and -2. Every reduction is checked against compensated `f64` arithmetic
outside timing. Native controls use faer's sum, inner product, and scaled norm,
nalgebra's sum, dot, and norm, and ndarray's sum and dot. Ndarray's norm control
takes the square root of its self-dot. The core vector control uses sequential
slice arithmetic.

Both release builds and all final measurements run in one exclusive
`compute bench` window on CPU 2, with Rust 1.98.1, 16 codegen units, generic
x86-64 code generation, and the `faer,nalgebra,ndarray` features. Six blocks
alternate baseline and candidate process order. Each process collects five
observations, reversing operation and method order on alternate observations.
Each observation repeats a reduction
`clamp(5_000_000 / max(length, 1), 8, 100_000)` times. Timings include the same
non-inlined benchmark dispatch in both builds, so dispatch matters most for
short vectors. Input construction and verification are outside timing.

The table reports baseline time divided by candidate time at length 5,000, using
the median of 30 observations per build. These are observed ratios on this
machine, rather than guarantees for other workloads. Native faer control ratios
vary by up to 15% in this run; the other contiguous native controls at this
length vary by at most 8%.

  | Backend  | Scalar | Sum   | Slice dot | Vector dot | L2 norm |
  | -------- | ------ | ----: | --------: | ---------: | ------: |
  | Vec      | f32    | 4.96x |     3.99x |      4.19x |   5.49x |
  | Vec      | f64    | 4.99x |     2.69x |      2.81x |   5.44x |
  | ndarray  | f32    | 5.05x |     4.31x |      5.38x |   5.68x |
  | ndarray  | f64    | 4.97x |     2.74x |      3.40x |   5.52x |
  | faer     | f32    | 4.38x |     4.17x |      4.45x |  native |
  | faer     | f64    | 4.67x |     2.50x |      2.63x |  native |
  | nalgebra | f32    | 4.96x |     4.05x |     native |  native |
  | nalgebra | f64    | 4.90x |     2.74x |     native |  native |

Here, `native` identifies an unchanged implementation. Nalgebra's unchanged dots
and norms measure within 3% of baseline, as does faer's unchanged norm, at this
length. The shared kernels replace serial accumulation dependencies with eight
independent accumulators, permitting packed arithmetic without fast-math.
Retaining native kernels also preserves their existing numerical strategies and
avoids unnecessary replacements.

The gains persist at length 262,144: changed sums improve by 4.4–4.9 times,
changed dots by 3.0–4.4 times for `f32` and 1.7–2.5 times for `f64`, and changed
norms by 4.6–5.7 times. The smaller dot gains with larger working sets are
consistent with memory traffic limiting more of their runtime, although these
measurements do not include hardware counters. Strided and reversed ndarray
fallbacks at lengths 256 and above range from 5% slower to 37% faster, with a
median improvement near 3–4%. Their arithmetic is unchanged, so those timing
differences do not establish a separate fallback optimization.

Short vectors do not improve uniformly. At lengths up to nine, some changed
operations cost about 0.3–2.2 ns more per call. For example, the length-seven
`f32` Vec norm rises from 4.18 to 6.36 ns. The slice checks, reduction setup,
and benchmark dispatch can outweigh the saved accumulation work at these sizes.
A separate small-vector policy would need a consumer that benefits enough to
justify another branch or kernel.

The consumer comparison calls the existing ridge solver from
[`ridge_primal_dual.rs`](../../examples/ridge_primal_dual.rs), with its
algorithm unchanged in both builds. Each sparse design has four random triplets
per row, including possible duplicates, center 0.001, scale 1.75, and ridge
penalty 2. Shapes are 512 by 512 and 32,768 by 256. Each observation times 30
complete fits, including solver allocations and result destruction. Design
construction is outside timing. An untimed fit in every process is checked with
independent triplet-based products, an objective, and a stationarity residual.
The fit benchmark covers `f64` with faer storage; it does not establish consumer
gains for the other backends or `f32`.

Complete-fit medians and inclusive interquartile intervals, in milliseconds:

  | Shape         | Baseline             | Candidate            | Time reduction |
  | ------------- | -------------------: | -------------------: | -------------: |
  | 512 by 512    | 0.079 [0.076, 0.081] | 0.068 [0.064, 0.070] |          14.1% |
  | 32,768 by 256 | 2.625 [2.592, 2.642] | 2.478 [2.464, 2.530] |           5.6% |

The smaller fit gains show that the reductions are only part of the solver's
work. The independent objectives agree between builds in every process:
879.4507901820218 for the square design and 64,971.86056845501 for the tall
design. The largest independently computed gradient norm is 3.56e-9.

[`solver_vector_timings.csv`](solver_vector_timings.csv) retains the timing
observations, [`solver_vector_errors.csv`](solver_vector_errors.csv) retains the
numerical checks, and [`solver_vector_summary.csv`](solver_vector_summary.csv)
reports medians and inclusive interquartile intervals for every case. In the
error CSV, ridge rows use `absolute_error` for the gradient norm and
`scaled_error_or_objective` for the independently computed objective.

The data contain 60,600 timing observations, 12,096 reduction checks, and 24
independent fit checks. Every numerical check passes. Among candidate contiguous
vector reductions, the largest observed error divided by `1 + abs(oracle)` is
4.71e-6 for `f32` and 6.16e-15 for `f64`. These fixtures do not imply uniformly
better accuracy: reduction order still affects rounding and intermediate
overflow, as documented above. Shared correctness tests also cover empty
vectors, short tails, nonfinite values, dimension mismatches, and the native
faer norm's protection against overflow and underflow.

Validation passed the full `task test` backend, parallelism, and feature
unification matrix, `task fmt`, `task lint`, and `task docs`. The benchmark
runner passed its shell syntax and formatting checks, and the summary script
passed Ruff checks.

Reproduce the comparison and its summary from the repository root:

```sh
compute bench --reason "Remaining solver-facing vector reductions" -- \
    env PROBE_BENCH=vector_reductions \
    PROBE_FEATURES=faer,nalgebra,ndarray PROBE_PREFIX=solver_vector \
    PROBE_BLOCKS=6 PROBE_SAMPLES=5 PROBE_ELEMENTS=5000000 PROBE_FITS=30 \
    bash benches/column_reductions/run-adapters.sh \
    target/solver-vector-reductions 8750c46
python benches/column_reductions/summarize-vectors.py \
    target/solver-vector-reductions/solver_vector_timings.csv \
    > target/solver-vector-reductions/solver_vector_summary.csv
```

The runner copies the same benchmark and ridge example into the exported
baseline. The example differs from the baseline only in exposing its existing
ridge entry point and backend aliases to the benchmark. `PROBE_LENGTHS`,
`PROBE_ELEMENTS`, and `PROBE_FITS` override the vector lengths, work per sample,
and fits per sample. The runner records source and binary hashes, compiler and
CPU information, and all sampling settings. The measured run used temporary
build directories in `/dev/shm` because the main filesystem initially ran out of
space. A shorter pilot showed substantial variation in unchanged native controls
and remains local; the final measurements use longer samples and more
observations.

## Vector sums and weighted column reductions

The follow-up comparison uses baseline revision `52b1bac` and the production
source identified in [`vector_metadata.json`](vector_metadata.json). It extends
the existing slice kernels to `VectorView::sum`, weighted column dots, and
weighted squared norms. Weighted norms normalize each entry before multiplying
and squaring, preserving the existing behavior at extreme scales. No fast path
allocates, gathers strided inputs, or changes sparse duplicate handling.

Both release builds and all measurements ran in one exclusive `compute bench`
window on CPU 2, with Rust 1.98.1, 16 codegen units, and the
`faer,nalgebra,ndarray` features. The backend releases match the adapter
measurements below. Inputs have 5,000 rows and 1,000 columns. Matrix entries,
vector entries, and signed weights are drawn uniformly from [-1, 1) with
`ChaCha8Rng` and seed 8421. Every column has center 0.375 and scale -1.75.

Four blocks alternate baseline and candidate process order. Each process checks
every result before collecting three observations of five passes, reversing
operation order on alternate observations. The table reports medians of 12
observations in milliseconds per pass. Vector sums repeat the same length-5,000
vector 1,000 times per pass, as ordinary dots do when visiting all columns.
Cached dot variants receive independently computed sums outside timing.

| Backend  | Operation               | Baseline | Candidate | Speedup |
| -------- | ----------------------- | -------: | --------: | ------: |
| ndarray  | Vector sum              |    2.271 |     0.475 |   4.78x |
| ndarray  | Dot                     |    3.947 |     1.998 |   1.98x |
| ndarray  | Weighted dot            |    7.842 |     2.763 |   2.84x |
| ndarray  | Cached-sum weighted dot |   10.453 |     1.836 |   5.69x |
| ndarray  | Weighted squared norm   |   12.428 |     2.686 |   4.63x |
| faer     | Vector sum              |    2.260 |     0.484 |   4.67x |
| faer     | Dot                     |    3.843 |     1.970 |   1.95x |
| faer     | Weighted dot            |    5.889 |     2.655 |   2.22x |
| faer     | Cached-sum weighted dot |    4.418 |     1.731 |   2.55x |
| faer     | Weighted squared norm   |   11.669 |     2.646 |   4.41x |
| nalgebra | Vector sum              |    2.225 |     0.481 |   4.62x |
| nalgebra | Dot                     |    3.864 |     1.996 |   1.94x |
| nalgebra | Weighted dot            |    7.200 |     2.714 |   2.65x |
| nalgebra | Cached-sum weighted dot |    4.671 |     1.849 |   2.53x |
| nalgebra | Weighted squared norm   |   12.383 |     2.710 |   4.57x |

Cached unweighted dots remain within 2% of baseline on contiguous inputs.
[`vector_timings.csv`](vector_timings.csv) retains all 864 observations,
including separate ndarray controls for strided vectors, strided weights, and
reversed rows. These layouts retain the indexed fallbacks where necessary.
Uncached operations can still benefit from faster vector sums when their columns
are noncontiguous. These are kernel measurements, without downstream solver
timings.

[`vector_errors.csv`](vector_errors.csv) records all 288 groups of 1,000
comparisons against independent, sequential arithmetic, with columns normalized
explicitly. Every comparison passes. The maximum absolute error is 4.48e-13, and
the maximum error divided by `1 + abs(expected)` is 1.24e-13. Input construction,
oracle materialization, and verification are outside timed passes.

Reproduce the comparison from the repository root:

```sh
compute bench --reason "Vector sums and weighted column reductions" -- \
    env PROBE_BENCH=column_vector_reductions \
    PROBE_FEATURES=faer,nalgebra,ndarray PROBE_PREFIX=vector \
    bash benches/column_reductions/run-adapters.sh \
    target/vector-reductions 52b1bac
```

The runner retains its adapter benchmark defaults. `PROBE_BENCH`,
`PROBE_FEATURES`, and `PROBE_PREFIX` select the follow-up harness, build
features, and output filenames. Sampling overrides are the same as below.

## Faer and nalgebra adapter measurements

The adapter comparison uses baseline revision `ac5f2e4`, before these backends
expose the hooks, and the working source identified in
[`adapter_metadata.json`](adapter_metadata.json). Both release builds and all
measurements ran in one exclusive `compute bench` window on CPU 2, with Rust
1.98.1, 16 codegen units, and the `faer,nalgebra` features. The measured releases
are faer 0.24.4 and nalgebra 0.35.0. Correctness tests exercise every supported
release through the same shared adapter implementations.

Each input has 5,000 rows and 1,000 columns, with `f64` values drawn uniformly
from [-1, 1) using `ChaCha8Rng` and seed 8421. The inputs have identity
normalization. Four blocks alternate baseline and candidate process order.
Each process checks every column result before collecting three observations
of five complete passes, alternating operation and method order by observation.
The table shows medians of 12 observations, in milliseconds per pass over all
1,000 contiguous columns. Cached dots use a borrowed contiguous backend vector.

| Backend  | Operation      | Baseline | Candidate | Native |
| -------- | -------------- | -------: | --------: | -----: |
| faer     | Sum            |    2.854 |     0.603 |  0.587 |
| faer     | Squared norm   |    2.885 |     0.670 |  0.601 |
| faer     | Cached-sum dot |    3.512 |     1.715 |  1.325 |
| nalgebra | Sum            |    2.852 |     0.617 |  3.116 |
| nalgebra | Squared norm   |    2.887 |     0.672 |  0.723 |
| nalgebra | Cached-sum dot |    3.553 |     1.743 |  2.157 |

Contiguous sums improve by 4.6–4.7 times, squared norms by 4.3 times, and cached
dots by about twice. Offset borrowed ranges show similar gains. Owned backend
vectors also benefit. The native controls use faer's `sum`, `squared_norm_l2`,
and `inner_prod`, and nalgebra's `sum`, `norm_squared`, and `dot`, on the same
borrowed layouts.

[`adapter_timings.csv`](adapter_timings.csv) preserves all 2,160 observations,
including row-major storage, reversed faer rows, nalgebra rows with stride two,
and strided input vectors. Noncontiguous columns retain the existing reductions
and measure within 1% of baseline. Strided vectors retain the existing dot path,
while sums and norms still benefit when their columns are contiguous.

[`adapter_errors.csv`](adapter_errors.csv) records the maximum absolute and
scaled errors for every process, layout, operation, and method. All 720 groups
of 1,000 column comparisons pass against independent sequential indexed
arithmetic. The maximum absolute error is 1.44e-11, and the maximum error divided
by `1 + abs(expected)` is 9.42e-14. No numerical check runs inside a timed pass.
Input construction, owned-vector copies, and cached vector sums are also outside
timing. These measurements cover individual raw reductions; the ndarray consumer
fit measurements below remain separate.

Reproduce the comparison from the repository root:

```sh
compute bench --reason "Faer and nalgebra column reductions" -- \
    bash benches/column_reductions/run-adapters.sh \
    target/adapter-column-reductions ac5f2e4
```

The script exports the baseline from Git and copies the same benchmark into it.
It uses separate Cargo target directories and rejects identical binaries before
measuring. It writes the combined timing and error CSVs, source and binary hashes,
compiler information, CPU information, and sampling settings to the output
directory. Override `PROBE_CPU`, `PROBE_ROWS`, `PROBE_COLS`, `PROBE_BLOCKS`,
`PROBE_SAMPLES`, or `PROBE_PASSES` as needed. The baseline revision is the second
argument.

Validation passed all 129 faer and nalgebra adapter tests, formatting, Clippy,
Rustdoc, Python checks and packaging, release checks, and website checks. The core
backend combinations also passed, including every supported version enabled
together with and without parallelism. The remaining individual-release,
coexistence, and BLAS checks were not completed in this run.

## ndarray production measurements

The production comparison uses the same preserved inputs, CPU 2, release
configuration, nine observations of 30 passes, and alternating method order as
the initial investigation. The following Rust 1.98.1 measurements use 16 codegen
units. Times are milliseconds for all 1,000 length-5,000 columns:

| Operation      | Published 0.9.0 | Production | ndarray |
| -------------- | --------------: | ---------: | ------: |
| Cached-sum dot |           3.743 |      1.804 |   1.836 |
| Sum            |           2.798 |      0.616 |   0.611 |
| Squared norm   |           2.797 |      0.683 |   0.672 |

[`implementation_timings.csv`](implementation_timings.csv) also includes one
codegen unit and the layout controls. [`implementation_errors.csv`](implementation_errors.csv)
records the numerical comparisons. Assembly confirms packed `mulpd` and `addpd`
instructions with independent accumulators in the ordinary release dot kernel.
[`implementation_metadata.json`](implementation_metadata.json) identifies the
source, binaries, toolchains, settings, and local assembly listings.

Both downstream comparisons use the baseline checkout and production source,
the same Shrinkage compatibility patch, Rust 1.89.0, and one codegen unit. Five
blocks alternate build order, with one warmup and five observations per case
per process. The following values are medians of 25 observations in milliseconds:

| Case       | Native baseline | Native production | Python baseline | Python production |
| ---------- | --------------: | ----------------: | --------------: | ----------------: |
| Original   |          18.872 |             8.356 |          24.835 |            12.312 |
| Larger     |          70.389 |            31.002 |          94.044 |            46.768 |
| Correlated |          74.494 |            34.324 |          84.004 |            35.338 |
| Weak       |         166.750 |           104.987 |         178.314 |           100.409 |
| Strong     |          17.542 |             8.044 |          24.122 |            12.038 |
| Zero       |          11.966 |             5.438 |          18.294 |             9.696 |

Native fits take 37–56% less time. Python fits take 44–58% less time, including
Python model construction, validation, copies into private Rust-owned storage,
preprocessing, solving, and destruction of the owned input storage before
returning. Loading arrays, diagnostics, and independent verification remain
outside timing. Native and Python inputs have different allocation paths, so
compare each mode against its own baseline. Individual observations are in
[`implementation_fits.csv`](implementation_fits.csv) and
[`python_fits.csv`](python_fits.csv).

All 600 measured fits pass the independent objective, KKT, dual-objective, gap,
and returned-certificate checks. The maximum relative gap is 5.56e-9, maximum
KKT violation is 1.24e-9, and maximum measured feasibility excess is zero. Both
builds take the same number of sweeps in every case.

The separate fused-summary experiment checks raw sums and centered squared
norms individually before timing. On the Fortran fixture, the production pair
takes 1.250 ms, separate slice reductions take 1.233 ms, and a simple fused
sixteen-accumulator prototype takes 1.489 ms. The explicitly centered fixture
shows the same ordering. [`summaries.csv`](summaries.csv) preserves the results.
These measurements support keeping the individual operations in the public API.

Production validation passed the full `task test` backend and feature matrix,
`task test:blas`, formatting, Clippy, Rustdoc, Python checks and packaging,
release checks, and website checks. The initial `task ci` invocation reached the
website check before finding a missing Diplodocus executable on `PATH`. Setting
the documented `DIPLODOCUS_BIN` override resolved it; that phase and the remaining
test phases then passed.

## Initial investigation

The November 3, 2026 Shrinkage kernel gap reproduces on the baseline checkout,
revision `b6fb52a`. On the
preserved 5,000 × 1,000 Fortran-contiguous input, one pass of
`LazyColumn::dot_with_sum` takes 3.769 ms versus 1.803 ms for ndarray's dot in the
baseline default release configuration. Contiguous prototypes with eight
independent accumulators take 1.799 ms for dots, 0.604 ms for sums, and 0.682 ms
for squared norms. An isolated library prototype also reduces native Shrinkage
fit times by 39–56% on six cases, with independently verified diagnostics.

The following original measurements and experimental patches retain the
evidence used to choose the production implementation.

## Kernel comparison

Times below are milliseconds per pass over all 1,000 columns, with 5,000 entries
per column. Each entry is the median of nine observations of 30 passes. Methods
alternate forward and reverse order between observations. Fixture construction,
normalization parameters, vector copies, correctness checks, and serialization
are outside timing. Each method receives one additional warmup pass.

All builds use release optimization, generic x86-64 instructions, no BLAS, and
no parallel feature. One exclusive `compute bench` window contains all four
builds and their comparisons, pinned to logical CPU 2 of an AMD Ryzen 9 7900.
The published crate and checkout share ndarray 0.17.2. The original input hashes,
toolchain versions, source hashes, and settings are in
[`metadata.json`](metadata.json).

| Rust / codegen units | Published dot | Checkout dot | ndarray dot | Slice dot, 8 accumulators |
| -------------------- | ------------: | -----------: | ----------: | ------------------------: |
| 1.98.1 / 16          |         3.819 |        3.769 |       1.803 |                     1.799 |
| 1.98.1 / 1           |         3.802 |        3.744 |       1.746 |                     1.732 |
| 1.89.0 / 16          |         3.993 |        3.971 |       1.678 |                     1.676 |
| 1.89.0 / 1           |         3.801 |        3.758 |       1.780 |                     1.833 |

The historical published result was 3.706 ms versus 1.667 ms. The current runs
confirm the gap with both the historical and current compiler. Using one codegen
unit does not close it.

The following controls use Rust 1.98.1 and 16 codegen units:

| Operation    | Checkout | Serial slice | Slice, 8 accumulators | ndarray |
| ------------ | -------: | -----------: | --------------------: | ------: |
| Dot          |    3.769 |        3.622 |                 1.799 |   1.803 |
| Sum          |    2.800 |        2.848 |                 0.604 |   0.614 |
| Squared norm |    2.803 |        2.875 |                 0.682 |   0.671 |

[`timings.csv`](timings.csv) retains all 1,728 observations, including a second
Fortran comparison and the additional layouts. `CurrentDot` uses a `Vec` input;
`CurrentDotView` uses the ndarray vector view. The `Vec` input is an untimed copy
with identical logical values. View methods preserve vector strides.

## Cause and implementation options

[`LazyColumn::dot_with_sum`](../../src/column.rs) reduces through
`RawColumn::for_each_stored`, with one accumulator and indexed vector access.
ndarray's contiguous dot instead uses eight accumulators. The independent
accumulations allow packed arithmetic without enabling fast-math. A serial
slice control saves only about 4%, while the contiguous eight-accumulator
prototype closes the dot gap. Reduction order accounts for most of the
difference on this fixture.

Assembly from the ordinary 16-unit builds also contains outlined ndarray
visitors with iterator-mode branches and an accumulator store inside the loop.
The generated listings are retained under the local evidence directory recorded
in metadata. The benchmark's `Inline*` methods copy the adapter's visitor into a
small backend wrapper with `#[inline]`; this control does not materially improve
the result. Inlining remains relevant to consumer regressions, but it does not
provide the missing independent accumulations.

Contiguity also helps the compiler eliminate indexing checks. Eight accumulators
with generic ndarray indexing take 2.420 ms for dots. Processing contiguous
column chunks against a generic `Vec` input takes 2.226 ms. Borrowing both inputs
as slices takes 1.799 ms. These controls support exposing contiguous vector
access as well as a backend reduction hook or contiguous raw-column access.

The isolated [`contiguous-prototype.patch`](contiguous-prototype.patch) adds
optional borrowed dense slices to `RawColumn` and contiguous slices to
`VectorView`. It uses them for raw sums, cached-sum dots, and directly centered
squared norms. Both hooks default to `None`. Only ndarray's dense columns and
the relevant vector implementations opt in. Sparse columns and strided columns
retain their existing paths, including duplicate coalescing and implicit zeros.
The fast paths use constant workspace and allocate nothing. The patch is an
experiment against the recorded baseline; production shares these reductions
through a separate private core module and documents both borrowing hooks.

The sum and norm results justify improving the individual reductions first.
Shrinking both scans makes a separate comparison of fused summaries useful;
these measurements do not establish the benefit of a fused API.

## Layout and numerical checks

All methods are checked column by column against a materialized logical-column
calculation accumulated with compensated summation. The full-size controls cover
Fortran storage, explicit centering and scaling, row-major storage, every other
row with a strided vector, and reversed rows with a reversed vector. Additional
fixtures exercise lengths 0, 1, 7, 8, 9, 15, 16, 17, 31, 32, and 33 with seven
columns on both compilers. [`errors.csv`](errors.csv) records maximum absolute
errors for the full-size fixtures.

For the raw Fortran fixture, maximum dot errors are 1.82e-11 for the checkout and
2.73e-12 for the contiguous prototype. Sum errors are 6.91e-13 and 2.25e-13, and
squared-norm errors are 3.55e-11 and 4.55e-12. The centered prototype subtracts the
center before squaring and retains the existing scale correction.

Layout changes affect the opportunity. The row-major dot takes 8.013 ms in the
checkout and 7.595 ms in ndarray; the indexed eight-accumulator prototype takes
7.235 ms. Reversed-row sums expose another opportunity: ndarray's sum can use
memory order, taking 0.610 ms versus 3.329 ms for the logical visitor. A
contiguous positive-stride fast path alone does not address these cases.

Reordering is not uniformly more accurate. [`numerics.csv`](numerics.csv)
contains two cancellation examples: one loses a unit that the serial reduction
retains, and one recovers units that the serial reduction loses. Alternating
`f64::MAX` and `-f64::MAX` eight times produces zero serially but NaN in both the
prototype and ndarray because intermediate pair sums overflow. Explicit NaNs,
infinities, and empty inputs are also exercised. Production accepts the usual
backend variation in addition order, including these effects. Typical-data
agreement does not establish bitwise or IEEE-class agreement for all finite
inputs. These original performance experiments cover `f64`.

Keep `VectorView` out of method-resolution scope when measuring ndarray's
inherent sum: its trait `sum` can shadow ndarray 0.17's method through `Deref`.
The probe uses the fully qualified trait name in generic bounds for this reason.

## Native Shrinkage fits

The isolated consumer uses Shrinkage revision
`3dd3c095547f9893c429646a991323e870eb0d9a`. The same
[`compatibility patch`](shrinkage-compat.patch) adapts both builds to the
checkout's normalization API without changing solver rules. The baseline uses
the checkout, and the candidate uses the contiguous prototype. Both use Rust
1.89.0, one codegen unit, ndarray 0.17.2, and no BLAS or parallel Rust feature.

Five blocks alternate build order. Each process warms each case once, then
measures five fits, giving 25 observations per build and case. Both builds share
one exclusive compute window and CPU 2. The timer covers the native `model.fit`
call. Input loading, model construction, returned-fit destruction,
serialization, and independent verification are outside timing. This measures
complete native fits with borrowed inputs; Python input copying and extension
overhead require a separate measurement.

Times are milliseconds, with inclusive interquartile intervals in brackets:

| Case       |                   Checkout |                  Prototype | Reduction | Sweeps, both builds |
| ---------- | -------------------------: | -------------------------: | --------: | ------------------: |
| Original   |    18.868 [18.754, 19.215] |       8.346 [8.267, 8.444] |       56% |                  10 |
| Larger     |    70.511 [70.261, 70.935] |    31.752 [31.412, 32.095] |       55% |                  10 |
| Correlated |    74.142 [73.771, 74.213] |    34.098 [33.822, 34.386] |       54% |                  90 |
| Weak       | 167.539 [166.718, 167.789] | 102.827 [100.262, 103.974] |       39% |                  60 |
| Strong     |    17.443 [17.356, 17.549] |       8.022 [7.937, 8.184] |       54% |                  10 |
| Zero       |    11.978 [11.894, 12.033] |       5.232 [5.127, 5.272] |       56% |                   0 |

The original matrix is 5,000 × 1,000, the larger matrix is 10,000 × 2,000, and
the correlated matrix is 5,000 × 1,000. The weak, strong, and zero cases multiply
the original penalty by 0.1, 5, and 20. All use centered input arrays, no fitted
intercept or scaling, automatic working sets, and relative duality-gap target
1e-8. Input identities are recorded in metadata.

NumPy independently recomputes the primal objective, KKT violation, reported
certificate feasibility, dual objective, and relative gap for every measured
fit. All 300 fits pass. The maximum relative gap is 5.56e-9, the maximum KKT
violation is 1.24e-9, and the maximum measured certificate-feasibility excess is
zero. Objectives agree between builds within rounding error. Sweep counts,
feature visits, and full-check counts agree. [`fits.csv`](fits.csv) preserves
each timing and independent diagnostic. A shorter pilot had wider timing
dispersion and remains in the local evidence directory; the table uses the
extended run.

## Reproduction and validation

Run from the repository root. The kernel probe is a standalone workspace, so it
does not add production dependencies or change the root lockfile:

```sh
compute bench --reason "Dense column reductions" -- env PROBE_LAYOUTS=1 \
    bash benches/column_reductions/run.sh \
    ../shrinkage/.build/sklearn-gap-20261103/data/original \
    target/column-reductions/reproduce rust-current
```

Use `-` instead of the data directory for seeded uniform synthetic inputs.
`PROBE_CPU`, `PROBE_SAMPLES`, and `PROBE_PASSES` override CPU 2, nine samples, and
30 passes. To repeat the historical compiler comparison, set `RUSTC` and
`PROBE_CARGO` to the desired compiler and Cargo executables and use a distinct
label. `run.sh` builds and measures both codegen-unit configurations, saves
assembly, checks short lengths, and writes the numerical counterexamples.
Set `PROBE_SUMMARIES=1` to include the separate and fused summary controls.

Prepare a fresh downstream directory with NumPy available in the interpreter:

```sh
compute run -- python scripts/python-tool.py python \
    benches/column_reductions/downstream.py prepare \
    target/column-reductions/reproduce-fits \
    --data ../shrinkage/.build/sklearn-gap-20261103/data
```

The preparer copies production source into the output directory, exports the
recorded baseline revision from Git, applies the Shrinkage compatibility patch,
and exports the three preserved datasets. Override the baseline with
`--baseline PATH`. It defaults to the sibling Shrinkage checkout; override that
with `--shrinkage PATH`. Choose an
interpreter and native library environment in which NumPy imports successfully.
The workspaces retain separate lockfiles for reproducible consumer builds.

```sh
compute bench --reason "Native column reduction fits" -- bash -c '
    export CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1
    for variant in current candidate; do
        CARGO_TARGET_DIR="$1/$variant-build" cargo build --release --locked \
            --manifest-path "$1/$variant/Cargo.toml" || exit
    done
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
        python scripts/python-tool.py python \
        benches/column_reductions/downstream.py run "$1"
' bash target/column-reductions/reproduce-fits
```

Select Rust 1.89.0 in the shell to reproduce the recorded native builds. Keep
both builds inside the same reservation. The runner defaults to five blocks of
five observations, with `--blocks`, `--repetitions`, and `--cpu` overrides.

For the Python comparison, add `--python` when preparing a fresh downstream
directory, then build the two isolated extension workspaces:

```sh
compute run -- bash -c '
    export CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1
    export PYO3_PYTHON="$PWD/.devenv/python/bin/python"
    for variant in current candidate; do
        CARGO_TARGET_DIR="$1/$variant-python-build" cargo build --release --locked \
            --manifest-path "$1/$variant-python/Cargo.toml" || exit
    done
' bash target/column-reductions/reproduce-fits
compute bench --reason "Python column reduction fits" -- \
    python scripts/python-tool.py python benches/column_reductions/python_fits.py \
    target/column-reductions/reproduce-fits
```

The Python runner creates a fresh process per build and block, sets numerical
libraries to one thread, pins each worker to the selected CPU, and independently
checks every fit. It uses the public Python `Lasso.fit` with its default
`copy=True`. The same block, observation-count, and CPU overrides are available.

Validation passed for the standalone probe and consumer harness with Clippy,
Rust formatting, Python lint and formatting, and shell syntax and formatting.
The library prototype passed 37 ndarray tests across releases 0.15, 0.16, and
0.17, 23 sprs tests, and the scalar-bounds test. Clippy and Rustdoc passed with
warnings denied for that feature set. The core source hashes identify the
prototype that was tested and timed.
