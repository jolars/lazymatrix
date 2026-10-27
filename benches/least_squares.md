# Complete fits and individual operations

The website's main benchmarks compare a complete ridge regression fit and
individual forward and transpose products. They use 2,048 × 128, 8,192 × 512,
and 16,384 × 1,024 matrices across seven target nonzero fractions: 0.001, 0.003,
0.01, 0.03, 0.1, 0.3, and 1.
Both paths fit mean centering and SD scaling. The lazy path keeps CSC storage;
the eager path materializes a normalized dense matrix. Backends are sprs with
ndarray as its dense destination and faer CSC with faer dense.

```sh
bash scripts/bench-least-squares.sh
mkdir -p benches/least_squares
cp target/least-squares/{results.csv,products.csv,metadata.json} benches/least_squares/
task website:check
```

`LAZYMATRIX_BENCH_QUICK=1` selects shorter timing settings. The default uses
30 samples, 0.3 seconds of warmup, at least one second of measurement, and
10,000 bootstrap resamples. Quick mode uses ten samples, 0.05 seconds of warmup,
at least 0.1 seconds of measurement, and 1,000 resamples. Long fits extend the
measurement period to collect the requested samples. Results are Criterion
means with 95% bootstrap confidence intervals. Builds and allocation profiling
finish before timing starts; collection executables run serially with one
thread. Collect timings on an otherwise idle machine.

Set `LAZYMATRIX_BENCH_SHAPES=2048x128,8192x512` to collect a subset of the
published sizes. The checked-in collection uses the default timing settings for
8,192 × 512 and quick settings for the other two sizes. Its `timing_runs`
metadata records each run's dimensions, date, and timing settings.

## Complete fit

The solver lives in `examples/common/least_squares.rs` and is shared with
`examples/least_squares_gd.rs`. It minimizes
`||A beta - y||^2 / (2n) + 0.1 * ||beta||^2 / 2` in normalized coordinates.
The response is `sin(0.13 * i) + 0.5 * cos(0.037 * i)` for zero-based row `i`.
Each fit starts from zero coefficients, estimates a step size with 30 power
iterations, and uses gradient descent with backtracking. It stops when the
gradient's L2 norm is at most `1e-7 * max(1, initial gradient L2 norm)` and
fails if 10,000 coefficient updates do not suffice.

Fixture generation, backend conversion, and response generation happen outside
measurement. Each timed fit includes fitting normalization parameters, eager
materialization when applicable, step-size estimation, solver allocations,
optimization to convergence, and dropping the result. Both strategies retain
the original sparse source. Before timing, the harness checks convergence,
coefficients, and objectives against the other representation. The example
also checks recovery of known coefficients for an unregularized noiseless fit.

`least_squares_memory` warms both paths and measures each complete lifecycle
with the shared allocation tracker. It adds the already allocated source and
response to the peak increase in live requested Rust bytes. It includes retained
parameters, the eager dense matrix, and solver workspace whenever live; it does
not sum phase peaks. Allocator overhead, native-library allocations, and
unrelated process memory are excluded. Reallocations count their resulting
size, not temporary allocator internals. Memory profiling and timing run in
separate executables so tracking does not affect reported wall-clock time.

The joined `results.csv` records both strategies' times, confidence bounds,
peaks, iteration counts, objectives, and final gradient norms. The metadata
records hardware, compiler, collection date, and settings.

## Individual products

The collector also runs `eager_normalization` and `normalization_memory` for
the same shapes and fixtures. `products.csv` follows the existing operation
[measurement format](eager_normalization.md). Forward and transpose timings
exclude parameter fitting and materialization, reuse output buffers, and include
scratch inside the public operation. Operation memory includes normalization,
construction, and ten product pairs, with the sparse source retained.

Only mean centering with SD scaling and the lazy/eager strategies appear in
the main plots. Scaling-only cases, raw sparse references, and memory breakdowns
remain in the CSV. The previous full size/shape sweep remains in
`normalization_tradeoffs/` with its original metadata, separately from this
collection.

## Smoke checks

```sh
LAZYMATRIX_BENCH_SMOKE=1 cargo bench --locked --bench least_squares \
  --features ndarray,sprs,faer -- --test
LAZYMATRIX_BENCH_SMOKE=1 cargo bench --locked --bench least_squares_memory \
  --features ndarray,sprs,faer
cargo run --locked --example least_squares_gd --features faer
```

Smoke fixtures use 256 × 16 matrices at all seven densities, including cases
with empty columns. Smoke collections are for numerical checks and must not
replace the published measurements.
