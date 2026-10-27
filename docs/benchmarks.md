---
title: Benchmarks
description: Compare the total time and peak memory of a complete fit, then examine individual operations with setup excluded.
aside: false
pageClass: benchmarks-page
---

<script setup>
import BenchmarkComparison from './.vitepress/theme/BenchmarkComparison.vue'
</script>

# Benchmarks

Lazy normalization preserves sparse storage. Eager normalization constructs a
dense normalized matrix. Compare a complete regression fit, including peak
memory, and then individual operations with setup excluded.

<BenchmarkComparison />

## Methodology

These are synthetic, single-threaded comparisons of sparse lazy normalization
and dense eager normalization. Each backend uses its own dense destination.
They illustrate the trade-off for this workload; the faster choice depends on
the data, operations, and hardware.

### Complete regression fit

Both paths solve the same ridge problem in normalized coordinates:

$$
\min_\beta\; \frac{1}{2n}\lVert \widetilde X\beta-y\rVert_2^2
             + \frac{0.1}{2}\lVert\beta\rVert_2^2.
$$

The shared [gradient descent example](https://github.com/jolars/lazymatrix/blob/main/examples/least_squares_gd.rs)
uses reusable output buffers and starts from zero coefficients. Each fit includes
30 power iterations to estimate an initial step size, followed by gradient
descent with backtracking. Both representations stop when the gradient's L2
norm is at most $10^{-7}\max(1,\lVert g_0\rVert_2)$, where $g_0$ is the initial
gradient. The harness rejects unconverged fits and checks that coefficients and
objectives agree before timing. The downloadable results include iteration
counts, objectives, and final gradient norms.

Timing starts with the raw matrix and response already allocated. It includes
fitting centers and scales, dense materialization for the eager path, all solver
setup, solving to convergence, and dropping the fitted result. Each timed fit
starts afresh. Fixture generation and backend conversion are excluded.

### Peak memory

A separate executable measures the largest increase in live requested Rust
bytes across normalization, materialization, and the complete solve. It adds
the original sparse input and response, which both paths retain throughout the
fit. This includes the eager dense matrix and solver workspace while they are
live. It measures the complete workflow directly; it does not add phase peaks.

The figures exclude allocator overhead, native-library allocations, and
unrelated process state. They measure allocation bytes rather than process RAM.
Reallocation is recorded at its resulting size. An application that releases its
source after eager construction may retain less memory during solving, though
construction still requires the source and destination to coexist.

### Fixtures and individual operations

The size selector offers 2,048 × 128, 8,192 × 512, and 16,384 × 1,024 matrices
and updates all four charts and their tables. Dimensions are rows × columns.
Matrices use canonical CSC storage, `f64` values, and `usize` indices.
Fixtures use `ChaCha8Rng` with seed 205 and target
nonzero fractions 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, and 1. Each cell is selected
independently. Stored values are positive, with varying column scales, so
centering generally turns structural zeros into nonzero logical values. The
response is the deterministic sequence $y_i=\sin(0.13i)+0.5\cos(0.037i)$, starting
at $i=0$. Mean centering and SD scaling are fitted separately for each complete
fit; exact zero scales are replaced with one.

Individual products reuse the matrix, fitted parameters, input vectors, and
output buffers. They include any scratch allocation inside the public
`*_into` methods. Forward and transpose inputs are deterministic, nonconstant
vectors with nonzero sums. The harness checks products against the eagerly
normalized matrix before timing.

Timings use Criterion means and 95% bootstrap confidence intervals. The
8,192 × 512 collection uses 30 samples, 0.3 seconds of warmup, at least one
second of measurement, and 10,000 bootstrap resamples. The other sizes use ten
samples, 0.05 seconds of warmup, at least 0.1 seconds of measurement, and 1,000
resamples. Long fits extend measurement to collect the requested samples.
Compilation and allocation profiling finish before timing begins. Intervals
describe variation on the collection machine; they do
not describe variability across hardware or sparse patterns. The
[run metadata](/benchmarks/least-squares-metadata.json) records the CPU,
compiler, collection date, thread settings, and solver settings.

## Reproduce

```sh
# Collect complete fits, operation timings, and allocation profiles.
bash scripts/bench-least-squares.sh

# Use shorter timing settings for a quick comparison.
LAZYMATRIX_BENCH_QUICK=1 bash scripts/bench-least-squares.sh

# Publish a checked collection and regenerate the website plots.
mkdir -p benches/least_squares
cp target/least-squares/{results.csv,products.csv,metadata.json} benches/least_squares/
task website:check

# Check small fixtures and convergence without collecting timing samples.
LAZYMATRIX_BENCH_SMOKE=1 cargo bench --locked --bench least_squares \
  --features ndarray,sprs,faer -- --test
```

The website builds its charts from checked-in measurements. It never runs
benchmarks. The [measurement notes](https://github.com/jolars/lazymatrix/blob/main/benches/least_squares.md)
describe the collector and its outputs.

<details>
<summary>Additional measurements</summary>

The operation CSV also contains SD scaling without centering, unnormalized
sparse products, setup timings, and memory breakdowns. Scaling alone preserves
structural zeros; its dense eager path compares with the crate's dense
materialization API rather than an optimized sparse scaling strategy.

The earlier [full operation sweep](/benchmarks/results.csv) covers tall, square,
and wide matrices at 2, 32, and 128 MiB of dense storage. It was collected on a
different machine; its [metadata](/benchmarks/metadata.json) and
[measurement notes](https://github.com/jolars/lazymatrix/blob/main/benches/eager_normalization.md)
remain available. Earlier [consumer fitting profiles](/benchmarks/eager-results.csv)
are also retained separately.

</details>
