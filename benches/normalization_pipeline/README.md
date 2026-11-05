# Ordered normalization benchmarks

Run paired eager and lazy comparisons in one exclusive reservation:

```sh
compute bench --reason "Compare ordered normalization runtime and workspace" -- \
  bash scripts/bench-normalization-pipeline.sh
```

The script writes timing estimates, allocation profiles, metadata, and the full
Criterion log to `target/normalization-pipeline/`. Set
`LAZYMATRIX_BENCH_SMOKE=1` for 256- and 1,024-row fixtures, or
`LAZYMATRIX_BENCH_QUICK=1` for shorter timing samples. Refresh `results.csv`,
`memory.csv`, `metadata.json`, and this description together after changing the
kernels or methodology.

The recorded run used Rust 1.98.1 on an Intel Core Ultra 7 155U on November 2, 2026. It enabled `ndarray_v0_17` and `sprs_v0_11` without parallelism. Fixtures
use `ChaCha8Rng` with seed 205, 128 columns, 2,048 or 8,192 rows, and stored
densities of 0.1% or 1%. Timing uses 20 samples, 0.2 seconds of warmup, 0.5
seconds of measurement, and 10,000 bootstrap resamples. The CSV reports median
times with 95% confidence intervals.

Column pipelines center by the mean and scale by the L2 norm, then standardize
columns. Both CSC and CSR retain their native storage. The dense mixed pipeline
standardizes rows, then columns. Each eager baseline retains the source, creates
one normalized dense matrix, and applies its second stage in place. Correctness
checks compare paired forward and transpose products before timing.

`fit_pair` measures fitting followed by one forward and one transpose product,
including vector allocation. `pair` measures products with fitting excluded and
reusable outputs. The allocation executable runs separately from timing and
measures peak additional live requested Rust bytes during fitting and one
product pair, excluding source storage. The `lifecycle_peak_bytes` column always
describes that complete lifecycle, including on product-only timing rows.

At 8,192 × 128 and 1% stored density:

| Pipeline              | Lazy fit and pair | Eager fit and pair | Lazy peak bytes | Eager peak bytes |
| --------------------- | ----------------: | -----------------: | --------------: | ---------------: |
| CSC columns           |           2.53 ms |           21.57 ms |         356,352 |        8,521,728 |
| CSR columns           |           2.95 ms |           18.26 ms |         347,136 |        8,521,728 |
| Dense row then column |          29.75 ms |           20.21 ms |         278,528 |        8,522,752 |

For the same sparse fixture, product-only pairs took 2.40 ms lazily versus
3.00 ms eagerly for CSC, and 2.81 ms versus 3.03 ms for CSR. These measurements
illustrate a workspace tradeoff: the dense mixed path uses about 31 times less
additional memory but takes about 1.47 times as long for fitting and one pair.
They do not establish a universal speedup.

Sparse fitting scans native storage, coalesces duplicates, and accounts for the
transformed implicit background. Sparse products use logarithmic queries over
missing index ranges instead of subtracting large background totals or visiting
every implicit cell. Their scratch grows with the matrix dimensions, while
the eager baseline grows with their product. Exceptional floating-point cases
can descend through individual missing contributions. This timing run measures
ordinary finite arithmetic; correctness suites cover those fallbacks.

## Fused vector products

Compare fused calls with explicit allocating products and output updates in one
exclusive reservation:

```sh
compute bench --reason "Compare fused and manual normalized vector products" -- \
  env LAZYMATRIX_BENCH_SMOKE=1 cargo bench --locked --no-default-features \
  --features ndarray,sprs --bench normalization_pipeline -- \
  fused_pipeline/256x128/0.01 --noplot --warm-up-time 0.1 \
  --measurement-time 0.3 --sample-size 20 --nresamples 10000
```

Each iteration computes one forward and one transpose product with
`alpha = -0.75` and `beta = 0.25`. The manual path allocates both product vectors
and combines them with the existing outputs. The fused path makes two fused
calls, retaining one temporary product vector at a time. Both paths reuse their
final outputs, preserve the fitted normalization, and include product-vector
allocation in the timing. Fitting and source construction happen before timing.
Every fused output is checked against the manual calculation before measurement.

The selected seed-205 fixture has 256 rows, 128 columns, and 1% stored density.
It exercises dense row normalization, a dense row-then-column chain, and CSC
and CSR column chains. Timing uses 20 samples, 0.1 seconds of warmup, 0.3 seconds
of measurement, and 10,000 bootstrap resamples.

The [measurements](fused.csv) and [metadata](fused-metadata.json) were collected
November 5, 2026, on an AMD Ryzen 9 7900 with Rust 1.98.1. The CSV reports medians
and 95% confidence intervals.

| Operator              | Manual pair median | Fused pair median |
| --------------------- | -----------------: | ----------------: |
| Dense rows            |           78.96 µs |          75.77 µs |
| Dense row then column |          122.16 µs |         125.44 µs |
| CSC column chain      |           36.86 µs |          37.03 µs |
| CSR column chain      |           37.47 µs |          36.90 µs |

On this fixture, fused and manual timings are close. Fused calls provide the
coefficient and output-reuse contract while retaining the existing kernels;
they still allocate product scratch and do not imply a general speedup.
