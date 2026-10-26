---
title: Benchmarks
description: See how nonzero density and matrix size affect runtime, memory, and the cost of repeated lazy and eager normalized products.
aside: false
pageClass: benchmarks-page
---

<script setup>
import BenchmarkComparison from './.vitepress/theme/BenchmarkComparison.vue'
</script>

# Benchmarks

Centering a sparse matrix generally turns its implicit zeros into nonzero values.
Lazy normalization keeps the original sparse storage and applies centering and
scaling during each operation. Eager normalization constructs a dense normalized
matrix once and uses it for subsequent operations.

The plots compare these choices across **seven nonzero densities, from 0.1% to
100%**, and three shapes: many more rows than columns ($n/p = 16$), square
($n = p$), and many more columns than rows ($p/n = 16$). Choose a matrix size
to compare all three shapes at the same number of entries. Their dense storage
occupies 2, 32, or 128 MiB.

Start with mean centering and SD scaling, then switch to scaling alone to see a
case where normalization preserves structural zeros. Each sparse backend has its
own dense destination; the backend control changes both together.

<BenchmarkComparison />

## Read the memory figures

The full lifecycle measurement starts with the sparse input already allocated.
It records the largest increase in live requested Rust bytes while fitting
parameters, constructing the operator, and running ten forward/transpose pairs,
then adds the input's values, indices, and column pointers. Both paths retain the
source input. The eager dense output stays included during all subsequent work.
Phase peaks are never summed.

The normalized-storage view counts the representation and retained parameters.
The workspace view subtracts that retained storage from the lifecycle peak;
its maximum can come from fitting, construction, or products. Input and output
vectors count as workspace. The detailed table reports all three views.

These figures exclude allocator overhead, native-library allocations, resident
memory effects, and unrelated process state. They count requested bytes, with
reallocation recorded at its resulting size. Dropping the sparse source after
eager construction would reduce retained storage during products. Dense inputs
that support `into_eager` can instead normalize their existing storage in place;
these plots compare sparse inputs with an explicitly allocated dense output.

## Separate setup from repeated work

The last plot projects total time for $k$ forward/transpose pairs:

$$
\begin{aligned}
T_{\mathrm{lazy}}(k) &= T_{\mathrm{fit}} + k\,t_{\mathrm{lazy}}, \\
T_{\mathrm{eager}}(k) &= T_{\mathrm{fit}} + T_{\mathrm{materialize}}
                       + k\,t_{\mathrm{eager}}.
\end{aligned}
$$

Here each $t$ is the sum of separately measured forward and transpose times.
Both paths use the same fitted parameters. When eager products are faster, the
estimated crossover is

$$
k = \frac{T_{\mathrm{materialize}}}
         {t_{\mathrm{lazy}} - t_{\mathrm{eager}}}.
$$

This projection assumes fixed product costs and reuses the same vectors and
matrix. It does not model solver convergence, changing cache behavior, or memory
pressure from other application state. If eager products are slower per pair,
the model has no crossover. Scaling alone preserves structural zeros, so its
dense eager conversion is a comparison with this crate's dense materialization
API, not an optimized sparse scaling baseline.

## Methodology and reproduction

Fixtures use `ChaCha8Rng` with seed 205 and the following shapes. Each row keeps
the element count fixed while changing the aspect ratio:

| Dense storage | $n/p = 16$          | $n = p$            | $p/n = 16$          |
| ------------- | ------------------- | ------------------ | ------------------- |
| 2 MiB         | $2048 \times 128$   | $512 \times 512$   | $128 \times 2048$   |
| 32 MiB        | $8192 \times 512$   | $2048 \times 2048$ | $512 \times 8192$   |
| 128 MiB       | $16384 \times 1024$ | $4096 \times 4096$ | $1024 \times 16384$ |

Target nonzero fractions are $0.001$, $0.003$, $0.01$, $0.03$, $0.1$, $0.3$, and $1$.
Each cell is selected independently. Stored values are positive, with varying
column scales, so nonempty columns have nonzero means. Very sparse fixtures
can have empty columns, especially when there are few rows; their counts and
realized densities appear in the table and CSV. Construction replaces their
zero SDs with one.

Fixture generation and backend conversion happen outside measurement. Inputs
use canonical CSC storage with `usize` indices. Dense destinations use each
backend's default layout. Products reuse output buffers and include any scratch
allocation performed by the public `*_into` methods. Forward inputs and
transpose inputs are deterministic, nonconstant vectors with nonzero sums.
The harness checks both product directions against the eagerly normalized
matrix before collecting timings.

The collector builds both executables first, then runs allocation profiling and
timing serially in separate executables. Timings use Criterion's mean estimate
and 95% bootstrap confidence intervals. The checked-in tall cases use 30 samples,
0.3 seconds of warmup, at least one second of measurement, and 10,000 bootstrap
resamples. Square and wide cases use a shorter run with ten samples, 0.05 seconds
of warmup, at least 0.1 seconds of measurement, and 1,000 bootstrap resamples.
Larger operations can extend the measurement period. The run metadata records
both collections. The default collector uses the longer settings for all cases.
The plots show intervals at sampled densities and
connect those samples. Confidence intervals describe timing variation on this
machine, not variability across hardware or sparse matrix patterns. Thread
settings, hardware, compiler, and collection date are available in the
[run metadata](/benchmarks/metadata.json).

```sh
# Collect the complete grid, including both backends and normalization modes.
bash scripts/bench-normalization-tradeoffs.sh

# Use shorter timing settings for a quick comparison.
LAZYMATRIX_BENCH_QUICK=1 bash scripts/bench-normalization-tradeoffs.sh

# Publish a checked run and regenerate the website plots.
mkdir -p benches/normalization_tradeoffs
cp target/normalization-tradeoffs/{results.csv,metadata.json} benches/normalization_tradeoffs/
task website:check

# Run the numerical checks quickly without collecting timing samples.
LAZYMATRIX_BENCH_SMOKE=1 cargo bench --locked --bench eager_normalization \
  --features ndarray,sprs,faer -- --test
```

The website generates downloadable SVGs from checked-in measurements during its
build. It never runs benchmarks. For performance decisions, collect a full run
on an otherwise idle machine with representative data and operations.

The earlier [Gaussian ridge fitting measurements](/benchmarks/eager-results.csv)
remain available as a consumer example. Those single instrumented runs used
smaller fixtures and a different machine; they are separate from this product
sweep. See the [measurement notes](https://github.com/jolars/lazymatrix/blob/main/benches/eager_normalization.md)
for details.
