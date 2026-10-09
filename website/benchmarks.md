---
title: Benchmarks
description: Compare construction time, fitting time, and additional memory for lazy and eager normalized matrices.
---

<script setup>
import BenchmarkComparison from './.vitepress/theme/BenchmarkComparison.vue'
</script>

# Benchmarks

Lazy normalization preserves the original matrix representation. Eager
normalization pays a construction cost and then operates on normalized dense
storage. Which choice helps depends on matrix size, sparsity, backend, and the
operations your algorithm repeats.

## Initial fitting measurements

These measurements compare a Gaussian ridge fit through lazy and eager operators,
with an intercept and penalty of 0.1. Both paths fit the same problem, and the
harness checks coefficients, intercepts, and objectives after converting eager
coefficients back with the fitted parameters.

::: info Exploratory measurements
These are single instrumented runs, collected while other builds were active.
They illustrate the tradeoffs and are not stable performance rankings. Run the
benchmarks on your target hardware before choosing a representation.
:::

<BenchmarkComparison />

[Download all raw measurements](/benchmarks/eager-results.csv), including the
centered-only cases. The chart and table are generated from the checked-in CSV.

## Read memory figures across phases

The allocator records the peak live requested Rust bytes **above the live bytes
at the start of each phase**. Eager construction includes its dense output, but
eager fitting starts with that output already allocated. A lower eager-fit peak
therefore does not imply lower overall memory.

The measurements exclude native BLAS/LAPACK allocations, allocator overhead,
and resident-memory effects. Total process memory also depends on whether the
original input stays live. For owned writable dense inputs, `into_eager` can
normalize the original storage instead of creating a second matrix.

## Separate setup from repeated work

For $k$ repetitions of a fixed operation, compare

$$
\begin{aligned}
T_{\mathrm{lazy}}(k) &= T_{\mathrm{fit}} + k\,t_{\mathrm{lazy}}, \\
T_{\mathrm{eager}}(k) &= T_{\mathrm{fit}} + T_{\mathrm{materialize}}
                       + k\,t_{\mathrm{eager}}.
\end{aligned}
$$

When $t_{\mathrm{eager}} < t_{\mathrm{lazy}}$, materialization pays for itself
after roughly

$$
k > \frac{T_{\mathrm{materialize}}}
         {t_{\mathrm{lazy}} - t_{\mathrm{eager}}}.
$$

This estimate assumes the same fitted parameters and fixed operation costs.
The fitting measurements above cover complete solver runs, whose operation counts
can differ slightly. They should not be used to infer a per-product break-even
point. Sparse lazy products can also remain faster than dense eager products.

## Methodology and reproduction

The allocation profiles were collected on October 8, 2026, with Rust 1.89.0 in
release mode on an Intel Core Ultra 7 155U. Fixtures use seed 205, sizes
$512 \times 32$ and $2048 \times 128$, and target densities of 100% and 10%.
OpenBLAS and OpenMP use one thread. Fixture generation and source-storage
conversion occur outside measurement, and each fitting path is warmed first.

```sh
# Construction and repeated forward/transpose products with Criterion.
cargo bench --locked --bench eager_normalization --features ndarray,sprs

# Consumer fitting times and Rust allocation profiles.
bash scripts/bench-normalization-allocations.sh > target/normalization-allocations.csv
```

The [measurement notes](https://github.com/jolars/lazymatrix/blob/main/benches/eager_normalization.md)
describe the initial runs. The
[Criterion harness](https://github.com/jolars/lazymatrix/blob/main/benches/eager_normalization.rs)
separates parameter fitting, dense materialization, and batches of forward and
transpose products. Use its longer default measurement settings on an idle
machine for runtime comparisons.
