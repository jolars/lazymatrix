# Fused consumer benchmarks

The current comparison keeps composed intercept and normalization workspaces in
a benchmark-only prototype. The public API remains unchanged while a concrete
consumer establishes whether it needs this capability. The historical timing
results below predate the stable centered dense kernels. Current dense product
and consumer timings appear in [the centered product investigation](dense_products/README.md).

Run the consumer comparison with:

```sh
cargo bench --locked --bench fused_consumer --features faer,nalgebra,ndarray,sprs,zarrs -- '^consumer/'
```

Criterion filters select individual cases. For example, use
`^consumer/faer_csc/2000x32/density_0.01/standardized/intercept/(workspace|composed_workspace)$`
to compare public workspace reuse with the private prototype. The benchmark
uses the newest enabled version of each backend. Versioned features can select
older releases.

The same benchmark also provides an [isolated overwrite comparison](overwrite_comparison.md)
using `alpha = 1` and `beta = 0` for both products. Its product and iteration
groups use the public paths with identical response subtraction and coefficient
updates.

Each timed batch resets coefficients and takes ten least-squares steps:

```text
r = A * coefficients - response
coefficients -= step * Aᵀ * r
```

Matrix construction, normalization, responses, reusable vectors, and workspace
creation happen outside timing. Every path uses the same initial coefficients,
response, and conservative step size. Final coefficients must agree before
timing starts. The benchmark contains consumer state; the library provides only
the matrix capabilities.

The fixtures use `ChaCha8Rng` seeds 203 and 204 and contain 2,000 × 32 or
10,000 × 128 predictors. Dense fixtures have density 100%; CSC and CSR fixtures
have density 1% or 10%. Every fixture runs raw, mean-centered, and standardized,
with and without `WithIntercept`. Zarr uses an uncompressed in-memory store and
chunk shapes 256 × 16 and 1,024 × 32. These measurements do not describe disk
latency or arrays larger than RAM.

The four public paths are:

- `allocating`: allocate forward and transpose results on each step, then
  subtract the response and update coefficients. Allocating `LazyMatrix`
  forward products can also clone the coefficient input for algebraic
  normalization.
- `overwrite`: reuse residual and gradient vectors with overwrite products,
  then perform the two vector updates separately.
- `fused`: initialize the residual to the negative response and accumulate
  the forward product, then fuse the transpose product with the coefficient
  update. Any required normalization and wrapper scratch is allocated internally.
- `workspace`: perform the same fused updates with caller-owned scratch.
  `LazyMatrix` reuses its normalization buffer. `WithIntercept` reuses its
  predictor buffer, but its inner normalized operator may still allocate.

Intercept consumers also run `composed_workspace`, a private prototype in
`benches/fused_consumer.rs`. It keeps a predictor buffer and a normalization
buffer, each with one entry per predictor, excluding the intercept. It reuses
the same pair for sequential forward and transpose products, calling the
existing `LazyMatrix` workspace methods and applying the intercept after
success. This experiment adds no library methods or trait hooks.

A benchmark-only allocator counts allocation and reallocation requests and
requested bytes in a separate warmed batch. Counting is disabled during
Criterion timing. The reported bytes are cumulative requests, not peak memory.
Use the default serial feature selection for reproducible counts; parallel
backend initialization and scheduling can add allocator traffic.

## Allocation results

The serial run recorded [810 consumer allocation measurements](intercept_workspaces/allocations.csv)
across both sizes, all normalizations, and every storage fixture. The
[metadata](intercept_workspaces/metadata.json) records versions and the command.
Each public consumer checks its final coefficients against the allocating path;
the prototype checks against the public workspace path. The benchmark also
asserts that the prototype adds no allocation calls or requested bytes to the
inner operator's workspace path. That assertion includes Zarr's decoding
allocations and is disabled with `parallel`, where scheduler traffic can affect
counts.

The following counts apply per ten steps at 32 predictors:

| Predictors | Backend | Intercept | Overwrite | Fused | Workspace | Prototype |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Raw or centered | Any in-memory | No | 0 | 0 | 0 | — |
| Raw or centered | Any in-memory | Yes | 20 | 20 | 0 | 0 |
| Standardized | Dense | No | 0 | 0 | 0 | — |
| Standardized | Dense | Yes | 20 | 20 | 0 | 0 |
| Standardized | CSC or CSR | No | 10 | 20 | 0 | — |
| Standardized | CSC or CSR | Yes | 30 | 40 | 20 | 0 |

Each counted predictor buffer requests 256 bytes at this size. Fused transpose
updates with algebraic scaling need a separate raw-product buffer to preserve
previous output before normalization. The overwrite transpose can normalize its
output in place, so ordinary fused sparse calls allocate more buffers. The
prototype removes both the wrapper and normalization allocations. Public
intercept workspace methods reuse only the wrapper buffer and allocate 20
normalization buffers per ten standardized sparse steps. That difference is
5,120 requested bytes at 32 predictors and 20,480 bytes at 128 predictors. Dense
normalized kernels already need no heap normalization scratch, so the prototype
saves no allocations there.

For Zarr with 256 × 16 chunks at 2,000 × 32, decoding and scanning make 4,800
allocation requests and request 21,064,320 bytes per ten steps. Both the inner
and composed-workspace paths retain that baseline. With 1,024 × 32 chunks,
both paths make 600 requests totaling 20,983,120 bytes. Public standardized
intercept workspace calls add 20 requests and 5,120 bytes to either baseline.
Add the sparse table's counts to those baselines for the other product paths.
Fused products retain one chunk scan per product and zero chunk reads for exact
zero `alpha`, including normalized and intercept wrappers.

These finite, owned-vector fixtures establish the allocation savings. They do
not establish that a new public interface is needed or that real solver fits
become faster. Keep the experiment in the benchmark until a downstream consumer
demonstrates that need.

## Prototype timing

One run on October 28, 2026, pinned to CPU 10 of an AMD Ryzen 9 7900, compared
the public intercept workspace methods with the private prototype. It used
twenty samples, a 0.2-second warmup, and a requested 0.8-second measurement
window, without `parallel`. Both paths ran in one exclusive `compute bench`
window. These point estimates are microseconds per ten
steps for standardized 2,000 × 32 CSC predictors at density 1%, with an
intercept. The [timing records](intercept_workspaces/timings.csv) include the
reported intervals.

| Backend | Public workspace | Prototype |
| --- | ---: | ---: |
| faer CSC | 54.498 | 51.971 |
| nalgebra CSC | 66.595 | 66.531 |
| sprs CSC | 47.821 | 49.579 |

This run shows no consistent time benefit from avoiding the 20 scratch
allocations. It supports keeping the prototype in the benchmark while public
workspace composition remains deferred.

## Historical evaluation

Measured on October 7, 2026, with Rust 1.89.0 on an AMD Ryzen 9 7900, pinned to
CPU 10 without `parallel`. The full 528-case matrix used ten samples, a
0.1-second warmup, and a requested 0.3-second measurement window. The following
point estimates are microseconds per ten steps, without an intercept, at 2,000 × 32.
Dense inputs have density 100%; CSC inputs have density 1%.

  | Backend               | Raw overwrite | Raw workspace | Standardized overwrite | Standardized workspace |
  | --------------------- | ------------: | ------------: | ---------------------: | ---------------------: |
  | faer dense            |        633.62 |        643.28 |                 650.46 |                 656.54 |
  | nalgebra dense        |        830.20 |        835.10 |                 843.13 |                 854.41 |
  | ndarray               |        493.13 |        495.48 |                 506.95 |                 506.33 |
  | faer CSC              |         16.30 |         13.15 |                  29.84 |                  32.60 |
  | nalgebra CSC          |         45.17 |         27.64 |                  58.34 |                  47.40 |
  | sprs CSC              |         20.50 |         17.79 |                  35.08 |                  37.58 |
  | Zarr, 256 × 16 chunks |      1,310.20 |      1,306.50 |               1,307.80 |               1,312.60 |

For standardized 1%-density CSC with an intercept at the same size, overwrite
versus workspace took 50.30 versus 52.15 µs for faer, 78.48 versus 66.82 µs for
nalgebra, and 53.02 versus 56.97 µs for sprs. At 10,000 × 128 without an
intercept, the corresponding standardized CSC figures were 249.59 versus 257.33
µs, 542.96 versus 493.39 µs, and 268.16 versus 279.31 µs.

An isolated build of the original operator implementations repeated the 22
standardized overwrite cases at 2,000 × 32 on the same pinned CPU, using a
0.2-second warmup and a 0.5-second measurement window. Existing overwrite
products remained within 5% of those baselines, including the unchanged ndarray
and sprs controls.

The slower centered and standardized faer/sprs workspace cases were repeated
with twenty samples, a 0.2-second warmup, and a 0.8-second measurement window.
They still showed mixed results: standardized faer without an intercept took
29.95 µs with overwrite and 32.39 µs with workspace; standardized sprs took
34.20 and 36.67 µs. These are comparisons between two consumer paths, rather
than regressions in the overwrite kernels. Accumulating into an initialized
residual and separating the scaled transpose result changes the output work;
eliminating allocations does not eliminate those passes.

Caller-owned scratch provides a predictable allocation benefit. Fused product
traits also let consumers accumulate or subtract directly into output across all
backends. They do not promise faster iterations: matrix traversal, output
passes, normalization, and storage decoding can dominate buffer allocation. Keep
the allocating and overwrite interfaces for consumers that prefer them.
