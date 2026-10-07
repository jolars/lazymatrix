# Fused consumer benchmarks

Run the consumer comparison with:

```sh
cargo bench --locked --bench fused_consumer --features faer,nalgebra,ndarray,sprs,zarrs
```

Criterion filters select individual cases. For example, append
`-- 'consumer/faer_csc/2000x32/density_0.01/standardized'` to compare the four
product paths with and without an intercept. The benchmark uses the newest
enabled version of each backend. Versioned features can select older releases.

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
10,000 × 128 predictors. Dense fixtures have density 100%; CSC fixtures have
density 1% or 10%. Every fixture runs raw, mean-centered, and standardized,
with and without `WithIntercept`. Zarr uses an uncompressed in-memory store
and chunk shapes 256 × 16 and 1,024 × 32. These measurements do not describe
disk latency or arrays larger than RAM.

The four paths are:

- `allocating`: allocate forward and transpose results on each step, then
  subtract the response and update coefficients. Allocating `LazyMatrix`
  forward products also clone the coefficient input.
- `overwrite`: reuse residual and gradient vectors with overwrite products,
  then perform the two vector updates separately.
- `fused`: initialize the residual to the negative response and accumulate
  the forward product, then fuse the transpose product with the coefficient
  update. Normalization and wrapper scratch are allocated internally.
- `workspace`: perform the same fused updates with caller-owned scratch.
  `LazyMatrix` reuses its normalization buffer; `WithIntercept` reuses its
  predictor buffer, while its inner `LazyMatrix` still allocates normalization
  scratch.

A benchmark-only allocator counts allocation and reallocation requests and
requested bytes in a separate warmed batch. Counting is disabled during
Criterion timing. The reported bytes are cumulative requests, not peak memory.
Use the default serial feature selection for reproducible counts; parallel
backend initialization and scheduling can add allocator traffic.

## Allocation results

All in-memory backends produced the following counts per ten steps at 32
predictors. Center-only products have the same counts as raw products.

| Predictors | Intercept | Overwrite | Fused | Workspace |
| --- | --- | ---: | ---: | ---: |
| Raw | No | 0 | 0 | 0 |
| Raw | Yes | 20 | 20 | 0 |
| Standardized | No | 10 | 20 | 0 |
| Standardized | Yes | 30 | 40 | 20 |

Each counted predictor buffer requests 256 bytes at this size. Fused transpose
updates with scaling need a separate raw-product buffer to preserve previous
output before normalization. The overwrite transpose can normalize its output
in place, so ordinary fused calls allocate more buffers. Caller-owned workspace
removes that cost for `LazyMatrix`; the wrapper's workspace methods remove its
two buffers per step, but do not also reuse the inner normalization buffers.

For Zarr with 256 × 16 chunks, decoding and scanning already make 5,440
allocation requests and request 21,074,560 bytes per ten steps. Add the table's
counts to that baseline. Fused products retain one chunk scan per product and
zero chunk reads for exact zero `alpha`, including normalized and intercept
wrappers. Those read guarantees are verified with counting and failing stores.

## Evaluation

Measured on October 7, 2026, with Rust 1.89.0 on an AMD Ryzen 9 7900, pinned
to CPU 10 without `parallel`. The full 528-case matrix used ten samples, a
0.1-second warmup, and a requested 0.3-second measurement window. The following
point estimates are microseconds per ten steps, without an intercept, at
2,000 × 32. Dense inputs have density 100%; CSC inputs have density 1%.

| Backend | Raw overwrite | Raw workspace | Standardized overwrite | Standardized workspace |
| --- | ---: | ---: | ---: | ---: |
| faer dense | 633.62 | 643.28 | 650.46 | 656.54 |
| nalgebra dense | 830.20 | 835.10 | 843.13 | 854.41 |
| ndarray | 493.13 | 495.48 | 506.95 | 506.33 |
| faer CSC | 16.30 | 13.15 | 29.84 | 32.60 |
| nalgebra CSC | 45.17 | 27.64 | 58.34 | 47.40 |
| sprs CSC | 20.50 | 17.79 | 35.08 | 37.58 |
| Zarr, 256 × 16 chunks | 1,310.20 | 1,306.50 | 1,307.80 | 1,312.60 |

For standardized 1%-density CSC with an intercept at the same size, overwrite
versus workspace took 50.30 versus 52.15 µs for faer, 78.48 versus 66.82 µs for
nalgebra, and 53.02 versus 56.97 µs for sprs. At 10,000 × 128 without an
intercept, the corresponding standardized CSC figures were 249.59 versus
257.33 µs, 542.96 versus 493.39 µs, and 268.16 versus 279.31 µs.

An isolated build of the original operator implementations repeated the 22
standardized overwrite cases at 2,000 × 32 on the same pinned CPU, using a
0.2-second warmup and a 0.5-second measurement window. Existing overwrite
products remained within 5% of those baselines, including the unchanged
ndarray and sprs controls.

The slower centered and standardized faer/sprs workspace cases were repeated
with twenty samples, a 0.2-second warmup, and a 0.8-second measurement window.
They still showed mixed results: standardized faer without an intercept took
29.95 µs with overwrite and 32.39 µs with workspace; standardized sprs took
34.20 and 36.67 µs. These are comparisons between two consumer paths, rather
than regressions in the overwrite kernels. Accumulating into an initialized
residual and separating the scaled transpose result changes the output work;
eliminating allocations does not eliminate those passes.

Caller-owned scratch provides a predictable allocation benefit. Fused product
traits also let consumers accumulate or subtract directly into output across
all backends. They do not promise faster iterations: matrix traversal, output
passes, normalization, and storage decoding can dominate buffer allocation.
Keep the allocating and overwrite interfaces for consumers that prefer them.
