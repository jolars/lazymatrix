# Batched products

This investigation compares `MatMat` and `MatTransposeMat` with repeated native
vector calls. All strategies use the same fitted normalization and source data.
The harness verifies their products and consumer updates before timing them.

Run the comparison in one exclusive compute window:

```sh
compute run -- cargo bench --locked --bench batched_products \
  --features faer,nalgebra,ndarray,sprs,zarrs --no-run
compute bench --reason "Compare batched products with vector calls" -- \
  bash -c 'cargo bench --locked --bench batched_products \
    --features faer,nalgebra,ndarray,sprs,zarrs > benches/batched_products/timings.csv'
```

Set `LAZYMATRIX_BENCH_SMOKE=1` for a smaller, shorter validation run. Benchmarks
select the newest enabled backend release; correctness suites exercise every
enabled release.

## Method

The regular fixture has 1,024 rows, 64 predictors, and target density 0.05, using
`ChaCha8Rng` seed 205. Dense fixtures contain the same values as CSC and CSR
fixtures. Products use 1, 4, 16, and 64 responses, with raw, centered, scaled, and
standardized operators. Zarr uses an uncompressed in-memory store with 128-by-32
chunks. Setup, normalization fitting, native vector conversion, and correctness
checks are outside product timings.

Each row reports the median of seven samples, each lasting at least 10 ms.
Allocations, requested bytes, peak additional live bytes, and storage reads
come from a separate warmed operation. Peak additional bytes exclude existing
inputs, outputs, and source storage. Vector allocating products retain all
response outputs in a vector; reusable vector outputs are allocated in advance.
Batched outputs use row-major ndarray storage for every input backend, exercising
the caller-selected output interface.

The consumer performs ten gradient updates with an implicit intercept, measuring
the entire call, including its coefficient, residual, and gradient allocations.
It uses four or sixteen standardized responses. Vector consumers copy columns
between those dense matrices and native vectors. The reusable vector consumer
retains its vector scratch between calls; the allocating consumer constructs
native vectors for each product. These ownership costs are included in consumer
timings and excluded from product timings.

## Results

For sixteen standardized responses, the table reports reusable vector time
divided by reusable batch time. Ratios above one favor batching. Consumer results
include ten gradient updates with an intercept.

| Backend          | Forward | Transpose | Ten updates |
| ---------------- | ------: | --------: | ----------: |
| ndarray dense    |    5.07 |      2.20 |        1.81 |
| faer dense       |    5.48 |     10.02 |        3.71 |
| nalgebra dense   |    2.53 |      4.13 |        1.97 |
| sprs CSC         |    0.36 |      1.09 |        0.62 |
| sprs CSR         |    0.46 |      1.60 |        0.74 |
| faer CSC         |    0.63 |      0.62 |        0.63 |
| faer CSR         |    0.48 |      0.65 |        0.62 |
| nalgebra CSC     |    0.49 |      0.46 |        0.52 |
| nalgebra CSR     |    0.52 |      1.01 |        0.83 |
| Zarr MemoryStore |    1.70 |      2.47 |        1.26 |

Dense standardized products allocate no additional heap storage when reusing
output. Native sparse batches allocate two working panels rather than a scaled
coefficient vector for each response, but require packing and scattering. sprs
uses no heap scratch and reads scaled coefficients through an indexed view.
These costs can outweigh batching benefits for a small, low-density matrix.

Zarr performs 16 chunk reads per batch: reusable vector calls perform 256 reads
for sixteen responses and 1,024 for sixty-four responses. Allocation profiles
include the chunk decoder's per-scan workspace. This fixture measures warm
MemoryStore access, so the read reduction does not establish disk throughput.

The full [CSV](timings.csv) includes raw and partially normalized products, all
response counts, and allocating baselines. [Metadata](metadata.json) records the
compiler, CPU, backend versions, and sampling settings.

## Workspace and interpretation

Centered dense kernels use fixed 16-by-16 accumulator panels and normalize each
entry before multiplication. Raw dense kernels pack native panels with at most
128 rows, 128 reduction entries, and 16 responses. Native sparse kernels reuse
two panels of at most 16 responses, taking O((n + p) min(k, 16)) scratch. sprs
and Zarr accumulate all responses during a storage traversal. Algebraic
normalization uses an indexed scaled-RHS view and applies corrections only after
the backend succeeds.

The output itself occupies O(nk) or O(pk) dense storage. No implementation
materializes the normalized design. Native kernels may allocate additional
workspaces internally. Zarr chunk buffers and codec workspaces depend on storage
chunk size. A batch with no responses performs no source reads.

These are warm, single-process CPU measurements of one fixture, not cold-I/O or
larger-than-RAM results. Batching can reduce vector allocations and storage scans
while still being slower for small batches or particular layouts. The CSV keeps
those cases visible.
