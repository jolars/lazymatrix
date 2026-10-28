# Submatrix product benchmarks

Run the benchmark and untimed peak heap workspace measurements with:

```sh
compute bench --reason "submatrix products and weighted operations" -- \
  cargo bench --locked --bench submatrix --features ndarray,sprs,zarrs
```

The source has 4,096 rows and 256 columns, with approximately 1% nonzero entries.
The selected rectangles start at row 17 and column 11. Explicit fitted centers
are one, and scales are two. Parameter fitting and input construction occur
outside measurements. Every product is checked against the native dense view
before timing.

The weights are `(local_row % 7) * 0.25 - 0.5`, including signed and zero
weights. Weighted sums and Gram products use reusable outputs. The Zarr fixture
uses zarrs 0.23, an uncompressed `MemoryStore`, and 128-by-32 storage chunks.
Its timings include retrieval and decoding from memory. Cold disk I/O and
larger-than-RAM inputs require separate measurements.

The dense baseline wraps an ndarray slice in `LazyMatrix`. The sparse baseline
filters the original CSC columns directly and applies normalization algebraically.
That baseline assumes unique entries and does not supply a general submatrix
API or duplicate handling. Both use caller-owned input and output vectors.

`timings.csv`, `allocations.csv`, and `metadata.json` record a short smoke run
in one exclusive `compute` window. Timings report Criterion's bootstrap means
and 95% confidence intervals with 10 samples, 0.1 seconds of warmup, and 0.2
seconds of measurement. Use Criterion's default settings for performance
conclusions. Historical Criterion change estimates are not part of these data.

Construction and dense reusable products allocated no heap workspace in this run.
Sparse products allocated working vectors proportional to the selected
rectangle's dimensions:

| Selected shape | Forward peak bytes | Transpose peak bytes |
| -------------- | ------------------ | -------------------- |
| 128 × 32       | 2,208              | 1,440                |
| 1,024 × 64     | 17,536             | 9,856                |

These figures exclude the source and caller-owned vectors. Sparse filtering
still visits every stored entry in the selected outer columns, including entries
outside the selected row range. No variant materializes a normalized design
matrix. Sparse kernels use additional workspace to combine duplicate cells
before products; the direct filtering baseline does not have that cost.

Submatrix weighted Gram products use two normalized panels and an accumulator,
each holding 32-by-32 scalars. These fixed stack arrays contain 24 KiB of `f64`
elements in total and are excluded from the heap measurements below. Additional
workspace comes from sparse cell accumulation or Zarr chunk retrieval and
decoding. The measured Zarr heap workspace stayed constant as the selection grew:

| Selected shape | Dense submatrix | CSC submatrix | Zarr submatrix | Native dense view |
| -------------- | --------------- | ------------- | -------------- | ----------------- |
| 128 × 32       | 0 bytes         | 288 bytes     | 65,576 bytes   | 139,264 bytes     |
| 1,024 × 64     | 0 bytes         | 288 bytes     | 65,576 bytes   | 270,336 bytes     |

The native dense Gram kernel uses GEMM. Its extra heap workspace accompanied
faster execution on this fixture. The submatrix path uses scalar accumulation
and revisits source panels for each column-panel pair. Sparse panels rescan
stored entries, and Zarr panels can decode the same storage chunk repeatedly.
The mean weighted Gram timings were:

| Selected shape | Native dense view | Dense submatrix | CSC submatrix | Zarr submatrix |
| -------------- | ----------------- | --------------- | ------------- | -------------- |
| 128 × 32       | 18.35 µs          | 36.66 µs        | 44.92 µs      | 54.72 µs       |
| 1,024 × 64     | 396.94 µs         | 929.49 µs       | 1,250.36 µs   | 1,578.22 µs    |

Zarr submatrix forward products, transpose products, eager conversion, and
weighted sums read each intersecting chunk once. Their chunk and codec workspace
depends on storage chunk size, including outer shards for sharded arrays. The
fixed Gram panels do not impose a byte budget on that workspace.
