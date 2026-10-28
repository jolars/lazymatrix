# Quantile fitting

This harness compares median centering and IQR scaling with an independent dense
reference that sorts each column. All source matrices, sparse conversions, and
Zarr storage are prepared before measurement. Each method fits the same values
and checks its parameters against the reference before measuring.

```sh
compute bench --reason "Compare exact quantile fitting" -- \
  cargo bench --locked --bench quantile_fitting --features ndarray,sprs,zarrs
```

The baseline and candidates run in one exclusive compute window. Set
`LAZYMATRIX_BENCH_SMOKE=1` for smaller fixtures and three timing samples instead
of nine. Output is CSV containing the median elapsed time in microseconds, peak
additional live heap allocation during one complete fit, and the number of
complete Zarr storage scans. Peak allocation includes fitted parameters,
selection state, sorting buffers, and chunk/codec allocations; it excludes the
existing source matrices. The global allocation tracker adds instrumentation
overhead to every measured method.

The sparse fixture has approximately 10% stored entries; the dense fixture
stores every entry. Both use `ChaCha8Rng` with seed 591. A third fixture spans
exponents from -512 to 508 to expose the data-dependent cost of exact streaming
selection. Zarr uses an uncompressed in-memory store with 128-row chunks and one
column chunk. These measurements describe fitting on warm in-memory data, not
cold I/O or larger-than-RAM throughput. Fitted operators retain their existing
product kernels.

## Recorded run

`results.csv` records the default command above on October 28, 2026, with Rust
1.98.1, an AMD Ryzen 9 7900, and 24 logical CPUs. The `parallel` feature was
disabled. Each row reports the median of nine samples, with allocation and scan
counts measured separately from timing.

For the 4,096-by-16 fixtures, dense fitting uses about 33 KB of additional live
allocation, comparable to the sorted reference. CSR uses about 9 KB, while Zarr
uses about 41 KB including its 128-by-16 chunk buffers and codec allocations.
CSC uses about 13 KB on the sparse fixture and about 66 KB on the dense fixture;
its working entries retain row indices to combine duplicates correctly.

Zarr takes 11 scans for the sparse fixture and 16 for the dense fixture. The
wide-exponent stress case takes 194 scans despite having only 256 rows and four
columns. CSR and Zarr preserve bounded memory at a substantial fitting-time cost
on that case. These counts are properties of these fixtures, not general bounds.
Refresh the CSV and this run description together when changing the algorithm
or methodology.
