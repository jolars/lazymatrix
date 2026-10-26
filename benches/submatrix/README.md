# Submatrix product benchmarks

Run the benchmark and untimed peak workspace measurements with:

```sh
cargo bench --locked --bench submatrix --features ndarray,sprs
```

The source has 4,096 rows and 256 columns, with approximately 1% nonzero entries.
The selected rectangles start at row 17 and column 11. Explicit fitted centers
are one, and scales are two. Parameter fitting and input construction occur
outside measurements. Every product is checked against the native dense view
before timing.

The dense baseline wraps an ndarray slice in `LazyMatrix`. The sparse baseline
filters the original CSC columns directly and applies normalization algebraically.
That baseline assumes unique entries and does not supply a general submatrix
API or duplicate handling. Both use caller-owned input and output vectors.

`timings.csv`, `allocations.csv`, and `metadata.json` record a short smoke run.
Other validation tasks were running concurrently, so the timing estimates are
illustrative. Use Criterion's default measurement settings on an idle machine
for performance conclusions.

Construction and dense reusable products allocated no workspace in this run.
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
