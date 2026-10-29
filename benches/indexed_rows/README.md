# Raw indexed-row selection

`IndexedRows::new(source, rows)` restricts raw data before fitting
normalization. It owns the supplied source handle and borrows strictly
increasing, unique row indices. Passing `&matrix` or a backend view avoids
copying matrix storage. Construction validates the indices in O(k) time,
allocates nothing, and reads no entries. Local row `i` maps to source row
`rows[i]`. Empty selections are valid; duplicate, descending, and out-of-bounds
indices panic.

All supported in-memory backend releases provide dense, CSC, or CSR traversal.
`LazyMatrix::new` fits only selected rows. Apply its parameters to held-out raw
rows with `LazyMatrix::from_normalization`; selection never manages folds,
scoring, solver state, or model selection.

## Allocation and traversal contracts

For k selected rows and p columns:

  | Source | One scan                               | Traversal workspace |
  | ------ | -------------------------------------- | ------------------- |
  | Dense  | O(kp)                                  | O(1)                |
  | CSC    | O(k + p + stored entries × log(k + 1)) | O(k)                |
  | CSR    | O(p + k + selected stored entries)     | O(p)                |

Sparse traversal combines duplicate cells in storage order before nonlinear
statistics or products. It includes selected implicit zeros in statistics and
never converts CSR to CSC. An empty selection reads no entries. Statistics use
O(p) additional state; standardization shares its mean with the centered
variance pass and needs two scans. Exact quantiles use O(p) selection state and
a data-dependent number of scans. The combined fitting hook shares scans for
median and quantile range. Nested raw selections allocate O(k) index mapping per
scan at each level.

Raw reusable products need only traversal workspace. Lazy scaling can allocate
O(p) coefficient scratch. Fused products with nonzero beta and centered dense
fused products use O(output length) scratch. No operation copies the selected
design. Allocating products additionally allocate their returned vectors.

`RawColumns` is available only when the parent naturally borrows columns.
Creating a filtered column allocates nothing; traversing it scans the parent
column, filters entries, and rebases their indices by binary search. This
preserves stored zeros, ordering, and duplicates. It does not provide contiguous
CSC index slices. `SparseRows` retains the parent's contiguous row slices,
including order and duplicates, without gathering or allocating.

## Measurements

Run both representations in one exclusive compute window:

```sh
compute bench --reason "LazyMatrix indexed rows versus explicit fold copies" -- \
  cargo bench --locked --features ndarray,sprs --bench indexed_rows
```

The fixture has 4,096 rows and 64 columns, with every fifth row excluded (3,276
selected rows). The dense source uses column-major ndarray storage. The sparse
source uses sprs CSC storage. Both contain the same deterministic values with
about 4% stored nonzeros. In this fixture, medians and interquartile ranges are
zero. Quantile scan counts can be substantially larger on other distributions.

Fitting includes selection and normalization. The copied baseline allocates a
subset with ndarray's `select` or a direct CSC filter, then fits it. Product
measurements reuse already fitted operators and an existing output vector, so
they exclude subset construction and fitting. Inputs, indices, and product
outputs are allocated before measurement. Peak live bytes include retained
copied designs and fitted parameters. The allocation tracker runs separately
from Criterion timing.

  | Operation             | Borrowed time | Copied time | Borrowed peak bytes | Copied peak bytes |
  | --------------------- | ------------: | ----------: | ------------------: | ----------------: |
  | Dense standardization |      1.682 ms |    1.030 ms |               7,168 |         1,808,352 |
  | Dense robust fitting  |      1.733 ms |    1.053 ms |              34,936 |         1,808,352 |
  | CSC standardization   |      245.2 µs |     94.7 µs |              38,700 |           264,704 |
  | CSC robust fitting    |      256.4 µs |    140.9 µs |              66,452 |           266,360 |
  | Dense forward product |      336.0 µs |    222.1 µs |                   0 |                 0 |
  | CSC forward product   |      107.3 µs |      5.2 µs |              32,044 |               512 |

Borrowing substantially reduces fitting allocations here, but repeated
filtering, index mapping, and duplicate handling cost time. Explicit copies can
amortize their setup cost over repeated products. These measurements support a
choice based on the consumer's memory budget and number of products; they do not
establish a general speed advantage for borrowed selection.

[timings.csv](timings.csv) contains Criterion means and 95% confidence bounds in
nanoseconds. [allocations.csv](allocations.csv) contains peak new live bytes.
[metadata.json](metadata.json) records the machine, compiler, fixture, and
sampling settings. Each timed case uses 10 samples, 100 ms of warmup, and a 300
ms target measurement interval. These short measurements describe this machine
and fixture, not universal performance bounds.
