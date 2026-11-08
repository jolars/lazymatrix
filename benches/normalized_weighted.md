# Weighted row and chain operations

The `normalized_weighted` group in `weighted_gram.rs` compares weighted column
sums and Gram products with reusable ndarray outputs. Run all cases in one
exclusive compute window:

```sh
compute run -- cargo bench --locked --no-default-features \
  --features ndarray,sprs --bench weighted_gram --no-run
compute bench --reason "Compare normalized weighted operations" -- \
  cargo bench --locked --no-default-features --features ndarray,sprs \
    --bench weighted_gram -- normalized_weighted
```

Each fixture has 1,024 rows and either 32 or 65 columns, crossing the kernels'
32-column panel boundary. Dense, CSC, and CSR storage contain the same values,
generated with `ChaCha8Rng` seed 517 and target density 0.1. Weights take both
signs, following `(i % 11) / 11 - 0.5` in floating-point arithmetic.

The row cases reuse fitted row standardization across dense, CSC, and CSR
storage. Their eager baseline uses an already materialized dense matrix. The
column-chain case applies column standardization followed by min-max
normalization on CSC storage. The mixed dense case applies row standardization
followed by column standardization. Fitting, materialization, storage conversion,
output allocation, and correctness checks are outside the timings. Each case
checks sums and Gram products against materialized logical entries before
measurement, with absolute tolerance `1e-8`.

Full-storage sums scan once. Sparse sums read 32-by-32 logical panels, including
transformed implicit zeros. Gram products use two panels and one accumulator of
the same size, rereading source panels for each column-panel pair. Sparse
coalescing adds bounded workspace but does not change storage orientation. The
eager baseline measures product cost alone; it excludes the time and full-matrix
storage required to materialize normalization.
