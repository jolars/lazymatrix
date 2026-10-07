# Downstream normalization allocations

Run the allocation profiles from the repository root:

```sh
bash scripts/bench-normalization-allocations.sh > target/normalization-allocations.csv
```

The script uses `OPENBLAS_LP64_LIB` from devenv and pins OpenBLAS and OpenMP to
one thread. The [isolated harness](normalization_allocations/Cargo.toml) calls
the downstream fitting APIs directly. Its dependencies are pinned to [Shrinkage
`165cd6e`](https://github.com/jolars/shrinkage/tree/165cd6e64c42c00bec15d09388d4c19793da4be4)
and [ndarray-glm
`0b727d8`](https://github.com/felix-clark/ndarray-glm/tree/0b727d8baaaf7c18980a8348fd410aa9d86a41b4).
Shrinkage pins LazyMatrix 0.5.0; the harness patches that dependency to this
checkout. The separate lockfile keeps these consumer dependencies out of the
library's dependency graph. No solver code or downstream APIs are changed.

The [complete CSV](normalization_allocations/results.csv) records 96 profiles
measured on October 7, 2026, with Rust 1.89.0 in release mode. Two runs produced
identical counts. Fixtures use `ChaCha8Rng` seed 205, 512 × 32 and 2,048 × 128
predictors, and target densities 100% and 10%. Input arrays use column-major
storage. Fixture creation and storage conversion happen outside measurement.
Every fitting path is warmed before sampling. The allocator counts Rust
allocation and reallocation requests and cumulative requested bytes, including
result construction. These counts exclude allocations made inside native BLAS
and LAPACK, and they do not measure peak memory or throughput.

## Shrinkage's backtracking proximal gradient

The harness fits Gaussian ridge problems with a fitted intercept, penalty 0.1,
initial step 64, and the solver's default mapping tolerance `1e-6`. The large
initial step makes every case exercise rejected line-search trials. Each fixture
runs raw, centered, and standardized through faer dense and CSC, ndarray dense,
and checked sprs CSC storage, using each backend's native vector type.

Thin wrappers tag native vector cloning and normalization-statistics calls and
count products. They delegate all numerical operations to the original backend.
Each instrumented fit must match an uninstrumented native fit in allocation
count, requested bytes, iterations, coefficients, intercept, and objective. Each
fit must converge. This checks that attribution does not add allocations or
change the fitted result.

At 2,048 × 128 and 10% density, faer CSC produced:

  | Normalization | Accepted updates | Forward products | Rejected trials | Coefficient clones | Clone bytes | Total allocations | Total bytes |
  | ------------- | ---------------: | ---------------: | --------------: | -----------------: | ----------: | ----------------: | ----------: |
  | Raw           |              183 |              375 |               7 |                  0 |           0 |                 9 |      70,656 |
  | Centered      |               93 |              194 |               6 |                  0 |           0 |                10 |      71,680 |
  | Standardized  |               23 |               55 |               7 |                 55 |      56,320 |                66 |     129,024 |

All four backends allocate one coefficient vector per standardized forward
product: `8 * p` bytes per clone at these sizes. Raw and centered fits allocate
no coefficient clones. Transpose normalization adds no coefficient scratch to
the overwrite path. The ridge penalty acts on normalized coefficients, so these
presets solve different objectives. Iteration counts also differ, so subtracting
total fit bytes would not isolate normalization costs. The tagged clone requests
provide that attribution directly.

For this standardized case, the 66 allocations comprise 55 coefficient clones,
two statistics buffers totaling 2,048 bytes, and nine other fitting allocations
totaling 70,656 bytes. Faer dense, ndarray dense, and sprs CSC use one
additional statistics buffer, bringing their totals to 67 allocations and
130,048 bytes. This statistics cost occurs once per fit.

The pinned proximal loop evaluates the current point and at least one trial
point in every pass, including the final convergence or budget check. With `k`
accepted updates and `b` rejected trials, it therefore performs `k + 1`
transpose products and `2 * (k + 1) + b` forward products. The harness verifies
that relationship and that every scaled forward product clones its input. In the
table's standardized case, the 55 clones include seven rejected trials and both
forward evaluations in the final pass. Reusing the solver's output vectors does
not remove these normalization allocations.

## ndarray-glm construction, IRLS, and prediction

The harness fits unregularized linear and logistic models with an intercept,
comparing the default standardization with `no_standardize()`. It measures
`ModelBuilder::build`, `Model::fit`, and `Fit::predict` separately. Predictions
from standardized and raw fits must agree within floating-point tolerance.
Normalization uses ndarray-glm's sample standard deviations. This profile does
not substitute LazyMatrix's population convention.

Model construction produced the same counts for both families and densities:

  | Predictors  | Raw allocations | Raw bytes | Standardized allocations | Standardized bytes | Additional normalization bytes |
  | ----------- | --------------: | --------: | -----------------------: | -----------------: | -----------------------------: |
  | 512 × 32    |              11 |   278,496 |                       20 |            677,088 |                        398,592 |
  | 2,048 × 128 |              13 | 4,259,808 |                       22 |         10,572,768 |                      6,312,960 |

Standardization adds nine requests and `24 * n * p + 8 * n + 40 * p` requested
bytes in these unweighted cases. The request-size counters identify three
additional `n * p` floating-point buffers. The [standardizer
implementation](https://github.com/felix-clark/ndarray-glm/blob/0b727d8baaaf7c18980a8348fd410aa9d86a41b4/src/data.rs#L303)
creates them while computing weighted means and centered variances, then
normalizes the owned design in place. Both raw and standardized builders also
copy the predictors and materialize the intercept column.

At 512 × 32 and 100% density, the separate fitting stage produced:

  | Family   | Normalization | IRLS iterations | Allocations | Requested bytes |
  | -------- | ------------- | --------------: | ----------: | --------------: |
  | Linear   | Raw           |               1 |         344 |       1,644,164 |
  | Linear   | Standardized  |               1 |         346 |       1,644,676 |
  | Logistic | Raw           |               5 |         408 |       2,981,356 |
  | Logistic | Standardized  |               4 |         394 |       2,647,408 |

The linear fits take the same number of iterations here; standardized fitting
adds two coefficient-sized requests during result and history transformations.
The logistic counts include different iteration totals and cannot be subtracted
to estimate a normalization cost. Prediction uses four requests and 147,456
bytes in all four cases, including a newly padded dense design.

ndarray-glm currently multiplies its owned standardized design directly. It does
not call LazyMatrix or repeatedly form `S^-1 x`, and it does not accept CSC
input. The downstream adoption TODO remains open. A future lazy integration must
remeasure fitting allocations and preserve weighted sample normalization. The
present evidence identifies its construction cost and provides a baseline for
that comparison.

## CSV fields and verification

`allocations` and `bytes` describe the measured stage. Shrinkage's `clones`,
`clone_allocations`, and `clone_bytes` describe actual native `Clone` calls;
`stats_allocations` and `stats_bytes` describe its forwarded
`normalization_stats` call. These are subsets of the total. Product, clone, and
statistics fields are blank for ndarray-glm because those internal operations
are not instrumented. `predictor_sized_calls` and `design_sized_calls` count
requests of exactly `8 * n * p` and `8 * n * (p + 1)` bytes. Request sizes alone
do not identify allocation sites.

The harness validates convergence and fit parity before printing each profile.
Its release runs also verify clone attribution and rejected trials across all 48
Shrinkage fits. Allocator traffic from reporting and numerical comparisons is
excluded. Formatting and Clippy are checked separately for this workspace:

```sh
cargo fmt --manifest-path benches/normalization_allocations/Cargo.toml -- --check
RUSTFLAGS="-L native=$OPENBLAS_LP64_LIB -l openblas -C link-arg=-Wl,-rpath,$OPENBLAS_LP64_LIB" \
  cargo clippy --release --locked --manifest-path benches/normalization_allocations/Cargo.toml --all-targets -- -D warnings
```
