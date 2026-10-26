# Weighted Gram benchmarks

Run the native ndarray and checked sprs CSC comparison with:

```sh
cargo bench --locked --bench weighted_gram --features ndarray,sprs
```

Include dense faer Gram, intercept, and weighted-column-sum cases with:

```sh
cargo bench --locked --bench weighted_gram --features ndarray,sprs,faer
```

Run the same cases with ndarray's BLAS kernels and one OpenBLAS thread with:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 bash scripts/bench-weighted-gram-blas.sh
```

The BLAS script uses `OPENBLAS_LP64_LIB`, supplied by the project's devenv,
and a separate target directory. It does not add system BLAS to ordinary builds
or the all-feature CI checks. Both commands accept Criterion filters and timing
options after `--` for Cargo or directly after the script name.

The fixtures contain 2,000 rows and 32 predictors or 10,000 rows and 128
predictors, with densities of 0.1%, 1%, 10%, and 100%. Values come from a seeded
`ChaCha8Rng`. Weights are positive and vary by row. Each case runs with and
without mean centering, without scaling. Matrix construction and center
computation take place outside the timed loops. Set both thread variables:
OpenBLAS builds that use OpenMP follow `OMP_NUM_THREADS`.

Each case measures ten operations:

- `dense`: the bounded-panel ndarray Gram kernel, including scratch allocation.
- `csc`: the sparse Gram kernel, including scratch allocation.
- `dense_column_sums`: directly centered ndarray column sums, with reusable output.
- `dense_operators` and `csc_operators`: assembly through repeated lazy forward
  and transpose products, with reusable vectors and output.
- `gemm_prepared`: dense matrix multiplication after materializing normalized
  and weighted matrices outside the timed loop.
- `gemm_with_preparation`: materialization, weighting, and dense multiplication
  inside the timed loop.
- `dense_intercept` and `csc_intercept`: the same predictors wrapped in
  `WithIntercept`, including stable weighted-column sums and coefficient scratch.
- `gemm_intercept`: a prepared dense `[1, X_tilde]` baseline, with preparation
  outside the timed loop.

Enabling a faer version adds `faer_dense`, `faer_dense_intercept`, and
`faer_column_sums`. These use the same values, weights, and centers in faer's
column-major storage, with reusable faer output. The Gram kernels call native
faer multiplication with `Par::Seq`, including when `parallel` is enabled.
Construction and conversion stay outside the timed loops. Version features
select the benchmark helper's newest enabled release.

All operations reuse their coefficient-space matrix or vector output. The
prepared GEMM baselines expose the multiplication cost separately from the cost
of storing and preparing two full design matrices. These matrices are benchmark
baselines, not workspace used by the Gram kernels.

For `f64`, each dense kernel's explicit panels and coefficient block occupy at
most 136 KiB, plus the multiplication library's workspace. Both use panels of
at most 256 rows and 32 columns. Nonfinite normalized or weighted entries use
direct accumulation to preserve IEEE arithmetic. Dense weighted-column sums
use constant scratch space and center each entry before weighting. Centered CSC
uses 24 KiB of panels for sufficiently populated, canonical inputs with at least
16 predictors. Below 2% density, centered canonical columns can use a
subtraction-free disjoint range-sum table. It answers implicit-zero weight
ranges in O(1) time after O(n log n) preparation, with table storage capped at
64 MiB. Larger inputs use the O(n) weight-sum tree. The sparse-pair path can
also cache normalized stored values, bounded to sparse input. Both paths need
O(p) column bookkeeping.
Uncentered CSC does not allocate the tree. Sparse fallback evaluation can
add two working columns for unsorted or duplicate indices and nonfinite
arithmetic. Every method needs the O(p²) output.

Intercept wrappers retain O(p) cross terms. Their CSC weighted-sum pass uses
a separate O(nrows) weight-sum tree when centered, or one working column for
noncanonical or nonfinite inputs. Predictor Gram and cross-term workspaces run
sequentially.

## Local measurements

Measured on September 23, 2026, with Rust 1.89.0 on an AMD Ryzen 9 7900.
The benchmark process was pinned to CPU 10. The BLAS build used OpenBLAS
0.3.33 with `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and
`OMP_DYNAMIC=FALSE`. Criterion used 10 samples, a 0.2-second warmup, and a
requested 0.5-second measurement window per case.

These tables show point estimates in milliseconds for 10,000 rows, 128
predictors, and mean centering. Full-matrix materialization is included only in
the last column; Gram kernel timings always include their internal preparation.

ndarray's default multiplication kernels:

| Density | Dense Gram | Dense operators | CSC Gram | CSC operators | GEMM prepared | GEMM + preparation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1% | 12.8 | 233 | 5.23 | 2.22 | 5.39 | 13.4 |
| 1% | 11.5 | 234 | 16.4 | 3.59 | 5.45 | 13.0 |
| 10% | 11.5 | 235 | 17.5 | 23.0 | 5.58 | 18.2 |
| 100% | 12.0 | 239 | 23.7 | 173 | 5.40 | 13.3 |

OpenBLAS:

| Density | Dense Gram | Dense operators | CSC Gram | CSC operators | GEMM prepared | GEMM + preparation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1% | 11.4 | 31.4 | 5.39 | 1.74 | 5.08 | 12.9 |
| 1% | 11.6 | 29.1 | 17.2 | 3.10 | 5.28 | 13.3 |
| 10% | 11.7 | 30.1 | 18.7 | 20.0 | 5.09 | 13.3 |
| 100% | 11.5 | 29.6 | 22.0 | 167 | 5.01 | 12.7 |

The CSC Gram kernel does not call BLAS. Its advantage at higher densities comes
from accumulating coefficient blocks directly. The measurements above predate
the sparse-pair optimization below. Repeated operators apply centering through
raw-product corrections and can lose small variations around large offsets.

## Sparse-pair optimization

Measured on October 6, 2026, on the same Ryzen 9 7900, pinned to CPU 10,
with 10 Criterion samples per case. These are the centered 10,000 by 128 CSC
cases with ndarray's default multiplication kernels. The original code and the
range-sum implementation were built from the same worktree and measured with
the same benchmark command.

| Density | Original CSC Gram | Range-sum CSC Gram |
| --- | ---: | ---: |
| 0.1% | 10.37 ms | 2.08 ms |
| 1% | 31.84 ms | 19.28 ms |

The range-sum table adds weights without subtracting a large prefix. Stored
values are normalized before pair accumulation, and noncanonical columns or
unsafe arithmetic still use the two-column fallback. The 10% and 100% cases
continue to use bounded panels.

A final pinned run with the optimized code measured these cases, in
milliseconds:

| Shape | Density | CSC Gram | CSC operators |
| --- | ---: | ---: | ---: |
| 2,000 by 32 | 0.1% | 0.062 | 0.134 |
| 2,000 by 32 | 1% | 0.243 | 0.149 |
| 10,000 by 128 | 0.1% | 1.98 | 3.05 |
| 10,000 by 128 | 1% | 17.22 | 6.18 |

The Gram kernel now wins at 0.1% in both fixtures. At 1%, repeated operators
remain faster, although their centering correction has the numerical limitation
described above.

## Dense faer comparison

Measured on October 26, 2026, with Rust 1.98.1 on the same Ryzen 9 7900,
pinned to CPU 10. These cases use faer 0.24.4 and ndarray 0.17.2 without BLAS
or the `parallel` feature. Criterion used 10 samples, a 0.2-second warmup, and
a requested 0.5-second measurement window. The fixture has 10,000 rows, 128
predictors, 100% density, positive weights, and no scaling.

Times are Criterion point estimates in milliseconds. Gram timings include
scratch allocation and panel preparation. Column sums reuse their output
vectors. ndarray uses row-major input, and faer uses column-major input.

| Centering | ndarray Gram | faer Gram | ndarray intercept | faer intercept | ndarray column sums | faer column sums |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| None | 10.01 | 5.45 | 13.24 | 7.13 | 2.30 | 1.63 |
| Mean | 10.69 | 7.10 | 13.52 | 9.14 | 2.57 | 2.02 |

Both dense kernels use directly centered panels with bounded workspace.
The intercept cases include the separate stable weighted-column-sum pass.
