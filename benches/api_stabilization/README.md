# API stabilization benchmark comparison

This comparison measures the existing submatrix and normalization pipeline
benchmarks before and after the normalization API stabilization. The baseline is
commit `dc8a0b3b820666b7e9f27d8aa99b75816769cf2e`; the candidate is the working
tree identified by `metadata.json` under `final.source_fingerprints_sha256`. Use
the `run=final` rows in both CSV files for the release comparison. Other labels
preserve the initial measurements and focused diagnostic rounds.

On the recorded AMD Ryzen 9 7900 system, final candidate-to-baseline ratios
across case medians have geometric means of 0.889 for the 36 submatrix cases and
0.995 for the 40 pipeline cases. All 56 allocation profiles match exactly
between revisions. Representative final medians are:

  | Case                                     | Baseline µs | Candidate µs | Ratio |
  | ---------------------------------------- | ----------: | -----------: | ----: |
  | Dense Gram, 1,024 × 64                   |    1,138.33 |       451.64 | 0.397 |
  | CSC Gram, 1,024 × 64                     |    1,235.77 |       572.80 | 0.464 |
  | Zarr Gram, 1,024 × 64                    |    1,373.91 |       902.42 | 0.657 |
  | Native dense Gram control, 1,024 × 64    |      390.24 |       395.49 | 1.013 |
  | CSC lazy pair, 8,192 × 128, 0.1% density |      184.95 |       182.18 | 0.985 |

Initial measurements exposed consistent regressions in Zarr Gram products and
the low-density CSC product pair. Focused ABBA checks led to direct unfused
full-operator dispatch and private helper inlining, exposing constant origins
and full-range selection to the compiler. Gram panels now skip unused lower
triangles through their loop ranges, preserving the order of included terms. The
final Zarr Gram medians improve by 34–37%; dense and sparse Gram medians improve
by 52–60% on these fixtures, with the same measured heap workspace.

The low-density CSC pair retains visible variability. Its final specialized ABBA
check measured baseline rounds of 203.7 and 186.9 µs and candidate rounds of
209.4 and 198.9 µs, a geometric-mean increase of 4.6%. The complete final run
measured a 1.5% decrease. Its small residual performance change remains
unresolved in these short runs. The largest final increase, 6.3%, occurred in an
unchanged eager control. Each bootstrap interval describes one Criterion run;
the diagnostic rounds expose additional variability between runs.

The `initial` and `confirmation-*` rows describe the first API implementation.
The `tuned-confirmation-*` rows retain the first unsuccessful tuning attempt.
The `specialized-confirmation-*` and `final` rows use the final source. The
corresponding source fingerprints are mapped in `metadata.json` under
`run_source_provenance`.

Both revisions use Rust 1.98.1, the same lockfile, and the `ndarray`, `sprs`,
and `zarrs` features. Both initial builds finish before either revision is
measured, using separate fresh Cargo target directories. The final comparison
reuses the unchanged baseline executable and rebuilds the final candidate. Its
focused ABBA confirmation and complete measurement suites share one exclusive
`compute bench` reservation. Separate Criterion output directories preserve all
estimates without comparisons against stale history.

Each complete run measures the baseline first, then the candidate, with one
thread for Rayon, OpenMP, OpenBLAS, and other configured numerical runtimes.
Focused rounds use baseline, candidate, candidate, baseline order. CPU policy,
compiler settings, source fingerprints, and each reservation are recorded in
`metadata.json`.

Criterion uses 20 samples, 0.2 seconds of warmup, 0.5 seconds of measurement,
and 10,000 bootstrap resamples. `results.csv` contains the median and its 95%
confidence interval for each case and revision. Full fixture sizes are used;
these are not smoke measurements. The raw Criterion estimates and logs remain in
the ignored `target/api-stabilization-bench/` directory.

The submatrix source has 4,096 rows and 256 columns, approximately 1% nonzero
entries, and selections of 128 by 32 and 1,024 by 64. Its fitted centers are
one, and scales are two. Products and weighted operations reuse caller-owned
output buffers. Zarr uses an uncompressed in-memory store with 128-by-32 chunks.
Native dense views and direct CSC filtering provide control cases. See [the
submatrix methodology](../submatrix/README.md) for their precise behavior.

Pipeline fixtures use `ChaCha8Rng` seed 205, 128 columns, 2,048 or 8,192 rows,
and stored densities of 0.1% or 1%. Column-only chains run on CSC and CSR;
row-then-column chains run on dense storage. `fit_pair` includes fitting and one
forward/transpose product pair. `pair` excludes fitting and reuses outputs. Each
eager control materializes one normalized dense matrix and applies its second
stage in place. See [the pipeline
methodology](../normalization_pipeline/README.md).

`memory.csv` records peak additional live requested Rust heap bytes. Submatrix
measurements exclude source storage and caller-owned buffers; pipeline lifecycle
measurements include fitting and one allocating product pair, excluding source
storage. Stack arrays, allocator overhead, and process RSS are excluded. The
submatrix Gram kernel's fixed stack panels hold an additional 24 KiB. Chunk and
codec heap workspaces are included in measured requested bytes and depend on the
full storage chunk, including outer shards when present.

To reproduce the comparison from the repository root, export the baseline and
run both revisions in one reservation. The candidate is the current working
tree, so use the same source state when comparing with these records.

```sh
comparison_baseline=$(mktemp -d)
comparison_output=$(mktemp -d)
git archive dc8a0b3b820666b7e9f27d8aa99b75816769cf2e | tar -x -C "$comparison_baseline"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1
export MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_DYNAMIC=FALSE
unset LAZYMATRIX_BENCH_SMOKE LAZYMATRIX_BENCH_QUICK
compute bench --reason "Compare API stabilization submatrix and pipeline paths" -- bash -c '
  set -euo pipefail
  for revision in baseline candidate; do
    source=$2
    if [[ $revision == candidate ]]; then source=$1; fi
    (
      cd "$source"
      CARGO_TARGET_DIR="$3/$revision" cargo bench --locked --no-default-features \
        --features ndarray,sprs,zarrs --bench submatrix \
        --bench normalization_pipeline --bench normalization_pipeline_memory --no-run
    )
  done
  for revision in baseline candidate; do
    source=$2
    if [[ $revision == candidate ]]; then source=$1; fi
    (
      cd "$source"
      export CARGO_TARGET_DIR="$3/$revision"
      cargo bench --locked --no-default-features --features ndarray,sprs,zarrs \
        --bench normalization_pipeline_memory > "$3/$revision-memory.csv"
      for suite in submatrix normalization_pipeline; do
        cargo bench --locked --no-default-features --features ndarray,sprs,zarrs \
          --bench "$suite" -- --noplot --warm-up-time 0.2 --measurement-time 0.5 \
          --sample-size 20 --nresamples 10000 > "$3/$revision-$suite.log" \
          2> "$3/$revision-$suite.stderr"
      done
    )
  done
' bash "$PWD" "$comparison_baseline" "$comparison_output"
```

Criterion writes estimates under each target directory's `criterion/` tree.
Submatrix workspace records appear in the corresponding `.stderr` files; the
pipeline memory executable writes its own CSV. The recorded run invokes the
compiled executables directly after the two builds, with the same arguments.
