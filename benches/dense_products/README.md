# Stable centered dense products

The stable dense products subtract each column's center and divide each entry
before multiplying by a coefficient. The former forward loop traversed every
matrix by row, and the transpose loop traversed every matrix by column. This
made one direction unfavorable for each storage order. Optional scaling also
left a branch in the inner loop, and the ordered reductions used scalar
arithmetic.

The selected optimization uses 128-output stack panels for column-major forward
products and row-major transpose products. Backend strides select traversal,
including negative and stepped views. Each output retains its original reduction
order. Scaled and unscaled panel kernels compile separately, so the optional
scale branch stays outside the inner loop. The existing scalar kernels remain
for directions that already follow storage. No normalized design copy, heap
scratch, reciprocal multiplication, or algebraic centering correction is added.
The forward panel holds 128 sums; the transpose panel also caches 128 center and
scale pairs. For `f64`, the arrays occupy 1 KiB and 3 KiB, respectively.

## Experiment

`benches/dense_products.rs` measures both directions for faer, nalgebra, and
ndarray at 2,000 × 32 and 10,000 × 128, with centering alone or centering and
scaling. The explicit centers are `1e6`, and scales cycle through positive and
negative values. Each backend has contiguous column-major and row-major cases.
The additional faer view reverses rows, the nalgebra view steps both axes by
two, and the ndarray view reverses both axes. Each candidate sees the same
matrix, coefficients, and output type within its fixture. An ordered scalar
oracle checks every result before measurement.

Six methods separate the costs:

- `operator` calls the public reusable-output product.
- `scalar` retains the original traversal and entry-wise division in a local
  `f64` kernel. Its timings need not equal the generic public operator's timings.
- `panel32` and `panel128` accumulate independent outputs in bounded panels.
  These prototypes still test optional scaling inside their loops.
- `terms128` stages individual products before an ordered reduction. Its extra
  staging did not help on this machine.
- `moved_scaling` prescales coefficients for forward products, including the
  coefficient scratch allocation, or divides after the transpose reduction.
  This experiment is valid only for the ordinary finite fixtures. The library
  does not use these arithmetic rearrangements.

`timings.csv` contains 432 candidate measurements, 192 product and consumer
measurements before the change, and 264 afterward. The additional 72 afterward
are the public operators across all microbenchmark layouts. The consumer
fixtures come from `fused_consumer`: ten least-squares steps through allocating,
overwrite, fused, and workspace paths, with and without an intercept. Separate
product measurements apply ten products per batch. Normalization fitting,
reference calculations, and allocation counting stay outside timing.
`allocations.csv` records the 384 matching warmed allocation measurements.
The exporter checks exact case coverage and requires twenty samples per case.

Measurements use Rust 1.98.1, Criterion 0.8.2, default release code generation,
and CPU 10 of an AMD Ryzen 9 7900, without the `parallel` feature. See
`metadata.json` for backend versions and measurement settings. These are local
microbenchmarks; they do not establish portable speedups or downstream fit times.

## Results

Large standardized products have the following before/after speed ratios. A
ratio above one means the selected operator is faster. Both contiguous storage
orders and the additional strided views are included.

| Backend  | Layout  | Forward | Transpose |
| -------- | ------- | ------: | --------: |
| faer     | column  |   2.19× |     1.01× |
| faer     | row     |   1.01× |     3.42× |
| faer     | strided |   1.31× |     1.09× |
| nalgebra | column  |   2.26× |     1.00× |
| nalgebra | row     |   1.01× |     1.97× |
| nalgebra | strided |   1.80× |     0.97× |
| ndarray  | column  |   2.57× |     1.00× |
| ndarray  | row     |   1.00× |     2.88× |
| ndarray  | strided |   0.99× |     1.51× |

On the large centered-only fixtures, contiguous column-major forward products
are 4.7–5.7× faster. Keeping scaling inside each entry still allows a 2.2–2.6×
improvement for the corresponding standardized products. The large standardized
cases that retain scalar traversal are within 4% of their former times.

The large faer standardized forward product takes 1372 µs before the change,
1239 µs with the plain 128-output prototype, and 627 µs with the selected kernel.
Tiling alone therefore explains only part of the improvement. Hoisting parameters
and specializing scaling expose independent output sums to compiler vectorization.
With the selected column-major traversal fixed, adding scaling increases the
large forward product from roughly 230–238 µs to 623–633 µs across the three
backends. Entry-wise division remains a substantial cost after the traversal
and vectorization improvements.

The fused consumer measurements below are milliseconds per ten least-squares
steps at 10,000 × 128 without an intercept. The CSV also includes the smaller
fixture, every other application path, and intercept cases.

| Backend        | Normalization | Before (ms) | After (ms) | Speed ratio |
| -------------- | ------------- | ----------: | ---------: | ----------: |
| faer_dense     | centered      |      18.949 |     10.687 |       1.77× |
| faer_dense     | standardized  |      25.464 |     18.429 |       1.38× |
| nalgebra_dense | centered      |      19.580 |      9.775 |       2.00× |
| nalgebra_dense | standardized  |      25.862 |     18.313 |       1.41× |
| ndarray        | centered      |      24.597 |     14.040 |       1.75× |
| ndarray        | standardized  |      25.807 |     20.017 |       1.29× |

Warmed reusable-output and fused products without an intercept make zero heap
allocations before and after the change. The intercept paths retain their
existing allocation counts. These kernel changes do not address composition of
caller-owned workspaces through nested wrappers.

Twenty-five of the 432 candidate measurements and one of the 192 repeated
original product/consumer measurements have 95% intervals wider than 10% of
their means. None of the 264 final measurements do. Retain the intervals in the
CSV when assessing individual cases, especially small apparent changes.

The generic unscaled panel kernels use packed subtraction, multiplication, and
addition. Their scaled paths also use packed division (`divpd` in the generated
x86-64 assembly; see `assembly.txt`), while each output's additions remain in the same order. The
original hot loop used scalar `subsd`, `divsd`, `mulsd`, and `addsd`, with an
optional-scale test inside the loop. This accounts for gains beyond changing the
memory traversal. Division remains necessary, but vector lanes process independent
outputs without reassociating a reduction.

`counters.csv` records a three-second `perf stat` profile of the large standardized
faer forward product before and after the change. Counts cover the whole process,
including fixture checks, and the runs execute different numbers of products.
They are supporting diagnostics, not instructions or cache misses per product.
Hardware sampling with `perf record` was unavailable under the host permissions.

## Numerical constraints

Moving division outside accumulation is faster in some ordinary finite fixtures,
but changes the documented entry-wise arithmetic:

| Rearrangement                                             | Entry-wise result | Rearranged result                     |
| --------------------------------------------------------- | ----------------- | ------------------------------------- |
| Prescale `1e-300 / 1e300`, then multiply an entry `1e300` | `1e-300`          | `0`                                   |
| Prescale `1e300 / 1e300`, then multiply an entry `1e-300` | `0`               | `1e-300`                              |
| Divide two entries `1e308` by `1e308` before summing      | `2`               | `infinity` if the raw sum comes first |
| Normalize `1e300 / 1e-300`, then multiply by zero         | `NaN`             | `0` if the coefficient is prescaled   |

The regression suite exercises these cases for every enabled dense backend
release, along with `f32` extremes, nonfinite centers and scales, negative scales,
panel boundaries, arbitrary views, empty dimensions, and fused and workspace
paths. Tiny expected values use relative comparisons so an absolute tolerance
cannot hide underflow. Signed-zero checks caught the need to initialize panel
sums with `Sum`'s identity rather than positive zero. Zero `alpha` bypasses matrix
and normalization arithmetic, and zero `beta` never reads the previous output.

## Reproduce

Run without competing builds. Prefix every timing command below with
`taskset -c 10` for the affinity used here, after building the benchmarks. Change
the CPU affinity on another machine.
Build and record the baseline on the commit identified in `metadata.json` before
applying the kernel change; the new benchmark and exporter can be copied there
without changing the library. Save each executable if comparing both builds in
one checkout. Criterion baseline directories must survive between builds.

```sh
cargo bench --locked --bench dense_products --features faer,nalgebra,ndarray -- \
  --sample-size 20 --warm-up-time 0.05 --measurement-time 0.15 \
  --nresamples 1000 --noplot --save-baseline dense-investigation
cargo bench --locked --bench fused_consumer \
  --features faer,nalgebra,ndarray,sprs,zarrs -- \
  '^(consumer|overwrite_products)/(faer_dense|nalgebra_dense|ndarray)/.*(centered|standardized)' \
  --sample-size 20 --warm-up-time 0.1 --measurement-time 0.4 \
  --nresamples 1000 --noplot --save-baseline stable-dense-before \
  > target/dense-consumer-before.log 2>&1
```

After applying the kernel change, run the following with the same affinity.

```sh
cargo bench --locked --bench dense_products --features faer,nalgebra,ndarray -- \
  '/operator$' --sample-size 20 --warm-up-time 0.05 --measurement-time 0.2 \
  --nresamples 1000 --noplot --save-baseline stable-dense-after
cargo bench --locked --bench fused_consumer \
  --features faer,nalgebra,ndarray,sprs,zarrs -- \
  '^(consumer|overwrite_products)/(faer_dense|nalgebra_dense|ndarray)/.*(centered|standardized)' \
  --sample-size 20 --warm-up-time 0.05 --measurement-time 0.2 \
  --nresamples 1000 --noplot --save-baseline stable-dense-after \
  > target/dense-consumer-after.log 2>&1
python3 scripts/summarize-dense-products.py \
  dense-investigation stable-dense-before stable-dense-after \
  --allocation-logs target/dense-consumer-before.log target/dense-consumer-after.log \
  --allocations benches/dense_products/allocations.csv \
  > benches/dense_products/timings.csv
```

For counter profiles, build first, then run the selected benchmark executable
under `perf stat -x, -e cycles:u,instructions:u,cache-references:u,cache-misses:u`,
using `--bench '^dense_products/faer/column/10000x128/standardized/forward/operator$'
--profile-time 3`. Inspect its disassembly with `objdump -d -C`. A numerical smoke
run uses `cargo bench --locked --bench dense_products --features
faer,nalgebra,ndarray -- --test`.

## Validation

`task test` and the complete `task ci` run passed. The CI run selected the
repository's required Diplodocus 0.8.0 through `DIPLODOCUS_BIN`; the default local
binary was 0.9.0. The standalone exporter also passed Ruff's lint and formatting
checks. Re-exporting the Criterion baselines and allocation logs reproduces both
checked-in CSV files exactly.
