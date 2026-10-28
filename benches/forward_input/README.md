# Forward input borrowing

Borrowing unscaled inputs in the default allocating forward hook removes one
input-sized allocation per product. Both original timing runs show a clear
benefit for wide sparse inputs: raw CSR products at 32 × 16,384 and 1% density
take about 33–40% less time.
The tall fixtures have little repeatable timing benefit. The default allocating
hook now implements this conditional borrow without adding a workspace API.

The former default allocating `MatVec::matvec_normalized` cloned its input even
when scaling was inactive. That clone supplied no modified coefficients in raw
or center-only products. Sparse backends and Zarr use the default hook. Dense
backends override it, allocate their result, and delegate to reusable-output
products, which already borrowed unscaled inputs.

## Original experiment

The original investigation compared the then-public allocating product with a
private prototype that borrowed the original input when scaling was absent.
The prototype preserved the backend product, output allocation, centering
correction, and error propagation. It delegated to the existing hook when
scaling was present. Dense controls also delegated to their existing hook,
preserving direct centering before accumulation. The original data retain the
method labels `current` and `borrow_unscaled` for those two paths.

The measurements cover faer, nalgebra, and sprs in CSC and CSR layouts, at 1%
and 10% density. Dense faer, nalgebra, and ndarray controls have 100% density.
All fixtures use owned `f64` vectors and raw, mean-centered, or standardized
normalization. Standardization uses population standard deviations. The shapes
are 2,000 × 32, 10,000 × 128, and 32 × 16,384. The last fixture makes input
copying large relative to output allocation. Separate input-clone measurements
cover the three vector types used by sparse backends.

Each product is checked against an independently materialized dense oracle
before timing. Warmed allocation measurements count successful allocation and
reallocation calls and cumulative requested bytes. They exclude fixture
construction, normalization fitting, and the oracle. The benchmark asserts the
expected allocation difference for every fixture and checks that inputs remain
unchanged. Timing includes allocating and destroying the returned vector. The
counting allocator forwards to `System` and disables counting during timing,
although each allocation still checks the counting flag.

Two complete runs share one exclusive `compute bench` reservation and CPU 10 of
an AMD Ryzen 9 7900. Each pair uses the same executable, matrix, coefficients,
and fitted parameters. The second run reverses the two product methods' order.
Each timing uses twenty samples, a 0.1-second warmup, a requested 0.4-second
measurement window, and 1,000 bootstrap resamples. [`metadata.json`](metadata.json)
records the toolchain, backend versions, seeds, and build configuration. The
crate's `parallel` feature is disabled.

[`timings.csv`](timings.csv) retains each run's mean and 95% confidence interval,
in nanoseconds per product or input clone. [`allocations.csv`](allocations.csv)
retains the corresponding untimed allocation measurements. There are 279 cases
per run: 216 sparse products, 54 dense controls, and nine input clones. These
microbenchmarks do not measure complete solver fits, strided input vectors,
storage decoding, or larger-than-RAM workloads.

## Results

The allocation measurements agree exactly across both runs. Every raw and
center-only sparse product saves one allocation of `8 * ncols` bytes: 256 bytes,
1 KiB, or 128 KiB for the three shapes. Output allocation remains. Dense and
standardized controls retain their allocation counts. At 32 × 16,384, for
example, sprs CSR drops from two allocations totaling 131,328 bytes to one
256-byte result allocation.

The following raw 1%-density measurements use the wide fixture. Times are
microseconds per product from the first run. A speed ratio above one favors
borrowing; the last column repeats the ratio with the measurement order reversed.

| Backend      | Current (µs) | Borrowed (µs) | Speed ratio | Reversed order |
| ------------ | -----------: | ------------: | ----------: | -------------: |
| faer CSC     |       11.394 |         9.681 |       1.18× |          1.19× |
| faer CSR     |        4.289 |         2.581 |       1.66× |          1.68× |
| nalgebra CSC |       31.582 |        29.914 |       1.06× |          1.05× |
| nalgebra CSR |        4.786 |         3.210 |       1.49× |          1.53× |
| sprs CSC     |       13.283 |        12.047 |       1.10× |          1.15× |
| sprs CSR     |        4.225 |         2.565 |       1.65× |          1.66× |

The isolated 16,384-element clones cost about 1.7 µs, which closely matches the
absolute product savings. Center-only wide CSR products also improve in both
runs, taking about 11–13% less time. Their coefficient dot product and output
correction make the removed copy a smaller fraction of the work. At 10%
density, matrix traversal also reduces its relative importance.

For the tall fixtures, the median speed ratios across the six sparse backends,
raw and center-only modes, and both run orders stay within 1% of one at each
density. Cloning their 32- and 128-element inputs costs roughly 7–24 ns.
Some larger individual differences reverse or disappear between
runs. The unchanged dense controls range from 0.97× to 1.04×, and some
unchanged standardized sparse controls have larger one-off changes. For
example, standardized nalgebra CSC at 2,000 × 32 and 10% density appears 1.55×
faster in the first run and 1.00× in the repeat, despite both methods calling
the same implementation. Five of the 558 timing intervals span more than 10%
of their mean; order-dependent variation can also exceed those within-run
confidence intervals. The retained records make these limits visible.

The allocation reduction and repeated wide-input benefit support a small change
to the default allocating hook. These measurements give no reason to add
scratch ownership or storage machinery for this case, and establish no
complete-solver speedup.

## Implementation

The [default hook](../../src/traits/operator.rs) now clones and divides only when
scales are present, then passes either that scratch or the original input to the
backend and centering correction. Backend overrides retain their directly
centered arithmetic. Reusable-output and fused paths already borrowed unscaled
coefficients. Public signatures and trait bounds are unchanged.

[Regression tests](../../tests/fallible.rs) count clones through a custom vector
and compare products with a dense oracle. Raw and center-only products make no
input clone, while explicit unit, positive, and negative scales retain one clone.
The tests also check input preservation and verify that backend failures skip
both the centering dot product and output correction with and without scaling.
Both new tests failed on the old unconditional clone and passed after the change.

The [benchmark](../forward_input.rs) now retains the former default as
`cloned_reference`, while `operator` calls the public `LazyMatrix::matvec`.
The constant `DEFAULT_HOOK` identifies sparse fixtures using that default; dense
controls preserve their specialized hook in both methods. The allocation
assertions still require exactly one input-sized allocation saved for each
unscaled sparse product and identical counts for the dense and scaled controls.

The implementation comparison repeats all 279 cases in each measurement order
with the original seeds and timing settings. Its records are separate from the
prototype: [timings](implementation_timings.csv),
[allocations](implementation_allocations.csv), and
[metadata](implementation_metadata.json). The metadata also record source hashes
for the library hook and retained benchmark reference.

Both production runs confirm the allocation saving, and their counts agree
exactly. Raw 1%-density CSR products at 32 × 16,384 take about 35–40% less time.
The following first-run times are microseconds per product. The last column
repeats the speed ratio with the measurement order reversed.

| Backend      | Cloned reference (µs) | Public operator (µs) | Speed ratio | Reversed order |
| ------------ | --------------------: | -------------------: | ----------: | -------------: |
| faer CSC     |                 9.663 |                8.090 |       1.19× |          1.19× |
| faer CSR     |                 4.265 |                2.623 |       1.63× |          1.65× |
| nalgebra CSC |                31.544 |               29.820 |       1.06× |          1.06× |
| nalgebra CSR |                 4.760 |                3.086 |       1.54× |          1.62× |
| sprs CSC     |                12.732 |               11.100 |       1.15× |          1.14× |
| sprs CSR     |                 4.227 |                2.555 |       1.65× |          1.65× |

The tall fixtures' median speed ratios again stay within 1% of one at each
density. One of the 558 production timing intervals spans more than 10% of its
mean. Individual controls still show variation between runs, so the allocation
counts and repeated wide-input results provide the clearest evidence.

## Reproduce

From the repository root, build and smoke-check the benchmark:

```sh
compute run -- cargo bench --locked --bench forward_input \
  --features faer,nalgebra,ndarray,sprs -- --test
```

Run the complete comparison in one reservation. Change CPU 10 to an available
CPU on another machine.

```sh
compute bench --reason "Validate implemented forward input borrowing against the cloned baseline" -- bash -c '
set -euo pipefail
taskset -c 10 cargo bench --locked --bench forward_input \
  --features faer,nalgebra,ndarray,sprs -- \
  --warm-up-time 0.1 --measurement-time 0.4 --sample-size 20 \
  --nresamples 1000 --noplot --save-baseline forward-input-implemented-first \
  > target/forward-input-implemented-first.log 2>&1
FORWARD_INPUT_REVERSE=1 taskset -c 10 cargo bench --locked --bench forward_input \
  --features faer,nalgebra,ndarray,sprs -- \
  --warm-up-time 0.1 --measurement-time 0.4 --sample-size 20 \
  --nresamples 1000 --noplot --save-baseline forward-input-implemented-reversed \
  > target/forward-input-implemented-reversed.log 2>&1
'
python3 scripts/summarize-forward-input.py \
  --baselines forward-input-implemented-first forward-input-implemented-reversed \
  --logs target/forward-input-implemented-first.log target/forward-input-implemented-reversed.log \
  --allocations benches/forward_input/implementation_allocations.csv \
  > benches/forward_input/implementation_timings.csv
```

The exporter rejects incomplete case coverage, duplicate records, unexpected
sample counts, and invalid timing estimates. Both raw Criterion baselines and
allocation logs are needed to re-export the CSV files.
Omitting `--baselines` re-exports the original investigation's records from
`target/forward-input-first.log` and `target/forward-input-reversed.log`.

## Validation

After implementation, `compute run -- task ci` passed, including the full
backend version, parallel, and feature-unification test matrix, Python checks
and packaging, and website checks. The benchmark smoke run passed all 279 cases.
Clippy for the benchmark and Ruff for the exporter passed. Re-exporting the
saved measurements reproduced the original and production timing and allocation
CSV files byte for byte.
