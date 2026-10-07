# Overwrite and fused workspace comparison

The `fused_consumer` benchmark includes an isolated comparison of the existing
overwrite products with fused workspace products using `alpha = 1` and
`beta = 0`. Run all of these cases with:

```sh
cargo bench --locked --bench fused_consumer --features faer,nalgebra,ndarray,sprs,zarrs --no-run
taskset -c 10 cargo bench --locked --bench fused_consumer \
  --features faer,nalgebra,ndarray,sprs,zarrs -- '^overwrite_' \
  --sample-size 20 --warm-up-time 0.1 --measurement-time 0.4 \
  --nresamples 10000 --noplot --save-baseline overwrite-comparison \
  > target/overwrite-comparison.log 2>&1
python3 scripts/summarize-overwrite-comparison.py target/overwrite-comparison.log \
  > benches/overwrite_comparison.csv
```

Change the CPU number to an available core on another machine. Run without
`parallel` and without competing builds for repeatable allocation counts and
timings. Criterion filters can select subsets, but the CSV exporter requires the
complete comparison. A smoke run with `-- '^overwrite_' --test` checks all
numerical comparisons without collecting timings.

## Measurement design

Both paths use the exact same fixtures as the [fused consumer
comparison](fused_consumer.md): 2,000 × 32 and 10,000 × 128 predictors, dense
storage at density 100%, CSC storage at densities 1% and 10%, raw,
mean-centered, and standardized predictors, and an optional intercept. Dense
backends are faer, nalgebra, and ndarray. CSC backends are faer, nalgebra, and
sprs. Zarr uses an uncompressed in-memory store with chunk shapes 256 × 16 and
1,024 × 32. These fixtures use the newest supported releases of each backend.

`overwrite_products` measures forward and transpose products separately. Each
batch performs ten identical products into a reusable output. Forward inputs use
`ChaCha8Rng` seed 205; transpose inputs use the existing response with seed 204.
Allocating products supply the numerical reference outside measurement. Outputs
start with NaNs, so verification also checks that overwrite application ignores
previous values. Each product exposes its output to `black_box`. Criterion
reports products per second.

`overwrite_iterations` measures ten least-squares steps. Each path overwrites
the residual and gradient, subtracts the response, and updates coefficients in
the same separate loops. The workspace path uses `alpha = 1` and `beta = 0` for
both products. Coefficients reset to zero at the start of every batch, and the
final coefficients must agree. The reset is included in both timings. These
cases isolate workspace use in an iteration without fusing response subtraction
or coefficient updates.

Matrix construction, normalization, input vectors, reference products, output
allocation, and workspace creation happen outside allocation sampling and
timing. A separate warmed batch counts allocation and reallocation requests and
cumulative requested bytes. Counting is disabled during Criterion timing. These
bytes do not measure peak memory. Zarr's counts include chunk decoding and scan
buffers; the store and chunk contents already exist before sampling.

`WithIntercept` workspace methods reuse the wrapper's predictor buffer. They
still call the inner `LazyMatrix` without its caller-owned normalization
workspace, including when `beta = 0`.

The [CSV exporter](../scripts/summarize-overwrite-comparison.py) joins
allocation samples with Criterion's mean batch time and confidence interval. It
requires all 792 measurements: 132 operator configurations, each with forward,
transpose, and iteration batches through both paths. Each row reports ten
products or ten iterations. `operations_per_second` is
`10 * 1e9 / mean_batch_ns`; an iteration includes one product in each direction
and both vector updates. Product throughput and iteration throughput have
different units and should be compared within their respective operations.

## Allocation results

Every in-memory backend produced the following counts per ten products or
iterations. Each cell shows overwrite calls followed by workspace calls.
Mean-centered predictors have the same counts as raw predictors.

  | Predictors   | Intercept | Forward | Transpose | Iteration |
  | ------------ | --------- | ------: | --------: | --------: |
  | Raw          | No        |   0 / 0 |     0 / 0 |     0 / 0 |
  | Raw          | Yes       |  10 / 0 |    10 / 0 |    20 / 0 |
  | Standardized | No        |  10 / 0 |     0 / 0 |    10 / 0 |
  | Standardized | Yes       | 20 / 10 |   10 / 10 |   30 / 20 |

Each request allocates one predictor buffer: 256 bytes for 32 predictors or
1,024 bytes for 128 predictors. For example, ten standardized iterations without
an intercept request 2,560 or 10,240 bytes through the overwrite API and zero
bytes through workspace application. With an intercept, workspace application
still requests 5,120 or 20,480 bytes for the inner normalization buffers. These
counts describe this measured loop; they do not establish the allocation
behavior of a downstream solver.

Ordinary transpose overwrite normalizes its output in place, so it already needs
no normalization scratch. The scaled workspace transpose uses its preallocated
scratch even with `beta = 0`, so this comparison tests its extra output work as
well as forward allocation savings.

Zarr adds the following baseline per ten products in either direction. Iteration
batches have twice these counts and bytes because each step scans chunks in both
directions. Add the normalization and intercept requests above to this baseline.

  | Predictors   | Chunk shape | Allocations | Requested bytes |
  | ------------ | ----------- | ----------: | --------------: |
  | 2,000 × 32   | 256 × 16    |       2,720 |      10,537,280 |
  | 2,000 × 32   | 1,024 × 32  |         340 |      10,492,200 |
  | 10,000 × 128 | 256 × 16    |      54,400 |     210,745,600 |
  | 10,000 × 128 | 1,024 × 32  |       6,800 |     209,844,000 |

## Throughput results

The [complete CSV](overwrite_comparison.csv) records 792 measurements collected
on October 7, 2026, with Rust 1.89.0 on an AMD Ryzen 9 7900, pinned to CPU 10
without `parallel`. Each case used twenty samples, a 0.1-second warmup, and a
requested 0.4-second measurement window. The CSV contains mean batch times, 95%
confidence intervals, and products or iterations per second. Allocation counts
matched a separate pinned smoke run in every case.

The following means are microseconds per ten standardized products without an
intercept at 2,000 × 32. Dense fixtures have density 100%; CSC fixtures have
density 1%.

  | Backend                 | Forward overwrite | Forward workspace | Transpose overwrite | Transpose workspace |
  | ----------------------- | ----------------: | ----------------: | ------------------: | ------------------: |
  | faer dense              |            262.86 |            268.51 |              375.11 |              375.88 |
  | nalgebra dense          |            440.96 |            436.54 |              406.09 |              408.28 |
  | ndarray                 |            184.13 |            121.17 |              445.66 |              384.37 |
  | faer CSC                |              7.05 |             10.92 |               13.86 |               13.97 |
  | nalgebra CSC            |             32.66 |             38.54 |               14.81 |               15.05 |
  | sprs CSC                |              7.88 |             13.91 |               14.78 |               14.98 |
  | Zarr, 256 × 16 chunks   |            816.28 |            819.31 |              500.21 |              501.74 |
  | Zarr, 1,024 × 32 chunks |            997.15 |            993.36 |              374.58 |              374.31 |

The iteration comparison keeps the two vector updates identical. These means are
microseconds per ten standardized iterations without an intercept, using the
same densities.

  | Backend                 | 2,000 × 32 overwrite | 2,000 × 32 workspace | 10,000 × 128 overwrite | 10,000 × 128 workspace |
  | ----------------------- | -------------------: | -------------------: | ---------------------: | ---------------------: |
  | faer dense              |               651.34 |               658.62 |              14,318.91 |              14,355.39 |
  | nalgebra dense          |               851.94 |               860.81 |              16,743.21 |              16,877.09 |
  | ndarray                 |               649.21 |               502.35 |              18,679.61 |              17,651.62 |
  | faer CSC                |                30.99 |                36.35 |                 248.57 |                 271.18 |
  | nalgebra CSC            |                59.28 |                66.19 |                 543.94 |                 574.56 |
  | sprs CSC                |                33.33 |                40.94 |                 264.33 |                 298.91 |
  | Zarr, 256 × 16 chunks   |             1,328.21 |             1,345.98 |              26,884.83 |              27,003.16 |
  | Zarr, 1,024 × 32 chunks |             1,373.37 |             1,401.74 |              27,695.20 |              27,421.69 |

Workspace application removes the scaled forward allocation for `LazyMatrix`,
but that saving does not consistently improve throughput. The standardized CSC
forward products are slower with workspace application at both sizes in this
run, while the small ndarray fixture is faster. Transpose timings generally
remain close. These APIs use different backend kernels and normalization passes,
so timing differences do not measure allocator cost alone.

Nineteen of the 792 cases have 95% intervals wider than 10% of their mean. In
particular, the large Zarr fixture with 1,024 × 32 chunks has a noisy
standardized workspace transpose measurement. The complete CSV retains those
intervals; the point estimates do not establish a general speed advantage for
either interface. These results cover in-memory Zarr storage and the measured
least-squares loop, rather than downstream fitting or disk latency.

The full repository checks passed with `task ci`. Both numerical smoke runs and
the timed run verified product parity and final-coefficient parity. The exporter
verified complete allocation and timing coverage; all 792 timing records contain
twenty samples. The Python exporter also passed Ruff formatting and lint checks.
