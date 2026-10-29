# PyTorch first layer

This comparison measures `lazymatrix.torch.NormalizedLinear` against
`torch.nn.Linear` operating on an explicitly normalized dense input. The lazy
layer uses native PyTorch sparse products and a dense centering correction.
Both strategies train weights in normalized coordinates.

Install the adapter and check correctness before measuring:

```sh
compute run -- task python:torch:check
```

Run all strategies in one exclusive window:

```sh
compute bench --reason "Compare PyTorch first-layer throughput and memory" -- \
  python scripts/python-tool.py python benches/torch_adapter/benchmark.py \
  --device all --output benches/torch_adapter
```

Use `--smoke` for a short harness check. `--outputs 32,128`, `--rows`,
`--columns`, `--densities`, and `--dtype` select other workloads. An explicit
`--device cuda` fails if CUDA is unavailable; `--device all` records that skip
and still measures CPU cases. Peak RSS measurement requires Linux or macOS.

## Method

The default fixture has 4,096 rows, 1,024 input features, 32 outputs, and target
densities 0.01 and 0.1. Each row contains a fixed number of distinct, sorted
column indices. NumPy's generator uses seed 205, and every strategy starts with
the same raw values, normalization parameters, weights, and bias. Both float32
and float64 are measured, using one PyTorch CPU thread by default.

The worker builds raw CSR input without first allocating a dense design. The
dense baseline converts it to dense and centers and scales it in place before
training. COO input is converted from the same CSR fixture. Preparation time
includes fixture construction, storage conversion, normalization, and layer
initialization. It excludes imports and the small correctness check.

Each measured step clears parameter gradients, computes predictions, and
backpropagates their mean squared value. It excludes optimizer updates and does
not change weights between samples. The dense baseline retains its normalized
input between steps, so it pays no repeated normalization cost. The lazy layer
derives its scaled weights and centering correction on every forward pass.

After three warmup steps, the harness doubles the iteration count until a sample
lasts at least 50 ms. It reports the median of seven samples, with CUDA
synchronization around each sample. Before measurement, a small independent
fixture checks sparse predictions and parameter gradients against explicit
dense normalization. This check never materializes the large fixture in a lazy
worker.

Every strategy runs in a fresh subprocess. CPU memory is the complete process's
peak resident set size, including imports, small correctness checks, input
preparation, warmup, and training. It is not a tensor allocation counter or an
increment over a common baseline. CUDA memory is PyTorch's peak allocated tensor
memory from preparation through training, excluding the small correctness check.
It excludes reserved allocator capacity and non-PyTorch device allocations.
The input-byte column counts only the retained input values and indices.

[`timings.csv`](timings.csv) contains preparation time, forward/backward
throughput, retained input bytes, and peak memory. [`metadata.json`](metadata.json)
records versions, hardware, the command, source hashes, sampling settings, and
individual timing samples. Setup and steady training are reported separately;
neither is an end-to-end model-fitting time.

## Results

Recorded October 29, 2026, on an AMD Ryzen 9 7900 with one CPU thread, Python
3.14.7, and PyTorch 2.15.0+cu132. CUDA was unavailable, so these results cover
CPU execution only. Times are median forward/backward milliseconds; RSS is
the complete process peak in MiB.

| Dtype   | Density | Dense ms | COO ms | CSR ms | Dense RSS | COO RSS | CSR RSS |
| ------- | ------: | -------: | -----: | -----: | --------: | ------: | ------: |
| float32 |    0.01 |     4.96 |   3.54 |   1.30 |     616.8 |   603.2 |   603.9 |
| float32 |     0.1 |     4.97 |  39.08 |  20.43 |     646.4 |   622.0 |   629.9 |
| float64 |    0.01 |    19.61 |   4.76 |   2.95 |     645.1 |   606.0 |   605.5 |
| float64 |     0.1 |    18.70 |  40.34 |  24.80 |     666.0 |   627.9 |   637.7 |

At 1% target density, CSR has lower step times and process peaks than the dense
baseline for both dtypes. At 10%, both sparse layouts are slower than dense.
The float32 CSR input at 1% density retains 0.50 MiB, versus 16 MiB for the
dense input. Framework startup and workspaces make the difference in total
process peaks much smaller than the difference in retained input storage.

## Interpretation

Sparse storage reduces the bytes needed to retain the design, but its products
can be slower than optimized dense products. Density, output width, dtype,
device, and framework workspaces affect the comparison. CPU process peaks can
be dominated by framework startup. These synthetic, warmed measurements do not
establish a general speedup or GPU performance when CUDA is unavailable.
