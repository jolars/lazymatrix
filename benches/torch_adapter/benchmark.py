"""Compare native lazy normalization with an eagerly normalized dense first layer."""

import argparse
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from lazymatrix.torch import NormalizedLinear


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def peak_rss():
    # ru_maxrss is a process high-water mark, so each strategy needs a fresh process.
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def fixture(rows, columns, density, dtype, device):
    generator = np.random.default_rng(205)
    per_row = max(1, min(columns, round(columns * density)))
    indices = np.concatenate(
        [
            np.sort(generator.choice(columns, per_row, replace=False))
            for _ in range(rows)
        ]
    )
    values = torch.tensor(generator.standard_normal(len(indices)), dtype=dtype)
    return torch.sparse_csr_tensor(
        torch.arange(0, (rows + 1) * per_row, per_row),
        torch.from_numpy(indices),
        values,
        size=(rows, columns),
    ).to(device)


def validate(dtype, device):
    raw = fixture(8, 12, 0.2, dtype, device)
    centers = torch.linspace(-0.5, 0.5, 12, dtype=dtype, device=device)
    scales = torch.linspace(0.75, 1.25, 12, dtype=dtype, device=device)
    layer = NormalizedLinear(
        12, 3, centers=centers, scales=scales, dtype=dtype, device=device
    )
    reference = torch.nn.Linear(12, 3, dtype=dtype, device=device)
    with torch.no_grad():
        reference.weight.copy_(layer.weight)
        reference.bias.copy_(layer.bias)
    expected = reference((raw.to_dense() - centers) / scales)
    expected.square().mean().backward()
    for source in (raw, raw.to_sparse_coo()):
        layer.zero_grad(set_to_none=True)
        actual = layer(source)
        actual.square().mean().backward()
        tolerance = {"rtol": 2e-5, "atol": 2e-6} if dtype == torch.float32 else {}
        torch.testing.assert_close(actual, expected, **tolerance)
        torch.testing.assert_close(
            layer.weight.grad, reference.weight.grad, **tolerance
        )
        torch.testing.assert_close(layer.bias.grad, reference.bias.grad, **tolerance)


def worker(config):
    torch.set_num_threads(config["threads"])
    dtype = getattr(torch, config["dtype"])
    device = config["device"]
    validate(dtype, device)
    synchronize(device)
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(205)
    started = time.perf_counter()
    source = fixture(
        config["rows"], config["columns"], config["density"], dtype, device
    )
    centers = torch.linspace(-0.5, 0.5, config["columns"], dtype=dtype, device=device)
    scales = torch.linspace(0.75, 1.25, config["columns"], dtype=dtype, device=device)
    if config["layout"] == "dense":
        source = source.to_dense().sub_(centers).div_(scales)
        layer = torch.nn.Linear(
            config["columns"], config["outputs"], dtype=dtype, device=device
        )
        input_bytes = source.numel() * source.element_size()
    else:
        if config["layout"] == "coo":
            source = source.to_sparse_coo()
        layer = NormalizedLinear(
            config["columns"],
            config["outputs"],
            centers=centers,
            scales=scales,
            dtype=dtype,
            device=device,
        )
        parts = (
            (source.values(), source.indices())
            if config["layout"] == "coo"
            else (source.values(), source.crow_indices(), source.col_indices())
        )
        input_bytes = sum(part.numel() * part.element_size() for part in parts)
    del centers, scales
    synchronize(device)
    preparation_seconds = time.perf_counter() - started

    def step():
        layer.zero_grad(set_to_none=True)
        layer(source).square().mean().backward()

    for _ in range(3):
        step()
    synchronize(device)
    iterations = 1
    while True:
        started = time.perf_counter()
        for _ in range(iterations):
            step()
        synchronize(device)
        elapsed = time.perf_counter() - started
        if elapsed >= config["min_run_time"]:
            break
        iterations *= 2
    samples = []
    for _ in range(config["samples"]):
        started = time.perf_counter()
        for _ in range(iterations):
            step()
        synchronize(device)
        samples.append((time.perf_counter() - started) / iterations)
    seconds = statistics.median(samples)
    return config | {
        "preparation_seconds": preparation_seconds,
        "step_seconds": seconds,
        "rows_per_second": config["rows"] / seconds,
        "input_bytes": input_bytes,
        "peak_rss_bytes": peak_rss(),
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated() if device == "cuda" else None
        ),
        "iterations_per_sample": iterations,
        "sample_seconds": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda", "all"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64", "all"], default="all")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--columns", type=int, default=1024)
    parser.add_argument("--densities", default="0.01,0.1")
    parser.add_argument("--outputs", default="32")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--min-run-time", type=float, default=0.05)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.worker:
        print(json.dumps(worker(json.loads(arguments.worker))))
        return
    if sys.platform == "win32":
        parser.error("peak RSS measurement requires Linux or macOS")
    if arguments.smoke:
        arguments.rows, arguments.columns = 64, 32
        arguments.densities, arguments.outputs = "0.1", "4"
        arguments.samples, arguments.min_run_time = 3, 0.01
    densities = [float(value) for value in arguments.densities.split(",")]
    outputs = [int(value) for value in arguments.outputs.split(",")]
    if (
        min(arguments.rows, arguments.columns, arguments.threads, arguments.samples)
        <= 0
        or arguments.min_run_time <= 0
        or any(not 0 < value <= 1 for value in densities)
        or any(value <= 0 for value in outputs)
    ):
        parser.error(
            "sizes, sample counts, and times must be positive; densities in (0, 1]"
        )
    if arguments.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    devices = ["cpu", "cuda"] if arguments.device == "all" else [arguments.device]
    skipped = []
    if "cuda" in devices and not torch.cuda.is_available():
        devices.remove("cuda")
        skipped.append("CUDA is unavailable")
    dtypes = ["float32", "float64"] if arguments.dtype == "all" else [arguments.dtype]
    results = []
    for device in devices:
        for dtype in dtypes:
            for density in densities:
                for width in outputs:
                    for layout in ("dense", "coo", "csr"):
                        config = {
                            "device": device,
                            "dtype": dtype,
                            "rows": arguments.rows,
                            "columns": arguments.columns,
                            "density": density,
                            "outputs": width,
                            "layout": layout,
                            "threads": arguments.threads,
                            "samples": arguments.samples,
                            "min_run_time": arguments.min_run_time,
                        }
                        completed = subprocess.run(
                            [sys.executable, __file__, "--worker", json.dumps(config)],
                            capture_output=True,
                            text=True,
                        )
                        if completed.returncode:
                            sys.stderr.write(completed.stderr)
                            completed.check_returncode()
                        results.append(json.loads(completed.stdout))
                        print(
                            f"{device} {dtype} density={density} outputs={width} {layout}: "
                            f"{results[-1]['step_seconds'] * 1e3:.3f} ms",
                            file=sys.stderr,
                        )
    cpu_model = platform.processor()
    if sys.platform == "linux":
        cpu_model = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu_model or platform.machine(),
        )
    metadata = {
        "measured_at": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "torch": torch.__version__,
        "lazymatrix": version("lazymatrix"),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu": cpu_model,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "seed": 205,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "adapter_sha256": hashlib.sha256(
            Path(sys.modules[NormalizedLinear.__module__].__file__).read_bytes()
        ).hexdigest(),
        "skipped": skipped,
        "command": sys.argv,
        "results": results,
    }
    if arguments.output is None:
        print(json.dumps(metadata, indent=2))
        return
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    fields = [name for name in results[0] if name != "sample_seconds"]
    with (arguments.output / "timings.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
