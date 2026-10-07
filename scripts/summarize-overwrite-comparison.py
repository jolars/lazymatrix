#!/usr/bin/env python3
"""Join warmed allocation counts and Criterion estimates for overwrite comparisons."""

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path


def expected_names():
    names = set()
    dense = [
        "faer_dense",
        "nalgebra_dense",
        "ndarray",
        "zarrs_[256, 16]",
        "zarrs_[1024, 32]",
    ]
    sparse = ["faer_csc", "nalgebra_csc", "sprs_csc"]
    for shape in ["2000x32", "10000x128"]:
        for backends, densities in [(dense, ["1"]), (sparse, ["0.01", "0.1"])]:
            for backend in backends:
                for density in densities:
                    for normalization in ["raw", "centered", "standardized"]:
                        for intercept in ["", "/intercept"]:
                            case = f"{backend}/{shape}/density_{density}/{normalization}{intercept}"
                            for path in ["overwrite", "workspace"]:
                                for direction in ["forward", "transpose"]:
                                    names.add(
                                        f"overwrite_products/{case}/{direction}/{path}"
                                    )
                                names.add(f"overwrite_iterations/{case}/{path}")
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log", type=Path, help="Combined stdout and stderr from the full run"
    )
    parser.add_argument("--criterion-dir", type=Path, default=Path("target/criterion"))
    parser.add_argument("--baseline", default="overwrite-comparison")
    args = parser.parse_args()

    counts = {}
    pattern = re.compile(
        r"^allocations (overwrite_(?:products|iterations)/.+): "
        r"(\d+) calls, (\d+) bytes per (\d+) operations$"
    )
    for line in args.log.read_text().splitlines():
        if match := pattern.fullmatch(line):
            name, calls, requested_bytes, batch_size = match.groups()
            if name in counts:
                raise ValueError(f"Duplicate allocation sample: {name}")
            counts[name] = tuple(map(int, (calls, requested_bytes, batch_size)))

    expected = expected_names()
    if counts.keys() != expected:
        raise ValueError(
            f"Allocation coverage differs: {len(expected - counts.keys())} missing, "
            f"{len(counts.keys() - expected)} unexpected"
        )

    estimates = {}
    for metadata_path in args.criterion_dir.rglob(f"{args.baseline}/benchmark.json"):
        metadata = json.loads(metadata_path.read_text())
        name = metadata["full_id"]
        if name in expected:
            if name in estimates:
                raise ValueError(f"Duplicate Criterion estimate: {name}")
            estimate = json.loads(metadata_path.with_name("estimates.json").read_text())
            estimates[name] = estimate["mean"]
    if estimates.keys() != expected:
        raise ValueError(
            f"Missing Criterion estimates: {sorted(expected - estimates.keys())}"
        )

    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(
        [
            "backend",
            "rows",
            "predictors",
            "density",
            "normalization",
            "intercept",
            "operation",
            "path",
            "batch_size",
            "allocations",
            "requested_bytes",
            "mean_batch_ns",
            "lower_batch_ns",
            "upper_batch_ns",
            "operations_per_second",
        ]
    )
    for name in sorted(expected):
        parts = name.split("/")
        prefix, backend, shape, density, normalization = parts[:5]
        intercept = "intercept" in parts[5:]
        operation = parts[-2] if prefix == "overwrite_products" else "iteration"
        path = parts[-1]
        calls, requested_bytes, batch_size = counts[name]
        estimate = estimates[name]
        mean = estimate["point_estimate"]
        interval = estimate["confidence_interval"]
        lower, upper = interval["lower_bound"], interval["upper_bound"]
        if batch_size != 10 or not all(
            math.isfinite(v) and v > 0 for v in (mean, lower, upper)
        ):
            raise ValueError(f"Invalid measurement: {name}")
        rows, predictors = map(int, shape.split("x"))
        writer.writerow(
            [
                backend,
                rows,
                predictors,
                density.removeprefix("density_"),
                normalization,
                str(intercept).lower(),
                operation,
                path,
                batch_size,
                calls,
                requested_bytes,
                f"{mean:.3f}",
                f"{lower:.3f}",
                f"{upper:.3f}",
                f"{batch_size * 1e9 / mean:.3f}",
            ]
        )


if __name__ == "__main__":
    main()
