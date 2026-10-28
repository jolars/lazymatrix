#!/usr/bin/env python3
"""Export the complete forward-input benchmark and its allocation measurements."""

import argparse
import csv
import json
import math
import sys
from itertools import product
from pathlib import Path

BASELINES = ("forward-input-first", "forward-input-reversed")
IMPLEMENTATION_BASELINES = (
    "forward-input-implemented-first",
    "forward-input-implemented-reversed",
)


def expected_names(baseline):
    methods = (
        ("cloned_reference", "operator")
        if baseline in IMPLEMENTATION_BASELINES
        else ("current", "borrow_unscaled")
    )
    names = {
        f"input_clone/{backend}/{columns}"
        for backend, columns in product(("faer", "nalgebra", "vec"), (32, 128, 16384))
    }
    for backend, shape, density, mode, method in product(
        (
            "faer_dense",
            "nalgebra_dense",
            "ndarray",
            "faer_csc",
            "faer_csr",
            "nalgebra_csc",
            "nalgebra_csr",
            "sprs_csc",
            "sprs_csr",
        ),
        ("2000x32", "10000x128", "32x16384"),
        ("1", "0.01", "0.1"),
        ("raw", "centered", "standardized"),
        methods,
    ):
        dense = backend in ("faer_dense", "nalgebra_dense", "ndarray")
        if dense == (density == "1"):
            names.add(
                f"forward_input/{backend}/{shape}/density_{density}/{mode}/{method}"
            )
    return names


def require_coverage(found, expected, source):
    if found != expected:
        raise ValueError(
            f"{source}: missing {sorted(expected - found)}, extra {sorted(found - expected)}"
        )


def timings(root, baseline, expected):
    found = set()
    records = []
    for path in root.glob(f"**/{baseline}/benchmark.json"):
        name = json.loads(path.read_text())["full_id"]
        if name not in expected:
            continue
        if name in found:
            raise ValueError(f"Duplicate timing: {baseline}/{name}")
        found.add(name)
        mean = json.loads((path.parent / "estimates.json").read_text())["mean"]
        interval = mean["confidence_interval"]
        values = [
            mean["point_estimate"],
            interval["lower_bound"],
            interval["upper_bound"],
        ]
        samples = len(json.loads((path.parent / "sample.json").read_text())["times"])
        if samples != 20 or not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError(f"Invalid timing: {baseline}/{name}")
        records.append([baseline, name, *values, samples])
    require_coverage(found, expected, baseline)
    return records


def allocations(path, baseline, expected):
    found = set()
    records = []
    for line in path.read_text().splitlines():
        if not line.startswith("forward_input_allocations,"):
            continue
        _, name, calls, size = next(csv.reader([line]))
        if name in found:
            raise ValueError(f"Duplicate allocation count: {baseline}/{name}")
        found.add(name)
        records.append([baseline, name, int(calls), int(size)])
    require_coverage(found, expected, path)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("target/criterion"))
    parser.add_argument(
        "--baselines",
        nargs=2,
        choices=BASELINES + IMPLEMENTATION_BASELINES,
        default=BASELINES,
    )
    parser.add_argument("--logs", nargs=2, type=Path, required=True)
    parser.add_argument("--allocations", type=Path, required=True)
    args = parser.parse_args()
    timing_records = []
    allocation_records = []
    for baseline, log in zip(args.baselines, args.logs, strict=True):
        expected = expected_names(baseline)
        timing_records.extend(timings(args.root, baseline, expected))
        allocation_records.extend(allocations(log, baseline, expected))
    with args.allocations.open("w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["baseline", "benchmark", "calls", "bytes"])
        writer.writerows(sorted(allocation_records))
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(
        ["baseline", "benchmark", "mean_ns", "lower_ns", "upper_ns", "samples"]
    )
    writer.writerows(sorted(timing_records))


if __name__ == "__main__":
    main()
