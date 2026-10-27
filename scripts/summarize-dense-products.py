#!/usr/bin/env python3
"""Export complete Criterion baselines for the centered dense product investigation."""

import argparse
import csv
import json
import math
import re
import sys
from itertools import product
from pathlib import Path

BASELINES = ("dense-investigation", "stable-dense-before", "stable-dense-after")


def expected_names(baseline):
    names = set()
    if baseline != "stable-dense-before":
        methods = (
            ("scalar", "panel32", "panel128", "terms128", "moved_scaling", "operator")
            if baseline == "dense-investigation"
            else ("operator",)
        )
        for backend, layout, shape, mode, direction, method in product(
            ("faer", "nalgebra", "ndarray"),
            ("column", "row", "strided"),
            ("2000x32", "10000x128"),
            ("centered", "standardized"),
            ("forward", "transpose"),
            methods,
        ):
            names.add(
                f"dense_products/{backend}/{layout}/{shape}/{mode}/{direction}/{method}"
            )
    if baseline != "dense-investigation":
        for backend, shape, mode, intercept in product(
            ("faer_dense", "nalgebra_dense", "ndarray"),
            ("2000x32", "10000x128"),
            ("centered", "standardized"),
            ("", "/intercept"),
        ):
            case = f"{backend}/{shape}/density_1/{mode}{intercept}"
            for path in ("allocating", "overwrite", "fused", "workspace"):
                names.add(f"consumer/{case}/{path}")
            for direction, path in product(
                ("forward", "transpose"), ("overwrite", "workspace")
            ):
                names.add(f"overwrite_products/{case}/{direction}/{path}")
    return names


def export_allocations(paths, destination):
    pattern = re.compile(
        r"^allocations (.+): (\d+) calls, (\d+) bytes per (\d+) operations$"
    )
    records = []
    for baseline, path in zip(BASELINES[1:], paths, strict=True):
        expected = expected_names(baseline)
        expected = {name for name in expected if not name.startswith("dense_products/")}
        found = set()
        for line in path.read_text().splitlines():
            match = pattern.fullmatch(line)
            if match and match[1] in expected:
                name = match[1]
                if name in found:
                    raise ValueError(f"Duplicate allocation sample: {name}")
                found.add(name)
                records.append([baseline, name, *map(int, match.groups()[1:])])
        if found != expected:
            raise ValueError(
                f"{baseline}: missing allocation samples: {sorted(expected - found)}"
            )
    with destination.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["baseline", "benchmark", "calls", "bytes", "operations"])
        writer.writerows(sorted(records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baselines", nargs="+", choices=BASELINES)
    parser.add_argument("--root", type=Path, default=Path("target/criterion"))
    parser.add_argument(
        "--allocation-logs", nargs=2, type=Path, metavar=("BEFORE", "AFTER")
    )
    parser.add_argument("--allocations", type=Path)
    args = parser.parse_args()
    if bool(args.allocation_logs) != bool(args.allocations):
        parser.error("--allocation-logs and --allocations must be supplied together")
    records = []
    for baseline in args.baselines:
        expected = expected_names(baseline)
        found = set()
        for path in args.root.glob(f"**/{baseline}/benchmark.json"):
            name = json.loads(path.read_text())["full_id"]
            if name not in expected:
                continue
            if name in found:
                raise ValueError(f"Duplicate timing sample: {baseline}/{name}")
            found.add(name)
            estimates = json.loads((path.parent / "estimates.json").read_text())["mean"]
            interval = estimates["confidence_interval"]
            values = [
                estimates["point_estimate"],
                interval["lower_bound"],
                interval["upper_bound"],
            ]
            samples = len(
                json.loads((path.parent / "sample.json").read_text())["times"]
            )
            if samples != 20 or not all(
                math.isfinite(value) and value > 0 for value in values
            ):
                raise ValueError(f"Invalid timing sample: {baseline}/{name}")
            records.append([baseline, name, *values, samples])
        if found != expected:
            parser.error(
                f"{baseline}: missing timing samples: {sorted(expected - found)}"
            )
    if args.allocations:
        export_allocations(args.allocation_logs, args.allocations)
    writer = csv.writer(sys.stdout)
    writer.writerow(
        ["baseline", "benchmark", "mean_ns", "lower_ns", "upper_ns", "samples"]
    )
    writer.writerows(sorted(records))


if __name__ == "__main__":
    main()
