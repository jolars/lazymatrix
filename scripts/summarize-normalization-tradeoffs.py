#!/usr/bin/env python3
"""Join complete lifecycle profiles with Criterion mean estimates and intervals."""

import argparse
import csv
import json
import math
from pathlib import Path

TIMINGS = ["fit_parameters", "materialize"] + [
    f"{strategy}_{direction}"
    for strategy in ["raw", "lazy", "eager"]
    for direction in ["forward", "transpose"]
]
KEY = ["backend", "n", "p", "target_density", "normalization"]
FIXTURE = ["backend", "n", "p", "target_density", "nnz", "empty_columns", "normalization"]
MEMORY = ["source_bytes", "representation_bytes", "workspace_bytes", "peak_total_bytes"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("memory", type=Path)
    parser.add_argument("criterion", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    profiles = list(csv.DictReader(args.memory.open()))
    groups = {}
    for row in profiles:
        key = tuple(row[field] for field in KEY)
        group = groups.setdefault(key, {})
        if row["strategy"] in group:
            raise ValueError(f"Duplicate memory profile: {key}")
        group[row["strategy"]] = row
    if not groups:
        raise ValueError("No memory profiles")

    expected = {}
    for key, strategies in groups.items():
        if set(strategies) != {"raw", "lazy", "eager"}:
            raise ValueError(f"Incomplete memory profiles: {key}")
        row = strategies["lazy"]
        prefix = f'normalization/{row["backend"]}/{row["normalization"]}/{row["n"]}x{row["p"]}/{row["target_density"]}'
        for timing in TIMINGS:
            expected[f"{prefix}/{timing}"] = (row, timing)
    estimates = {}
    for path in args.criterion.rglob("new/benchmark.json"):
        name = json.loads(path.read_text())["full_id"]
        if name not in expected:
            continue
        if name in estimates:
            raise ValueError(f"Duplicate timing: {name}")
        estimate = json.loads(path.with_name("estimates.json").read_text())["mean"]
        interval = estimate["confidence_interval"]
        point, lower, upper = estimate["point_estimate"], interval["lower_bound"], interval["upper_bound"]
        if not all(math.isfinite(value) and value > 0 for value in [point, lower, upper]) or not lower <= point <= upper:
            raise ValueError(f"Invalid interval: {name}")
        estimates[name] = (point, lower, upper)
    if estimates.keys() != expected.keys():
        raise ValueError(f"Missing timings: {sorted(expected.keys() - estimates.keys())}")

    with args.output.open("w") as output:
        writer = csv.DictWriter(output, fieldnames=FIXTURE + ["measurement", "estimate_ns", "lower_ns", "upper_ns"] + MEMORY, lineterminator="\n")
        writer.writeheader()
        for name, (row, timing) in sorted(expected.items()):
            point, lower, upper = estimates[name]
            writer.writerow({**{field: row[field] for field in FIXTURE}, "measurement": timing,
                             "estimate_ns": f"{point:.6f}", "lower_ns": f"{lower:.6f}", "upper_ns": f"{upper:.6f}"})
        for key, strategies in sorted(groups.items()):
            for strategy, row in sorted(strategies.items()):
                writer.writerow({**{field: row[field] for field in FIXTURE + MEMORY}, "measurement": f"{strategy}_memory"})


if __name__ == "__main__":
    main()
