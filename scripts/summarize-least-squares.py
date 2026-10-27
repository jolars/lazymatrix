#!/usr/bin/env python3
"""Join complete-fit allocation and convergence profiles with Criterion timings."""

import argparse
import csv
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("memory", type=Path)
    parser.add_argument("criterion", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.memory.open() as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames
        profiles = list(reader)
    if not profiles:
        raise ValueError("No complete-fit profiles")
    expected = {}
    for row in profiles:
        if row["strategy"] not in {"lazy", "eager"}:
            raise ValueError("Unknown strategy")
        name = f"least_squares/{row['backend']}/{row['n']}x{row['p']}/{row['target_density']}/{row['strategy']}"
        if name in expected:
            raise ValueError(f"Duplicate profile: {name}")
        expected[name] = row
    estimates = {}
    for path in args.criterion.rglob("new/benchmark.json"):
        name = json.loads(path.read_text())["full_id"]
        if name not in expected:
            continue
        if name in estimates:
            raise ValueError(f"Duplicate timing: {name}")
        estimate = json.loads(path.with_name("estimates.json").read_text())["mean"]
        interval = estimate["confidence_interval"]
        point, lower, upper = (
            estimate["point_estimate"],
            interval["lower_bound"],
            interval["upper_bound"],
        )
        if (
            not all(
                math.isfinite(value) and value > 0 for value in [point, lower, upper]
            )
            or not lower <= point <= upper
        ):
            raise ValueError(f"Invalid interval: {name}")
        estimates[name] = (point, lower, upper)
    if estimates.keys() != expected.keys():
        raise ValueError(
            f"Missing timings: {sorted(expected.keys() - estimates.keys())}"
        )
    with args.output.open("w") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=fields + ["estimate_ns", "lower_ns", "upper_ns"],
            lineterminator="\n",
        )
        writer.writeheader()
        for name, row in sorted(expected.items()):
            point, lower, upper = estimates[name]
            writer.writerow(
                {
                    **row,
                    "estimate_ns": f"{point:.6f}",
                    "lower_ns": f"{lower:.6f}",
                    "upper_ns": f"{upper:.6f}",
                }
            )


if __name__ == "__main__":
    main()
