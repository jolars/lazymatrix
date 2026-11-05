"""Summarize the matched vector-reduction measurements as CSV."""

import csv
import statistics
import sys
from collections import defaultdict


def main():
    groups = defaultdict(lambda: defaultdict(list))
    with open(sys.argv[1], newline="") as source:
        for row in csv.DictReader(source):
            key = tuple(
                row[name]
                for name in ["backend", "layout", "scalar", "operation", "length"]
            )
            variant = row["label"].split("-b")[0]
            groups[key][variant, row["method"]].append(float(row["nanoseconds"]))

    writer = csv.writer(sys.stdout, lineterminator="\n")
    columns = ["backend", "layout", "scalar", "operation", "length"]
    for variant, method in [
        ("baseline", "lazy"),
        ("candidate", "lazy"),
        ("baseline", "native"),
        ("candidate", "native"),
    ]:
        columns.extend(
            f"{variant}_{method}_{stat}"
            for stat in ["samples", "median_ns", "q25_ns", "q75_ns"]
        )
    writer.writerow([*columns, "speedup", "native_speedup"])
    for key, observations in sorted(groups.items()):
        result = list(key)
        for variant, method in [
            ("baseline", "lazy"),
            ("candidate", "lazy"),
            ("baseline", "native"),
            ("candidate", "native"),
        ]:
            values = observations.get((variant, method), [])
            if values:
                q25, _, q75 = statistics.quantiles(values, n=4, method="inclusive")
                result.extend([len(values), statistics.median(values), q25, q75])
            else:
                result.extend([0, "", "", ""])
        speedup = statistics.median(
            observations["baseline", "lazy"]
        ) / statistics.median(observations["candidate", "lazy"])
        native_speedup = ""
        if ("baseline", "native") in observations:
            native_speedup = statistics.median(
                observations["baseline", "native"]
            ) / statistics.median(observations["candidate", "native"])
        writer.writerow([*result, speedup, native_speedup])


if __name__ == "__main__":
    main()
