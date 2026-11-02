#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
output_dir=${1:-target/normalization-pipeline}
mkdir -p "$output_dir"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target}"
warm_up=0.2
measurement=0.5
samples=20
resamples=10000
if [[ ${LAZYMATRIX_BENCH_QUICK:-0} == 1 ]]; then
	warm_up=0.05
	measurement=0.1
	samples=10
	resamples=1000
fi

cargo bench --locked --no-default-features --features ndarray,sprs \
	--bench normalization_pipeline --bench normalization_pipeline_memory --no-run
cargo bench --locked --no-default-features --features ndarray,sprs \
	--bench normalization_pipeline_memory >"$output_dir/memory.csv"
cargo bench --locked --no-default-features --features ndarray,sprs \
	--bench normalization_pipeline -- --noplot --warm-up-time "$warm_up" \
	--measurement-time "$measurement" --sample-size "$samples" --nresamples "$resamples" \
	>"$output_dir/timing.log" 2>&1
python3 - "$output_dir" "$CARGO_TARGET_DIR/criterion" "$warm_up" "$measurement" "$samples" "$resamples" <<'PY'
import csv
import datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

output, criterion = map(Path, sys.argv[1:3])
memory = {}
with (output / "memory.csv").open() as source:
    for row in csv.DictReader(source):
        memory[(row["rows"], row["cols"], row["density"], row["orientation"])] = row
fields = ["rows", "cols", "density", "orientation", "phase", "method",
          "median_ns", "ci_lower_ns", "ci_upper_ns", "lifecycle_peak_bytes"]
rows = []
for manifest in criterion.rglob("new/benchmark.json"):
    benchmark = json.loads(manifest.read_text())
    parts = benchmark["full_id"].split("/")
    if parts[0] != "pipeline":
        continue
    nrows, ncols = parts[1].split("x")
    density, operation = parts[2:4]
    method = operation.rsplit("_", 1)[1]
    orientation = parts[4] if len(parts) > 4 else "dense_mixed"
    key = (nrows, ncols, density, orientation)
    if key not in memory:
        continue
    median = json.loads(manifest.with_name("estimates.json").read_text())["median"]
    interval = median["confidence_interval"]
    rows.append([nrows, ncols, density, orientation,
                 "fit_pair" if "fit_pair" in operation else "pair", method,
                 median["point_estimate"], interval["lower_bound"], interval["upper_bound"],
                 memory[key][f"{method}_peak_bytes"]])
with (output / "results.csv").open("w") as target:
    writer = csv.writer(target, lineterminator="\n")
    writer.writerow(fields)
    writer.writerows(sorted(rows))
cpu = platform.processor()
for line in subprocess.check_output(["lscpu"], text=True).splitlines():
    if line.startswith("Model name:"):
        cpu = line.split(":", 1)[1].strip()
metadata = {
    "collected_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "cpu": cpu,
    "platform": platform.platform(),
    "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
    "features": ["ndarray_v0_17", "sprs_v0_11"],
    "seed": 205,
    "threads": 1,
    "warm_up_seconds": float(sys.argv[3]),
    "measurement_seconds": float(sys.argv[4]),
    "sample_size": int(sys.argv[5]),
    "bootstrap_resamples": int(sys.argv[6]),
    "confidence_level": 0.95,
    "timing_estimator": "Criterion median",
    "smoke": "LAZYMATRIX_BENCH_SMOKE" in os.environ,
    "memory_measure": "Peak additional live requested Rust bytes, source excluded",
    "baseline": "One eager normalized dense matrix, with the second stage applied in place",
}
(output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
PY
