#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
output_dir=${1:-target/normalization-tradeoffs}
mkdir -p "$output_dir"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target}"
warm_up=0.3
measurement=1
samples=30
resamples=10000
if [[ ${LAZYMATRIX_BENCH_QUICK:-0} == 1 ]]; then
	warm_up=0.05
	measurement=0.1
	samples=10
	resamples=1000
fi

# Compile both executables before collection so builds cannot overlap timings.
cargo bench --locked --features ndarray,sprs,faer \
	--bench eager_normalization --bench normalization_memory --no-run
cargo bench --locked --features ndarray,sprs,faer --bench normalization_memory \
	>"$output_dir/memory.csv"
cargo bench --locked --features ndarray,sprs,faer --bench eager_normalization -- \
	--noplot --warm-up-time "$warm_up" --measurement-time "$measurement" \
	--sample-size "$samples" --nresamples "$resamples" \
	>"$output_dir/timing.log" 2>&1
python3 scripts/summarize-normalization-tradeoffs.py \
	"$output_dir/memory.csv" "$CARGO_TARGET_DIR/criterion" "$output_dir/results.csv"
python3 - "$output_dir/metadata.json" "$warm_up" "$measurement" "$samples" "$resamples" <<'PY'
import datetime
import json
import platform
import subprocess
import sys

cpu = platform.processor()
try:
    for line in subprocess.check_output(["lscpu"], text=True).splitlines():
        if line.startswith("Model name:"):
            cpu = line.split(":", 1)[1].strip()
except (OSError, subprocess.CalledProcessError):
    pass
metadata = {
    "collected_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "cpu": cpu,
    "platform": platform.platform(),
    "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
    "seed": 205,
    "sample_size": int(sys.argv[4]),
    "warm_up_seconds": float(sys.argv[2]),
    "measurement_seconds": float(sys.argv[3]),
    "bootstrap_resamples": int(sys.argv[5]),
    "confidence_level": 0.95,
    "lifecycle_pairs": 10,
    "threads": 1,
    "features": ["ndarray", "sprs", "faer"],
    "timing_estimator": "Criterion mean",
    "source_retained": True,
}
with open(sys.argv[1], "w") as output:
    json.dump(metadata, output, indent=2)
    output.write("\n")
PY
printf 'Measurements written to %s\n' "$output_dir"
