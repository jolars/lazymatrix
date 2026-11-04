#!/usr/bin/env bash
set -euo pipefail

# Reserve one compute bench window around this script, including both builds.
repository=$(git rev-parse --show-toplevel)
output=${1:?Specify an output directory.}
baseline_ref=${2:-ac5f2e4}
benchmark=${PROBE_BENCH:-adapter_column_reductions}
features=${PROBE_FEATURES:-faer,nalgebra}
prefix=${PROBE_PREFIX:-adapter}
mkdir -p "$output"
output=$(realpath "$output")
baseline="$output/baseline-source"
mkdir -p "$baseline"
git -C "$repository" archive "$baseline_ref" | tar -x -C "$baseline"
cp "$repository/benches/$benchmark.rs" "$baseline/benches/"
if ! rg -q "^name = \"$benchmark\"$" "$baseline/Cargo.toml"; then
    cat >>"$baseline/Cargo.toml" <<TOML

[[bench]]
name = "$benchmark"
harness = false
TOML
fi

export CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16
export PROBE_ROWS=${PROBE_ROWS:-5000}
export PROBE_COLS=${PROBE_COLS:-1000}
export PROBE_SAMPLES=${PROBE_SAMPLES:-3}
export PROBE_PASSES=${PROBE_PASSES:-5}
cpu=${PROBE_CPU:-2}
blocks=${PROBE_BLOCKS:-4}

for label in baseline candidate; do
    export CARGO_TARGET_DIR="$output/$label-build"
    source_dir=$repository
    if [[ $label == baseline ]]; then
        source_dir=$baseline
    fi
    cargo build --manifest-path "$source_dir/Cargo.toml" --release --locked \
        --no-default-features --features "$features" --bench "$benchmark" \
        --message-format=json >"$output/$label-build.json" 2>"$output/$label-build.log"
    executable=$(
        python - "$output/$label-build.json" "$benchmark" <<'PY'
import json
import sys

for line in open(sys.argv[1]):
    item = json.loads(line)
    if item.get("executable") and item.get("target", {}).get("name") == sys.argv[2]:
        print(item["executable"])
PY
    )
    cp "$executable" "$output/$label"
done

if cmp -s "$output/baseline" "$output/candidate"; then
    echo "Baseline and candidate binaries are identical; refusing to measure." >&2
    exit 1
fi

for ((block = 0; block < blocks; block++)); do
    labels=(baseline candidate)
    if ((block % 2 == 1)); then
        labels=(candidate baseline)
    fi
    for label in "${labels[@]}"; do
        PROBE_LABEL="$label-b$block" taskset -c "$cpu" "$output/$label" \
            >"$output/$label-b$block.csv" 2>"$output/$label-b$block-errors.csv"
    done
done

python - "$repository" "$output" "$baseline_ref" "$cpu" "$blocks" "$benchmark" "$features" "$prefix" <<'PY'
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

repository, output, baseline_ref, cpu, blocks, benchmark, features, prefix = sys.argv[1:]
repository, output = Path(repository), Path(output)
for suffix, destination in [("", f"{prefix}_timings.csv"), ("-errors", f"{prefix}_errors.csv")]:
    with (output / destination).open("w") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        paths = [output / f"{label}-b{block}{suffix}.csv"
                 for block in range(int(blocks)) for label in ["baseline", "candidate"]]
        for index, path in enumerate(paths):
            with path.open() as source:
                reader = csv.reader(source)
                header = next(reader)
                if index == 0:
                    writer.writerow(header)
                writer.writerows(reader)

def command(*args):
    return subprocess.check_output(args, cwd=repository, text=True).strip()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

sources = ["Cargo.toml", "Cargo.lock", "src/backends/faer/dense.rs",
           "src/backends/nalgebra/dense.rs", "src/column.rs", "src/reductions.rs",
           "src/traits/vectors.rs", "src/backends/ndarray/vector.rs",
           f"benches/{benchmark}.rs", "benches/column_reductions/run-adapters.sh"]
metadata = {
    "measured_at": datetime.now(timezone.utc).isoformat(),
    "baseline_revision": command("git", "rev-parse", baseline_ref),
    "candidate_base_revision": command("git", "rev-parse", "HEAD"),
    "candidate_has_uncommitted_changes": bool(command("git", "status", "--porcelain")),
    "candidate_source_sha256": {source: sha(repository / source) for source in sources},
    "binary_sha256": {label: sha(output / label) for label in ["baseline", "candidate"]},
    "rustc": command("rustc", "-Vv"),
    "cargo": command("cargo", "-V"),
    "platform": platform.platform(),
    "cpu": command("lscpu"),
    "pinned_cpu": int(cpu),
    "compute_window": "One exclusive compute bench window for both builds and all measurements.",
    "profile": "release, opt-level=3, 16 codegen units, Cargo default LTO, generic x86-64",
    "rustflags": os.environ.get("RUSTFLAGS", ""),
    "features": features.split(","),
    "backend_versions": {"faer": "0.24.4", "nalgebra": "0.35.0", "ndarray": "0.17.2"},
    "seed": 8421,
    "scalar": "f64",
    "rows": int(os.environ["PROBE_ROWS"]),
    "cols": int(os.environ["PROBE_COLS"]),
    "blocks": int(blocks),
    "samples_per_process": int(os.environ["PROBE_SAMPLES"]),
    "passes_per_sample": int(os.environ["PROBE_PASSES"]),
    "warmup": "All column results checked before each process starts timing each layout.",
    "order": "Baseline/candidate process order alternates by block; operation/method order alternates by sample.",
    "comparison": "Each column checked against sequential indexed arithmetic, tolerance 1e-10 * (1 + abs(expected)).",
}
(output / f"{prefix}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
PY
