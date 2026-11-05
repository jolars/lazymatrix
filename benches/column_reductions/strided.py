"""Build, check, measure, and summarize the strided-column investigation.

Run inside one compute bench reservation so every method shares the same window.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path


def command(*args, **kwargs):
    return subprocess.check_output(args, text=True, **kwargs).strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def combine(paths, destination):
    with destination.open("w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        for index, path in enumerate(paths):
            with path.open(newline="") as source:
                reader = csv.reader(source)
                header = next(reader)
                if index == 0:
                    writer.writerow(header)
                writer.writerows(reader)


def summarize(source, destination):
    groups = defaultdict(list)
    keys = ["backend", "layout", "scalar", "operation", "method", "rows", "cols"]
    with source.open(newline="") as handle:
        for row in csv.DictReader(handle):
            groups[tuple(row[key] for key in keys)].append(float(row["ns_per_pass"]))
    with destination.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([*keys, "samples", "median_ns", "q25_ns", "q75_ns"])
        for key, values in sorted(groups.items()):
            quartiles = statistics.quantiles(values, n=4, method="inclusive")
            writer.writerow(
                [
                    *key,
                    len(values),
                    statistics.median(values),
                    quartiles[0],
                    quartiles[2],
                ]
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--elements", type=int, default=10_000_000)
    parser.add_argument("--shapes", default="33x7,5000x64,5000x1000")
    args = parser.parse_args()
    if (
        min(args.blocks, args.samples, args.elements) < 1
        or args.blocks * args.samples < 2
    ):
        parser.error("Use positive counts and at least two timing observations.")
    repository = Path(__file__).resolve().parents[2]
    os.chdir(repository)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "CARGO_TARGET_DIR": str(output / "build"),
        "CARGO_PROFILE_RELEASE_CODEGEN_UNITS": "16",
        "PROBE_SAMPLES": str(args.samples),
        "PROBE_ELEMENTS": str(args.elements),
        "PROBE_SHAPES": args.shapes,
    }
    sources = [
        "Cargo.toml",
        "Cargo.lock",
        "rust-toolchain.toml",
        "src/reductions.rs",
        "src/column.rs",
        "src/traits/columns.rs",
        "src/traits/vectors.rs",
        "src/backends/ndarray/vector.rs",
        "src/backends/faer/dense.rs",
        "src/backends/nalgebra/dense.rs",
        "tests/common/backend_aliases.rs",
        "benches/strided_column_reductions.rs",
        "benches/column_reductions/strided.py",
    ]
    source_hashes = {source: sha(repository / source) for source in sources}
    print(f"Building probe in {output}", flush=True)
    with (
        (output / "build.jsonl").open("w") as stdout,
        (output / "build.log").open("w") as stderr,
    ):
        subprocess.run(
            [
                "cargo",
                "rustc",
                "--release",
                "--locked",
                "--no-default-features",
                "--features",
                "faer,nalgebra,ndarray",
                "--bench",
                "strided_column_reductions",
                "--message-format=json",
                "--",
                "--emit=asm",
            ],
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=True,
        )
    artifacts = [
        json.loads(line) for line in (output / "build.jsonl").read_text().splitlines()
    ]
    executable = next(
        Path(item["executable"])
        for item in artifacts
        if item.get("executable")
        and item["target"]["name"] == "strided_column_reductions"
    )
    binary = output / "probe"
    shutil.copy2(executable, binary)
    assembly = next(executable.parent.glob("strided_column_reductions-*.s"))
    shutil.copy2(assembly, output / "probe.s")

    def run(label, **overrides):
        print(f"Checking and measuring {label}", flush=True)
        with (
            (output / f"{label}.csv").open("w") as stdout,
            (output / f"{label}-errors.csv").open("w") as stderr,
        ):
            subprocess.run(
                ["taskset", "-c", str(args.cpu), str(binary)],
                env={**environment, "PROBE_LABEL": label, **overrides},
                stdout=stdout,
                stderr=stderr,
                check=True,
            )

    with (output / "strided_numerics.csv").open("w") as stdout:
        subprocess.run(
            ["taskset", "-c", str(args.cpu), str(binary)],
            env={**environment, "PROBE_NUMERICS": "1"},
            stdout=stdout,
            check=True,
        )
    run(
        "tails",
        PROBE_SAMPLES="0",
        PROBE_SHAPES=",".join(
            f"{n}x3" for n in [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33]
        ),
    )
    for block in range(args.blocks):
        run(f"b{block}", PROBE_ORDER_OFFSET=str(block % 2))
    combine(
        [output / f"b{block}.csv" for block in range(args.blocks)],
        output / "strided_timings.csv",
    )
    combine(
        [
            output / "tails-errors.csv",
            *(output / f"b{block}-errors.csv" for block in range(args.blocks)),
        ],
        output / "strided_errors.csv",
    )
    summarize(output / "strided_timings.csv", output / "strided_summary.csv")
    assert source_hashes == {source: sha(repository / source) for source in sources}, (
        "Source changed during the comparison."
    )
    metadata = {
        "measured_at": datetime.now(UTC).isoformat(),
        "revision": command("git", "rev-parse", "HEAD"),
        "source_sha256": source_hashes,
        "binary_sha256": sha(binary),
        "assembly_sha256": sha(output / "probe.s"),
        "rustc": command("rustc", "-Vv"),
        "cargo": command("cargo", "-V"),
        "platform": platform.platform(),
        "cpu": command("lscpu"),
        "pinned_cpu": args.cpu,
        "compute_window": "One exclusive compute bench window for the build, validation, and all methods.",
        "profile": "release, opt-level=3, 16 codegen units, Cargo default LTO",
        "rustflags": environment.get("RUSTFLAGS", ""),
        "features": ["faer", "nalgebra", "ndarray"],
        "backend_versions": command(
            "cargo",
            "tree",
            "--locked",
            "--no-default-features",
            "--features",
            "faer,nalgebra,ndarray",
            "--depth",
            "1",
        ),
        "seed": 8421,
        "scalars": ["f32", "f64"],
        "shapes": args.shapes,
        "blocks": args.blocks,
        "samples_per_process": args.samples,
        "elements_per_sample": args.elements,
        "passes": "clamp(elements_per_sample / max(rows * cols, 1), 1, 10000)",
        "warmup": "Every method and every column checked before timing each layout.",
        "order": "Operation and method order reverses on alternate samples and blocks.",
        "comparison": "Compensated f64 oracle; tolerance 32 * epsilon * max(sqrt(rows), 1) * (1 + abs(oracle)).",
        "borrowing": "NaN padding and offset 3; logical slice and visitor contracts checked; memory order is private to the prototype.",
        "alignment": "Untimed dot, signed weighted dot, and signed weighted norm checked in every layout against independent logical arithmetic.",
        "artifacts": str(output),
    }
    (output / "strided_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Measurements and summary saved in {output}", flush=True)


if __name__ == "__main__":
    main()
