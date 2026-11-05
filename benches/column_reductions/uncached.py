"""Build and compare fused uncached dots inside one compute bench reservation."""

import argparse
import csv
import json
import os
import platform
import shutil
import statistics
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from strided import combine, command, sha


def summarize(source, destination):
    keys = [
        "build",
        "scalar",
        "regime",
        "rows",
        "cols",
        "working_set_bytes",
        "operation",
        "method",
    ]
    groups = defaultdict(list)
    with source.open(newline="") as handle:
        for row in csv.DictReader(handle):
            row["build"] = row["label"].split("-b")[0]
            groups[tuple(row[key] for key in keys)].append(float(row["ns_per_dot"]))
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
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--elements", type=int, default=8_000_000)
    parser.add_argument("--bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument(
        "--lengths", default="0,1,7,8,9,33,256,5000,65536,1048576,4194304"
    )
    parser.add_argument(
        "--builds",
        nargs="+",
        choices=["generic", "native"],
        default=["generic", "native"],
    )
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args()
    if (
        min(args.blocks, args.samples, args.elements, args.bytes) < 1
        or args.blocks * args.samples < 2
    ):
        parser.error("Use positive counts and at least two timing observations.")
    if len(args.builds) != len(set(args.builds)):
        parser.error("Build names must be unique.")
    repository = Path(__file__).resolve().parents[2]
    os.chdir(repository)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = [
        "Cargo.toml",
        "Cargo.lock",
        "rust-toolchain.toml",
        "src/reductions.rs",
        "src/column.rs",
        "src/traits/columns.rs",
        "src/traits/vectors.rs",
        "benches/uncached_column_dots.rs",
        "benches/column_reductions/uncached.py",
        "benches/column_reductions/strided.py",
    ]
    source_hashes = {name: sha(repository / name) for name in sources}
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_RUSTFLAGS"}
        and not key.startswith("CARGO_PROFILE_RELEASE_")
    }
    environment.update(
        {
            "CARGO_PROFILE_RELEASE_CODEGEN_UNITS": "16",
            "PROBE_SAMPLES": str(args.samples),
            "PROBE_ELEMENTS": str(args.elements),
            "PROBE_BYTES": str(args.bytes),
            "PROBE_LENGTHS": args.lengths,
        }
    )
    binaries = {}
    for build in args.builds:
        build_environment = {
            **environment,
            "RUSTFLAGS": "-C target-cpu=native"
            if build == "native"
            else "-C target-cpu=x86-64",
            "CARGO_TARGET_DIR": str(output / f"{build}-build"),
        }
        print(f"Building {build}", flush=True)
        with (
            (output / f"{build}-build.jsonl").open("w") as stdout,
            (output / f"{build}-build.log").open("w") as stderr,
        ):
            subprocess.run(
                [
                    "cargo",
                    "rustc",
                    "--release",
                    "--locked",
                    "--no-default-features",
                    "--bench",
                    "uncached_column_dots",
                    "--message-format=json",
                    "--",
                    "--emit=asm",
                ],
                env=build_environment,
                stdout=stdout,
                stderr=stderr,
                check=True,
            )
        artifacts = [
            json.loads(line)
            for line in (output / f"{build}-build.jsonl").read_text().splitlines()
        ]
        executable = next(
            Path(item["executable"])
            for item in artifacts
            if item.get("executable")
            and item["target"]["name"] == "uncached_column_dots"
        )
        binary = output / build
        shutil.copy2(executable, binary)
        assembly_files = sorted(executable.parent.glob(f"{executable.name}*.s"))
        assert assembly_files, "Cargo emitted no probe assembly."
        with (output / f"{build}.s").open("w") as assembly:
            for path in assembly_files:
                assembly.write(f"# {path.name}\n{path.read_text()}\n")
        binaries[build] = binary

    def run(build, label, **overrides):
        print(f"Checking and measuring {label}", flush=True)
        with (
            (output / f"{label}.csv").open("w") as stdout,
            (output / f"{label}-errors.csv").open("w") as stderr,
        ):
            subprocess.run(
                ["taskset", "-c", str(args.cpu), str(binaries[build])],
                env={**environment, "PROBE_LABEL": label, **overrides},
                stdout=stdout,
                stderr=stderr,
                check=True,
            )

    for build in args.builds:
        for switch, suffix in [
            ("PROBE_NUMERICS", "numerics"),
            ("PROBE_SPARSE", "sparse"),
        ]:
            with (output / f"{build}-{suffix}.csv").open("w") as stdout:
                subprocess.run(
                    ["taskset", "-c", str(args.cpu), str(binaries[build])],
                    env={**environment, switch: "1"},
                    stdout=stdout,
                    check=True,
                )
        run(
            build,
            f"{build}-tails",
            PROBE_SAMPLES="0",
            PROBE_LENGTHS="0,1,7,8,9,15,16,17,31,32,33",
        )
    if not args.checks_only:
        for block in range(args.blocks):
            builds = args.builds if block % 2 == 0 else args.builds[::-1]
            for build in builds:
                run(build, f"{build}-b{block}", PROBE_ORDER=str(block))
        combine(
            [
                output / f"{build}-b{block}.csv"
                for block in range(args.blocks)
                for build in args.builds
            ],
            output / "uncached_timings.csv",
        )
        summarize(output / "uncached_timings.csv", output / "uncached_summary.csv")
    combine(
        [
            output / f"{build}-{label}-errors.csv"
            for build in args.builds
            for label in [
                "tails",
                *([] if args.checks_only else [f"b{b}" for b in range(args.blocks)]),
            ]
        ],
        output / "uncached_errors.csv",
    )
    assert source_hashes == {name: sha(repository / name) for name in sources}, (
        "Source changed during measurement."
    )
    metadata = {
        "measured_at": datetime.now(UTC).isoformat(),
        "revision": command("git", "rev-parse", "HEAD"),
        "source_sha256": source_hashes,
        "binary_sha256": {name: sha(binary) for name, binary in binaries.items()},
        "assembly_sha256": {name: sha(output / f"{name}.s") for name in binaries},
        "rustc": command("rustc", "-Vv"),
        "cargo": command("cargo", "-V"),
        "platform": platform.platform(),
        "cpu": command("lscpu"),
        "cpu_caches": command("lscpu", "-C"),
        "pinned_cpu": args.cpu,
        "compute_window": "One exclusive compute bench window for both builds and every method.",
        "profile": "release, opt-level=3, 16 codegen units, Cargo default LTO",
        "builds": {
            name: {
                "rustflags": "-C target-cpu=native"
                if name == "native"
                else "-C target-cpu=x86-64"
            }
            for name in binaries
        },
        "features": [],
        "seed": 8421,
        "scalars": ["f32", "f64"],
        "lengths": args.lengths,
        "rotating_target_bytes": args.bytes,
        "blocks": args.blocks,
        "samples_per_process": args.samples,
        "elements_per_sample": args.elements,
        "passes": "clamp(elements / max(rows * cols, 1), 1, 100000)",
        "working_set": "Three independently generated padded column banks, cached sums, and saved centers and scales; each dot uses its own vector and weights.",
        "warmup": "All methods and columns checked before timing; one extra pass before each sample.",
        "order": "Build order reverses per block. Operation and method orders rotate by sample and block; the default six blocks place every method at each position twice.",
        "verification": "Compensated f64 directly centered oracle on random inputs; tolerance 64*epsilon*max(sqrt(rows),1)*(1+abs(oracle)). All methods except Fused4 must also match production bits, treating NaNs by classification.",
        "allocation_check": "Zero allocation or reallocation calls around every verified dot and timed sample.",
        "numerics": "Tails, cancellation, large offsets, reduction and correction overflow, weighted multiply order, nonfinite values and parameters, and signed zeros.",
        "sparse": "Public cached methods on unsorted duplicate entries and explicit zeros: five stored visits and vector reads at 17 and one billion logical rows; no vector sum or allocations.",
        "checks_only": args.checks_only,
        "artifacts": str(output),
    }
    (output / "uncached_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"Results saved in {output}", flush=True)


if __name__ == "__main__":
    main()
