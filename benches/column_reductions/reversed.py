"""Compare production reversed-column reductions and complete Shrinkage fits.

Wrap this runner once in compute bench, including all builds and measurements.
"""

import argparse
import csv
import io
import json
import os
import shutil
import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path

from strided import combine, command, sha, summarize


def export(repository, revision, destination):
    archive = subprocess.check_output(
        ["git", "-C", str(repository), "archive", revision]
    )
    destination.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        source.extractall(destination, filter="data")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline", default="fe41613")
    parser.add_argument("--shrinkage", type=Path)
    parser.add_argument(
        "--shrinkage-revision", default="87a735cdac7880cfe43a1219b19465587c1e9754"
    )
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--samples", type=int, default=3)
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
    shrinkage = (args.shrinkage or repository.parent / "shrinkage").resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    export(repository, args.baseline, output / "baseline-source")
    shutil.copytree(output / "baseline-source", output / "candidate-source")
    shutil.copytree(
        repository / "src", output / "candidate-source/src", dirs_exist_ok=True
    )
    for variant in ["baseline", "candidate"]:
        source = output / f"{variant}-source"
        shutil.copy2(repository / "Cargo.toml", source / "Cargo.toml")
        shutil.copy2(repository / "Cargo.lock", source / "Cargo.lock")
        shutil.copy2(
            repository / "benches/strided_column_reductions.rs",
            source / "benches/strided_column_reductions.rs",
        )
    export(shrinkage, args.shrinkage_revision, output / "shrinkage-source")

    sources = [
        "Cargo.toml",
        "Cargo.lock",
        "rust-toolchain.toml",
        "src/column.rs",
        "src/reductions.rs",
        "src/traits/columns.rs",
        "src/traits/vectors.rs",
        "src/backends/faer/dense.rs",
        "src/backends/ndarray/vector.rs",
        "src/backends/nalgebra/dense.rs",
        "tests/common/backend_aliases.rs",
        "benches/strided_column_reductions.rs",
        "benches/column_reductions/strided.py",
        "benches/column_reductions/reversed.py",
        "benches/column_reductions/reversed_consumer.rs",
    ]
    source_hashes = {name: sha(repository / name) for name in sources}
    environment = {
        **os.environ,
        "CARGO_TARGET_DIR": os.environ.get("CARGO_TARGET_DIR", str(output / "build")),
        "CARGO_PROFILE_RELEASE_CODEGEN_UNITS": "16",
        "PROBE_SAMPLES": str(args.samples),
        "PROBE_ELEMENTS": str(args.elements),
        "PROBE_SHAPES": args.shapes,
    }
    binaries = {}

    def build(variant, program, manifest, target_arguments):
        label = f"{variant}-{program}"
        print(f"Building {label}", flush=True)
        with (
            (output / f"{label}-build.jsonl").open("w") as stdout,
            (output / f"{label}-build.log").open("w") as stderr,
        ):
            subprocess.run(
                [
                    "cargo",
                    "rustc",
                    "--manifest-path",
                    str(manifest),
                    "--release",
                    "--locked",
                    *target_arguments,
                    "--message-format=json",
                    *(["--", "--emit=asm"] if program == "kernel" else []),
                ],
                env=environment,
                stdout=stdout,
                stderr=stderr,
                check=True,
            )
        artifacts = [
            json.loads(line)
            for line in (output / f"{label}-build.jsonl").read_text().splitlines()
        ]
        executable = next(
            Path(item["executable"])
            for item in artifacts
            if item.get("executable")
            and item["target"]["name"]
            == (
                "strided_column_reductions"
                if program == "kernel"
                else "reversed-consumer"
            )
        )
        binary = output / label
        shutil.copy2(executable, binary)
        if program == "kernel":
            assembly = sorted(executable.parent.glob(executable.name + "*.s"))
            assert assembly, "The compiler did not preserve kernel assembly."
            with (output / f"{label}.s").open("wb") as destination:
                for part in assembly:
                    destination.write(part.read_bytes())
        binaries[label] = sha(binary)

    for variant in ["baseline", "candidate"]:
        environment["CARGO_TARGET_DIR"] = str(
            Path(os.environ.get("CARGO_TARGET_DIR", output / "build")) / variant
        )
        source = output / f"{variant}-source"
        build(
            variant,
            "kernel",
            source / "Cargo.toml",
            [
                "--no-default-features",
                "--features",
                "faer,nalgebra,ndarray",
                "--bench",
                "strided_column_reductions",
            ],
        )
        workspace = output / f"{variant}-consumer-source"
        workspace.mkdir()
        shutil.copy2(
            repository / "benches/column_reductions/reversed_consumer.rs",
            workspace / "main.rs",
        )
        (workspace / "Cargo.toml").write_text(f"""[package]
name = "reversed-consumer"
version = "0.0.0"
edition = "2021"

[workspace]

[[bin]]
name = "reversed-consumer"
path = "main.rs"

[dependencies]
shrinkage = {{ path = "../shrinkage-source", default-features = false, features = ["ndarray_v0_17"] }}
ndarray = {{ version = "=0.17.2", default-features = false, features = ["std"] }}
rand = {{ version = "=0.10.3", default-features = false, features = ["std"] }}
rand_chacha = "=0.10.0"
serde_json = "1"

[patch.crates-io]
lazymatrix = {{ path = "../{variant}-source" }}
""")
        lock = repository / "benches/column_reductions/reversed_consumer.lock"
        if lock.exists():
            shutil.copy2(lock, workspace / "Cargo.lock")
        else:
            subprocess.run(
                [
                    "cargo",
                    "generate-lockfile",
                    "--offline",
                    "--manifest-path",
                    str(workspace / "Cargo.toml"),
                ],
                env=environment,
                check=True,
            )
        build(
            variant,
            "consumer",
            workspace / "Cargo.toml",
            ["--bin", "reversed-consumer"],
        )

    assert sha(output / "baseline-consumer-source/Cargo.lock") == sha(
        output / "candidate-consumer-source/Cargo.lock"
    )
    shutil.copy2(
        output / "candidate-consumer-source/Cargo.lock",
        output / "reversed_consumer.lock",
    )
    assert binaries["baseline-kernel"] != binaries["candidate-kernel"]
    records = []
    for block in range(args.blocks):
        variants = (
            ["baseline", "candidate"] if block % 2 == 0 else ["candidate", "baseline"]
        )
        for variant in variants:
            label = f"{variant}-b{block}"
            print(f"Checking and measuring {label}", flush=True)
            env = {
                **environment,
                "PROBE_LABEL": label,
                "PROBE_ORDER_OFFSET": str(block % 2),
            }
            with (
                (output / f"{label}.csv").open("w") as stdout,
                (output / f"{label}-errors.csv").open("w") as stderr,
            ):
                subprocess.run(
                    ["taskset", "-c", str(args.cpu), str(output / f"{variant}-kernel")],
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    check=True,
                )
            with (output / f"{label}-fits.jsonl").open("w") as stdout:
                subprocess.run(
                    [
                        "taskset",
                        "-c",
                        str(args.cpu),
                        str(output / f"{variant}-consumer"),
                    ],
                    env=env,
                    stdout=stdout,
                    check=True,
                )
            records.extend(
                json.loads(line)
                for line in (output / f"{label}-fits.jsonl").read_text().splitlines()
            )

    paths = [
        output / f"{variant}-b{block}"
        for block in range(args.blocks)
        for variant in ["baseline", "candidate"]
    ]
    combine(
        [path.with_suffix(".csv") for path in paths], output / "reversed_timings.csv"
    )
    combine(
        [path.with_name(path.name + "-errors.csv") for path in paths],
        output / "reversed_errors.csv",
    )
    # Keep variants separate when applying the investigation's summary routine.
    for variant in ["baseline", "candidate"]:
        combine(
            [output / f"{variant}-b{block}.csv" for block in range(args.blocks)],
            output / f"{variant}-timings.csv",
        )
        summarize(output / f"{variant}-timings.csv", output / f"{variant}-summary.csv")
    with (output / "reversed_summary.csv").open("w") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for index, variant in enumerate(["baseline", "candidate"]):
            with (output / f"{variant}-summary.csv").open() as source:
                reader = csv.reader(source)
                header = next(reader)
                if index == 0:
                    writer.writerow(["variant", *header])
                writer.writerows([variant, *row] for row in reader)
    with (output / "reversed_fits.csv").open("w") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(records[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)
    assert source_hashes == {name: sha(repository / name) for name in sources}, (
        "Source changed during comparison."
    )
    metadata = {
        "measured_at": datetime.now(UTC).isoformat(),
        "baseline_revision": command("git", "rev-parse", args.baseline),
        "candidate_base_revision": command("git", "rev-parse", "HEAD"),
        "candidate_source_sha256": source_hashes,
        "binary_sha256": binaries,
        "baseline_source_sha256": {
            name: sha(output / "baseline-source" / name)
            for name in sources
            if (output / "baseline-source" / name).exists()
        },
        "shrinkage_revision": args.shrinkage_revision,
        "consumer_lock_sha256": sha(output / "reversed_consumer.lock"),
        "rustc": command("rustc", "-Vv"),
        "cargo": command("cargo", "-V"),
        "cpu": command("lscpu"),
        "pinned_cpu": args.cpu,
        "compute_window": "Both builds, kernel controls, and complete consumer fits in one exclusive compute bench window.",
        "profile": "release, 16 codegen units, default LTO",
        "rustflags": environment.get("RUSTFLAGS", ""),
        "features": ["faer", "nalgebra", "ndarray"],
        "seed": 8421,
        "shapes": args.shapes,
        "scalars": ["f32", "f64"],
        "blocks": args.blocks,
        "samples_per_process": args.samples,
        "elements_per_sample": args.elements,
        "order": "Process order alternates by block; kernel method and operation order alternates by sample and block.",
        "consumer": "Shrinkage Lasso, 5000 x {64, 1000}, penalties 0.01 and 2, centered columns, fitted intercept, relative gap target 1e-8; forward and reverse borrowed ndarray views with aligned responses.",
        "consumer_timing": "Complete fit; construction, verification, serialization, and returned-fit destruction outside timing.",
        "consumer_checks": "Every warmup and timed fit independently checked with raw input products, primal and dual objectives, KKT violation, dual feasibility, and relative gap.",
        "artifacts": str(output),
    }
    (output / "reversed_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"Measurements saved in {output}", flush=True)


if __name__ == "__main__":
    main()
