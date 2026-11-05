"""Compare indexed column updates, a mutable-slice prototype, and complete fits.

Run once inside compute bench to keep all builds and measurements together.
"""

import argparse
import csv
import json
import os
import re
import shutil
import statistics
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from reversed import export
from strided import combine, command, sha

VARIANTS = ("baseline", "candidate", "source_slice")
ORDERS = [
    ["baseline", "candidate", "source_slice"],
    ["source_slice", "candidate", "baseline"],
    ["candidate", "source_slice", "baseline"],
    ["baseline", "source_slice", "candidate"],
    ["source_slice", "baseline", "candidate"],
    ["candidate", "baseline", "source_slice"],
]


def summary(source, destination, keys, value):
    groups = defaultdict(list)
    with source.open(newline="") as handle:
        for row in csv.DictReader(handle):
            variant = row["label"].split("-b")[0]
            groups[(variant, *(row[key] for key in keys))].append(float(row[value]))
    with destination.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["variant", *keys, "samples", "median_ns", "q25_ns", "q75_ns"])
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


def assembly_summary(output):
    records = []
    for variant in VARIANTS:
        assembly = (output / f"{variant}-kernel.s").read_text()

        def body(symbol, assembly=assembly):
            matches = re.findall(
                r"^" + re.escape(symbol) + r":\n(.*?)^\.Lfunc_end\d+:",
                assembly,
                re.MULTILINE | re.DOTALL,
            )
            assert len(matches) == 1, f"Missing or ambiguous assembly for {symbol}"
            return matches[0]

        for backend in ["ndarray", "faer", "nalgebra"]:
            for scalar in ["f32", "f64"]:
                entry = body(f"updates_{backend}_{scalar}")
                symbol = re.search(r"\tjmp\t([^\s]+)", entry)[1]
                code = body(symbol)
                helpers = [
                    name
                    for name in re.findall(r"\t(?:callq|jmp)\t([^\s]+)", code)
                    if "affine_add" in name
                ]
                if helpers:
                    symbol = helpers[0]
                    code = body(symbol)
                artifact = output / f"{variant}-{backend}-{scalar}-update.s"
                artifact.write_text(symbol + ":\n" + code)
                records.append(
                    {
                        "variant": variant,
                        "backend": backend,
                        "scalar": scalar,
                        "symbol": symbol,
                        "assembly_sha256": sha(artifact),
                        "lines": len(code.splitlines()),
                        "packed_multiply_instructions": len(
                            re.findall(r"\tmulp[sd]\t", code)
                        ),
                        "packed_add_instructions": len(
                            re.findall(r"\taddp[sd]\t", code)
                        ),
                        "scalar_multiply_instructions": len(
                            re.findall(r"\tmuls[sd]\t", code)
                        ),
                        "fused_multiply_add_instructions": code.count("fmadd"),
                    }
                )
    (output / "updates_assembly.json").write_text(json.dumps(records, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline", default="d551db8")
    parser.add_argument("--shrinkage", type=Path)
    parser.add_argument(
        "--shrinkage-revision", default="87a735cdac7880cfe43a1219b19465587c1e9754"
    )
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--elements", type=int, default=2_000_000)
    parser.add_argument(
        "--shapes", default="0x7,1x7,7x7,8x7,9x7,33x7,5000x64,5000x1000,262144x16"
    )
    parser.add_argument("--checks-only", action="store_true")
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
    shrinkage = (args.shrinkage or repository.parent / "shrinkage").resolve()
    sources = [
        "Cargo.toml",
        "Cargo.lock",
        "rust-toolchain.toml",
        "rustfmt.toml",
        "tests/common/backend_aliases.rs",
        "benches/column_updates.rs",
        "benches/column_reductions/updates.py",
        "benches/column_reductions/updates_consumer.rs",
        "benches/column_reductions/updates-prototype.patch",
        "benches/column_reductions/updates-source-slice.patch",
        "benches/column_reductions/updates_consumer.lock",
        "benches/column_reductions/reversed.py",
        "benches/column_reductions/strided.py",
    ]
    source_hashes = {name: sha(repository / name) for name in sources}
    export(repository, args.baseline, output / "baseline-source")
    for name in ["Cargo.toml", "Cargo.lock", "benches/column_updates.rs"]:
        shutil.copy2(repository / name, output / "baseline-source" / name)
    for variant, patch in [
        ("candidate", "updates-prototype.patch"),
        ("source_slice", "updates-source-slice.patch"),
    ]:
        shutil.copytree(output / "baseline-source", output / f"{variant}-source")
        subprocess.run(
            [
                "patch",
                "--batch",
                "--forward",
                "-p1",
                "-i",
                str(repository / "benches/column_reductions" / patch),
            ],
            cwd=output / f"{variant}-source",
            check=True,
        )
    export(shrinkage, args.shrinkage_revision, output / "shrinkage-source")
    environment = {
        **os.environ,
        "CARGO_TARGET_DIR": os.environ.get("CARGO_TARGET_DIR", str(output / "build")),
        "CARGO_PROFILE_RELEASE_CODEGEN_UNITS": "16",
        "PROBE_SAMPLES": str(args.samples),
        "PROBE_ELEMENTS": str(args.elements),
        "PROBE_SHAPES": args.shapes,
    }
    binaries = {}
    build_root = Path(environment["CARGO_TARGET_DIR"])
    build_sources = {
        variant: {
            str(path.relative_to(output / f"{variant}-source")): sha(path)
            for path in (output / f"{variant}-source").rglob("*.rs")
        }
        for variant in VARIANTS
    }

    def build(variant, program, manifest, target):
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
                    *target,
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
            == ("column_updates" if program == "kernel" else "updates-consumer")
        )
        shutil.copy2(executable, output / label)
        if program == "kernel":
            assembly = sorted(executable.parent.glob(executable.name + "*.s"))
            assert assembly, "Missing emitted kernel assembly."
            with (output / f"{label}.s").open("wb") as destination:
                for part in assembly:
                    destination.write(part.read_bytes())
        binaries[label] = sha(output / label)

    for variant in VARIANTS:
        # Cargo can reuse a root package's library across different source paths.
        # Each variant needs its own fingerprints as well as its own executable.
        environment["CARGO_TARGET_DIR"] = str(build_root / variant)
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
                "column_updates",
            ],
        )
        workspace = output / f"{variant}-consumer-source"
        workspace.mkdir()
        shutil.copy2(
            repository / "benches/column_reductions/updates_consumer.rs",
            workspace / "main.rs",
        )
        (workspace / "Cargo.toml").write_text(f"""[package]
name = "updates-consumer"
version = "0.0.0"
edition = "2021"

[workspace]

[[bin]]
name = "updates-consumer"
path = "main.rs"

[dependencies]
shrinkage = {{ path = "../shrinkage-source", default-features = false, features = ["ndarray_v0_17", "faer_v0_24", "nalgebra_v0_34"] }}
ndarray = {{ version = "=0.17.2", default-features = false, features = ["std"] }}
faer = {{ version = "=0.24.4", default-features = false, features = ["std", "linalg"] }}
nalgebra = "=0.34.2"
rand = {{ version = "=0.10.3", default-features = false, features = ["std"] }}
rand_chacha = "=0.10.0"
serde_json = "1"

[patch.crates-io]
lazymatrix = {{ path = "../{variant}-source" }}
""")
        lock = repository / "benches/column_reductions/updates_consumer.lock"
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
            variant, "consumer", workspace / "Cargo.toml", ["--bin", "updates-consumer"]
        )
    assert (
        len(
            {
                sha(output / f"{variant}-consumer-source/Cargo.lock")
                for variant in VARIANTS
            }
        )
        == 1
    )
    shutil.copy2(
        output / "candidate-consumer-source/Cargo.lock",
        output / "updates_consumer.lock",
    )
    assert len(set(binaries.values())) == len(binaries)
    assembly_summary(output)
    records = []
    for block in range(args.blocks):
        for variant in ORDERS[block % len(ORDERS)]:
            label = f"{variant}-b{block}"
            print(f"Checking and measuring {label}", flush=True)
            env = {
                **environment,
                "PROBE_LABEL": label,
                "PROBE_ORDER_OFFSET": str(block % 2),
            }
            if args.checks_only:
                env["PROBE_CHECKS_ONLY"] = "1"
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
        for variant in VARIANTS
    ]
    combine(
        [path.with_suffix(".csv") for path in paths], output / "updates_timings.csv"
    )
    combine(
        [path.with_name(path.name + "-errors.csv") for path in paths],
        output / "updates_errors.csv",
    )
    if not args.checks_only:
        summary(
            output / "updates_timings.csv",
            output / "updates_summary.csv",
            ["backend", "layout", "destination", "scalar", "method", "rows", "cols"],
            "ns_per_pass",
        )
    with (output / "updates_fits.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(records[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)
    summary(
        output / "updates_fits.csv",
        output / "updates_fit_summary.csv",
        ["backend", "layout", "rows", "cols", "signal", "penalty"],
        "ns",
    )
    assert source_hashes == {name: sha(repository / name) for name in sources}, (
        "Source changed during comparison."
    )
    metadata = {
        "measured_at": datetime.now(UTC).isoformat(),
        "baseline_revision": command("git", "rev-parse", args.baseline),
        "source_sha256": source_hashes,
        "build_source_sha256": build_sources,
        "variants": {
            "baseline": "Indexed production loops",
            "candidate": "Optional mutable slices and a shared slice kernel",
            "source_slice": "Ndarray source-slice branch with indexed destination access",
        },
        "binary_sha256": binaries,
        "shrinkage_revision": args.shrinkage_revision,
        "consumer_lock_sha256": sha(output / "updates_consumer.lock"),
        "rustc": command("rustc", "-Vv"),
        "cargo": command("cargo", "-V"),
        "cpu": command("lscpu"),
        "pinned_cpu": args.cpu,
        "compute_window": "All three builds, controls, and complete fits in one exclusive compute bench window.",
        "profile": "release, 16 codegen units, default LTO",
        "rustflags": environment.get("RUSTFLAGS", ""),
        "features": ["faer", "nalgebra", "ndarray"],
        "scalars": ["f32", "f64"],
        "seed": 8421,
        "blocks": args.blocks,
        "samples_per_process": args.samples,
        "elements_per_sample": args.elements,
        "shapes": args.shapes,
        "order": "Six balanced permutations of variant order; kernel method order alternates by sample and block; consumer backend order alternates by block.",
        "kernel_checks": "Bitwise equality (NaNs by classification), untouched padding, zero allocations around each complete update pass.",
        "consumer": "Unchanged Shrinkage Hybrid with ClusteredLasso(penalty, 0), centered columns, fitted intercept, mapping tolerance 1e-8, 5000 x {64,1000}, signals {8,64}, zero-solution controls, and reversed ndarray controls.",
        "consumer_timing": "Complete fit; input construction, verification, serialization, and result destruction outside timing.",
        "consumer_checks": "Independent raw-input residuals, primal objective, L1 KKT, and feasible Gaussian dual gap for every fit.",
        "checks_only": args.checks_only,
        "artifacts": str(output),
        "cargo_targets": str(build_root / "{baseline,candidate,source_slice}"),
    }
    (output / "updates_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Results saved in {output}", flush=True)


if __name__ == "__main__":
    main()
