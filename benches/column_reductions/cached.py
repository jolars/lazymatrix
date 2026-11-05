"""Diagnose cached dots in one compute bench window, including both builds."""

import argparse
import csv
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from strided import combine, command, sha


def summarize(source, destination, keys, field):
    groups = defaultdict(list)
    with source.open(newline="") as handle:
        for row in csv.DictReader(handle):
            row["build"] = row["label"].split("-b")[0]
            groups[tuple(row[key] for key in keys)].append(float(row[field]))
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


def assembly(output, build):
    listing = command("objdump", "-Cd", "--no-show-raw-insn", str(output / build))
    (output / f"{build}-disassembly.txt").write_text(listing + "\n")
    functions = re.split(r"(?=^[0-9a-f]+ <[^\n]+>:\n)", listing, flags=re.MULTILINE)
    selected = []
    records = []
    for function in functions:
        name = function.splitlines()[0]
        if not (
            re.search(
                r"cached_column_dots::(dispatch|accumulated|slice_f|acc16_f|acc32_f|avx8_f|avx32_f|fma32_f|native_f)",
                name,
            )
            or "matmul::dot::" in name
            or ("pulp::" in name and "WithSimd" in name)
            or ("LazyColumn<" in name and "dot_with_sum" in name)
        ):
            continue
        selected.append(function)
        records.append(
            {
                "build": build,
                "symbol": name,
                "lines": len(function.splitlines()),
                "xmm_operands": len(re.findall(r"%xmm\d+", function)),
                "ymm_operands": len(re.findall(r"%ymm\d+", function)),
                "packed_multiplies": len(re.findall(r"\b(?:v?mulp[sd])\b", function)),
                "packed_adds": len(re.findall(r"\b(?:v?addp[sd])\b", function)),
                "fma_instructions": len(re.findall(r"\bvfmadd\w+\b", function)),
            }
        )
    assert selected, "Missing probe functions in final binary."
    return "\n".join(selected), records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--elements", type=int, default=8_000_000)
    parser.add_argument(
        "--shapes",
        default="0x1,1x1,7x1,8x1,9x1,33x1,256x1,5000x1,5000x64,5000x1000,65536x1,1048576x1",
    )
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args()
    if (
        min(args.blocks, args.samples, args.elements) < 1
        or args.blocks * args.samples < 2
    ):
        parser.error("Use positive counts and at least two observations.")
    repository = Path(__file__).resolve().parents[2]
    os.chdir(repository)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = [
        "Cargo.toml",
        "Cargo.lock",
        "rust-toolchain.toml",
        "rustfmt.toml",
        "src/reductions.rs",
        "src/column.rs",
        "src/traits/columns.rs",
        "src/traits/vectors.rs",
        "src/backends/faer/dense.rs",
        "tests/common/backend_aliases.rs",
        "benches/cached_column_dots.rs",
        "benches/column_reductions/cached.py",
        "benches/column_reductions/strided.py",
    ]
    hashes = {name: sha(repository / name) for name in sources}
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_RUSTFLAGS"}
        and not key.startswith("CARGO_PROFILE_RELEASE_")
        and not key.startswith("PROBE_")
    }
    environment.update(
        {
            "CARGO_PROFILE_RELEASE_CODEGEN_UNITS": "16",
            "PROBE_SAMPLES": str(args.samples),
            "PROBE_ELEMENTS": str(args.elements),
            "PROBE_SHAPES": args.shapes,
        }
    )
    binaries = {}
    dispatch = {}
    assembly_records = []
    excerpts = []
    for build in ["generic", "native"]:
        build_env = {
            **environment,
            "RUSTFLAGS": f"-C target-cpu={'native' if build == 'native' else 'x86-64'}",
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
                    "--features",
                    "faer",
                    "--bench",
                    "cached_column_dots",
                    "--message-format=json",
                    "--",
                    "--emit=asm",
                ],
                env=build_env,
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
            if item.get("executable") and item["target"]["name"] == "cached_column_dots"
        )
        shutil.copy2(executable, output / build)
        files = sorted(executable.parent.glob(executable.name + "*.s"))
        assert files, "Missing emitted assembly."
        with (output / f"{build}.s").open("wb") as handle:
            for path in files:
                handle.write(path.read_bytes())
        binaries[build] = sha(output / build)
        dispatch[build] = subprocess.check_output(
            ["taskset", "-c", str(args.cpu), str(output / build)],
            env={**environment, "PROBE_ARCH": "1"},
            text=True,
        ).strip()
        excerpt, records = assembly(output, build)
        excerpts.append(f"# {build}\n{excerpt}")
        assembly_records.extend(records)
        with (output / f"{build}-numerics.csv").open("w") as stdout:
            subprocess.run(
                ["taskset", "-c", str(args.cpu), str(output / build)],
                env={**environment, "PROBE_NUMERICS": "1"},
                stdout=stdout,
                check=True,
            )
    assert len(set(binaries.values())) == 2
    (output / "cached_assembly.txt").write_text("\n".join(excerpts))
    (output / "cached_assembly.json").write_text(
        json.dumps(assembly_records, indent=2) + "\n"
    )

    def run(build, label, consumer=False, **overrides):
        print(
            f"Checking and measuring {label}{' fits' if consumer else ''}", flush=True
        )
        suffix = "-fits" if consumer else ""
        with (
            (output / f"{label}{suffix}.csv").open("w") as stdout,
            (output / f"{label}{suffix}-errors.csv").open("w") as stderr,
        ):
            subprocess.run(
                ["taskset", "-c", str(args.cpu), str(output / build)],
                env={
                    **environment,
                    "PROBE_LABEL": label,
                    **({"PROBE_CONSUMER": "1"} if consumer else {}),
                    **overrides,
                },
                stdout=stdout,
                stderr=stderr,
                check=True,
            )

    for build in binaries:
        run(
            build,
            f"{build}-tails",
            PROBE_SAMPLES="0",
            PROBE_SHAPES=",".join(f"{n}x3" for n in range(66)),
        )
    if not args.checks_only:
        for block in range(args.blocks):
            for build in (
                ["generic", "native"] if block % 2 == 0 else ["native", "generic"]
            ):
                label = f"{build}-b{block}"
                run(build, label, PROBE_ORDER=str(block))
                run(build, label, consumer=True, PROBE_ORDER=str(block))
        paths = [
            output / f"{build}-b{block}"
            for block in range(args.blocks)
            for build in binaries
        ]
        combine([p.with_suffix(".csv") for p in paths], output / "cached_timings.csv")
        combine(
            [p.with_name(p.name + "-fits.csv") for p in paths],
            output / "cached_fits.csv",
        )
        summarize(
            output / "cached_timings.csv",
            output / "cached_summary.csv",
            [
                "build",
                "scalar",
                "regime",
                "rows",
                "cols",
                "offset",
                "working_set_bytes",
                "method",
            ],
            "ns_per_dot",
        )
        summarize(
            output / "cached_fits.csv",
            output / "cached_fit_summary.csv",
            ["build", "rows", "cols", "signal", "penalty", "method"],
            "ns",
        )
    combine(
        [
            output / f"{build}-{suffix}-errors.csv"
            for build in binaries
            for suffix in [
                "tails",
                *([] if args.checks_only else [f"b{b}" for b in range(args.blocks)]),
            ]
        ],
        output / "cached_errors.csv",
    )
    assert hashes == {name: sha(repository / name) for name in sources}, (
        "Source changed during measurement."
    )
    metadata = {
        "measured_at": datetime.now(UTC).isoformat(),
        "revision": command("git", "rev-parse", "HEAD"),
        "source_sha256": hashes,
        "binary_sha256": binaries,
        "faer_cpu_dispatch": dispatch,
        "assembly_sha256": {build: sha(output / f"{build}.s") for build in binaries},
        "disassembly_sha256": {
            build: sha(output / f"{build}-disassembly.txt") for build in binaries
        },
        "rustc": command("rustc", "-Vv"),
        "cargo": command("cargo", "-V"),
        "platform": platform.platform(),
        "cpu": command("lscpu"),
        "cpu_caches": command("lscpu", "-C"),
        "pinned_cpu": args.cpu,
        "compute_window": "One exclusive compute bench window for both builds, every method, and all fits.",
        "profile": "release, opt-level=3, 16 codegen units, Cargo default LTO",
        "builds": {"generic": "-C target-cpu=x86-64", "native": "-C target-cpu=native"},
        "features": ["faer"],
        "faer_version": "0.24.4",
        "pulp_version": "0.22.3",
        "seed": 8421,
        "scalars": ["f32", "f64"],
        "shapes": args.shapes,
        "blocks": args.blocks,
        "samples_per_process": args.samples,
        "elements_per_sample": args.elements,
        "passes": "clamp(elements / max(rows*cols, 1), 1, 100000)",
        "working_set": "Padded faer matrix, one reused vector, and centers and scales. Includes allocated leading dimension and outer padding.",
        "normalization": "Center 0.375, scale -1.75; every method applies the same cached correction.",
        "order": "Build order alternates by block; method order rotates by block and sample, including fits. Eight blocks put each method at each position twice.",
        "verification": "Compensated f64 directly centered oracle; tolerance 64*epsilon*max(sqrt(rows),1)*(1+abs(oracle)). Slice8 and Avx8 must also match production bits, treating NaNs by classification.",
        "allocation_check": "Every verified kernel and timed pass must make zero allocation or reallocation calls.",
        "numerics": "All lengths 0 through 65, cancellation, intermediate overflow, offsets, FMA rounding, nonfinite values and parameters, and signed zeros.",
        "consumer": "Complete f64 cyclic Lasso coordinate descent, adapted from examples/coordinate_descent.rs to dense faer. Fits include mean centering, summaries, residual state, coordinate updates, and per-sweep KKT checks. No screening or working sets. Only cached dots vary.",
        "consumer_cases": "5000x64 and 5000x1000, (8 signals, penalty .01), (64 signals, penalty .005), and (8 signals, penalty 2).",
        "consumer_verification": "Outside timing, reconstruct residuals from raw X and fitted coefficients/intercept. Require KKT <= 2e-9, coefficient difference <= 1e-10, objective difference <= 1e-12, and the same sweep count as Public8.",
        "warmup": "Check every column and method before timings, plus an extra pass immediately before each sample. Each timed fit has an untimed warmup.",
        "checks_only": args.checks_only,
        "artifacts": str(output),
    }
    (output / "cached_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Results saved in {output}", flush=True)


if __name__ == "__main__":
    main()
