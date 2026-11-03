"""Prepare isolated sources and verify paired native Shrinkage fits."""

import argparse
import csv
import io
import json
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def copy_source(source, destination):
    destination.mkdir()
    shutil.copytree(source / "src", destination / "src")
    manifest = (source / "Cargo.toml").read_text()
    # Consumer builds do not need the source repositories' examples or benches.
    manifest = re.sub(
        r"\n\[\[(?:example|bench)\]\]\n.*?(?=\n\[|\Z)", "", manifest, flags=re.DOTALL
    )
    (destination / "Cargo.toml").write_text(manifest)


def apply_patch(destination, name):
    with (HERE / name).open() as patch:
        subprocess.run(
            ["patch", "--batch", "--forward", "-p1", "-d", str(destination)],
            stdin=patch,
            check=True,
        )


def prepare(output, shrinkage, data, baseline=None, python=False):
    import numpy as np

    output.mkdir(parents=True)
    if baseline is None:
        revision = json.loads((HERE / "metadata.json").read_text())[
            "lazymatrix_revision"
        ]
        archive = subprocess.check_output(
            ["git", "-C", str(ROOT), "archive", revision, "src", "Cargo.toml"]
        )
        baseline = output / "baseline-checkout"
        baseline.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            source.extractall(baseline, filter="data")
    copy_source(baseline, output / "baseline-lazymatrix")
    copy_source(ROOT, output / "candidate-lazymatrix")
    copy_source(shrinkage, output / "shrinkage")
    apply_patch(output / "shrinkage", "shrinkage-compat.patch")
    for name, library in [
        ("current", output / "baseline-lazymatrix"),
        ("candidate", output / "candidate-lazymatrix"),
    ]:
        workspace = output / name
        workspace.mkdir()
        (workspace / "Cargo.toml").write_text(
            f"""[package]
name = "column-downstream-{name}"
version = "0.0.0"
edition = "2021"
rust-version = "1.85"

[workspace]

[[bin]]
name = "column-downstream"
path = {json.dumps(str(HERE / "downstream.rs"))}

[dependencies]
shrinkage = {{ path = "../shrinkage", default-features = false, features = ["ndarray_v0_17"] }}
ndarray = {{ version = "=0.17.2", default-features = false, features = ["std"] }}
serde_json = "1"

[patch.crates-io]
lazymatrix = {{ path = {json.dumps(str(library))} }}
"""
        )
        shutil.copyfile(HERE / f"downstream-{name}.lock", workspace / "Cargo.lock")
        if python:
            wrapper = output / f"{name}-python"
            wrapper.mkdir()
            shutil.copytree(shrinkage / "packages/python/rust", wrapper / "rust")
            manifest = (shrinkage / "packages/python/Cargo.toml").read_text()
            manifest = manifest.replace('path = "../.."', 'path = "../shrinkage"')
            manifest += f"\n[patch.crates-io]\nlazymatrix = {{ path = {json.dumps(str(library))} }}\n"
            (wrapper / "Cargo.toml").write_text(manifest)
            lockfile = HERE / "downstream-python.lock"
            if lockfile.exists():
                shutil.copyfile(lockfile, wrapper / "Cargo.lock")
    for name in ["original", "larger", "correlated"]:
        destination = output / "data" / name
        destination.mkdir(parents=True)
        with np.load(data / name / "data.npz", allow_pickle=False) as archive:
            x, y, alpha = archive["x"], archive["y"], float(archive["alpha"])
        x.astype("<f8").ravel(order="F").tofile(destination / "x.bin")
        y.astype("<f8").tofile(destination / "y.bin")
        (destination / "metadata.json").write_text(
            json.dumps({"n": x.shape[0], "p": x.shape[1], "alpha": alpha}) + "\n"
        )


def verify(row, data):
    import numpy as np

    name = row["case"] if row["case"] in {"larger", "correlated"} else "original"
    x, y = data[name]
    beta = np.asarray(row["coefficients"])
    dual = np.asarray(row["dual_certificate"])
    alpha, n = row["alpha"], len(y)
    assert np.all(np.isfinite(beta)) and np.all(np.isfinite(dual))
    residual = y - x @ beta
    gradient = x.T @ residual / n
    objective = residual @ residual / (2 * n) + alpha * np.abs(beta).sum()
    violation = np.where(
        beta != 0,
        np.abs(gradient - alpha * np.sign(beta)),
        np.maximum(np.abs(gradient) - alpha, 0),
    )
    kkt = float(np.max(violation))
    feasibility = max(0.0, float(np.max(np.abs(x.T @ dual))) - alpha)
    dual_objective = y @ dual - n * (dual @ dual) / 2
    reference_loss = y @ y / (2 * n)
    gap = max(0.0, float(objective - dual_objective))
    relative_gap = gap / reference_loss
    assert relative_gap <= 1e-8 + 1e-13, row["case"]
    assert feasibility <= 2e-12 * (1 + alpha), row["case"]
    for actual, reported in [
        (objective, row["objective"]),
        (kkt, row["kkt"]),
        (dual_objective, row["dual_objective"]),
        (gap, row["duality_gap"]),
        (reference_loss, row["reference_loss"]),
    ]:
        assert abs(actual - reported) <= 2e-11 * (1 + abs(actual)), row["case"]
    return {
        key: row[key]
        for key in [
            "build",
            "case",
            "sample",
            "ns",
            "iterations",
            "termination",
            "feature_visits",
            "full_checks",
        ]
        if key in row
    } | {
        "objective": float(objective),
        "kkt": kkt,
        "relative_gap": float(relative_gap),
        "feasibility": feasibility,
    }


def run(output, cpu, blocks, repetitions):
    import numpy as np

    data = {}
    for name in ["original", "larger", "correlated"]:
        directory = output / "data" / name
        metadata = json.loads((directory / "metadata.json").read_text())
        data[name] = (
            np.fromfile(directory / "x.bin", dtype="<f8").reshape(
                (metadata["n"], metadata["p"]), order="F"
            ),
            np.fromfile(directory / "y.bin", dtype="<f8"),
        )
    records = []
    for block in range(blocks):
        variants = ["current", "candidate"]
        if block % 2:
            variants.reverse()
        for variant in variants:
            executable = output / f"{variant}-build" / "release" / "column-downstream"
            raw = output / f"{variant}-{block}.jsonl"
            with raw.open("w") as stream:
                subprocess.run(
                    [
                        "taskset",
                        "-c",
                        str(cpu),
                        str(executable),
                        str(output / "data"),
                        variant,
                        str(repetitions),
                    ],
                    stdout=stream,
                    check=True,
                )
            for line in raw.read_text().splitlines():
                record = verify(json.loads(line), data)
                records.append({"block": block} | record)
    with (output / "fits.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("output", type=Path)
    parser.add_argument("--shrinkage", type=Path, default=ROOT.parent / "shrinkage")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--python", action="store_true")
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.data is None:
            parser.error("prepare requires --data")
        prepare(
            args.output.resolve(),
            args.shrinkage.resolve(),
            args.data.resolve(),
            args.baseline.resolve() if args.baseline else None,
            args.python,
        )
    else:
        run(args.output.resolve(), args.cpu, args.blocks, args.repetitions)


if __name__ == "__main__":
    main()
