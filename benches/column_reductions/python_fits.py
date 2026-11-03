"""Measure paired Python Shrinkage fits, including owned input copies."""

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from downstream import verify


def worker(args):
    import numpy as np

    library = args.output / f"{args.variant}-python-build" / "release/lib_shrinkage.so"
    spec = importlib.util.spec_from_file_location("shrinkage._shrinkage", library)
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    sys.path.insert(0, str(args.shrinkage / "packages/python/src"))
    from shrinkage import Lasso, StoppingCriterion

    data = {}
    for name in ["original", "larger", "correlated"]:
        directory = args.output / "data" / name
        meta = json.loads((directory / "metadata.json").read_text())
        data[name] = (
            np.fromfile(directory / "x.bin", dtype="<f8").reshape(
                (meta["n"], meta["p"]), order="F"
            ),
            np.fromfile(directory / "y.bin", dtype="<f8"),
            meta["alpha"],
        )
    stopping = StoppingCriterion.duality_gap(1e-8)
    records = []
    for case, source, multiplier in [
        ("original", "original", 1.0),
        ("larger", "larger", 1.0),
        ("correlated", "correlated", 1.0),
        ("weak", "original", 0.1),
        ("strong", "original", 5.0),
        ("zero", "original", 20.0),
    ]:
        x, y, alpha = data[source]
        alpha *= multiplier
        for sample in range(-1, args.repetitions):
            start = time.perf_counter_ns()
            fit = Lasso(
                alpha,
                fit_intercept=False,
                normalization="none",
                stopping=stopping,
                max_iterations=10_000,
            ).fit(x, y)
            elapsed = time.perf_counter_ns() - start
            row = {
                "build": args.variant,
                "case": case,
                "sample": sample,
                "ns": elapsed,
                "alpha": alpha,
                "iterations": fit.iterations,
                "termination": fit.termination,
                "objective": fit.objective,
                "kkt": fit.kkt_violation,
                "duality_gap": fit.duality_gap,
                "dual_objective": fit.dual_objective,
                "reference_loss": fit.reference_loss,
                "coefficients": fit.coefficients,
                "dual_certificate": fit.dual_certificate,
            }
            checked = verify(row, {name: values[:2] for name, values in data.items()})
            if sample >= 0:
                records.append(checked)
            del fit
    (args.output / f"python-{args.variant}-{args.block}.json").write_text(
        json.dumps(records) + "\n"
    )


def run(args):
    records = []
    environment = os.environ | {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    for block in range(args.blocks):
        variants = ["current", "candidate"]
        if block % 2:
            variants.reverse()
        for variant in variants:
            subprocess.run(
                [
                    "taskset",
                    "-c",
                    str(args.cpu),
                    sys.executable,
                    str(Path(__file__).resolve()),
                    str(args.output),
                    "--worker",
                    "--variant",
                    variant,
                    "--block",
                    str(block),
                    "--repetitions",
                    str(args.repetitions),
                    "--shrinkage",
                    str(args.shrinkage),
                ],
                env=environment,
                check=True,
            )
            path = args.output / f"python-{variant}-{block}.json"
            records.extend(
                {"block": block} | row for row in json.loads(path.read_text())
            )
    with (args.output / "python-fits.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--shrinkage",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "shrinkage",
    )
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--variant", choices=["current", "candidate"])
    parser.add_argument("--block", type=int)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.shrinkage = args.shrinkage.resolve()
    if args.worker:
        worker(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
