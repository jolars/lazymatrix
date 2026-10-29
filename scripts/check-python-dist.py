"""Check distributions by installing wheels and rebuilding an isolated sdist."""

import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from pathlib import Path


def run(
    *command: str, cwd: Path | None = None, env: dict[str, str] | None = None
) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def check_wheel(wheel: Path, directory: Path, tests: Path, torch_version: str) -> None:
    environment = directory / f"{wheel.parent.name}-venv"
    run("uv", "venv", "--python", sys.executable, str(environment))
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    run("uv", "pip", "install", "--python", str(python), str(wheel), "pytest")
    run(
        str(python),
        "-c",
        "import importlib.util; import lazymatrix; "
        "assert importlib.util.find_spec('torch') is None",
        cwd=directory,
    )
    run(
        str(python),
        "-m",
        "pytest",
        "-q",
        str(tests),
        cwd=directory,
        env=os.environ | {"LAZYMATRIX_REQUIRE_TORCH": "0"},
    )
    run(
        "uv",
        "pip",
        "install",
        "--python",
        str(python),
        "--torch-backend=cpu",
        f"{wheel}[torch]",
        f"torch=={torch_version}",
    )
    run(
        str(python),
        "-m",
        "pytest",
        "-q",
        str(tests),
        cwd=directory,
        env=os.environ | {"LAZYMATRIX_REQUIRE_TORCH": "1"},
    )


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    project = root / "python"
    metadata = tomllib.loads((project / "pyproject.toml").read_text())
    version = metadata["project"]["version"]
    lock = tomllib.loads((project / "uv.lock").read_text())
    torch_version = next(
        item["version"] for item in lock["package"] if item["name"] == "torch"
    )
    distributions = project / "dist"
    wheels = sorted(distributions.glob(f"lazymatrix-{version}-*.whl"))
    source = distributions / f"lazymatrix-{version}.tar.gz"
    if not wheels or not source.is_file():
        raise RuntimeError("Build a wheel and sdist before checking distributions")
    run(sys.executable, "-m", "twine", "check", str(source), *map(str, wheels))
    maturin = os.environ.get("LAZYMATRIX_MATURIN") or shutil.which("maturin")
    if not maturin:
        raise RuntimeError("maturin must be installed")
    with tempfile.TemporaryDirectory(prefix="lazymatrix-dist-") as temporary:
        directory = Path(temporary)
        tests = directory / "tests"
        shutil.copytree(
            project / "tests", tests, ignore=shutil.ignore_patterns("__pycache__")
        )
        check_wheel(wheels[0], directory, tests, torch_version)
        unpacked = directory / "source"
        with tarfile.open(source) as archive:
            archive.extractall(unpacked, filter="data")
        candidates = [
            path.parent
            for path in unpacked.rglob("pyproject.toml")
            if tomllib.loads(path.read_text()).get("project", {}).get("name")
            == "lazymatrix"
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                "Expected one self-contained Python project in the sdist"
            )
        output = directory / "rebuilt"
        # The sdist must resolve its bundled Rust source with the lockfile intact.
        run(
            maturin,
            "build",
            "--release",
            "--locked",
            "--out",
            str(output),
            cwd=candidates[0],
        )
        (rebuilt,) = output.glob("*.whl")
        check_wheel(rebuilt, directory, tests, torch_version)


if __name__ == "__main__":
    main()
