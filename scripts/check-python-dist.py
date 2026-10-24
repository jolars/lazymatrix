"""Check distributions by installing wheels and rebuilding an isolated sdist."""

import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from pathlib import Path


def run(*command: str, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    project = root / "python"
    metadata = tomllib.loads((project / "pyproject.toml").read_text())
    version = metadata["project"]["version"]
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
        environment = directory / "venv"
        run("uv", "venv", "--python", sys.executable, str(environment))
        python = environment / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        tests = directory / "tests"
        shutil.copytree(
            project / "tests", tests, ignore=shutil.ignore_patterns("__pycache__")
        )
        run("uv", "pip", "install", "--python", str(python), str(wheels[0]), "pytest")
        run(str(python), "-m", "pytest", "-q", str(tests), cwd=directory)
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
        run(
            "uv", "pip", "install", "--python", str(python), "--reinstall", str(rebuilt)
        )
        run(str(python), "-m", "pytest", "-q", str(tests), cwd=directory)


if __name__ == "__main__":
    main()
