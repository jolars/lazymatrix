"""Run a tool in the bindings' environment, preferring native system tools."""

import os
import shutil
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    environment = root / ".devenv" / "python"
    binaries = environment / ("Scripts" if os.name == "nt" else "bin")
    name, *arguments = sys.argv[1:]
    suffix = ".exe" if os.name == "nt" else ""
    if name == "python":
        executable = str(binaries / f"python{suffix}")
    else:
        # NixOS needs native executables instead of generic Linux tool wheels.
        executable = shutil.which(name) or str(binaries / f"{name}{suffix}")
    env = os.environ | {
        "VIRTUAL_ENV": str(environment),
        "PATH": str(binaries) + os.pathsep + os.environ.get("PATH", ""),
        "LAZYMATRIX_MATURIN": shutil.which("maturin")
        or str(binaries / f"maturin{suffix}"),
    }
    os.execvpe(executable, [executable, *arguments], env)


if __name__ == "__main__":
    main()
