# Contributing to the Python package

Work from the repository root and enter the pinned development environment:

```sh
devenv shell
```

## Checks

```sh
task python:check
task python:torch:check
task python:package
```

`task python:check` builds an editable extension and runs formatting, linting,
and correctness checks, including the optional PyTorch adapter.
`task python:torch:check` runs just the adapter checks. CUDA tests run when
hardware is available and report skips otherwise. The development environment
lives in `.devenv/python`.

`task python:package` builds a wheel and source distribution, installs the wheel
in an isolated environment, rebuilds from the source distribution, and tests
both installations with and without the PyTorch extra. Distribution checks use
CPU PyTorch wheels.

## Releases

Python and Rust use independent versions and share one Versionary release PR.
Rust releases also release Python bindings. Python-only releases leave the Rust
version unchanged. Python tags use `python-v<version>`, and Rust tags use
`v<version>`.

PyPI trusted publishing must name `jolars/lazymatrix`, `publish-python.yml`, and
the `release` environment. The Versionary release token must retain permission
to create tags that trigger publishing workflows.
