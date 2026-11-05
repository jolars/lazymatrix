# Contributing to LazyMatrix

Contributions are welcome, whether you want to report a bug, improve an example,
clarify the documentation, or change the code. You do not need to work on both
Rust and Python to contribute.

## Find something to work on

Check the [open issues](https://github.com/jolars/lazymatrix/issues) for
existing discussions. [TODO.md](TODO.md) describes planned design work. For a
substantial feature or API change, open an issue first so we can discuss the
approach before you spend time implementing it. Small fixes can go straight to a
pull request.

When reporting a bug, include a small example that reproduces it, the result you
expected, and what happened instead. Include your LazyMatrix version, operating
system, and relevant Rust or Python and matrix library versions. For a
performance issue, include the matrix dimensions, storage format, and how you
measured it.

## Set up a checkout

Fork the [repository](https://github.com/jolars/lazymatrix), clone your fork,
and create a branch for your change. Run the commands below from the repository
root.

The repository provides a development environment through Nix and devenv. With
those installed, enter it with:

```sh
devenv shell
```

The environment supplies the Rust toolchain and the tools for Python and website
development. It also configures native dependencies on NixOS. The first build or
check may take a while to download dependencies and compile them.

You can also use your own tools. Rust and Python extension work require the
toolchain in [rust-toolchain.toml](rust-toolchain.toml), including rustfmt and
Clippy. The `task` commands use [go-task](https://taskfile.dev). Python work
also needs Python 3.11 or newer and uv. Website work uses Node.js 24 and pnpm
10, plus the API extraction tools described in [docs/README.md](docs/README.md).
You only need the tools for the part of the project you are changing.

## Find your way around

- `src/` contains the Rust library, public traits, and matrix backends.
- `tests/` contains Rust integration tests. `tests/common/runner.rs` holds
  checks shared by the backends.
- `examples/` contains algorithms that use the library, and `benches/` contains
  benchmarks and their results.
- `python/src/` contains the Python package, `python/rust/` contains its native
  extension, and `python/tests/` contains Python tests.
- `docs/` contains the website, guides, and API reference integration.

LazyMatrix is still in early development. A consistent API takes priority over
backward compatibility before 1.0.0. Keep lazy operations from materializing the
full normalized matrix, preserve sparse storage, and keep the core library's
dependencies small. Solver algorithms belong in examples or downstream projects.
See [AGENTS.md](AGENTS.md) for detailed numerical and backend conventions when
working on those parts of the library.

## Check your changes

Choose checks that cover the behavior you changed. Run `task --list` to see the
available commands, or read [Taskfile.yml](Taskfile.yml) for their definitions.

### Rust

For a change to one backend, run its version-specific suite. For example:

```sh
bash scripts/test-backends.sh version ndarray_v0_17
task fmt
task lint
task docs
```

The backend suite checks both serial and parallel operation. No backend is
enabled by default, so a plain `cargo test` does not cover backend behavior. Run
`task test` for changes to shared behavior or backend features, and
`task test:blas` for BLAS integration changes. BLAS tests require an LP64
OpenBLAS installation, which the development environment provides.

Add regression tests for bug fixes. Put behavior shared by matrix backends in
`tests/common/runner.rs`, and use backend-specific tests for storage details.
Document public APIs and update examples or guides when their behavior changes.
For performance changes, include measurements and describe how to reproduce
them.

### Python

```sh
task python:check
```

This builds an editable native extension and runs formatting, linting, and
tests, including the optional PyTorch adapter. The Python environment lives in
`.devenv/python`. The tasks use `scripts/python-tool.py` to select compatible
native tooling on NixOS.

Run `task python:torch:check` for focused PyTorch adapter checks. CUDA tests run
when hardware is available and report skips otherwise.

For changes to packaging or installation, also run:

```sh
task python:package
```

This builds a wheel and source distribution, tests the installed wheel, and
rebuilds and tests the source distribution in isolated environments. It checks
installations with and without the PyTorch extra, using CPU PyTorch wheels.

### Documentation and website

For changes to Markdown files outside the website, check formatting and make
sure paths, links, and example commands are correct. Rust tests are unnecessary
for prose-only changes.

For website changes, preview and check the site with:

```sh
task website:dev
```

Stop the preview server, then run `task website:check`. These tasks prepare the
Python extension and website dependencies automatically. See
[docs/README.md](docs/README.md) for API extraction and benchmark workflows.
Generated reference pages and website build output should stay out of commits.

### Changes across the project

Run `task ci` for broad changes affecting multiple components, dependencies,
packaging, or releases. It runs the local equivalent of the CI checks, including
Python distribution tests and the website build.

## Open a pull request

Keep each pull request focused on one change. Explain the problem it solves,
describe any API or performance impact, link related issues, and list the checks
you ran. Mention checks you could not run so reviewers know what remains to be
verified. Draft pull requests are welcome if you want feedback while working.

Use short Conventional Commit subjects, such as `fix: handle empty columns` or
`docs: clarify fitted parameters`. Include `Fixes #123` for a bug fix or
`Closes #123` for a completed feature when applicable.

Rust and Python have independent versions. Versionary manages version bumps and
changelogs through a shared release pull request. You do not need to prepare a
release as part of a normal contribution.
