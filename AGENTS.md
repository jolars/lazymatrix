# Repository Guidelines

Keep this file focused on durable instructions. Record implementation history in
commits and put detailed background in focused documentation.

## Project Scope and Layout

LazyMatrix is in early development. Prefer a robust, consistent API over backward
compatibility before 1.0.0.

- `src/` contains the Rust core, public traits, and backend implementations.
- `tests/` contains integration suites and isolated consumer workspaces.
- `examples/` contains consumer algorithms; `benches/` contains benchmarks.
- `python/` contains the independently versioned Python package and private
  native extension. See [python/README.md](python/README.md).
- `docs/` contains the website and API reference integration. See
  [docs/README.md](docs/README.md) for development and benchmark workflows.
- [TODO.md](TODO.md) tracks planned design work.

Keep backend-independent logic outside backend implementation directories. The
core must remain dependency-light, with only `num-traits` when no features are
enabled. Keep Python dependencies and FFI in the companion package.

Keep solver algorithms and state in examples or downstream crates. Add reusable
matrix capabilities when consumers need them; keep residual policies, cached
solver state, and update rules outside the library and its correctness suites.

## Normalization and Numerical Invariants

`LazyMatrix<M, P>` stores raw data and typed fitted parameters. Column
normalization represents the design matrix as a linear operator:

```text
X_tilde = (X - 1c^T) S^-1
X_tilde v = X(S^-1 v) - 1(c^T S^-1 v)
X_tilde^T u = S^-1(X^T u - c sum(u))
```

Row normalization uses `D^-1(X - r1^T)` and supports fitting and vector products
on dense, sparse CSC/CSR, and chunked Zarr storage. `RowStats` shares row fitting
hooks, and `RowNormalizationKernel` supplies products. CSR fitting buffers one
coalesced row; CSC fitting streams columns, using repeated scans for exact quantiles.
Ordered recipes compose with `.then()`. Fitted `NormalizationChainParams`
preserve stage order and retain row recipes for
`.for_new_samples()`, which refits row stages while reusing training column
parameters. `.apply_frozen()` reuses every stage unchanged. Fitting and products
for mixed chains require dense storage; column-only chains also support sparse
and chunked storage. Borrowed logical rows require only `SparseRows`; columns
require `RawColumns` or `SparseColumns`, including for explicit mixed chains.
Eager conversion supports every normalization type. Row and chain conversion
requires `MatrixTraversal`, including explicit mixed chains on sparse or chunked
storage, and uses one scan plus traversal buffers. In-place row and chain
conversion requires writable dense indexed storage.
Select the axis through `ColumnNormalizationParams`
or `RowNormalizationParams`, and inspect centers and scales through
`normalization()`. Both `LazyMatrix` and `EagerMatrix` return `(data, params)`
from `into_parts()`. The parameter types return
`(dimension, centers, scales)` from their own `into_parts()`, preserving
inactive identity dimensions.
`LazyMatrix::from_parts(data, params)` rebuilds the lazy operator. Supply
explicit vectors with `from_column_parts` or `from_row_parts`; infer the
dimension from the matrix.

Centering and scaling are independently optional. Lazy operations must preserve
sparse storage and never materialize the full normalized design matrix. Explicit
eager conversion may materialize dense storage. Eager operators must not apply
their retained fitted parameters again. Materialization must use bounded
workspace rather than a second full design matrix.

- Store owned centers and scales as `Vec<F>`; parameter views borrow slices.
  Application consumes owned parameters or explicit `.view()` values without
  fitting or copying. Row application uses the explicit `.apply_frozen()` name.
- Replace only exact computed zero scales with one. Explicit construction must
  panic on positive and negative zero scales, while preserving negative scales and
  IEEE nonfinite values.
- Compute scales from centered columns when centering is enabled. Include
  implicit zeros in sparse statistics and centered norms.
- Sum duplicate raw entries at each cell before applying normalization or
  computing nonlinear statistics.
- Center values before accumulating weighted Gram products. Do not correct raw
  moments afterward; large offsets can erase centered variation. Use bounded
  workspace and preserve contributions from implicit zeros.

## Errors, Borrowing, and Storage

Products, column statistics, and built-in normalization fitting return `Result`,
sharing the backend error through `MatrixErrorType`; in-memory backends use
`Infallible`. Custom `NormalizationRecipe` implementations select their fitting
error type and return supported fitted parameters. Parameter types and operator
kernels remain library controlled. Dimension mismatches panic. Reusable outputs
may be partial after an error; apply normalization corrections only after the
backend succeeds.
Forward `normalization_stats` through references and wrappers so storage backends
can share scans.

Keep operators independent of storage orientation. Expose column and row
borrowing through capabilities:

- `RawColumns` requires natural borrowed column access.
- `SparseColumns` requires contiguous CSC slices; `SparseRows` requires
  contiguous CSR slices. Their indices are `usize`.
- Borrowing must not hide gathering, allocation, decoding, or I/O. Unsupported
  borrowing should fail at compile time.

`MatrixTraversal` supplies operational rectangle scans in source coordinates.
Coalesce sparse cells before visiting them in the declared native order, retain
explicit zeros, and restrict storage reads to selected outer ranges or chunks.
`Submatrix<'a, O>` borrows a logical parent and ranges; nested selections flatten
without refitting. `RectangularProducts` applies the parent's saved
transformation, including row parameters fitted outside the selection. Keep
column-only advanced capabilities separate from general vector products.

Keep logical column operations on borrowed views and document their complexity.
Row and composed column views apply stages in order after coalescing raw cells,
including transformed implicit zeros. Borrowing requires no fitting or product
capabilities; scalar center and scale access belongs to column-normalized views.
Sparse row views must preserve storage order and duplicates, have logical length
`ncols`, and expose the column-dependent background without allocating a dense
row. Respect backend slice ranges so spare capacity and sliced-view offsets do
not become logical entries.

CSR statistics must scan rows with O(ncols) workspace, combine duplicates, and
include implicit zeros. Do not convert CSR to CSC for products or statistics.
Chunked storage must read serially, preserve fill values, exclude edge padding,
and stream lazy products and statistics without materializing the full array.
Account for storage chunk size, including outer shards, when assessing memory.

## Backend Features

Version features are additive: compile independent implementations for every
enabled backend release. Unversioned aliases select the newest supported release.
Keep API differences in version wrappers and compatibility helpers, and gate
implementations on matching version features. Gate shared helpers on internal
`*_all` markers. Keep heavier backend bounds on matrix and operator implementations.

Keep aliases in `src/lib.rs` and `tests/common/backend_aliases.rs` synchronized.
Correctness suites must exercise every enabled release; selected-version aliases
are only for examples and benchmarks. When adding a release, update the consumers
in `tests/feature_unification`, `scripts/test-backends.sh`, and CI. Keep the
consumer workspace lockfile compatible with the root lockfile.

## Testing and Validation

Extend `tests/common/runner.rs` for behavior shared across backends. Use
backend-specific tests for adapter or representation details. Preserve parity
with the dense normalization oracle and the adjoint identity. Seed random
generators with `ChaCha8Rng` and compare floats with `approx`. Import shared
modules with `#[path = "common/runner.rs"] mod common;`; do not add `mod.rs` files.

Select validation by the behavior affected. Run the smallest set of checks that
covers the change, and expand only when failures or unresolved risks justify it:

- Instruction and prose changes: verify referenced paths, commands, links, and
  applicable formatting. Rust tests are unnecessary for these changes.
- Rust changes: run the affected suites, formatting, Clippy, and documentation
  checks. `bash scripts/test-backends.sh version FEATURE` checks one backend
  release. Run `task test` for shared behavior or backend feature changes.
- Python bindings: run `task python:check`; add `task python:package` for changes
  that affect distributions or installation.
- Website changes: run `task website:check`. For isolated scripts, focused Node
  tests and formatting checks suffice.
- CI workflows and validation scripts: run the affected commands or jobs. Use the
  matching toolchain for MSRV changes. Changes confined to CI orchestration or
  dependency overrides do not require unrelated Rust, Python, or website suites.
- Broad changes spanning components or affecting dependencies, packaging, or
  releases: run `task ci` before submitting. For narrowly scoped changes, run the
  affected checks instead.

Use `task --list` and [Taskfile.yml](Taskfile.yml) for available checks. Read the
active toolchains and compiler requirements from `rust-toolchain.toml`, Cargo
manifests, and CI. Use the MSRV modes in `scripts/test-backends.sh` with the
toolchains selected by CI when changing compatibility requirements. On NixOS,
use `scripts/python-tool.py` to select native Python tooling.

## Coding and Review

Document public APIs and explain reasons in comments. `Scalar` does not imply
`AddAssign`; use iterator sums or folds instead of `+=` in generic code.

Use Conventional Commits with short, imperative subjects. Pull requests should
explain the motivation, API or performance impact, related issues, and validation.
Benchmark hot-path changes, update documentation when public behavior changes,
and keep benchmark data and methodology synchronized.
