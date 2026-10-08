# Repository Guidelines

## Project Structure & Module Organization

`src/lib.rs` contains crate documentation, module declarations, and public
re-exports. The main implementation is divided as follows:

- `src/normalization.rs` defines `Centering`, `Scaling`, `Normalization`, and
  reusable fitted `NormalizationParams`.
- `src/matrix.rs` implements `LazyMatrix`, construction, column and row access, and
  operator behavior.
- `src/eager.rs` implements `EagerMatrix`, which retains fitted parameters as
  metadata and forwards operations to already normalized dense storage.
  `src/materialize.rs` and `src/traits/materialize.rs` provide explicit dense
  conversion and in-place dense normalization capabilities.
- `src/column.rs` contains logical and sparse borrowed column views.
- `src/row.rs` contains the borrowed normalized sparse row view.
- `src/intercept.rs` implements `WithIntercept`, an implicit leading column of
  ones added after predictor normalization. `src/weighted_sums.rs` and
  `src/traits/weighted_sums.rs` provide directly centered weighted-column sums
  for its Gram cross terms.
- `src/traits/operator.rs` defines matrix shape and allocating or
  reusable-output matrix-vector products.
- `src/traits/gram.rs` defines writable dense output and weighted Gram
  capabilities. `src/gram.rs` contains shared validation and sparse Gram
  kernels.
- `src/traits/vectors.rs`, `stats.rs`, and `columns.rs` define vector algebra,
  sparse-aware statistics, and column capabilities.
- `src/traits/rows.rs` defines the contiguous sparse-row borrowing capability.
- `src/traits/blocks.rs` defines experimental fallible rectangular reads into
  caller-owned buffers and the borrowed packed `DenseBlock` view.
- `src/backends/faer/` and `src/backends/nalgebra/` contain feature-gated dense,
  sparse, and vector implementations. `src/backends/ndarray/` contains dense
  matrix and vector implementations. `src/backends/sprs/` contains sparse
  statistics and the checked `SprsCsc` and `SprsCsr` wrappers;
  `src/backends/sprs.rs` implements CSC and CSR products.
  `src/backends/zarrs.rs` and its statistics module scan synchronous Zarr arrays
  one chunk at a time. Shared helpers live in `src/backends/support.rs`.

Keep backend-independent logic out of backend implementation directories. With
no features enabled, the crate provides its traits and `LazyMatrix` with only
`num-traits`; adding a backend means implementing the existing trait surface for
a matrix/vector pair. This crate contains no FFI.

Each supported backend release has a version feature, such as `faer_v0_22`. The
unversioned `faer`, `nalgebra`, `ndarray`, `sprs`, and `zarrs` features select
the newest supported release. Version features are additive: compile independent
trait implementations for every enabled release. Shared backend files use the
crate aliases supplied by version-specific wrapper modules; keep API differences
in those wrappers and their compatibility helpers. Gate backend-independent
helpers on the internal `*_all` markers.

Keep version aliases in `src/lib.rs` and `tests/common/backend_aliases.rs`
synchronized. Correctness suites must exercise every enabled release. Only
examples and benchmarks use the helper's selected-version convenience aliases.
The isolated `tests/feature_unification` workspace checks separate consumer
crates with different backend dependencies. Keep its lockfile compatible with
the root lockfile and include new releases in its consumers, the backend test
matrix in `scripts/test-backends.sh`, and CI.

Integration tests are in `tests/`. The backend suites reuse
`tests/common/runner.rs`, while `tests/cross_backend.rs` checks agreement
between implementations. Runnable demonstrations belong in `examples/`, and
Criterion benchmarks belong in `benches/`. See `TODO.md` for planned API work.

## Development Status

LazyMatrix is in early development. Breaking changes are expected. Prefer a
robust and consistent API over backwards compatibility until the crate reaches
1.0.0.

## Architecture & Invariants

`LazyMatrix` presents a normalized design matrix as a linear operator:

```text
X_tilde = (X - 1c^T) S^-1
X_tilde v = X(S^-1 v) - 1(c^T S^-1 v)
X_tilde^T u = S^-1(X^T u - c sum(u))
```

Centering and scaling are independently optional. Lazy operators never
materialize the centered matrix: doing so turns structural zeros into nonzeros
and defeats sparse storage. Lazy backends multiply the original `X` and fold
normalization into the operator calculation. Explicit `to_eager` and
`to_eager_into` conversions may materialize any supported input into dense
storage; `into_eager` modifies writable dense storage in place. Eager operators
must not apply their retained fitted parameters again. Materialization sums raw
sparse duplicates before normalizing, including implicit zeros, and uses one
working row or column rather than a second full design matrix.

Preserve these normalization semantics:

- Replace exact zero scales with one so constant columns do not divide by zero.
- When centering and scaling are both enabled, compute scales from centered
  columns. SD and range are centering-invariant; L1, L2, and max-absolute use
  the sparse closed-form `*_centered` methods, including contributions from
  implicit zeros. Preserve IEEE nonfinite values.
- Store centers and scales as `Vec<F>`, not backend-specific vector types.

## Fallible Operations and Storage

Products, `ColumnStats` methods, and `LazyMatrix::new` return `Result`. Share
one associated error through `MatrixErrorType`; in-memory backends use
`Infallible`. Dimension mismatches still panic. Reusable outputs may be partial
after an error; normalization corrections must run only after the backend
succeeds. Forward the combined `normalization_stats` hook through references and
wrappers so storage backends retain their shared scans. `NormalizationStats<F>`
is the pair of optional center and raw-scale vectors; only computed `LazyMatrix`
construction replaces exact zero scales with one. Explicit construction panics
on zero scales, including negative zero, and otherwise preserves parameters,
including negative scales and nonfinite values.

Weighted Gram kernels receive normalization parameters through
`WeightedGramKernel` and center values before accumulating products. Do not
correct raw moments afterward: large offsets can erase centered variation. Dense
ndarray kernels use bounded panels; CSC kernels choose bounded panels or
borrowed column pairs with subtraction-free sums for implicit rows. Noncanonical
columns and nonfinite arithmetic may use two working columns, but never a full
dense design matrix. `MatrixWrite` keeps dense coefficient output independent of
the input backend.

The zarrs 0.22 backend supports synchronous two-dimensional floating-point
arrays and Rust 1.87. Read chunks serially, preserve configured fill values, and
exclude edge padding. Keep working vectors in RAM. Lazy products and statistics
never materialize the full array. Explicit eager conversion streams chunks into
caller-selected dense storage. Chunk buffers and codec workspaces depend on storage chunk size,
including outer shards. Borrowing capabilities must not hide decoding or I/O.
Tests use counting and failing stores; larger-than-RAM benchmarks remain manual.

## Column Access & Capability Boundaries

Keep storage orientation out of the operator and express contiguous borrowing as
a capability:

- `MatVec`, `MatTransposeVec`, and `ColumnStats` are orientation-agnostic.
- Dense backends and storage with natural column views implement `RawColumns`.
- Only storage that can borrow contiguous CSC slices implements `SparseColumns`.
- Only storage that can borrow contiguous CSR slices implements `SparseRows`.

`LazyMatrix::column(j)` requires `RawColumns`; `sparse_column(j)` retains the
stronger `SparseColumns` bound. Do not hide an O(nnz)-per-column gather behind
`SparseColumns`. An incompatible storage layout should fail at compile time
rather than silently add a gather.

Views borrow rather than copy. `LazyColumn` wraps a backend raw view and the two
normalization scalars; `LazySparseColumn` also exposes CSC slices and the
sparse-plus-offset representation. Put logical column operations—dots, weighted
products, scaled additions, and norms—on the view so callers do not rederive
normalization formulas. Keep raw slices available for specialized algorithms,
and document complexity: a centered dot requiring a dense vector sum is O(n +
nnz), while a cached-sum path is O(nnz). sprs supports CSC and CSR operators
directly because orientation is a runtime flag. Only the checked `SprsCsc`
wrapper implements column borrowing; `SprsCsr` checks CSR storage for row
borrowing. `SparseColumns` and `SparseRows` require `usize` row and column
indices, respectively. faer CSR matrices and views and nalgebra CSR matrices
provide `MatrixShape` and `SparseRows`; use faer's row ranges to exclude spare
capacity and sprs's adjusted outer ranges for sliced views. `LazyMatrix::row`
requires `SparseRows` and returns a `LazyRow` borrowing raw storage and the full
optional center and scale slices. Its logical length is the column count, even
when no entries are stored. The background varies by column; expose it and the
stored scaled corrections without allocating a dense row. Preserve storage
order and duplicates, and sum duplicate raw values before direct normalization.
Faer and nalgebra CSR operators and statistics remain prospective work described
in `TODO.md`.

## Example-Driven Design

Keep solver algorithms and state in `examples/` or downstream crates. Examples
such as `least_squares_gd` and `coordinate_descent` demonstrate consumers and
expose missing matrix capabilities, but solvers do not belong in the library or
its correctness suite. Add a reusable operator or logical-column operation when
an example reveals a genuine matrix need; do not add residual policies, cached
sums, update rules, or an entire solver to the crate.

## Build, Test, and Development Commands

- `cargo build --locked` builds the dependency-light core.
- `cargo build --all-features --locked` checks all supported backends.
- `task test` runs every backend version with and without parallelism, version
  pairs, separate-consumer feature unification, cross-backend comparisons, and
  the all-feature test matrix.
- `task ci` runs formatting, Clippy, documentation, and all tests—the local
  equivalent of GitHub CI.
- `cargo bench --locked --bench column_sds` runs the column-statistics
  benchmarks.
- `cargo run --locked --example coordinate_descent --features faer` runs an
  example.

The repository's devenv supplies Rust 1.89, `go-task`, and the configured
pre-commit hooks. The core and sprs 0.11.4 MSRV is Rust 1.85. Other backend checks
retain Rust 1.87, except for nalgebra 0.35 (including the `nalgebra` alias), which
requires Rust 1.89. The lockfile uses
sprs 0.11.4 for Rust 1.85 compatibility; sprs 0.11.5 requires Rust 1.88. Run
`bash scripts/test-backends.sh msrv` with Rust 1.87 to check the compatible
feature selections, including tests, examples, and benchmarks.

## Coding Style & Naming Conventions

Use standard `rustfmt` formatting (four-space indentation) and keep
`cargo clippy --all-features --all-targets --locked -- -D warnings` clean. Name
modules, functions, and files in `snake_case`; types and traits use
`UpperCamelCase`. Document public APIs and explain the reason—not the
mechanics—in comments. Keep feature-specific dependencies and trait
implementations behind their matching Cargo feature. `Scalar` does not imply
`AddAssign`; in generic code, accumulate with iterator sums or folds instead of
`+=`, and keep heavier backend bounds on matrix and operator implementations.

## Testing Guidelines

Prefer test-driven changes: add a failing regression or behavior test before the
implementation. Use `#[test]` integration tests with descriptive `snake_case`
names. `tests/common/runner.rs` is a backend-generic suite driven by thin
`faer_backend.rs` and `nalgebra_backend.rs` adapters. Extend the generic runner
when behavior should apply to every backend; use a backend-specific file only
for adapter or representation details. A new backend should provide construction
and conversion closures, then call `run_backend_suite`.

The test oracle materializes the normalized matrix densely and compares naive
products with lazy output up to floating-point noise. Preserve oracle parity and
the adjoint identity; `cross_backend.rs` separately verifies agreement between
implementations. Seed random generators with `ChaCha8Rng`, and compare floats
with `approx`. Import shared modules using
`#[path = "common/runner.rs"] mod common;`; do not introduce `mod.rs` files. Run
`task test` during development and `task ci` before submitting.

## Commit & Pull Request Guidelines

Recent history generally follows Conventional Commits, such as `feat:`,
`refactor:`, `docs:`, and `chore(deps):`. Write short, imperative subjects and
keep each commit focused. Pull requests should explain the motivation and API or
performance impact, link relevant issues (for example, `Closes #123`), and list
the commands run. Include benchmarks when changing hot paths and update examples
or documentation when public behavior changes.
