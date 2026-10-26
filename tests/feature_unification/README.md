# Feature unification regression fixtures

Each consumer crate depends on one backend release and enables only its matching
`lazymatrix` adapter. The harness depends on multiple consumers as ordinary
optional dependencies, so Cargo unifies their adapter features in one build.
Consumers check normalization and allocating and reusable forward and transpose
products with their own backend types.

Run the harness with, for example:

```sh
cargo test --manifest-path tests/feature_unification/Cargo.toml --locked \
  --features ndarray_v0_16,ndarray_v0_17
```

`scripts/test-backends.sh coexistence` runs this consumer harness for every
version pair, both with and without parallelism, and all supported releases
together. It also checks each family's unversioned alias alongside all of its
version features in both parallel modes using `cargo check --lib`. Full
correctness suites and solver examples run in the individual-version and core
jobs rather than repeating for each pair. CI caches this workspace's build
artifacts separately from the root workspace.

MSRV checks exclude nalgebra 0.35 and zarrs 0.23; the latter has a separate
Rust 1.91 check. The
separate lockfile starts from the root lockfile's dependency versions; update both
deliberately when changing supported releases.
