# TODO

This file records design work identified during the initial API audit. The crate
should expose normalized matrices and their storage capabilities; solver state
and solver-specific update logic belong in consuming crates.

## Python and neural-framework integration

Identified October 12, 2026, for the `normreg-nn` research consumer. Lazy
centering and scaling, reusable fitted parameters, and sparse CSR/CSC access
already exist. The blockers below concern exposing those capabilities to Python
and using them in neural first layers. Keep FFI and framework adapters in
companion packages; the Rust core remains independent of Python and solvers.

- [ ] Prototype a PyTorch adapter using supplied centers and scales before
  requiring Rust bindings or Rust execution in the training loop.
  - Compute `(X - 1c^T) S^-1 W + b` through sparse multiplication and a dense
    centering correction without materializing the normalized input.
  - Keep `W` trainable in normalized coordinates and derive `S^-1 W` during each
    forward pass. Training independent raw-scale weights would change the
    optimization dynamics despite equivalent initial predictions.
  - Store fixed normalization parameters as buffers and verify predictions and
    parameter gradients against an explicitly normalized dense reference.
    Establish supported sparse layouts, dtypes, devices, and bias behavior.
  - Measure peak memory and throughput before claiming a performance benefit. A
    native framework forward pass does not require Rust autograd or GPU kernels.
    TensorFlow support can follow the first working consumer.

- [x] Add an owned float64 Python binding for dense and CSR/CSC input, products,
  transpose, column access, a SciPy operator adapter, and explicit dense
  conversion. Python and Rust use independent versions in one release PR,
  with both API references generated inside VitePress through Diplodocus.

- [ ] Expose reusable fitted `NormalizationParams` through the Python binding.
  - Export and reconstruct centers and scales with column-count validation,
    preserving constant-column and nonfinite-value semantics. Specify a portable
    representation for reuse in framework buffers and checkpoints.
  - Fit statistics on training data and reuse them unchanged for validation and
    prediction. Online updates of statistics are a separate algorithm.
  - Keep research scaling exponents in the consumer initially, using explicit
    scales; determine whether a reusable core extension is actually needed.
  - Verify reconstructed parameters and products against a dense reference,
    including reuse across languages and recoverable parameter errors.

- [ ] If a consumer needs Rust-backed neural products, use the existing
  multiple-right-hand-side operator task to add sparse matrix--dense matrix
  products and transpose products. The current public product traits operate
  on vectors; repeated vector calls are a reference path, not evidence of
  efficient batched execution.
  - Benchmark against native framework products before choosing Rust for the
    training loop. Define backward integration and device transfers only if that
    route is justified. These are not blockers for the native PyTorch adapter
    using exported normalization parameters.

## Backend version compatibility

- [x] Make versioned backend features additive under Cargo feature unification.
  - Compile shared implementations independently for every enabled release, with
    version-specific aliases and compatibility adapters.
  - Run full backend suites for every enabled release, including version pairs,
    all-feature builds, and parallelism. Separate consumer crates verify Cargo
    feature unification across dependencies.
  - Cover sprs and Zarr interoperability with every enabled ndarray release, and
    compare products and Gram outputs across backend versions.

## Out-of-core storage

- [x] Demonstrate memory-mapped ndarray views with a private `.npy` file.
- [x] Make products, column statistics, and computed normalization fallible,
  sharing each backend's error type through `MatrixErrorType`.
- [x] Add synchronous zarrs 0.22 products and all normalization options, with
  serial chunk reads and at most two scans for computed normalization.
- [x] Prototype a separate fallible block-reader capability with a Shrinkage
  consumer before stabilizing buffered access.
  - Specify caller-owned buffer capacity, logical block dimensions, and how
    returned views borrow the buffer until the next read. Keep memory bounded
    independently of the full design size.
  - Keep decoding and I/O out of infallible `RawColumns`, `SparseColumns`, and
    `SparseRows` borrowing. Preserve the backend's error type and define which
    buffer contents remain usable after a failed read.
  - Inject failures during normalization statistics and a later solver pass.
    Verify buffer reuse and that consumers discard partially written blocks.
    Keep solver iteration and residual policies in Shrinkage.
  - `ReadBlock` and `DenseBlock` provide raw rectangular Zarr reads into
    caller-owned storage. Shrinkage's experimental buffered proximal consumer
    reuses iteration and normalization, with parity and injected-failure tests.
    Its development Cargo override remains local; the interfaces are
    provisional.
- [ ] Measure cold-I/O throughput and peak memory on larger-than-RAM inputs
  before adding chunk caching, prefetching, async reads, or stricter
  budgets.

## Foundation

- [x] Add an orientation-independent `MatrixShape` trait.
  - Implement `nrows()` and `ncols()` for every backend matrix and for
    `LazyMatrix`.
  - Infer dimensions in `LazyMatrix` constructors instead of accepting caller
    supplied dimensions that can disagree with the backend.
  - Let generic consumers obtain dimensions from the operator.
  - Test rectangular and zero-column matrices; caller/backend dimension
    disagreement is eliminated by construction.

- [x] Make standard-deviation calculations numerically stable.
  - Replace `E[x^2] - E[x]^2` with a stable two-pass or combined-variance
    calculation that accounts for implicit sparse zeros.
  - Add a regression test using a large offset and small variation, such as
    values near `1e12` whose true variance is nonzero.
  - Apply the same algorithm and edge-case policy to every backend.

- [x] Define the policy for empty matrices and nonfinite statistics.
  - Allow normalization of matrices with zero rows. Undefined means and standard
    deviations are `NaN`; zero L2 and max-absolute scales use the
    degenerate-column convention below.
  - Follow IEEE behavior for nonfinite values instead of rejecting them. Ensure
    aggregations propagate `NaN` rather than accidentally masking it.
  - Replace only exact computed zero scales with `1`, leaving nonfinite scales
    untouched. Such degenerate columns are left unscaled.
  - Document and test the behavior for every backend.

- [x] Validate explicit normalization parameters.
  - Reject exact zero scales, including negative zero, in `from_parts` and
    `with_scales`. Panic with the column index, as for invalid dimensions.
  - Preserve all other supplied values, including negative scales, tiny nonzero
    scales, and nonfinite centers or scales. Reconstruction from `into_parts`
    preserves parameters even when computed statistics are nonfinite.
  - Keep zero-to-one replacement exclusive to computed normalization.
  - Defer a fallible constructor until a consumer needs recoverable validation.

## Operator performance

- [x] Investigate the performance regression in stable centered dense products.
  - Profile forward and transpose products across faer, nalgebra, and ndarray,
    including contiguous and strided layouts, to separate division costs from
    traversal and vectorization costs.
  - Evaluate bounded panels and moving scaling outside inner loops while
    preserving centered accumulation, overflow, underflow, nonfinite values, and
    the fused zero-coefficient contracts.
  - Rerun product and consumer benchmarks before choosing an optimization.
    Preserve the stable default and avoid a full normalized design copy.
  - Results: [centered dense product
    investigation](benches/dense_products/README.md).

- [ ] Evaluate native-vector borrowing in Shrinkage's proximal consumer.
  - `VectorOwned`, `VectorView`, and `VectorViewMut` already cover buffer
    construction and indexed access. Check whether adopting them lets Shrinkage
    remove its separate `ProximalVector` backend implementations.
  - Shrinkage currently copies between native product vectors and slice buffers
    used by predictor datafits and complete proximal terms. Measure these copies
    separately from normalization allocations.
  - Evaluate borrowed contiguous slice access only if consumer measurements
    justify it. Keep contiguity an explicit capability, preserve strided views,
    and avoid implicit gathers or scalar runtime dispatch.

- [x] Avoid cloning the forward input when scaling is inactive.
  - The default allocating hook borrows raw and center-only inputs and clones
    coefficients only when scales are present. Backend overrides retain their
    existing paths, and centering corrections run only after a successful
    product.
  - Regression tests cover clone counts, input preservation, and backend
    failures. The benchmark retains the former cloning path for comparison with
    the public operator. See [the
    investigation](benches/forward_input/README.md).

- [x] Add multiple-right-hand-side operator capabilities.
  - `MatMat`, `MatTransposeMat`, and their reusable-output forms support every
    backend release, fitted normalization, eager operators, and intercepts.
    `MatrixRead` borrows dense inputs, and callers select allocating output
    through `MatrixOwned`. Products remain independent of response and solver
    types and never materialize a normalized design.
  - The multiresponse least-squares example demonstrates batched predictions
    and gradients. [Measurements](benches/batched_products/README.md) compare
    vector and batch products, allocations, storage reads, and consumer updates.
    Dense and Zarr consumers benefit on the measured fixture; sparse batching
    can be slower despite fewer allocations.
  - Batched submatrix products, fused `alpha`/`beta` forms, and explicit
    workspace methods remain follow-up work.

## Operator algebra

- [ ] Evaluate a borrowed transpose operator view with a concrete consumer.
  - A transpose view should swap `MatVec` and `MatTransposeVec` without copying
    or changing the normalization represented by the original operator.
  - Do not imply that transposed sparse storage has acquired the opposite
    orientation-specific borrowing capability.

- [ ] Evaluate lightweight scaled, sum, and composition operator wrappers.
  - Candidate forms are `Scaled<A>`, `Sum<A, B>`, and `Composition<A, B>`;
    require compatible dimensions and implement products without materializing
    their operands.
  - Prefer named constructors or methods over `Add` and `Mul` until ownership,
    borrowing, scalar-zero behavior, and error messages are settled.
  - Add a wrapper only when a consumer benefits beyond spelling two existing
    operator calls explicitly.
  - Keep broadcast scalar/vector addition out of generic arithmetic: its row
    versus column semantics are ambiguous, and the general case needs a low-rank
    expression rather than altered normalization metadata.

## Lazy design matrices

- [ ] Prototype a programmatic API for numeric column selection and
  interactions.
  - Borrow source columns and store term descriptions, including an optional
    intercept. For numeric inputs, `~ x1 + x2 + x2:x3` represents the columns
    `[1, x1, x2, x2 * x3]` without allocating the interaction column.
  - Implement `MatrixShape`, `MatrixErrorType`, forward and transpose products,
    and their reusable-output counterparts. Evaluate interactions directly into
    outputs or accumulators.
  - Start with borrowed dense columns. Efficient interactions require aligned
    access to source values; operator products alone are insufficient.
  - Expose logical column operations for dots, norms, weighted products, and
    scaled additions so coordinate-wise consumers can use the same terms.
  - Test products and column operations against a materialized dense oracle,
    including the adjoint identity, empty inputs, and nonfinite values.

- [ ] Compose lazy design terms with column normalization.
  - Implement `ColumnStats` and its combined `normalization_stats` hook by
    scanning source values without materializing expanded columns.
  - Normalize the expanded terms by default. Centering an interaction is
    different from multiplying centered predictors; document and test that
    distinction.
  - Define how to preserve an intercept with effective center zero and scale one
    when normalizing the remaining columns.

- [ ] Evaluate sparse and chunked interaction strategies with concrete
  consumers.
  - Preserve `SparseColumns` as a borrowed-slice capability; computed sparse
    interactions must not claim to expose stored product values as slices.
  - Preserve IEEE nonfinite behavior when exploiting structural zeros.
  - For storage-backed inputs, evaluate terms from aligned chunks and share
    reads across terms while retaining fallible operations.
  - Document scan costs and working memory. Benchmark repeated evaluation
    against materialization before adding optional caching.

- [ ] Add an optional formula frontend after the numeric representation settles.
  - Translate formula syntax into the programmatic representation. In R-style
    syntax, `a:b` denotes an interaction and `a*b` expands to `a + b + a:b`.
  - Keep parsing and data-schema concerns separate from the dependency-light
    numerical core; evaluate a separate crate or optional feature.
  - Define categorical levels, contrasts, term and column ordering, and
    missing-value handling. Preserve the training schema for prediction,
    including an explicit policy for unseen levels.
  - Keep response handling and model fitting in consuming crates.

## Submatrices

- [x] Add borrowed contiguous submatrices after row and column views.
  - `lazy.submatrix(rows, cols)` restricts the already normalized matrix and
    borrows the selected columns' fitted centers and scales without refitting.
  - Nested selections use local coordinates and flatten to the original parent.
    A block is a rectangular submatrix, not a separate logical abstraction.
  - Support allocating, reusable, and fused forward and transpose products,
    filtered columns and sparse rows, and explicit dense conversion across all
    in-memory backend versions.
  - Filter and rebase sparse indices through view adapters; preserve the full
    CSC/CSR slice-borrowing capabilities without pretending restricted indices
    remain contiguous borrowed slices.
  - Raw selection followed by normalization remains a distinct operation with
    different statistical semantics.
- [x] Extend submatrices to bounded Zarr reads and weighted operations.
  - Zarr selections stream forward and transpose products, eager conversion,
    and weighted column sums through intersecting chunks without refitting.
    Empty ranges and zero-alpha products skip storage reads.
  - Weighted Gram products normalize entries in bounded panels across all
    backend releases, combining sparse duplicates before multiplication. Weights
    use local row indices; signed and nonfinite values retain IEEE behavior.
  - Shared oracle tests and injected storage failures cover selection semantics
    and errors. [Measurements](benches/submatrix/README.md) document bounded
    workspace and the cost of repeated scans and chunk decoding.
- [ ] Evaluate arbitrary index selections after defining ordering,
  duplicate-index, mapping, allocation, and complexity contracts.
- [ ] Expose submatrix selection in Python after defining indexing and
  ownership.

## Tests and maintenance

- [ ] Exercise the public scalar claim with shared `f32` tests.
- [ ] Test explicit stored zeros and fully dense sparse columns/rows.
- [ ] Test every new view through dense reconstruction and relevant algebraic
  identities rather than solver convergence.
- [ ] Consider sharing private CSC statistics helpers between backends to
  prevent their numerical behavior from diverging.
- [ ] Replace the ignored crate-level example with a small compiling doctest
  once the shape-aware constructors settle.

## Surface ownership decisions

- [ ] Revisit whole-operator logical reductions only with a concrete consumer.
  - Column sums and squared norms are already available through `LogicalColumn`;
    do not duplicate them as matrix-wide allocation-returning methods merely for
    symmetry.
  - Spectral norms, bilinear forms such as `u^T A v`, and quadratic forms such
    as `A^T A v` remain compositions of existing primitives unless fusion is
    shown to matter.

- [ ] Keep general vector algebra outside the normalization trait surface.
  - Hadamard products, arbitrary maps and reductions, vector construction, and
    proximal operations belong to backends or solver-facing crates.
  - Promote a vector primitive here only when it is required to implement a
    matrix capability across multiple backends.

## Outreach targets

- [x] RustyStats: <https://github.com/PricingFrontier/rustystats>
- [x] anofox-regression: <https://github.com/sipemu/anofox-regression>
- [ ] Greeners: <https://github.com/sheep-farm/Greeners>
- [ ] Polars for Data Science
  <https://github.com/abstractqqq/polars_ds_extension>
- [ ] lmutils <https://github.com/GMELab/lmutils.r>
