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

## Sparse access capabilities

- [x] Add `SparseColumns` for contiguous CSC column access.
  - Return borrowed row-index and raw-value slices without copying.
  - Implement it only for storage types that provide efficient contiguous column
    access.
  - Keep it separate from `ColumnStats`, which remains orientation-independent.

- [x] Add `LazyMatrix::column` and a borrowed `LazyColumn` view.
  - Gate generic column access on `M: RawColumns`; retain `M: SparseColumns` for
    explicit access to contiguous CSC slices.
  - Expose raw storage, logical length, center, and scale. Sparse views also
    expose row indices and raw stored values.
  - Use the canonical `center` and `scale` terminology used by `LazyMatrix`;
    inverse scales and affine background values are derived quantities.
  - Represent inactive centering and scaling as effective values `0` and `1` in
    the view where that simplifies consumers.
  - Do not hide a dense centered-column update behind a method that appears to
    be sparse.
  - Test reconstruction of logical columns against a dense oracle.

- [x] Add `SparseRows` for contiguous CSR row access.
  - Return borrowed column-index and raw-value slices without copying.
  - Implement it only for storage types that provide efficient contiguous row
    access; do not gather rows from CSC under this trait.
  - Support faer CSR matrices and views, nalgebra-sparse CSR matrices, and
    checked `SprsCsr` matrices and views, with `usize` column indices.

- [x] Add `LazyMatrix::row` and a borrowed `LazyRow` view.
  - Gate the method on `M: SparseRows`.
  - Expose column indices, raw stored values, logical length, and borrowed
    normalization parameters.
  - Make the sparse-plus-affine structure explicit: a centered logical row is
    generally dense even when its raw row is sparse.
  - Test reconstruction of logical rows against a dense oracle.

- [x] Complete CSR operator and statistics support, exercised by a sparse SGD
  example.
  - sprs already supports CSC and CSR operators and statistics because its
    matrix type stores orientation at runtime. `SprsCsc` and `SprsCsr` check
    storage orientation for borrowed columns and rows, respectively.
  - faer `SparseRowMat` and nalgebra-sparse `CsrMatrix` already implement
    `MatrixShape` and `SparseRows`. They now also implement `MatVec`,
    `MatTransposeVec`, their reusable-output and fused scaled counterparts, and
    `ColumnStats`, including faer's borrowed and mutable views.
  - `examples/least_squares_sgd.rs` demonstrates single-observation and
    minibatch updates over lazily normalized CSR rows, with deferred centering
    state kept in the example.
  - Reuse the backend-generic oracle and adjoint tests.
  - Do not implement `SparseColumns` by performing an expensive gather.

## Operator performance

- [ ] Investigate the performance regression in stable centered dense products.
  - Profile forward and transpose products across faer, nalgebra, and ndarray,
    including contiguous and strided layouts, to separate division costs from
    traversal and vectorization costs.
  - Evaluate bounded panels and moving scaling outside inner loops while
    preserving centered accumulation, overflow, underflow, nonfinite values, and
    the fused zero-coefficient contracts.
  - Rerun product and consumer benchmarks before choosing an optimization.
    Preserve the stable default and avoid a full normalized design copy.

- [x] Add explicit eager normalization and dense materialization.
  - `to_eager` selects an owned dense backend; `to_eager_into` fills existing
    dense storage; `into_eager` reuses writable dense storage in place.
  - Support dense, sparse, and Zarr inputs. Sparse conversion includes implicit
    zeros and sums duplicates before normalization; Zarr streams chunks.
  - Preserve fitted centers and scales through `NormalizationParams` for reuse
    on prediction data and downstream coefficient transformations.
  - Compare conversion, products, downstream fitting time, and peak live Rust
    allocations with the lazy path. Keep lazy normalization the default.

- [x] Allow `LazyMatrix` to wrap a borrowed backend matrix.
  - Add forwarding implementations for the matrix capability traits on `&M`, or
    provide an explicit borrowed wrapper with equivalent ergonomics.
  - Support construction such as `LazyMatrix::new(&x, spec)` so fitting paths,
    cross-validation, and prediction do not need to consume the design matrix.

- [ ] Evaluate reusable-output operator methods before stabilizing the traits.
  - [x] Add `matvec_into` and `mat_transpose_vec_into` capabilities, with the
    backend implementations as the primitive path for allocating products.
  - [x] Allow input and output vector types to differ where backend APIs permit
    it, so borrowed or strided inputs can write into owned backend vectors.
  - [x] Specify dimension-checking and overwrite semantics, and implement
    backend-specific fast paths.
  - [x] Measure allocation costs in an iterative consumer before extending the
    fused prototype's reusable workspace API. [Consumer
    benchmarks](benches/fused_consumer.md) measure allocation count,
    requested bytes, and throughput for allocating, overwrite, fused, and
    workspace paths. They cover dense and CSC input with raw, centered, and
    standardized designs, with dataset construction and workspace setup
    outside measurement.
  - [x] Measure normalization allocations in ndarray-glm fitting and Shrinkage's
    backtracking proximal gradient. [Downstream
    profiles](benches/normalization_allocations.md) verify one coefficient
    clone per scaled forward product in Shrinkage, including rejected trials
    and the final pass. ndarray-glm currently standardizes an owned dense
    design during construction; the profiles measure that cost separately
    from fitting and prediction. Lazy adoption remains open.
  - [x] Compare the existing overwrite API with fused workspace application
    using `alpha = 1` and `beta = 0`. The [isolated
    comparison](benches/overwrite_comparison.md) measures allocation count,
    requested bytes, and throughput for forward and transpose products and
    identical iteration batches across the existing inputs and normalization
    modes, with setup outside measurement.
  - [ ] Use the isolated comparison and downstream consumer measurements to
    decide whether the overwrite API also needs an explicit workspace
    method. Verify the integrated consumer before claiming that normalized
    iterations allocate nothing.
  - [x] Keep allocating convenience methods if they materially improve
    ergonomics. The consumer benchmarks recommend retaining both allocating
    and overwrite interfaces because workspace allocation savings do not
    consistently improve throughput.

- [x] Prototype fused scaled operator application for ndarray and sprs.
  - Add forward and transpose capabilities for `y = alpha * A * x + beta * y`.
    Exact zero `alpha` skips the product, and exact zero `beta` ignores prior
    output.
  - Express overwrite, accumulation, and subtraction through the same primitive
    in the prototype backends rather than allocating intermediate vectors.
  - Let callers reuse coefficient scratch through optional `LazyMatrix` methods;
    keep backend-specific scratch types out of the core operator traits.
  - Test zero, one, negative, and nonfinite coefficients, plus empty and
    rectangular operators.

- [x] Evaluate fused application for the remaining backends and `WithIntercept`
  after consumer benchmarks.
  - Added fused products for faer, nalgebra, zarrs, and `WithIntercept`, plus
    reusable predictor scratch on the intercept wrapper.
  - [Consumer benchmarks](benches/fused_consumer.md) show allocation savings
    with caller-owned scratch and mixed throughput gains. Keep allocating and
    overwrite products; ordinary fused calls can allocate more with scaling.

- [ ] Make workspace reuse compose through `WithIntercept<LazyMatrix<M>>`.
  - The current intercept workspace reuses only the wrapper's buffer. Its inner
    normalized operator still allocates coefficient scratch, as measured in
    `benches/fused_consumer.md`.
  - Prototype a way to pass or retain both workspaces without moving solver
    state into the matrix crate. Preserve separate allocating convenience paths.
  - Extend the consumer allocation checks to the nested wrapper, including
    scaled forward and transpose products, zero `alpha` and `beta`, and backend
    failures after partial output writes. Normalization and intercept
    corrections must run only after the inner product succeeds.

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

- [ ] Avoid cloning the forward input when scaling is inactive.
  - Preserve the direct backend path for raw and center-only products.
  - Benchmark before adding more elaborate scratch-storage machinery.

- [ ] Add multiple-right-hand-side operator capabilities.
  - Prototype `MatMat` and `MatTransposeMat` for multiresponse and multinomial
    consumers rather than requiring one allocation and backend call per
    response.
  - Fold normalization into the batched products using the same identities as
    `MatVec` and `MatTransposeVec`.
  - Keep the capability independent of any response, loss, or solver type.
  - Consider reusable-output and fused `alpha`/`beta` variants only after the
    vector forms establish their ownership and workspace contracts.

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

## ndarray-glm integration

These items come from comparing this crate with ndarray-glm 0.1.0 at
[`0b727d8`](https://github.com/felix-clark/ndarray-glm/tree/0b727d8). The first
target is borrowed ndarray and sparse CSC input while retaining ndarray-glm's
IRLS algorithm, regularization, and ndarray-linalg solves. Tall matrices with
moderate numbers of predictors are the initial use case; the coefficient-space
system still requires O(p^2) storage.

- [x] Add efficient weighted Gram-matrix products as the first priority.
  - Compute `A^T diag(weights) A` for the logical normalized matrix, with a
    fallible reusable-output capability and a dense coefficient-space result.
  - Keep the core trait independent of ndarray while allowing ndarray-glm to
    receive an `Array2` for its existing solver.
  - Start with dense ndarray and sparse CSC implementations. Preserve sparse
    input storage and avoid materializing normalized or weighted design
    matrices.
  - Benchmark dense kernels against BLAS matrix-matrix multiplication and
    compare both backends with assembly through repeated operator products.
  - Test against a dense oracle, including large offsets with small variation.
    Use numerically stable centering; subtracting large raw moments can erase
    the centered cross-product.
  - Implemented for dense ndarray and faer, plus faer, nalgebra, and checked
    sprs CSC inputs, with backend-independent reusable output. See
    `examples/weighted_gram.rs` and `benches/weighted_gram.md`.

- [x] Reduce centered CSC Gram overhead at very low densities.
  - Sparse pairs now use a subtraction-free disjoint range-sum table for
    implicit rows and cache normalized stored values on sparse input. Direct
    centering, IEEE fallbacks, and bounded-panel kernels remain in place.
  - Pinned 10,000 by 128 benchmarks improved from 10.37 to 2.08 ms at 0.1%
    density and from 31.84 to 19.28 ms at 1% density. See
    `benches/weighted_gram.md`.

- [ ] Demonstrate compatible weighted normalization in the consuming crate.
  - ndarray-glm uses weighted means and sample standard deviations, including an
    effective-sample-size correction for frequency and variance weights.
    Preserve this policy without changing `Scaling::Sd`'s population convention.
  - Compute weighted means through a transpose product and centered weighted
    sums of squares through existing logical-column operations, then pass the
    adjusted centers and scales to `LazyMatrix::from_parts`.
  - With no intercept, disable centering but retain scales computed from
    deviations about the weighted mean.
  - Preserve downstream handling of empty inputs, single observations, constant
    columns, and nonfinite values. Keep sample corrections and validation policy
    in ndarray-glm.
  - Evaluate a shared weighted-statistics capability after the integration
    establishes a need, especially for backends without borrowed columns.

- [x] Add an implicit intercept wrapper as a first lazy design-matrix component.
  - Represent `A = [1, X_tilde]` by normalizing predictors before adding the
    constant column, so centering cannot erase the intercept.
  - Implement shape, error forwarding, forward and transpose products, and
    reusable-output variants without allocating a column of ones.
  - Construct weighted Gram blocks from the predictor Gram matrix,
    `X_tilde^T weights`, and `sum(weights)`.
  - Implemented as `WithIntercept`, with directly centered cross terms through
    `WeightedColumnSumsInto` and backend normalization kernels. See
    `examples/weighted_gram.rs`.
  - Keep intercept fitting, penalty exclusions, and coefficient transformations
    in the consuming crate. The wrapper should not require a formula frontend.

- [ ] Prototype adoption through ndarray-glm's data and fitting interfaces.
  - Its public `Dataset.x` is an owned `Array2`; supporting borrowed and sparse
    storage requires a downstream API design spanning `Dataset`, `Model`, `Fit`,
    and `Glm`.
  - Replace design-matrix products and weighted transposes in initialization,
    IRLS, and Fisher-information calculations with matrix capabilities.
  - Retain coefficient, score, and covariance transformations downstream, using
    the stored centers and scales.
  - Keep solver experiments in examples or the downstream crate. Validate
    lazymatrix capabilities through algebraic tests rather than solver
    convergence tests.

- [ ] Evaluate matrix capabilities needed for scalable diagnostics.
  - ndarray-glm currently obtains leverage by constructing the full O(n^2) hat
    matrix. Compute its diagonal directly downstream, evaluating batched
    products or row quadratic forms as reusable matrix capabilities.
  - Exact leave-one-out fitting already excludes observations through zero
    frequency weights. Reuse borrowed raw storage and recompute normalization
    for each refit; row-selection support is not a prerequisite.
  - Evaluate column-selection views for fits that exclude individual predictors,
    coordinated with the lazy design-matrix work above.
  - Keep full hat matrices and other explicitly dense diagnostic outputs opt-in,
    with their allocation costs documented downstream.

- [ ] Validate and benchmark the first integration milestone.
  - Cover dense ndarray and sparse CSC input, frequency and variance weights,
    offsets, and models with and without an intercept.
  - Compare Gaussian ridge coefficients and logistic IRLS systems with
    ndarray-glm, then extend downstream coverage to fitted results and
    diagnostics, including regularization and normalization edge cases.
  - Measure fitting time, allocation costs, and peak memory against the existing
    dense path. Include the reusable normalization workspace work under operator
    performance.
  - Establish the benefit before expanding to additional storage backends or
    changing the solver for very large predictor counts.

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
- [ ] Extend submatrices to bounded Zarr reads and weighted operations.
- [ ] Evaluate arbitrary index selections after defining ordering, duplicate-index,
  mapping, allocation, and complexity contracts.
- [ ] Expose submatrix selection in Python after defining indexing and ownership.

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

- [ ] RustyStats: <https://github.com/PricingFrontier/rustystats>
- [ ] anofox-regression: <https://github.com/sipemu/anofox-regression>
- [ ] Greeners: <https://github.com/sheep-farm/Greeners>
- [ ] Polars for Data Science
  <https://github.com/abstractqqq/polars_ds_extension>
- [ ] lmutils <https://github.com/GMELab/lmutils.r>
