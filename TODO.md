# TODO

This file records design work identified during the initial API audit. The crate
should expose normalized matrices and their storage capabilities; solver state
and solver-specific update logic belong in consuming crates.

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

- [ ] Complete CSR operator and statistics support when it has a concrete
  consumer.
  - sprs already supports CSC and CSR operators and statistics because its
    matrix type stores orientation at runtime. `SprsCsc` and `SprsCsr` check
    storage orientation for borrowed columns and rows, respectively.
  - faer `SparseRowMat` and nalgebra-sparse `CsrMatrix` already implement
    `MatrixShape` and `SparseRows`. Add `MatVec`, `MatTransposeVec`, their
    reusable-output counterparts, and `ColumnStats` when needed.
  - Reuse the backend-generic oracle and adjoint tests.
  - Do not implement `SparseColumns` by performing an expensive gather.

## Operator performance

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
  - [ ] Measure allocation costs in an iterative consumer before extending the
    fused prototype's reusable workspace API.
  - [ ] Use ndarray-glm fitting and Shrinkage's backtracking proximal gradient
    to evaluate repeated `S^-1 x` allocations. Shrinkage currently pins
    0.3.0 and clones a native coefficient vector on every scaled forward
    product, including trial steps and final diagnostics.
  - [ ] Compare the existing overwrite API with fused workspace application
    using `alpha = 1` and `beta = 0`. Measure allocation count, requested
    bytes, and throughput for dense and CSC input with raw, centered, and
    standardized designs. Keep dataset construction and workspace setup
    outside measurement.
  - [ ] Use those results to decide whether the overwrite API also needs an
    explicit workspace method. Verify the integrated consumer before
    claiming that normalized iterations allocate nothing.
  - [ ] Keep allocating convenience methods if they materially improve
    ergonomics.

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
  - Implemented for dense ndarray and faer, nalgebra, and checked sprs CSC
    inputs, with backend-independent reusable output. See
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

## SLOPE rewrite support

These items come from comparing the normalization code in `../libslope` with the
current operator and column-view API. The JIT-normalization enum and its
four-way branches should not be ported: optional centers and scales already
represent the same four states.

- [x] Add weighted logical-column products.
  - [x] Provide a weighted dot product for `x_tilde_j^T (weights * vector)`
    without materializing the elementwise product.
  - [x] Provide a weighted squared norm `sum_i weights_i * x_tilde_ij^2` for
    coordinate-wise Hessian calculations.
  - [x] Offer a weighted-dot variant accepting cached `sum(weights * vector)` so
    repeated dots remain O(nnz_j). Weighted norms scan row weights directly
    in O(n + nnz_j) to preserve small implicit-zero contributions. Their
    cached-total variant uses the same calculation.
  - [x] Accept borrowed inputs without forcing copies of dense matrix columns;
    account explicitly for contiguous versus strided vector views.
  - [x] Test each formula against a dense oracle for all four center/scale
    combinations, including implicit and explicitly stored zeros.

- [x] Make the sparse-plus-offset decomposition of `LazyColumn` easier to use.
  - Provide `implicit_value()` (`-center / scale`), `raw_sum()`, and an iterator
    over stored corrections (`raw_value / scale`).
  - Keep these as representation-level column operations. Residual offsets,
    cached residual sums, and coordinate-update policy remain in the consuming
    solver.
  - Use the coordinate-descent example to verify that a centered residual update
    can stay O(nnz_j) without rederiving normalization formulas.

- [x] Add the remaining normalization statistics used by `libslope`.
  - Add minimum centering and L1 and range scaling.
  - Extend `ColumnStats` with sparse-aware minima, ranges, L1 norms, and
    centered L1 norms. Implicit zeros must participate in every statistic.
  - Use the sparse closed form
    `sum_stored |value - center| + (n - nnz) * |center|` for centered L1 norms.
  - Preserve the current rule that non-translation-invariant scales such as L1,
    L2, and max-absolute are computed after centering. `libslope` computes its
    scales from raw `X`; exact legacy behavior for unusual combinations can be
    reproduced with `from_parts`.

- [x] Add a generic dense/sparse logical-column interface.
  - Use `RawColumns` with an associated borrowed view for backend storage and
    `Columns` with an associated `LogicalColumn` for generic consumers.
  - Accept contiguous and strided inputs and destinations through `VectorView` /
    `VectorViewMut` without forcing copies.
  - Keep `SparseColumns` as the stronger contiguous-CSC capability and expose
    sparse representation helpers separately from common logical operations.

- [x] Add dense faer and nalgebra backends.
  - Implement operators, statistics, and raw-column access for owned matrices
    and immutable backend-native matrix views.
  - Exercise the same logical-column oracle over dense and sparse storage.
  - Demonstrate SLOPE-shaped weighted derivatives and active-column updates in
    the `slope_primitives` example.

- [ ] Benchmark active-set products before adding a restricted-operator API.
  - Full gradients map directly to `MatTransposeVec`, and active-set gradients
    can be computed through `LazyColumn` views.
  - Add a restricted `matvec` or transpose product only if working-set and
    screening benchmarks show that zero-filled full products are a bottleneck.
  - Do not encode flattened feature-response indices or SLOPE working-set policy
    in the matrix API.

- [ ] Benchmark logical column-pair products before adding a Gram-entry API.
  - Prototype `X_tilde_j^T X_tilde_k` and `X_tilde_j^T diag(weights) X_tilde_k`
    in a consuming coordinate or block method.
  - For CSC inputs, merge stored row indices and account for the centered
    background analytically rather than materializing either logical column.
  - Define the capability at matrix level if accepting two backend-specific
    associated column-view types makes a reusable `LogicalColumn` method
    awkward.
  - Document backend-specific complexity; do not promise sparse intersection
    costs for dense storage or for an incompatible sparse orientation.

- [ ] Pressure-test signed combinations of normalized columns in the SLOPE
  consumer.
  - Build cluster directions from `LazyColumn` views into a consumer-owned
    sparse-plus-offset workspace instead of materializing a normalized sparse
    matrix for every cluster.
  - Promote a general column-combination abstraction into this crate only if a
    second non-SLOPE consumer establishes a reusable contract.

## Submatrices

- [ ] Add submatrix support only after row and column views are established.
  - Define `lazy.submatrix(rows, cols)` as a restriction of the already
    normalized matrix; inherit the selected columns' existing centers and scales
    rather than recomputing them.
  - Start with contiguous ranges, where global-to-local index mapping is cheap.
  - Treat a block as the range-selected form of a submatrix rather than as an
    unrelated abstraction.
  - Use filtering view adapters when restriction breaks slice contiguity; do not
    weaken full CSC/CSR access merely to give every view the same type.
  - Defer arbitrary index selections until their ordering, duplicate-index,
    mapping, allocation, and complexity contracts are clear.
  - Keep selecting raw data and then normalizing it as a distinct operation with
    distinct statistical semantics.

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

## Explicit non-goals

- Coordinate descent, SGD, residual offsets, coefficient offsets, sampling,
  optimization workspaces, and convergence logic remain in consuming crates.
- Orientation capability traits must not conceal full scans or sparse-to-dense
  materialization.
- Column or row views expose matrix structure and normalization metadata, not
  solver-specific state or update operations.
- Arbitrary affine-expression machinery is deferred until more transformations
  than column centering and scaling require it.
- Determinants, inverses, factorizations, direct solves, mutable entrywise
  matrix arithmetic, rank-one updates, and general matrix Hadamard products are
  not linear-operator capabilities and remain out of scope.
- Intercept fitting, coefficient rescaling, working and strong sets, screening,
  KKT policy, SLOPE clusters, and cluster merging or splitting remain in the
  consuming model crate.
- Dense in-place normalization and a `modify_x` mode are not part of the lazy
  operator. Consumers that deliberately materialize normalized dense data can do
  so outside this crate.
