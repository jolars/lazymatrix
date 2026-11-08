# TODO

## Out-of-core storage

- [ ] Measure cold-I/O throughput and peak memory on larger-than-RAM inputs
  before adding chunk caching, prefetching, async reads, or stricter
  budgets.

## Normalization methods

Use [scikit-learn's preprocessing
methods](https://scikit-learn.org/stable/modules/preprocessing.html) as a
coverage reference. Evaluate additions against the library's scope rather than
assuming full API or numerical parity.

- [x] Give the Rust column-normalization types explicit axis names.
  - Use `ColumnNormalization`, `ColumnNormalizationParams`, and
    `ColumnNormalizationStats` before introducing row normalization.

- [x] Add median centering and configurable quantile-range scaling.
  - Use exact linear quantiles, with the interquartile range as the default.
    Reuse `ColumnNormalizationParams` and preserve sparse duplicates and
    implicit zeros.
  - Dense and CSC fitting reuse one working column; CSR and Zarr use O(ncols)
    selection state and exact repeated scans without storage conversion.
    Document scan costs, workspace, and numerical differences from scikit-learn.

- [ ] Evaluate optional standard-normal calibration (`unit_variance`) for
  quantile-range scaling, as provided by `RobustScaler`.

- [ ] Consider explicitly approximate quantiles if storage-backed consumers need
  fewer fitting scans. Keep exact quantiles as the default.

- [ ] Extend min-max scaling beyond the default `[0, 1]` range (`MinMaxScaler`).
  - Evaluate a parameterized `feature_range` recipe that computes explicit
    centers and scales, including constant-column behavior and reuse on new
    data.
  - Evaluate optional clipping for `MinMaxScaler` and `MaxAbsScaler` separately
    from the affine normalization metadata.

- [ ] Evaluate weighted and incremental fitting of normalization parameters.
  - Cover weighted means and population standard deviations, as supported by
    `StandardScaler.fit(sample_weight=...)`. Weighted products and Gram matrices
    do not provide weighted fitting.
  - Consider mergeable statistics for `partial_fit` equivalents of
    `StandardScaler`, `MinMaxScaler`, and `MaxAbsScaler`. Streaming a fixed Zarr
    array is distinct from updating fitted parameters as new batches arrive.

- [x] Expose the existing Rust normalization choices in Python.
  - Include minimum and median centering and range, quantile-range,
    maximum-absolute, L1, and L2 scaling.

- [ ] Document intentional numerical differences from scikit-learn.
  - Compare IEEE NaN propagation with skipping missing values during fitting,
    and exact-zero scale replacement with near-constant feature handling.
  - Evaluate any compatibility options explicitly against the current numerical
    invariants before changing behavior.

- [ ] Assess broader transforms before extending the column-normalization scope.
  - `QuantileTransformer` maps features to uniform or normal distributions.
  - `PowerTransformer` supports Box-Cox and Yeo-Johnson transformations, with
    optional standardization afterward.
  - `KernelCenterer` requires kernel double centering, including fitted
    statistics for transforming test-versus-training kernel matrices.
  - PCA whitening mixes features rather than only scaling columns.
  - Evaluate separate operators, adapters, or downstream implementations for
    these methods. Specify sparse-storage and bounded-workspace contracts rather
    than representing them only through column centers and scales.

## Fitting and parameter reuse

Make fitted parameters useful independently of a matrix, building on
`LazyMatrix<M, P>` and the existing axis-specific parameter types. Add explicit
fitting and borrowed application while retaining the current constructors. Start
with single-axis fitting and reuse before introducing fitted pipelines.

- [x] Add explicit `.fit(&data)` methods to column and row recipes.
  - Return `ColumnNormalizationParams<F>` or `RowNormalizationParams<F>` without
    retaining the fitting data. Keep `LazyMatrix::new(data, recipe)` as the
    convenient operation that fits and wraps in one call.
  - Use the same fitting paths, validation, and numerical policies as the
    existing constructors.

- [x] Apply fitted parameters with explicit ownership and borrowed views.
  - Generalize `ColumnNormalizationParams<F, S = Vec<F>>` and
    `RowNormalizationParams<F, S = Vec<F>>` over parameter storage. Owned forms
    keep vectors; `.view()` returns `ColumnNormalizationParamsView<'a, F>` or
    `RowNormalizationParamsView<'a, F>` with borrowed slices. Keep fields
    private and construction validated.

  - Make column `.apply(data)` consume its receiver and retain its storage in
    the lazy operator. Borrow explicitly to share fitted values without copying:

    ```rust
    let params = ColumnNormalization::standardize().fit(&training)?;
    let train = params.view().apply(&training);
    let valid = params.view().apply(&validation);
    ```

  - Use `params.apply(data)` to move owned vectors. Add recipe
    `.fit_apply(data)` to fit and return an operator owning its parameters. Each
    form stores the supplied data handle, which can be an owned matrix,
    reference, or backend view. Returning owned parameters must not borrow
    temporary fitted values.

  - Name consuming row application `.apply_frozen(data)` on both owned
    parameters and views. Row parameters describe particular rows; fit new
    samples independently, even when row counts match.

  - Preserve `from_parts`, `from_column_parts`, and `from_row_parts`.
    Application uses supplied values unchanged, checks dimensions, and performs
    no fitting or data reads. Share kernels and existing axis capabilities
    between owned parameters and borrowed views, including submatrices and
    column eager conversion.

- [x] Validate fitting and application signatures with external consumers.
  - Check scalar inference for `f32` and `f64`, generic recipes and matrices,
    fallible backend errors, and owned and borrowed data and parameters.
  - Cover shared-parameter reuse, lifetimes, dimension validation, and parity
    with existing constructors across supported backends.

- [ ] Evaluate a public fitting trait for downstream estimators.
  - Keep explicit parameter construction available for custom fitting. Add an
    open trait only when it improves integration with generic consumers.
  - Specify scalar inference, output types, errors, and validation before
    opening the interface. Custom recipes should produce supported fitted
    parameter types; operator kernels and parameter invariants remain library
    controlled. Keep backend statistics capabilities and shared scan hooks
    independent of recipe extensibility.

## Row normalization

- [x] Add dense row normalization as a companion to column normalization.
  - `RowNormalization` offers independent centering and scaling choices matching
    `ColumnNormalization`, including standardization, min-max, and robust
    recipes. L1, L2, and maximum-absolute scaling leave zero rows unchanged.
  - `LazyMatrix` supports fitted and explicit row parameters, allocating and
    reusable forward and transpose vector products, and dense views across every
    supported backend release. Explicit parameters permit custom fitting.
  - Fit new samples' row parameters independently. Exact quantiles reuse one
    working row; products normalize entries before accumulation without
    materializing the design.

- [x] Unify row and column normalization under `LazyMatrix`.
  - Accept either axis's recipe in `LazyMatrix::new`, and select capabilities
    through the fitted parameter type. Remove the separate row wrapper.
  - Keep column-only use concise:
    `LazyMatrix::new(data, ColumnNormalization::standardize())?`. Retain
    specialized column-only kernels without row-parameter allocations or runtime
    axis dispatch.
  - Store typed metadata in `LazyMatrix<M, P>` and `EagerMatrix<M, P>`. Inspect
    it with `normalization()` and recover `(data, params)` with `into_parts()`.
    Rebuild from those parts with `LazyMatrix::from_parts`. Supply explicit
    vectors with `from_column_parts` or `from_row_parts`, inferring the
    dimension from the matrix.
  - Migrate Rust consumers, Python's native extension, tests, and documentation
    together, preferring the consistent API over backward compatibility.

- [x] Compose normalization recipes with `.then()`, read from left to right.
  - Support explicit fitting of an ordered recipe chain and retain
    `LazyMatrix::new(data, recipe)` as a convenience.

  - Support row-then-column standardization for the Duke breast-cancer
    preprocessing order, with separate frozen and new-sample application:

    ```rust
    let recipe = RowNormalization::standardize()
        .then(ColumnNormalization::standardize());
    let fitted = recipe.fit(&training)?;
    let train = fitted.view().apply_frozen(training.view());
    let valid = fitted.view().for_new_samples(validation.view())?;
    ```

  - Fit each stage on the logical output of all preceding stages. Reversing the
    recipe changes the fitted transformation; allow repeated axes as well. Later
    stages may change statistics established by earlier stages.

  - Preserve stage order in the types, and expose capabilities only when the
    backend and the entire sequence support them.

  - Apply the fitted transformation lazily, preserving sparse storage and
    bounded workspace without materializing intermediate design matrices.

- [x] Represent reusable fitted pipelines in the same order as their recipes.
  - Retain the axis-specific `RowNormalizationParams` and
    `ColumnNormalizationParams` types. Retain row recipes alongside fitted
    values so that the pipeline can fit new rows. Have `normalization()` expose
    the fitted sequence and `from_parts` apply it unchanged, without refitting.
  - Use `.apply_frozen(data)` to apply every saved stage unchanged to matching
    rows. Use a fallible `.for_new_samples(data)` to fit row stages at the
    correct points while reusing fitted training column stages. Avoid an
    ambiguous plain `.apply()` on mixed pipelines.
  - Follow single-axis ownership: consume owned fitted stages or explicit
    `.view()` values. Define lifetimes for both operations, including newly
    fitted row values and borrowed training column parameters. Keep all
    refitting in explicit preparation calls; ordinary products only execute
    fitted stages.
  - Implement only sequences whose intermediate statistics and products have
    supported bounded-workspace paths. Unsupported capabilities should fail at
    compile time; composable recipe syntax alone does not imply backend support.

- [x] Validate ordered normalization against an eager dense oracle.
  - Cover both axis orders, repeated stages, explicit and computed parameters,
    frozen application, new-sample row fitting with reused column parameters,
    borrowed and owning operators, the adjoint identity, and existing zero,
    empty, and nonfinite cases across supported backends. Preserve the existing
    numerical invariants.

- [x] Extend row normalization (`RowNormalization`) to sparse storage.
  - Combine duplicate cells before nonlinear statistics, include implicit zeros,
    and preserve CSC and CSR storage with bounded workspace.

- [x] Stream row fitting and products over Zarr chunks without materializing the
  array. Preserve fill values, exclude padding, and account for outer
  shards.

- [x] Add a public row-statistics capability when sparse or storage-backed
  fitting needs it, keeping shared fitting hooks and reference forwarding.
  - `RowStats::row_normalization_stats` returns raw row centers and scales.
    References and checked sparse wrappers forward the hook without refitting.

- [ ] Extend `LazyMatrix` capabilities to row and composed normalization.

  - [x] Add fused vector products.
  - [x] Add vector products with caller-owned workspace.
    - Row and chain fused calls reuse a product vector of length `nrows` for
      forward products or `ncols` for transpose products. Sparse traversal and
      chunk decoding retain their internal bounded workspace.
  - [x] Add batched products.
    - Row normalization and column-only chains traverse dense, sparse, and
      chunked storage. Mixed chains require dense indexed access. Sparse batches
      scan once per panel of up to 16 responses with bounded workspace; dense
      and chunked batches scan once for all responses.
  - [x] Add borrowed logical rows.
    - `SparseRows` views borrow raw storage and fitted row or chain parameters.
      Logical reads coalesce duplicates before applying stages in order; sparse
      background and correction helpers allocate nothing. Explicit mixed chains
      can borrow CSR rows independently of fitting and product capabilities.
  - [x] Add borrowed logical columns.
    - `RawColumns` and `SparseColumns` views borrow storage and fitted row or
      chain parameters, including explicit mixed chains on sparse storage.
      Logical operations coalesce duplicates before applying stages in order
      and include transformed implicit zeros. Sparse backgrounds vary by row;
      borrowing, individual reads, and correction helpers allocate nothing.
  - [x] Add weighted operations.
    - Weighted column sums and Gram products preserve fitted stage order and
      include transformed implicit zeros. Row normalization and column-only
      chains support dense, sparse, and chunked storage; mixed chains require
      dense indexed access. Full-storage sums scan once. Sparse sums and Gram
      products use bounded logical panels without materializing the design.
  - [ ] Add eager conversion.

- [ ] Expose row normalization in Python, including independent fitting for new
  samples and explicit parameters.

- [ ] Benchmark row fitting and products across storage layouts and document
  scan counts, workspace, and tradeoffs against eager normalization.

## Operator algebra

- [x] Evaluate a borrowed transpose operator view with a concrete consumer.
  - `Transpose::new(&operator)` swaps dimensions and vector, reusable-output,
    fused, and batched products without copying or changing fitted
    normalization. The view exposes no sparse row or column borrowing
    capabilities.
  - The [ridge regression example](examples/ridge_primal_dual.rs) reuses one
    normal-equation solver for primal and dual systems, choosing the smaller
    system without materializing a Gram matrix.

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

## Column kernel performance

- [x] Investigate the dense column-reduction gap measured in Shrinkage.
  - The [investigation](benches/column_reductions/README.md) reproduces the gap
    on this checkout with Rust 1.89 and 1.98, at one and 16 codegen units.
    Contiguous eight-accumulator prototypes close the dot gap and accelerate
    sums and squared norms. Separate native Shrinkage fits improve by 39–56%,
    with independently checked objectives, KKT violations, and certificates.
  - Reproducible probes, raw timings, numerical counterexamples, and isolated
    prototype patches accompany the report.

- [x] Implement production dense column-reduction fast paths for ndarray.
  - `RawColumn::dense_values` and `VectorView::contiguous_values` expose
    optional borrowed slices. Contiguous ndarray columns use shared
    eight-accumulator sums, dots, and directly centered squared norms. Default
    paths preserve strided access, sparse duplicates, implicit zeros, and O(nnz)
    products.
  - Reduction order follows the usual backend convention. Shared tests cover
    `f32`, `f64`, empty inputs, short tails, nonfinite values, and large
    offsets. All backend releases, feature combinations, parallelism, and BLAS
    checks pass.
  - The [implementation measurements](benches/column_reductions/README.md)
    include kernel assembly, fused-summary controls, and separate native and
    Python Shrinkage fits with independently checked objectives, KKT violations,
    and certificate feasibility. Python timings include owned input copies.

- [x] Extend dense column-reduction fast paths to faer and nalgebra.
  - faer 0.22–0.24 and nalgebra 0.32–0.35 expose borrowed contiguous columns and
    owned and borrowed vectors through both reduction hooks. Slice access takes
    O(1) time and preserves view offsets, lengths, and logical order.
    Noncontiguous views retain their existing paths.
  - Adapter tests check hook availability and borrowed ranges, including
    mutable, strided, and reversed views. The shared reduction suite covers
    native vectors, `f32`, `f64`, empty inputs, short tails, normalization, and
    exceptional values.
  - The [adapter
    measurements](benches/column_reductions/README.md#faer-and-nalgebra-adapter-measurements)
    compare baseline, candidate, and native reductions in one exclusive compute
    window. Contiguous sums improve by 4.6–4.7 times, squared norms by 4.3
    times, and cached dots by about twice. Strided controls retain baseline
    performance, and all numerical comparisons pass. Raw data and reproduction
    commands are included in the report.

- [x] Extend contiguous reductions to vector sums and weighted columns.
  - Reuse the shared kernels for `VectorView::sum`, weighted column dots, and
    weighted squared norms. Preserve indexed fallbacks and normalize entries
    before squaring to support extreme scales and signed weights.
  - The
    [measurements](benches/column_reductions/README.md#vector-sums-and-weighted-column-reductions)
    include uncached dots, cached-dot controls, and strided layouts. Shared
    tests cover both scalar types, borrowed ranges, short tails, and nonfinite
    values.

- [x] Reuse contiguous kernels in the remaining vector reductions.
  - Core `Vec` and contiguous ndarray reductions, faer sums and dots, and
    nalgebra's `sum_entries` and `dot_slice` reuse the shared kernels. Faer
    retains its scaled L2 norm, and nalgebra retains its native dots and norms.
    Ndarray's strided and reversed views preserve their original fallbacks.
  - Shared tests cover both scalar types, empty inputs, reduction tails,
    nonfinite values, dimension checks, and extreme faer norms across releases.
  - The
    [measurements](benches/column_reductions/README.md#remaining-solver-facing-vector-reductions)
    include native controls, short and long vectors, borrowed ranges, and
    strided layouts. Length-5,000 sums improve by 4.4–5.1 times, and the
    existing ridge consumer takes 6–14% less time. Raw data, numerical checks,
    short-vector tradeoffs, and reproduction commands accompany the report.

- [x] Investigate reductions over reversed and strided columns.
  - The [measurements](benches/column_reductions/README.md#reversed-and-strided-column-investigation)
    compare memory-order sums and centered norms with indexed and pointer-based
    independent accumulators on ndarray, faer, and nalgebra.
  - Reversed unit-stride reductions benefit from memory order. General strided
    kernels help smaller working sets but do not consistently help larger
    stride-three or row-major views. The report records implementation options
    that preserve logical borrowing contracts.
  - Separate reversed, positive- and negative-stride, and row-major measurements
    cover both scalar types, short tails, and multiple working-set sizes. Logical
    dots and weighted operations pass independent alignment checks. Raw data,
    numerical counterexamples, and reproduction commands accompany the report.

- [x] Implement memory-order reductions for reversed unit-stride columns.
  - Ndarray and faer reuse the shared sum and directly centered norm kernels.
    `RawColumn::centered_sum_squares` separates reduction order from logical
    borrowing. Sparse and general strided fallbacks retain their behavior.
  - The [production measurements](benches/column_reductions/README.md#reversed-unit-stride-production-reductions)
    compare kernels and complete Shrinkage fits. Reversed fits use 14–29% less
    time, and all numerical checks pass. Tests exercise every supported release,
    both scalar types, borrowed ranges, nonfinite values, and logical alignment.

- [x] Benchmark contiguous fast paths for logical column updates.
  - The [investigation](benches/column_reductions/README.md#contiguous-logical-column-update-investigation)
    compares compiled production loops, an optional mutable-slice capability,
    and a source-only ndarray slice branch. Faer and nalgebra already vectorize
    indexed destination updates; ndarray benefits from borrowing its source.
  - Source-only ndarray updates improve length-5,000 `f64` kernels by 2.3–2.9
    times and the smaller hybrid fit with 64 active coefficients by 10.1%. The
    evidence favors the existing vector API.
  - Ndarray 0.15–0.17 now uses the source-slice branch. Regression tests cover
    both scalar types, borrowed ranges, strided destinations, and floating-point
    update order. The mutable-slice capability remains experimental.
  - Numerical and allocation checks preserve borrowing, strided destinations,
    arithmetic order, and sparse duplicate behavior. Raw data, assembly, locked
    consumer dependencies, sampling limits, and reproduction commands accompany
    the report.

- [x] Evaluate fused dense kernels for uncached column dots.
  - The
    [investigation](benches/column_reductions/README.md#fused-uncached-column-dot-investigation)
    compares three allocation-free prototypes with production, separate
    optimized reductions, and cached-sum controls across vector lengths and
    working-set sizes, both scalar types, and generic and CPU-specific builds.
  - Fusion improves the largest tested vectors by 1.29–1.71 times but regresses
    cache-resident cases. Retain the separate production kernels and cached
    methods; length alone does not explain the crossover.
  - Eight accumulators preserve production arithmetic, including cancellation,
    overflow, and nonfinite behavior. Cached sparse products retain O(nnz) work.
    Raw data, assembly, numerical counterexamples, and reproduction commands
    accompany the report. The sum-and-norm summary API remains unchanged.

- [x] Explain the remaining cached-dot gap against native faer.
  - The
    [investigation](benches/column_reductions/README.md#cached-column-dot-gap-against-native-faer)
    identifies narrower packed arithmetic in the shared kernel and faer's
    runtime AVX2/FMA dispatch. Eight methods cover both scalar types, borrowed
    offsets, multiple lengths, and generic and CPU-specific builds.
  - Native faer improves the larger CPU-specific Lasso fits by 19–20%, but gives
    no gain for those generic fits. Explicit SIMD also has regressions, and FMA
    changes numerical behavior. Retain the shared default and require downstream
    evidence before adding a targeted specialization.
  - Raw timings, complete consumer fits, numerical diagnostics, assembly, source
    hashes, and reproduction commands accompany the report. The dependency-light
    core remains unchanged.

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
    borrows the logical parent operator without refitting. `Submatrix<'a, O>`
    exposes its parent with `as_inner()` and its source ranges with
    `row_range()` and `column_range()`.
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
- [x] Generalize submatrix vector products to row normalization and supported
  chains through `RectangularProducts`.
  - Preserve the parent's fitted transformation and original coordinates,
    including row stages fitted using columns outside the selection. Retain
    column-only weighted operations, eager conversion, and borrowing.
- [x] Extend submatrices to bounded Zarr reads and weighted operations.
  - Zarr selections stream forward and transpose products, eager conversion, and
    weighted column sums through intersecting chunks without refitting. Empty
    ranges and zero-alpha products skip storage reads.
  - Weighted Gram products normalize entries in bounded panels across all
    backend releases, combining sparse duplicates before multiplication. Weights
    use local row indices; signed and nonfinite values retain IEEE behavior.
  - Shared oracle tests and injected storage failures cover selection semantics
    and errors. [Measurements](benches/submatrix/README.md) document bounded
    workspace and the cost of repeated scans and chunk decoding.
- [ ] Evaluate arbitrary index selections after defining ordering,
  duplicate-index, mapping, allocation, and complexity contracts.
- [x] Add raw indexed-row selection for fitting normalization on a subset.
  - `IndexedRows::new(source, indices)` selects raw rows before fitting, without
    copying dense, CSC, or CSR designs. It supports every in-memory backend
    release and accepts owned storage, references, and backend views. Existing
    `Submatrix` views continue to restrict an already normalized matrix.
  - Provides `MatrixShape`, `MatrixErrorType`, `ColumnStats`, its combined
    `normalization_stats` hook, and native allocating, reusable, and fused
    forward and transpose products. `RawColumns` requires natural parent column
    borrowing; sparse rows retain their original slices. Statistics and products
    combine duplicate cells and preserve implicit zeros.
  - Indices must be strictly increasing, unique, and in bounds. Empty selections
    are valid. The view borrows indices and exposes local-to-source mapping.
    [Measurements](benches/indexed_rows/README.md) document ownership,
    allocation, scan costs, and the tradeoff between borrowed traversal and
    explicit copies.
  - Shared tests compare all normalization choices and products with explicit
    raw subsets, including sparse inputs, empty selections, local indices,
    unselected outliers, and held-out parameter reuse without refitting.
    Borrowed sparse views also cover duplicates, stored zeros, offsets, and
    spare capacity. Folds, scoring, path workspaces, and model selection remain
    in Shrinkage.
- [ ] Expose submatrix selection in Python after defining indexing and
  ownership.

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
