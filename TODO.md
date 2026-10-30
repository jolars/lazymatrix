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
    maximum-absolute, L1, and L2 scaling. Coordinate this with the reusable
    fitted-parameter task above.

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

## Row normalization

- [x] Add dense row normalization as a companion to column normalization.
  - `RowNormalization` offers independent centering and scaling choices matching
    `ColumnNormalization`, including standardization, min-max, and robust
    recipes. L1, L2, and maximum-absolute scaling leave zero rows unchanged.
  - `LazyMatrix` supports fitted and explicit row parameters, allocating
    and reusable forward and transpose vector products, and dense views across
    every supported backend release. Explicit parameters permit custom fitting.
  - Fit new samples' row parameters independently. Exact quantiles reuse one
    working row; products normalize entries before accumulation without
    materializing the design.

- [x] Unify row and column normalization under `LazyMatrix`.
  - Accept either axis's recipe in `LazyMatrix::new`, and select capabilities
    through the fitted parameter type. Remove the separate row wrapper.
  - Keep column-only use concise:
    `LazyMatrix::new(data, ColumnNormalization::standardize())?`.
    Retain specialized column-only kernels without row-parameter allocations or
    runtime axis dispatch.
  - Store typed metadata in `LazyMatrix<M, P>` and `EagerMatrix<M, P>`. Inspect it
    with `normalization()` and recover `(data, params)` with `into_parts()`.
    Rebuild from those parts with `LazyMatrix::from_parts`. Supply explicit
    vectors with `from_column_parts` or `from_row_parts`, inferring the
    dimension from the matrix.
  - Migrate Rust consumers, Python's native extension, tests, and documentation
    together, preferring the consistent API over backward compatibility.

- [ ] Compose normalization recipes with `.then()`, read from left to right.
  - Accept an ordered recipe chain in `LazyMatrix::new`.
  - Support row-then-column standardization for the Duke breast-cancer
    preprocessing order:

    ```rust
    let recipe = RowNormalization::standardize()
        .then(ColumnNormalization::standardize());
    let matrix = LazyMatrix::new(data, recipe)?;
    ```

  - Fit each stage on the logical output of all preceding stages. Reversing the
    recipe changes the fitted transformation; allow repeated axes as well.
    Later stages may change statistics established by earlier stages.
  - Preserve stage order in the types, and expose capabilities only when the
    backend and the entire sequence support them.
  - Apply the fitted transformation lazily, preserving sparse storage and
    bounded workspace without materializing intermediate design matrices.

- [ ] Represent fitted parameters in the same order as their recipes.
  - Retain the axis-specific `RowNormalizationParams` and
    `ColumnNormalizationParams` types. Have `normalization()` expose the fitted
    sequence and `from_parts` apply it unchanged, without refitting.
  - Define an explicit path for new samples that fits their row stages at the
    correct points in the sequence while reusing fitted training column stages.
    Keep this distinct from applying frozen parameters to the same rows.

- [ ] Validate ordered normalization against an eager dense oracle.
  - Cover both axis orders, repeated stages, explicit and computed parameters,
    new samples, the adjoint identity, and existing zero, empty, and nonfinite
    cases across supported backends. Preserve the existing numerical invariants.

- [ ] Extend row normalization (`Normalizer` and `normalize(axis=1)`) to sparse
  storage.
  - Combine duplicate cells before nonlinear statistics, include implicit
    zeros, and preserve CSC and CSR storage with bounded workspace.

- [ ] Stream row fitting and products over Zarr chunks without materializing
  the array. Preserve fill values, exclude padding, and account for outer shards.

- [ ] Add a public row-statistics capability when sparse or storage-backed
  fitting needs it, keeping shared fitting hooks and reference forwarding.

- [ ] Extend `LazyMatrix` capabilities to row and composed normalization:
  fused and workspace products, batched products, borrowed logical rows and
  columns, weighted operations, and eager conversion.

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
