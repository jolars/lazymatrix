# Python API

LazyMatrix applies column or row normalization during matrix operations on NumPy
arrays and SciPy sparse matrices. Start with `standardize` for mean centering and
standard-deviation scaling, or use `normalize` to choose other methods. Both
accept `axis=0` for columns or `axis=1` for rows and return a `LazyMatrix`.
`normalize(X)` defaults to the same standardization as `standardize(X)`.
Direct `LazyMatrix` construction accepts the same options as `normalize`.

`ColumnNormalization` and `RowNormalization` retain reusable fitting recipes.
Their `fit` methods return parameters, and `fit_apply` fits and wraps raw input
in a lazy matrix. `ColumnNormalizationParams.transform` reuses training column
parameters on new samples. `RowNormalizationParams.apply_frozen` reuses saved
values on the same rows; fit new samples independently with the row recipe.
Both parameter classes accept explicit centers and scales, with `ncols` or
`nrows` retaining the normalized dimension when components are inactive.

The optional `lazymatrix.torch.NormalizedLinear` layer applies fitted
normalization in a trainable PyTorch first layer.

See the [Python guide](https://lazymatrix.org/guide/python) for installation,
worked examples, and saving fitted parameters.
