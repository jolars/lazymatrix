# Python API

LazyMatrix applies column normalization during matrix operations on NumPy arrays
and SciPy sparse matrices. Start with `standardize` for mean centering and
standard-deviation scaling, or use `LazyMatrix` to choose other methods.
`NormalizationParams` stores fitted centers and scales for use with new data.

The optional `lazymatrix.torch.NormalizedLinear` layer applies fitted
normalization in a trainable PyTorch first layer.

See the [Python guide](https://lazymatrix.org/guide/python) for installation,
worked examples, and saving fitted parameters.
