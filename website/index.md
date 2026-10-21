---
layout: home
hero:
  name: LazyMatrix
  text: Give your matrix a break.
  tagline: Normalized design matrices for Rust. Keep sparse matrices sparse,
    reuse fitted parameters, and choose when to materialize.
  image:
    light: /logo.png
    dark: /logo-dark.png
    alt: A matrix relaxing in a beach chair under a palm tree.
  actions:
    - theme: brand
      text: Get started
      link: /guide/getting-started
    - theme: alt
      text: Compare lazy and eager
      link: /benchmarks
    - theme: alt
      text: API documentation
      link: https://docs.rs/lazymatrix
features:
  - title: Preserve sparse storage
    details: Apply centering and scaling through matrix operations without turning structural zeros into stored entries.
  - title: Choose your backend
    details: Use faer, nalgebra, ndarray, sprs, or chunked Zarr arrays through a shared trait interface.
  - title: Materialize when it helps
    details: Convert to dense storage explicitly, normalize writable dense storage in place, and retain fitted parameters for prediction.
---

<script setup>
import NormalizationDemo from "./.vitepress/theme/components/NormalizationDemo.vue";
</script>

## A normalized matrix, used as an operator

LazyMatrix represents

$$
  \widetilde{X} = (X - \mathbf{1}c^\top)S^{-1}
$$

without storing the centered matrix. Here, $c$ holds the column centers, and $S$
is diagonal with the column scales. Products use the original storage and apply
normalization algebraically.

Here is what each one represents, with $C = \mathbf{1}c^\top$:

<NormalizationDemo />

## Example

First, add the crate to your project with the `ndarray` feature:

```sh
cargo add lazymatrix --features ndarray
cargo add ndarray@0.17
```

Then, use the `LazyMatrix` type to wrap an existing matrix and apply
normalization. Here, we center by the mean and scale by the maximum absolute
value of each column. Then, we multiply the normalized matrix by a vector.

```rust
use lazymatrix::{Centering, LazyMatrix, MatVec, Normalization, Scaling};
use ndarray::array;

let x = array![[3.0, 6.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 9.0]];
let lazy = LazyMatrix::new(
    x.view(),
    Normalization::new(Centering::Mean, Scaling::MaxAbs),
).unwrap();

let y = lazy.matvec(&array![1.0, 1.0, 1.0]).unwrap();
assert_eq!(y, array![1.0, -0.5, -0.5]);
```

## Where to go next

Read the [guide](/guide/getting-started) for installation and eager conversion,
or see [how it works](/guide/how-it-works) for the normalization identities. The
[benchmark comparisons](/benchmarks) explain the costs of construction, repeated
operations, and additional memory.

LazyMatrix is in early development, and breaking changes are expected. Find
releases on [crates.io](https://crates.io/crates/lazymatrix) and source code on
[GitHub](https://github.com/jolars/lazymatrix).
