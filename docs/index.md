---
layout: home
hero:
  name: LazyMatrix
  text: Give your matrix a break.
  tagline:
    Normalized design matrices for Rust and Python. Keep sparse matrices sparse
    and materialize at will.
  image:
    light: /logo.png
    dark: /logo-dark.png
    alt: A matrix relaxing in a beach chair under a palm tree.
  actions:
    - theme: brand
      text: What is LazyMatrix?
      link: /guide/what-is-lazymatrix
    - theme: alt
      text: Get started
      link: /guide/getting-started
    - theme: alt
      text: Source code
      link: https://github.com/jolars/lazymatrix
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
