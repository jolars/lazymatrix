# Changelog

## [0.6.0](https://github.com/jolars/lazymatrix-rs/compare/v0.5.0...v0.6.0) (2026-10-08)

### Features

- add eager normalization and dense conversion ([`519190a`](https://github.com/jolars/lazymatrix-rs/commit/519190ad44fbfb82f410eb55c3e27863afcad0d6))
- add normalized product hooks and bounded CSR Gram kernels ([`42a6aef`](https://github.com/jolars/lazymatrix-rs/commit/42a6aef6f053306de7f8a6391ba43f837a25230e))

## [0.5.0](https://github.com/jolars/lazymatrix-rs/compare/v0.4.0...v0.5.0) (2026-10-07)

### Features

- prototype fallible rectangular block reads ([`82e1581`](https://github.com/jolars/lazymatrix-rs/commit/82e1581e30da96511722572e72718174837fa0d1))
- extend fused operators across backends ([`91e06fa`](https://github.com/jolars/lazymatrix-rs/commit/91e06fa2d6e341fe11ed80fd26dbda274e208fbc))
- add borrowed normalized row views ([`effa08c`](https://github.com/jolars/lazymatrix-rs/commit/effa08c1f0535d06d564f4a3bd36be9309d3bbb9))
- prototype fused scaled operators ([`271458e`](https://github.com/jolars/lazymatrix-rs/commit/271458e5f0c774d161f08b7ba8a237b458097b8a))

### Performance Improvements

- speed up sparse centered Gram products ([`ced9620`](https://github.com/jolars/lazymatrix-rs/commit/ced962073b2228e98828129b4b75c9ec12d52f1c))

## [0.4.0](https://github.com/jolars/lazymatrix-rs/compare/v0.3.0...v0.4.0) (2026-09-25)

### Features
- add implicit intercept operators ([`800a3fc`](https://github.com/jolars/lazymatrix-rs/commit/800a3fcde56c489954735b72011ec1faadffb93b))
- add weighted Gram matrix products ([`ea63d8c`](https://github.com/jolars/lazymatrix-rs/commit/ea63d8c789738c08d89967bdf599e058a0c866ad))

### Bug Fixes
- preserve implicit weights in column norms ([`51204f5`](https://github.com/jolars/lazymatrix-rs/commit/51204f5016ccd547635b1ce9e1822b75e4554314))
- clear sparse product destinations ([`d821440`](https://github.com/jolars/lazymatrix-rs/commit/d8214408dbebfb0bb37591591efbdea5167ed7aa))
- make backend version features additive ([`e2ae23e`](https://github.com/jolars/lazymatrix-rs/commit/e2ae23eb2e6e71d82a338a47e8a843f1d02776d7))

## [0.3.0](https://github.com/jolars/lazymatrix-rs/compare/v0.2.0...v0.3.0) (2026-09-22)

### Breaking changes
- add out-of-core matrix support ([`c99caa6`](https://github.com/jolars/lazymatrix-rs/commit/c99caa6b79b284f03afc0ce2e0e4b3ffdce7355b))
- add version-specific backend features ([`b86950e`](https://github.com/jolars/lazymatrix-rs/commit/b86950e4fe35bb5a231a9bfb0c6eeb317ab4b602))

### Features
- add out-of-core matrix support ([`c99caa6`](https://github.com/jolars/lazymatrix-rs/commit/c99caa6b79b284f03afc0ce2e0e4b3ffdce7355b))
- add `SparseRows` for CSR backends ([`4114935`](https://github.com/jolars/lazymatrix-rs/commit/4114935b8d9492491eccec80f81e1d9f67437209))
- add `sprs` backend ([`fe40fee`](https://github.com/jolars/lazymatrix-rs/commit/fe40feebef212069af87467deb4218c8f8840a5f))
- add version-specific backend features ([`b86950e`](https://github.com/jolars/lazymatrix-rs/commit/b86950e4fe35bb5a231a9bfb0c6eeb317ab4b602))
- add `ndarray` backend ([`231a622`](https://github.com/jolars/lazymatrix-rs/commit/231a622ba38df79bef12410b3569da19073080fb))

## [0.2.0](https://github.com/jolars/lazymatrix-rs/compare/v0.1.0...v0.2.0) (2026-09-16)

### Features
- add reusable-output products ([`9df246a`](https://github.com/jolars/lazymatrix-rs/commit/9df246a3cb0dd5a49291a67f0ff696f754ef7e39))
- add opt-in parallelism ([`3f94728`](https://github.com/jolars/lazymatrix-rs/commit/3f94728cbbdac6b10cce896d964750fad77e3104))
- add normalization statistics ([`bc37b8f`](https://github.com/jolars/lazymatrix-rs/commit/bc37b8fce400c45de9092a8884bae3cf08b6fba4))
- generalize column views ([`8fef143`](https://github.com/jolars/lazymatrix-rs/commit/8fef14307d499b78724df9771b8b64939f6501cf))
- add weighted column products ([`8fe5d79`](https://github.com/jolars/lazymatrix-rs/commit/8fe5d79cab891b340f72e1ad9235fe576b8729d7))
- add lazy column operations ([`06dc82a`](https://github.com/jolars/lazymatrix-rs/commit/06dc82a3dfe50082486e18697881d7fede056cc2))
- add vector algebra traits ([`a849b79`](https://github.com/jolars/lazymatrix-rs/commit/a849b79949b53154183537cccbbfd9f5e60995c6))
- add sparse column views ([`a300375`](https://github.com/jolars/lazymatrix-rs/commit/a300375d2bc5fe5b2018b333bfa7ea1bc85f87c0))
- infer dimensions from matrix ([`1c0e158`](https://github.com/jolars/lazymatrix-rs/commit/1c0e15848cd6fa81a8d3b2cba79cb2b78302088c))
- **examples:** least-squares gradient descent on the lazy operator ([`451c99e`](https://github.com/jolars/lazymatrix-rs/commit/451c99e0be095e1c99c86f16fc445c3c7f7d3cf4))
- lazy column-normalized matrix operator ([`45ab747`](https://github.com/jolars/lazymatrix-rs/commit/45ab747f16303ea8bdfeabe29bca4a5a28ad8220))

### Bug Fixes
- define nonfinite statistics policy ([`19e0a10`](https://github.com/jolars/lazymatrix-rs/commit/19e0a105c4af15119607eaddde77133e3f8824e0))
- stabilize column standard deviations ([`2e97084`](https://github.com/jolars/lazymatrix-rs/commit/2e97084c85602443d40ea8ed4fb3596222e530ce))
