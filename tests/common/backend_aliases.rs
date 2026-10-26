#![allow(clippy::single_component_path_imports, unused_imports)]

// Examples and benchmarks select one release for their demonstrations.

#[cfg(feature = "sprs_v0_11")]
pub(crate) use sprs;

#[cfg(all(
    feature = "nalgebra_v0_32",
    not(any(
        feature = "nalgebra_v0_33",
        feature = "nalgebra_v0_34",
        feature = "nalgebra_v0_35"
    ))
))]
pub(crate) use nalgebra_0_32 as nalgebra;

#[cfg(all(
    feature = "nalgebra_v0_32",
    not(any(
        feature = "nalgebra_v0_33",
        feature = "nalgebra_v0_34",
        feature = "nalgebra_v0_35"
    ))
))]
pub(crate) use nalgebra_sparse_0_9 as nalgebra_sparse;

#[cfg(all(
    feature = "nalgebra_v0_33",
    not(any(feature = "nalgebra_v0_34", feature = "nalgebra_v0_35"))
))]
pub(crate) use nalgebra_0_33 as nalgebra;

#[cfg(all(
    feature = "nalgebra_v0_33",
    not(any(feature = "nalgebra_v0_34", feature = "nalgebra_v0_35"))
))]
pub(crate) use nalgebra_sparse_0_10 as nalgebra_sparse;

#[cfg(all(feature = "nalgebra_v0_34", not(feature = "nalgebra_v0_35")))]
pub(crate) use nalgebra_0_34 as nalgebra;

#[cfg(all(feature = "nalgebra_v0_34", not(feature = "nalgebra_v0_35")))]
pub(crate) use nalgebra_sparse_0_11 as nalgebra_sparse;

#[cfg(feature = "nalgebra_v0_35")]
pub(crate) use nalgebra;

#[cfg(feature = "nalgebra_v0_35")]
pub(crate) use nalgebra_sparse;

#[cfg(all(
    feature = "ndarray_v0_15",
    not(any(feature = "ndarray_v0_16", feature = "ndarray_v0_17"))
))]
pub(crate) use ndarray_0_15 as ndarray;

#[cfg(all(feature = "ndarray_v0_16", not(feature = "ndarray_v0_17")))]
pub(crate) use ndarray_0_16 as ndarray;

#[cfg(feature = "ndarray_v0_17")]
pub(crate) use ndarray;

#[cfg(all(
    feature = "faer_v0_22",
    not(any(feature = "faer_v0_23", feature = "faer_v0_24"))
))]
pub(crate) use faer_0_22 as faer;

#[cfg(all(
    feature = "faer_v0_22",
    not(any(feature = "faer_v0_23", feature = "faer_v0_24"))
))]
pub(crate) use faer_traits_0_22 as faer_traits;

#[cfg(all(feature = "faer_v0_23", not(feature = "faer_v0_24")))]
pub(crate) use faer_0_23 as faer;

#[cfg(all(feature = "faer_v0_23", not(feature = "faer_v0_24")))]
pub(crate) use faer_traits_0_23 as faer_traits;

#[cfg(feature = "faer_v0_24")]
pub(crate) use faer;

#[cfg(feature = "faer_v0_24")]
pub(crate) use faer_traits;

#[cfg(feature = "zarrs_v0_22")]
#[path = "zarrs_compat_v0_22.rs"]
pub(crate) mod zarrs_compat_v0_22;
#[cfg(feature = "zarrs_v0_23")]
#[path = "zarrs_compat_v0_23.rs"]
pub(crate) mod zarrs_compat_v0_23;
#[cfg(feature = "zarrs_all")]
pub(crate) use zarrs_compat::zarrs;
#[cfg(all(feature = "zarrs_v0_22", not(feature = "zarrs_v0_23")))]
pub(crate) use zarrs_compat_v0_22 as zarrs_compat;
#[cfg(feature = "zarrs_v0_23")]
pub(crate) use zarrs_compat_v0_23 as zarrs_compat;

#[cfg(feature = "zarrs_v0_23")]
pub(crate) use ::zarrs as zarrs_0_23;
#[cfg(feature = "zarrs_v0_22")]
pub(crate) use ::zarrs_0_22;

// Correctness tests name each release explicitly so feature unification cannot hide it.
#[cfg(feature = "faer_v0_24")]
pub(crate) use ::faer as faer_0_24;
#[cfg(feature = "faer_v0_22")]
pub(crate) use ::faer_0_22;
#[cfg(feature = "faer_v0_23")]
pub(crate) use ::faer_0_23;
#[cfg(feature = "faer_v0_24")]
pub(crate) use ::faer_traits as faer_traits_0_24;
#[cfg(feature = "faer_v0_22")]
pub(crate) use ::faer_traits_0_22;
#[cfg(feature = "faer_v0_23")]
pub(crate) use ::faer_traits_0_23;
#[cfg(feature = "nalgebra_v0_35")]
pub(crate) use ::nalgebra as nalgebra_0_35;
#[cfg(feature = "nalgebra_v0_32")]
pub(crate) use ::nalgebra_0_32;
#[cfg(feature = "nalgebra_v0_33")]
pub(crate) use ::nalgebra_0_33;
#[cfg(feature = "nalgebra_v0_34")]
pub(crate) use ::nalgebra_0_34;
#[cfg(feature = "nalgebra_v0_35")]
pub(crate) use ::nalgebra_sparse as nalgebra_sparse_0_12;
#[cfg(feature = "nalgebra_v0_32")]
pub(crate) use ::nalgebra_sparse_0_9;
#[cfg(feature = "nalgebra_v0_33")]
pub(crate) use ::nalgebra_sparse_0_10;
#[cfg(feature = "nalgebra_v0_34")]
pub(crate) use ::nalgebra_sparse_0_11;
#[cfg(feature = "ndarray_v0_17")]
pub(crate) use ::ndarray as ndarray_0_17;
#[cfg(feature = "ndarray_v0_15")]
pub(crate) use ::ndarray_0_15;
#[cfg(feature = "ndarray_v0_16")]
pub(crate) use ::ndarray_0_16;
