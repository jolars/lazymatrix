pub(crate) mod support;

#[cfg(feature = "faer_all")]
mod faer;

#[cfg(feature = "nalgebra_all")]
mod nalgebra;

#[cfg(feature = "ndarray_all")]
mod ndarray;

#[cfg(feature = "sprs_all")]
pub(crate) mod sprs;

#[cfg(feature = "zarrs_all")]
pub(crate) mod zarrs;
