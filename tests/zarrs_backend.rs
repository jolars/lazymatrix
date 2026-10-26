#![cfg(feature = "zarrs_all")]
// Each enabled release must run the full storage suite with its own array types.
#![allow(clippy::duplicate_mod)]
//! Chunked storage behavior and parity with the shared dense oracle.

#[path = "common/backend_aliases.rs"]
mod backend_aliases;
#[path = "common/runner.rs"]
mod common;

#[cfg(feature = "zarrs_v0_22")]
#[path = "common/zarrs_suite_v0_22.rs"]
mod v0_22;
#[cfg(feature = "zarrs_v0_23")]
#[path = "common/zarrs_suite_v0_23.rs"]
mod v0_23;
