#!/usr/bin/env bash
set -euo pipefail

: "${OPENBLAS_LP64_LIB:?Set OPENBLAS_LP64_LIB to the directory containing LP64 OpenBLAS}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-L native=$OPENBLAS_LP64_LIB -l openblas -C link-arg=-Wl,-rpath,$OPENBLAS_LP64_LIB"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target}/normalization-allocations"
exec cargo run --release --locked --manifest-path benches/normalization_allocations/Cargo.toml
