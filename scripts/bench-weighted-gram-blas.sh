#!/usr/bin/env bash
set -euo pipefail

: "${OPENBLAS_LP64_LIB:?Set OPENBLAS_LP64_LIB to the directory containing LP64 OpenBLAS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$OPENBLAS_NUM_THREADS}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-L native=$OPENBLAS_LP64_LIB -l openblas -C link-arg=-Wl,-rpath,$OPENBLAS_LP64_LIB"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target}/gram-blas"
printf 'OpenBLAS threads: %s; OpenMP threads: %s\n' "$OPENBLAS_NUM_THREADS" "$OMP_NUM_THREADS"
exec cargo bench --locked --bench weighted_gram --features 'ndarray-blas,sprs' -- "$@"
