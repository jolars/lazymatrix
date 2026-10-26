#!/usr/bin/env bash
set -euo pipefail

faer_versions=(faer_v0_22 faer_v0_23 faer_v0_24)
nalgebra_versions=(nalgebra_v0_32 nalgebra_v0_33 nalgebra_v0_34 nalgebra_v0_35)
ndarray_versions=(ndarray_v0_15 ndarray_v0_16 ndarray_v0_17)
sprs_versions=(sprs_v0_11)
zarrs_versions=(zarrs_v0_22 zarrs_v0_23)
versions=("${faer_versions[@]}" "${nalgebra_versions[@]}" "${ndarray_versions[@]}" "${sprs_versions[@]}" "${zarrs_versions[@]}")

test_features() {
    cargo test --locked --no-default-features "$@"
}

fixture_manifest=tests/feature_unification/Cargo.toml

test_fixtures() {
    cargo test --manifest-path "$fixture_manifest" --locked --no-default-features "$@"
}

test_version() {
    test_features --features "$1"
    test_features --features "$1,parallel"
    if [[ $1 == faer_v* ]]; then
        cargo test --locked --no-default-features --features "$1" --example least_squares_sgd
        cargo run --locked --no-default-features --features "$1" --example least_squares_sgd
    fi
}

test_core() {
    cargo test --manifest-path tests/core_consumer/Cargo.toml --locked
    test_features
    for features in parallel faer nalgebra ndarray sprs zarrs \
        faer,nalgebra,ndarray,sprs faer,nalgebra,ndarray,sprs,parallel \
        faer_v0_22,nalgebra_v0_32,ndarray_v0_15,sprs \
        faer_v0_22,nalgebra_v0_32,ndarray_v0_15,sprs,parallel \
        sprs,ndarray_v0_16 sprs,ndarray_v0_16,parallel \
        zarrs,ndarray_v0_15 zarrs,ndarray_v0_16 zarrs,ndarray_v0_17; do
        test_features --features "$features"
    done
    test_features --features "$(IFS=,; echo "${versions[*]}")"
    test_features --all-features
}

test_pairs() {
    local versions=("$@")
    local i j
    for ((i = 0; i < ${#versions[@]}; i++)); do
        for ((j = i + 1; j < ${#versions[@]}; j++)); do
            local pair="${versions[i]},${versions[j]}"
            test_version "$pair"
            test_fixtures --features "$pair"
            test_fixtures --features "$pair,parallel"
        done
    done
}

test_coexistence() {
    test_pairs "${faer_versions[@]}"
    test_pairs "${nalgebra_versions[@]}"
    test_pairs "${ndarray_versions[@]}"
    test_pairs "${zarrs_versions[@]}"
    local feature family
    for feature in "${faer_versions[@]:0:2}" "${nalgebra_versions[@]:0:3}" "${ndarray_versions[@]:0:2}" "${zarrs_versions[@]:0:1}"; do
        family=${feature%%_v*}
        test_version "$feature,$family"
        test_fixtures --features "$feature,$family"
        test_fixtures --features "$feature,$family,parallel"
    done
    local combined
    combined=$(IFS=,; echo "${faer_versions[*]},${nalgebra_versions[*]},${ndarray_versions[*]},${zarrs_versions[*]}")
    test_fixtures --features "$combined"
    test_fixtures --all-features
}

check_msrv() (
    if [[ $(rustc --version) != "rustc 1.87.0 "* ]]; then
        echo "Run the MSRV checks with the Rust 1.87.0 toolchain." >&2
        exit 1
    fi
    local lockfile
    lockfile=$(mktemp)
    cp Cargo.lock "$lockfile"
    trap 'cp "$lockfile" Cargo.lock; rm -f "$lockfile"' EXIT
    # Newer sprs patch releases require Rust 1.88, so test the compatible release.
    cargo update --package sprs --precise 0.11.4
    cargo check --all-targets --locked --no-default-features
    cargo check --all-targets --locked --no-default-features --features parallel
    for feature in "${versions[@]}"; do
        if [[ $feature != nalgebra_v0_35 && $feature != zarrs_v0_23 ]]; then
            cargo check --all-targets --locked --no-default-features --features "$feature"
            cargo check --all-targets --locked --no-default-features --features "$feature,parallel"
        fi
    done
    local compatible=() fixture_compatible=()
    for feature in "${versions[@]}"; do
        if [[ $feature != nalgebra_v0_35 && $feature != zarrs_v0_23 ]]; then
            compatible+=("$feature")
            if [[ $feature != sprs_* ]]; then
                fixture_compatible+=("$feature")
            fi
        fi
    done
    local combined fixtures
    combined=$(IFS=,; echo "${compatible[*]}")
    fixtures=$(IFS=,; echo "${fixture_compatible[*]}")
    cargo check --all-targets --locked --no-default-features --features "$combined"
    cargo check --all-targets --locked --no-default-features --features "$combined,parallel"
    cargo check --manifest-path "$fixture_manifest" --all-targets --locked --no-default-features --features "$fixtures"
    cargo check --manifest-path "$fixture_manifest" --all-targets --locked --no-default-features --features "$fixtures,parallel"
    # The Gram example and benchmark require both dense and sparse adapters.
    for feature in "${ndarray_versions[@]}"; do
        cargo check --all-targets --locked --no-default-features --features "$feature,sprs_v0_11"
    done
)

check_zarrs_msrv() {
    if [[ $(rustc --version) != "rustc 1.91.0 "* ]]; then
        echo "Run the zarrs 0.23 MSRV checks with Rust 1.91.0." >&2
        exit 1
    fi
    local features
    for features in zarrs_v0_23 zarrs_v0_23,parallel zarrs_v0_22,zarrs_v0_23 zarrs_v0_22,zarrs_v0_23,parallel; do
        cargo check --all-targets --locked --no-default-features --features "$features"
        cargo check --manifest-path "$fixture_manifest" --all-targets --locked --no-default-features --features "$features"
    done
}

case "${1:-all}" in
all)
    test_core
    for feature in "${versions[@]}"; do
        test_version "$feature"
    done
    test_coexistence
    ;;
core) test_core ;;
version) test_version "${2:?Specify a version feature.}" ;;
coexistence | selection) test_coexistence ;;
msrv) check_msrv ;;
zarrs-msrv) check_zarrs_msrv ;;
*)
    echo "Usage: $0 [all|core|version FEATURE|coexistence|msrv|zarrs-msrv]" >&2
    exit 1
    ;;
esac
