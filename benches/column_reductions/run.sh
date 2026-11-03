#!/usr/bin/env bash
set -euo pipefail

# Reserve one compute bench window around this script, including both builds.
probe_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
data=${1:--}
output=${2:?Specify an output directory.}
tag=${3:?Specify a build label.}
mkdir -p "$output"
output=$(realpath "$output")

"${RUSTC:-rustc}" -Vv >"$output/$tag-rustc.txt"
"${PROBE_CARGO:-cargo}" -V >"$output/$tag-cargo.txt"
sha256sum "$probe_dir/src/main.rs" "$probe_dir/Cargo.toml" "$probe_dir/Cargo.lock" >"$output/$tag-source.sha256"

arguments=(--samples "${PROBE_SAMPLES:-9}" --passes "${PROBE_PASSES:-30}")
if [[ $data != - ]]; then
    arguments+=(--data "$data")
fi

for units in 16 1; do
    export CARGO_PROFILE_RELEASE_CODEGEN_UNITS=$units
    export CARGO_TARGET_DIR="$output/$tag-cgu$units-build"
    "${PROBE_CARGO:-cargo}" build --release --locked --manifest-path "$probe_dir/Cargo.toml"
    executable="$CARGO_TARGET_DIR/release/column-reduction-probe"
    taskset -c "${PROBE_CPU:-2}" "$executable" "${arguments[@]}" --label "$tag-cgu$units" \
        >"$output/$tag-cgu$units.csv" 2>"$output/$tag-cgu$units-errors.csv"
    objdump -d -C "$executable" >"$output/$tag-cgu$units.asm"
    if [[ ${PROBE_LAYOUTS:-0} == 1 && $units == 16 ]]; then
        taskset -c "${PROBE_CPU:-2}" "$executable" "${arguments[@]}" --layouts --label "$tag-cgu$units-layouts" \
            >"$output/$tag-layouts.csv" 2>"$output/$tag-layouts-errors.csv"
    fi
    if [[ ${PROBE_SUMMARIES:-0} == 1 && $units == 16 ]]; then
        taskset -c "${PROBE_CPU:-2}" "$executable" "${arguments[@]}" --layouts --summaries --label "$tag-cgu$units-summaries" \
            >"$output/$tag-summaries.csv" 2>"$output/$tag-summaries-errors.csv"
    fi
done

"$executable" --numerics >"$output/$tag-numerics.csv"
for rows in 0 1 7 8 9 15 16 17 31 32 33; do
    "$executable" --rows "$rows" --cols 7 --samples 0 --layouts \
        >/dev/null 2>"$output/$tag-rows$rows-errors.csv"
done
