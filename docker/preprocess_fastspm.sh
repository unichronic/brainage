#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: fastbrainage preprocess INPUT_T1_NII_OR_GZ OUTPUT_MWC1_NII" >&2
    exit 2
fi

input_path=$1
output_path=$2
spm_path=${SPM12_DIR:-/opt/spm12}
driver_path=${FASTBRAINAGE_MATLAB_DIR:-/opt/fastbrainage/matlab}
recipe_path=${FASTSPM_RECIPE:-$driver_path/fastspm_v1.m}

if [[ ! -f "$input_path" ]]; then
    echo "input does not exist: $input_path" >&2
    exit 1
fi
if [[ ! -d "$spm_path" ]]; then
    echo "SPM12 directory does not exist: $spm_path" >&2
    exit 1
fi
if [[ ! -f "$driver_path/run_fastspm_batch.m" ]]; then
    echo "FastSPM Octave driver does not exist: $driver_path/run_fastspm_batch.m" >&2
    exit 1
fi
if [[ ! -f "$recipe_path" ]]; then
    echo "FastSPM recipe does not exist: $recipe_path" >&2
    exit 1
fi

work_dir=$(mktemp -d)
cleanup() {
    rm -rf "$work_dir"
}
trap cleanup EXIT

mkdir -p "$(dirname "$output_path")" "$work_dir/out"
if [[ "$input_path" == *.nii.gz ]]; then
    gzip -dc "$input_path" > "$work_dir/input.nii"
else
    cp "$input_path" "$work_dir/input.nii"
fi

# Octave strings escape a literal quote by doubling it. Paths in normal
# container mounts do not contain quotes, but handling them here is cheap.
octave_quote() {
    local value=$1
    value=${value//\'/\'\'}
    printf '%s' "$value"
}

input_expr=$(octave_quote "$work_dir/input.nii")
output_expr=$(octave_quote "$work_dir/out")
spm_expr=$(octave_quote "$spm_path")
driver_expr=$(octave_quote "$driver_path")
recipe_expr=$(octave_quote "$recipe_path")
octave --quiet --no-gui --eval \
    "addpath('$driver_expr'); run_fastspm_batch('$input_expr','$output_expr','$spm_expr','$recipe_expr')"

result=$(find "$work_dir/out" -maxdepth 1 -type f -name '*_mwc1.nii' -print -quit)
if [[ -z "$result" ]]; then
    echo "FastSPM did not create an output map" >&2
    exit 1
fi
cp "$result" "$output_path"
echo "saved $output_path"
