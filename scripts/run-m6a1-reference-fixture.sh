#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 || $# > 5 )); then
    echo "usage: $0 LLAMA_CPP_ROOT MODEL.gguf OUTPUT_DIR [PROMPT] [MAX_GENERATED]" >&2
    exit 2
fi

llama_root=$1
model=$2
output=$3
prompt=${4:-hello}
max_generated=${5:-64}
tool_dir=$(mktemp -d)
trap 'rm -rf "$tool_dir"' EXIT

c++ -std=c++17 -O2 \
    -I"$llama_root/include" -I"$llama_root/ggml/include" \
    "$(dirname "$0")/../tools/m6a1_reference_fixture.cpp" \
    -L"$llama_root/build/bin" -Wl,-rpath,"$llama_root/build/bin" \
    -lllama -lggml-base -o "$tool_dir/m6a1_reference_fixture"

"$tool_dir/m6a1_reference_fixture" "$model" "$output" "$prompt" "$max_generated"
python3 "$(dirname "$0")/check-m6a1-fixture.py" "$output"
