#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 MODEL.gguf" >&2
    exit 2
fi

model_path=$1
if [[ ! -f "$model_path" ]]; then
    echo "M4-C3 acceptance: model does not exist: $model_path" >&2
    exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
generator="$repo_root/build/mi50-release/miinfer-qwen3-generate"
if [[ ! -x "$generator" ]]; then
    echo "M4-C3 acceptance: build the mi50-release preset first" >&2
    exit 2
fi

echo "M4-C3 physical acceptance: MI50 Release text generation"
expected_text=$') {\n        return "Hello, "'
"$generator" "$model_path" \
    --prompt hello \
    --max-tokens 8 \
    --expect-generated-ids 8,341,286,470,330,9707,11,330 \
    --expect-generated-text "$expected_text"
echo "M4-C3 physical acceptance: PASS"
