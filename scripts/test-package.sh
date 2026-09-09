#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'usage: %s PACKAGE.tar.gz\n' "$0" >&2
    exit 2
fi

archive=$(realpath "$1")
if [[ ! -f "$archive" ]]; then
    printf 'package not found: %s\n' "$archive" >&2
    exit 2
fi

stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
tar -xzf "$archive" -C "$stage"

package_root=$(find "$stage" -mindepth 1 -maxdepth 1 -type d -name 'miinfer-*' -print -quit)
if [[ -z "$package_root" ]]; then
    printf 'package has no MIInfer root directory\n' >&2
    exit 1
fi

for binary in miinfer miinfer-device-info; do
    path="$package_root/bin/$binary"
    if [[ ! -x "$path" ]]; then
        printf 'package is missing executable: %s\n' "$path" >&2
        exit 1
    fi
    if ldd "$path" | grep -q 'not found'; then
        printf 'missing dynamic dependency for %s\n' "$binary" >&2
        ldd "$path" >&2
        exit 1
    fi
done
if [[ ! -x "$package_root/install.sh" ]]; then
    printf 'package is missing install.sh\n' >&2
    exit 1
fi

"$package_root/bin/miinfer" --help >/dev/null
"$package_root/bin/miinfer" --version >/dev/null
"$package_root/bin/miinfer-device-info" --version >/dev/null
"$package_root/bin/miinfer" config > "$stage/config.txt"
"$package_root/bin/miinfer" doctor --port 0 > "$stage/doctor.txt"
if ! grep -q '^gpu=PASS:' "$stage/doctor.txt"; then
    printf 'doctor did not validate the gfx906 GPU\n' >&2
    cat "$stage/doctor.txt" >&2
    exit 1
fi
if ! grep -q '^prefill_path=validated-default$' "$stage/config.txt"; then
    printf 'configuration contract missing validated prefill path\n' >&2
    exit 1
fi
if ! grep -q '^runtime_context_capacity=65536$' "$stage/config.txt" || \
   ! grep -q '^model_context_length=262144$' "$stage/config.txt" || \
   ! grep -q '^qualified_context_length=1024$' "$stage/config.txt"; then
    printf 'configuration contract missing context limits\n' >&2
    exit 1
fi
mkdir -p "$stage/models/nested"
: > "$stage/models/nested/sample.gguf"
if ! "$package_root/bin/miinfer" models "$stage/models" | grep -q 'sample.gguf'; then
    printf 'model discovery did not find the GGUF fixture\n' >&2
    exit 1
fi
"$package_root/install.sh" "$archive" "$stage/installed"
if [[ ! -x "$stage/installed/bin/miinfer" ]]; then
    printf 'installer did not create a runnable release layout\n' >&2
    exit 1
fi
"$stage/installed/bin/miinfer" config >/dev/null
"$stage/installed/bin/miinfer" models "$stage/models" | grep -q 'sample.gguf'
"$package_root/bin/miinfer-device-info" > "$stage/device-info.txt"

if ! grep -q 'MIInfer contract: gfx906 compatible' "$stage/device-info.txt"; then
    printf 'device probe did not confirm the gfx906 contract\n' >&2
    cat "$stage/device-info.txt" >&2
    exit 1
fi

printf 'package smoke test passed: %s\n' "$archive"
