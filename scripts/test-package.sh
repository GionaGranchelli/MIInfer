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
for document in \
    share/doc/MIInfer/README.md \
    share/doc/MIInfer/developer-guide.md \
    share/doc/MIInfer/docs/current-state-of-the-art.md \
    share/doc/MIInfer/docs/cli-product-contract.md \
    share/doc/MIInfer/experiments/V2-0045-definitive-competitive-qualification.md; do
    if [[ ! -f "$package_root/$document" ]]; then
        printf 'package is missing documentation: %s\n' "$document" >&2
        exit 1
    fi
done

"$package_root/bin/miinfer" --help > "$stage/help.txt"
if grep -Eq 'M26-C|EXP-[0-9]+|MIINFER_PRESET|kernel selector|profil' "$stage/help.txt"; then
    printf 'public help exposes developer controls\n' >&2
    exit 1
fi
for command in doctor models inspect run chat serve config; do
    "$package_root/bin/miinfer" "$command" --help > "$stage/$command-help.txt"
    if grep -Eq 'M26-C|EXP-[0-9]+|MIINFER_PRESET|kernel selector' "$stage/$command-help.txt"; then
        printf 'public %s help exposes developer controls\n' "$command" >&2
        exit 1
    fi
done
"$package_root/bin/miinfer" --version >/dev/null
"$package_root/bin/miinfer-device-info" --version >/dev/null
XDG_CONFIG_HOME="$stage/config" "$package_root/bin/miinfer" config --json > "$stage/config.json"
XDG_CONFIG_HOME="$stage/config" "$package_root/bin/miinfer" doctor --port 0 > "$stage/doctor.txt"
if ! grep -q '^GPU      PASS' "$stage/doctor.txt"; then
    printf 'doctor did not validate the gfx906 GPU\n' >&2
    cat "$stage/doctor.txt" >&2
    exit 1
fi
for key in model_dir default_model host port context api_key_file; do
    if ! grep -q "\"$key\"" "$stage/config.json"; then
        printf 'configuration output missing public key: %s\n' "$key" >&2
        exit 1
    fi
done
if ! grep -q '"value": "8192"' "$stage/config.json"; then
    printf 'configuration output missing production context default\n' >&2
    exit 1
fi
mkdir -p "$stage/models/nested"
: > "$stage/models/nested/sample.gguf"
if ! "$package_root/bin/miinfer" models "$stage/models" | grep -q 'sample.gguf'; then
    printf 'model discovery did not find the GGUF fixture\n' >&2
    exit 1
fi
mkdir -p "$stage/models/ambiguous"
: > "$stage/models/ambiguous/Ambiguous-Q4.gguf"
: > "$stage/models/ambiguous/Ambiguous-Q5.gguf"
XDG_CONFIG_HOME="$stage/config" "$package_root/bin/miinfer" config set model-dir \
    "$stage/models/ambiguous" >/dev/null
if XDG_CONFIG_HOME="$stage/config" "$package_root/bin/miinfer" inspect Ambiguous \
    > "$stage/ambiguity.txt" 2>&1; then
    printf 'model resolution silently accepted ambiguous matches\n' >&2
    exit 1
fi
for name in Ambiguous-Q4.gguf Ambiguous-Q5.gguf; do
    if ! grep -q "$name" "$stage/ambiguity.txt"; then
        printf 'ambiguity error omitted matching model: %s\n' "$name" >&2
        exit 1
    fi
done
"$package_root/install.sh" "$archive" "$stage/installed"
if [[ ! -x "$stage/installed/bin/miinfer" ]]; then
    printf 'installer did not create a runnable release layout\n' >&2
    exit 1
fi
XDG_CONFIG_HOME="$stage/installed-config" "$stage/installed/bin/miinfer" config >/dev/null
"$stage/installed/bin/miinfer" models "$stage/models" | grep -q 'sample.gguf'
"$package_root/bin/miinfer-device-info" > "$stage/device-info.txt"

if ! grep -q 'MIInfer contract: gfx906 compatible' "$stage/device-info.txt"; then
    printf 'device probe did not confirm the gfx906 contract\n' >&2
    cat "$stage/device-info.txt" >&2
    exit 1
fi

printf 'package smoke test passed: %s\n' "$archive"
