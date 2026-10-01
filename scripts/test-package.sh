#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    printf 'usage: %s PACKAGE.tar.gz [MODEL.gguf]\n' "$0" >&2
    exit 2
fi

archive=$(realpath "$1")
if [[ ! -f "$archive" ]]; then
    printf 'SKIP: package archive has not been built: %s\n' "$archive"
    exit 0
fi
if [[ $# -eq 2 && ! -f "$2" ]]; then
    printf 'model not found: %s\n' "$2" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
stage=$(mktemp -d)
cleanup() {
    status=$?
    if [[ $status -eq 0 ]]; then
        rm -rf "$stage"
    else
        printf 'package smoke failed; diagnostics retained at %s\n' "$stage" >&2
    fi
    return "$status"
}
trap cleanup EXIT
tar -xzf "$archive" -C "$stage"

package_root=$(find "$stage" -mindepth 1 -maxdepth 1 -type d -name 'miinfer-*' -print -quit)
if [[ -z "$package_root" ]]; then
    printf 'package has no MIInfer root directory\n' >&2
    exit 1
fi

expected_files=(
    bin/miinfer
    bin/miinfer-device-info
    install.sh
    share/doc/MIInfer/LICENSE
    share/doc/MIInfer/README.md
    share/doc/MIInfer/docs/cli-product-contract.md
    share/doc/MIInfer/docs/hardware.md
    share/doc/MIInfer/docs/release.md
)
actual_files=$(cd "$package_root" && find . ! -type d -printf '%P\n' | sort)
expected_manifest=$(printf '%s\n' "${expected_files[@]}" | sort)
if [[ "$actual_files" != "$expected_manifest" ]]; then
    printf 'package file manifest differs from the release allowlist\n' >&2
    diff -u <(printf '%s\n' "$expected_manifest") <(printf '%s\n' "$actual_files") >&2 || true
    exit 1
fi

for binary in miinfer miinfer-device-info; do
    path="$package_root/bin/$binary"
    if [[ ! -x "$path" ]]; then
        printf 'package is missing executable: %s\n' "$binary" >&2
        exit 1
    fi
    if ldd "$path" | grep -q 'not found'; then
        printf 'missing dynamic dependency for %s\n' "$binary" >&2
        ldd "$path" >&2
        exit 1
    fi
done
if [[ ! -x "$package_root/install.sh" ]]; then
    printf 'package is missing executable installer\n' >&2
    exit 1
fi

archive_name=$(basename "$archive")
package_version=${archive_name#miinfer-}
package_version=${package_version%-gfx906-Linux.tar.gz}
version_output=$("$package_root/bin/miinfer" --version)
grep -Fq "MIInfer version: $package_version" <<< "$version_output"
grep -Eq '^Git commit: [[:xdigit:]]{7,12}$' <<< "$version_output"
grep -q '^Target architecture: gfx906$' <<< "$version_output"
"$package_root/bin/miinfer-device-info" --version \
    | grep -Fq "MIInfer version: $package_version"

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

XDG_CONFIG_HOME="$stage/config" "$package_root/bin/miinfer" config --json > "$stage/config.json"
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

mkdir -p "$stage/home"
HOME="$stage/home" "$package_root/install.sh" "$archive"
if [[ ! -x "$stage/home/.local/miinfer/bin/miinfer" ]]; then
    printf 'installer did not create the default user-prefix layout\n' >&2
    exit 1
fi

mkdir -p "$stage/installed"
"$package_root/install.sh" "$archive" "$stage/installed"
if [[ ! -x "$stage/installed/bin/miinfer" ]]; then
    printf 'installer did not create a runnable release layout\n' >&2
    exit 1
fi
touch "$stage/installed/upgrade-marker"
printf 'stale release doc\n' > "$stage/installed/share/doc/MIInfer/docs/release.md"
"$package_root/install.sh" "$archive" "$stage/installed"
if [[ ! -f "$stage/installed/upgrade-marker" ]]; then
    printf 'repeat install unexpectedly removed existing prefix contents\n' >&2
    exit 1
fi
if ! cmp -s "$package_root/share/doc/MIInfer/docs/release.md" \
    "$stage/installed/share/doc/MIInfer/docs/release.md"; then
    printf 'repeat install did not overwrite the packaged release file\n' >&2
    exit 1
fi
XDG_CONFIG_HOME="$stage/installed-config" "$stage/installed/bin/miinfer" config >/dev/null
"$stage/installed/bin/miinfer" models "$stage/models" | grep -q 'sample.gguf'

if [[ $# -eq 2 ]]; then
    bash "$script_dir/test-cli-first-run.sh" \
        "$stage/installed/bin/miinfer" "$(realpath "$2")"
fi

printf 'package smoke test passed: %s\n' "$archive"
