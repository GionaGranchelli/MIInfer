#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 miinfer-*.tar.gz [install-directory]" >&2
    exit 2
fi
archive=$(realpath "$1")
destination=${2:-"$HOME/.local/miinfer"}
if [[ ! -f "$archive" ]]; then echo "package not found: $archive" >&2; exit 1; fi
mkdir -p "$destination"
tar -xzf "$archive" --strip-components=1 -C "$destination"
echo "Installed MIInfer to $destination"
echo "Add $destination/bin to PATH, then run: miinfer doctor"
