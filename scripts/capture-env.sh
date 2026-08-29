#!/usr/bin/env bash
set -euo pipefail

output_path="${1:-}"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)

json_quote() {
    local value=${1-}
    if command -v jq >/dev/null 2>&1; then
        printf '%s' "$value" | jq -Rs .
        return
    fi
    value="$(printf '%s' "$value" | tr -d '\000-\011\013\014\016-\037')"
    printf '"'
    printf '%s' "$value" | sed ':a;N;$!ba;s/\\/\\\\/g;s/"/\\\"/g;s/\n/\\n/g'
    printf '"'
}

unavailable="UNAVAILABLE"
first_line() {
    local value
    value="$($@ 2>/dev/null || true)"
    value="${value%%$'\n'*}"
    if [[ -n "$value" ]]; then
        printf '%s' "$value"
    else
        printf '%s' "$unavailable"
    fi
}

full_output() {
    local value
    value="$($@ 2>/dev/null || true)"
    if [[ -n "$value" ]]; then
        printf '%s' "$value"
    else
        printf '%s' "$unavailable"
    fi
}

git_commit="$unavailable"
git_dirty="$unavailable"
if command -v git >/dev/null 2>&1 && git -C "$repo_root" rev-parse --git-dir >/dev/null 2>&1; then
    git_commit="$(git -C "$repo_root" rev-parse --short=12 HEAD 2>/dev/null || printf '%s' "$unavailable")"
    if [[ -n "$(git -C "$repo_root" status --porcelain 2>/dev/null || true)" ]]; then
        git_dirty=true
    else
        git_dirty=false
    fi
fi

distro="$unavailable"
if [[ -r /etc/os-release ]]; then
    distro="$(sed -n 's/^PRETTY_NAME=//p' /etc/os-release | head -n 1 | sed 's/^"//; s/"$//')"
    [[ -n "$distro" ]] || distro="$unavailable"
fi

cpu="$unavailable"
if command -v lscpu >/dev/null 2>&1; then
    cpu="$(lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p' | head -n 1)"
    [[ -n "$cpu" ]] || cpu="$unavailable"
fi

ram_bytes="$unavailable"
if [[ -r /proc/meminfo ]]; then
    ram_kib="$(sed -n 's/^MemTotal:[[:space:]]*\([0-9][0-9]*\) kB/\1/p' /proc/meminfo | head -n 1)"
    if [[ -n "$ram_kib" ]]; then
        ram_bytes=$((ram_kib * 1024))
    fi
fi

rocm_version="$unavailable"
for version_file in /opt/rocm/.info/version /opt/rocm/.info/version-dev; do
    if [[ -r "$version_file" ]]; then
        rocm_version="$(head -n 1 "$version_file")"
        break
    fi
done
hip_compiler_version="$(first_line hipcc --version)"
[[ "$rocm_version" != "$unavailable" ]] || rocm_version="$hip_compiler_version"

rocm_smi_text="$unavailable"
if command -v rocm-smi >/dev/null 2>&1; then
    rocm_smi_text="$(full_output rocm-smi --showproductname --showmeminfo vram --showclocks --showtemp --showpower --showmaxpower)"
fi
rocm_smi_json="$unavailable"
if command -v rocm-smi >/dev/null 2>&1; then
    rocm_smi_json="$(full_output rocm-smi --json --showproductname --showmeminfo vram --showclocks --showtemp --showpower --showmaxpower)"
fi
amd_smi_text="$unavailable"
if command -v amd-smi >/dev/null 2>&1; then
    amd_smi_text="$(full_output amd-smi list)"
fi
rocminfo_summary="$unavailable"
if command -v rocminfo >/dev/null 2>&1; then
    rocminfo_summary="$(full_output rocminfo)"
fi

json_body=$(cat <<EOF
{
  "timestamp_utc": $(json_quote "$(date -u +%Y-%m-%dT%H:%M:%SZ)"),
  "git": {
    "commit": $(json_quote "$git_commit"),
    "dirty": $(json_quote "$git_dirty")
  },
  "host": {
    "hostname": $(json_quote "$(hostname 2>/dev/null || printf '%s' "$unavailable")"),
    "distribution": $(json_quote "$distro"),
    "kernel": $(json_quote "$(uname -srvm 2>/dev/null || printf '%s' "$unavailable")"),
    "cpu": $(json_quote "$cpu"),
    "ram_bytes": $(json_quote "$ram_bytes")
  },
  "software": {
    "rocm_version": $(json_quote "$rocm_version"),
    "hip_compiler_version": $(json_quote "$hip_compiler_version"),
    "tools": {
      "rocminfo": $(json_quote "$rocminfo_summary"),
      "rocm_smi": $(json_quote "$rocm_smi_text"),
      "amd_smi": $(json_quote "$amd_smi_text")
    }
  },
  "gpu": {
    "target_architecture": "gfx906",
    "rocm_smi_json": $(json_quote "$rocm_smi_json"),
    "metrics_note": "Parse rocm_smi_json for GPU identity, GFX Version, VRAM, SCLK, MCLK, temperature, power, and power limit; unavailable fields remain UNAVAILABLE."
  }
}
EOF
)

if [[ -n "$output_path" ]]; then
    mkdir -p "$(dirname "$output_path")"
    printf '%s\n' "$json_body" > "$output_path"
else
    printf '%s\n' "$json_body"
fi
