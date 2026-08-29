#!/usr/bin/env bash
set -Eeuo pipefail

# Controlled platform experiment for the MI50 host. It changes no packages or
# source files; it only unbinds gfx802 for the A/B test and restores it on exit.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
target_bdf="0000:03:00.0"
target_driver_path="/sys/bus/pci/drivers/amdgpu"
results_root="${MIINFER_PLATFORM_RESULTS_DIR:-$repo_root/bench/platform-results}"
command_timeout_sec="${MIINFER_PLATFORM_COMMAND_TIMEOUT_SEC:-30}"
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
run_dir="$results_root/gfx802-isolation/$run_id"
mkdir -p "$run_dir"
restart_gdm=0
leave_unbound=0

usage() {
    cat <<'EOF'
Usage: scripts/diagnose-gfx802-isolation.sh [--restart-gdm] [--leave-unbound]

Run the gfx802-unbound A/B test and retain all results. Use SSH as the
control channel. --restart-gdm asks the script to restart Fedora's display
manager after the GPU has been rebound; this ends the current graphical
session. --leave-unbound skips the rebind step; reboot after collecting the
results to restore the normal two-GPU state.
EOF
}

while (($#)); do
    case "$1" in
        --restart-gdm)
            restart_gdm=1
            ;;
        --leave-unbound)
            leave_unbound=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'unknown option: %s\n\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

log_file="$run_dir/commands.log"
printf 'MI50 gfx802 isolation diagnostic\n' | tee "$run_dir/README.txt" "$log_file" >/dev/null
printf 'started_utc=%s\nrepo_root=%s\ntarget_bdf=%s\nrun_dir=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$repo_root" "$target_bdf" "$run_dir" \
    | tee -a "$run_dir/README.txt" "$log_file"
printf 'restart_gdm=%s\n' "$restart_gdm" | tee -a "$run_dir/README.txt" "$log_file"
printf 'leave_unbound=%s\n' "$leave_unbound" | tee -a "$run_dir/README.txt" "$log_file"

if [[ "${EUID}" -eq 0 ]]; then
    printf '%s\n' 'Run this script as your normal user; it uses sudo only for device binding.' >&2
    exit 2
fi
if ! command -v sudo >/dev/null 2>&1; then
    printf '%s\n' 'sudo is required.' >&2
    exit 2
fi
if [[ ! -e "/sys/bus/pci/devices/$target_bdf" ]]; then
    printf 'target PCI device is not present: %s\n' "$target_bdf" >&2
    exit 2
fi

driver_link="/sys/bus/pci/devices/$target_bdf/driver"
if [[ ! -e "$driver_link" ]] || [[ "$(readlink -f "$driver_link")" != "$target_driver_path" ]]; then
    printf 'target GPU is not currently bound to amdgpu: %s\n' "$target_bdf" >&2
    exit 2
fi

run_capture() {
    local name=$1
    shift
    local output="$run_dir/$name.log"
    {
        printf '\n$'
        printf ' %q' "$@"
        printf '\n'
    } | tee -a "$log_file" > "$output"

    set +e
    timeout --foreground "${command_timeout_sec}s" "$@" 2>&1 | tee -a "$output" | tee -a "$log_file"
    LAST_STATUS=${PIPESTATUS[0]}
    set -e
    printf 'exit_status=%s\n' "$LAST_STATUS" | tee -a "$output" "$log_file"
}

capture_env() {
    local phase=$1
    local output="$run_dir/environment-$phase.json"
    set +e
    timeout --foreground "${command_timeout_sec}s" "$script_dir/capture-env.sh" "$output" \
        >"$run_dir/capture-env-$phase.log" 2>&1
    LAST_STATUS=$?
    set -e
    printf 'capture-env-%s exit_status=%s\n' "$phase" "$LAST_STATUS" | tee -a "$log_file"
}

capture_topology() {
    local phase=$1
    run_capture "$phase-lspci" lspci -nnk
    run_capture "$phase-drm" bash -c 'ls -l /sys/class/drm'
    run_capture "$phase-kfd" bash -c '
        for node in /sys/class/kfd/kfd/topology/nodes/*; do
            [[ -d "$node" ]] || continue
            printf "=== %s ===\n" "$node"
            for field in properties gpu_id name io_links; do
                printf "--- %s ---\n" "$field"
                if [[ -r "$node/$field" ]]; then cat "$node/$field"; else printf "UNAVAILABLE\n"; fi
            done
        done
    '
    run_capture "$phase-bindings" bash -c '
        for bdf in 0000:03:00.0 0000:06:00.0; do
            printf "%s: " "$bdf"
            readlink -f "/sys/bus/pci/devices/$bdf/driver" 2>/dev/null || printf "UNBOUND\n"
        done
    '
    run_capture "$phase-kernel-log" bash -c 'journalctl -k -b --no-pager 2>/dev/null | rg -i "amdgpu|kfd|hsa|iommu|gfx|CRAT" | tail -200'
    run_capture "$phase-rocm-smi" rocm-smi --showproductname --showmeminfo vram --showclocks --showtemp --showpower --showmaxpower
    run_capture "$phase-processes" bash -c 'pgrep -af "ollama|llama-server|lmstudio" || true'
    run_capture "$phase-env" bash -c 'env | sort | rg "ROCM|HIP|HSA|AMD|GPU" || true'
    run_capture "$phase-tools" bash -c 'for tool in hipcc rocminfo; do printf "== %s ==\n" "$tool"; command -v "$tool" || true; done; command -v rocminfo >/dev/null 2>&1 && ldd "$(command -v rocminfo)" || true'
}

unbind_done=0
isolation_attempted=0
rebind_target() {
    if (( unbind_done == 0 )); then
        return 0
    fi

    if (( leave_unbound == 1 )); then
        printf '%s\n' 'Leaving gfx802 unbound as requested; reboot to restore it.' \
            | tee -a "$run_dir/README.txt" "$log_file"
        unbind_done=0
        return 0
    fi

    printf '%s\n' 'Restoring gfx802/amdgpu binding...' | tee -a "$log_file"
    if printf '%s\n' "$target_bdf" | sudo -n tee "$target_driver_path/bind" >/dev/null; then
        unbind_done=0
        printf '%s\n' 'gfx802 rebind completed.' | tee -a "$log_file"
    else
        printf '%s\n' "ERROR: failed to rebind $target_bdf; restore it manually." | tee -a "$log_file" >&2
        printf 'sudo sh -c '\''printf %%s %s > %s/bind'\''\n' "$target_bdf" "$target_driver_path" \
            | tee -a "$run_dir/README.txt" "$log_file" >&2
    fi
}

finish() {
    local status=$?
    trap - EXIT INT TERM
    rebind_target
    capture_topology post-cleanup
    capture_env after
    if (( isolation_attempted == 1 && restart_gdm == 1 && leave_unbound == 0 )); then
        printf '%s\n' 'Restarting GDM to restore the graphical session...' | tee -a "$log_file"
        if sudo -n systemctl restart gdm >>"$log_file" 2>&1; then
            printf '%s\n' 'GDM restart completed.' | tee -a "$log_file"
        else
            printf '%s\n' 'WARNING: GDM restart failed; reboot or run sudo systemctl restart gdm manually.' \
                | tee -a "$run_dir/README.txt" "$log_file" >&2
        fi
    elif (( isolation_attempted == 1 && restart_gdm == 1 && leave_unbound == 1 )); then
        printf '%s\n' 'WARNING: GDM restart skipped because gfx802 was left unbound; reboot to restore display.' \
            | tee -a "$run_dir/README.txt" "$log_file" >&2
    fi
    printf 'finished_utc=%s\nexit_status=%s\nresults=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" "$run_dir" \
        | tee -a "$run_dir/README.txt" "$log_file"
    exit "$status"
}

trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

capture_topology pre
capture_env before
run_capture pre-rocminfo rocminfo
run_capture pre-mi50-device-info "$repo_root/build/mi50-release/miinfer-device-info"
run_capture pre-release-ctest ctest --preset mi50-release --output-on-failure

printf '%s\n' 'Requesting administrator authentication once...' | tee -a "$log_file"
sudo -v

printf '%s\n' 'Unbinding gfx802 display GPU for controlled test...' | tee -a "$log_file"
isolation_attempted=1
if ! printf '%s\n' "$target_bdf" | sudo -n tee "$target_driver_path/unbind" >/dev/null; then
    printf '%s\n' 'Unable to unbind the target GPU.' | tee -a "$log_file" >&2
    exit 1
fi
unbind_done=1

sleep 2
capture_topology gfx906-only
capture_env gfx906-only
run_capture gfx906-only-rocminfo rocminfo
run_capture gfx906-only-mi50-device-info "$repo_root/build/mi50-release/miinfer-device-info"
run_capture gfx906-only-release-ctest ctest --preset mi50-release --output-on-failure

if rg -qi 'gfx906' "$run_dir/gfx906-only-rocminfo.log" && \
   ! rg -qi 'HSA_STATUS_ERROR|no ROCm-capable device' "$run_dir/gfx906-only-rocminfo.log"; then
    gfx906_only_verdict=PASS
else
    gfx906_only_verdict=FAIL
fi

cat > "$run_dir/summary.txt" <<EOF
gfx802 isolation diagnostic
===========================
target_bdf: $target_bdf
gfx906_only_rocminfo_verdict: $gfx906_only_verdict
gfx906_only_rocminfo_log: $run_dir/gfx906-only-rocminfo.log
gfx906_only_miinfer_device_info_log: $run_dir/gfx906-only-mi50-device-info.log
gfx906_only_release_ctest_log: $run_dir/gfx906-only-release-ctest.log

Interpretation:
- PASS means rocminfo printed gfx906 without the observed HSA/no-device errors.
- This is the gfx802-unbound half of the A/B test; post-cleanup results are captured by the EXIT trap.
- With --leave-unbound, reboot after the script finishes to restore the normal two-GPU state.
- Review raw logs and topology files before marking the root cause confirmed.
EOF
cat "$run_dir/summary.txt" | tee -a "$log_file"
