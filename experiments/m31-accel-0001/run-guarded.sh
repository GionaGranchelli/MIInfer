#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
[[ -f "$repo/CMakeLists.txt" ]] || { echo "invalid repository root: $repo" >&2; exit 2; }
temp_warning() { awk -v t="$1" 'BEGIN { exit !(t >= 85.0) }'; }
temp_limit() { awk -v t="$1" 'BEGIN { exit !(t >= 90.0) }'; }
vram_limit() { awk -v b="$1" 'BEGIN { exit !(b >= 33200000000) }'; }
if [[ ${1:-} == --self-check ]]; then
  ! temp_warning 84.9 && temp_warning 85 && ! temp_limit 89.9 && temp_limit 90 \
    && ! vram_limit 33199999999 && vram_limit 33200000000
  echo 'guard self-check: PASS'
  exit 0
fi
run_dir=${1:?usage: run-guarded.sh CAPTURE_DIR}
mode=${2:-native}
model=${M31_MODEL:-/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf}
image=localhost/miinfer-dev:rocm-7.2.1
expected_model=7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169
name=m31-accel-0001-$(basename "$run_dir")
model_file=$(basename "$model")
mkdir -p "$run_dir"
[[ ! -e "$run_dir/exit.status" ]] || { echo "capture directory already used: $run_dir" >&2; exit 2; }
podman container exists "$name" && { echo "refusing existing container $name" >&2; exit 2; }

read_snapshot() { rocm-smi --showtemp --showmeminfo vram --showpower --showclocks 2>&1; }
parse_temp() { awk '/Temperature \(Sensor junction\)/ { print $NF; exit }'; }
parse_used() { awk '/VRAM Total Used Memory \(B\)/ { print $NF; exit }'; }
pre=$(read_snapshot)
temp=$(printf '%s\n' "$pre" | parse_temp)
used=$(printf '%s\n' "$pre" | parse_used)
[[ $temp =~ ^[0-9]+([.][0-9]+)?$ && $used =~ ^[0-9]+$ ]] || { echo 'preflight telemetry unavailable' >&2; exit 3; }
temp_warning "$temp" && { echo "preflight is already at warning temperature (85C): $temp" >&2; exit 4; }
(( used < 1073741824 )) || { echo "preflight VRAM already in use: $used" >&2; exit 5; }
model_sha=$(sha256sum "$model" | cut -d' ' -f1)
[[ $model_sha == "$expected_model" ]] || { echo "unexpected model SHA: $model_sha" >&2; exit 6; }
source_sha=$(git -C "$repo" rev-parse HEAD)
printf 'source_sha=%s\nmodel_sha=%s\nimage=%s\nimage_id=%s\nrocr_visible_devices=0\nstart_junction_c=%s\nstart_vram_used_bytes=%s\nthermal_warning_c=85\nthermal_stop_c=90\nfree_vram_floor_bytes=1073741824\n' \
  "$source_sha" "$model_sha" "$image" "$(podman image inspect --format '{{.Id}}' "$image")" "$temp" "$used" > "$run_dir/environment.txt"
printf '%s %s\n' "$(date -u +%FT%TZ)" "$pre" > "$run_dir/telemetry.log"

case "$mode" in
  native)
    command=(/workspace/MIInfer/build/m31-accel-0001/miinfer-m31-accel-0001-agent64k
      "/models/$model_file" /captures/session.miinfer)
    ;;
  agent)
    command=(bash /workspace/MIInfer/experiments/m31-accel-0001/run-agent-server.sh)
    ;;
  *) echo "unknown workload mode: $mode (expected native or agent)" >&2; exit 2 ;;
esac
if ! podman run --detach --name "$name" --device /dev/kfd --device /dev/dri \
  --security-opt=label=disable --group-add=keep-groups -e ROCR_VISIBLE_DEVICES=0 \
  -e "M31_MODEL_FILE=$model_file" -e "M31_SOURCE_SHA=$source_sha" \
  -v "$repo:/workspace/MIInfer:Z" -v "$(dirname "$model"):/models:ro" \
  -v "$run_dir:/captures:Z" "$image" "${command[@]}" > "$run_dir/container.id"; then
  echo 125 > "$run_dir/exit.status"
  exit 125
fi
podman logs --follow "$name" >> "$run_dir/runner.log" 2>&1 &
log_pid=$!
rc=0
warned=0
while :; do
  state=$(podman inspect --format '{{.State.Status}}' "$name" 2>/dev/null || echo unknown)
  [[ $state != created ]] || { sleep 1; continue; }
  [[ $state == running ]] || { [[ $state == stopping ]] && sleep 1 && continue; break; }
  sleep 2
  state=$(podman inspect --format '{{.State.Status}}' "$name" 2>/dev/null || echo unknown)
  [[ $state == running ]] || continue
  snapshot=$(read_snapshot)
  now=$(date -u +%FT%TZ)
  printf '%s %s\n' "$now" "$snapshot" >> "$run_dir/telemetry.log"
  temp=$(printf '%s\n' "$snapshot" | parse_temp)
  used=$(printf '%s\n' "$snapshot" | parse_used)
  if [[ ! $temp =~ ^[0-9]+([.][0-9]+)?$ || ! $used =~ ^[0-9]+$ ]]; then
    echo "ABORT telemetry missing at $now" >> "$run_dir/guard.log"; rc=125
  elif temp_warning "$temp" && (( warned == 0 )); then
    echo "WARN junction=${temp}C threshold=85C at $now; operator inspection required" >> "$run_dir/guard.log"
    warned=1
  fi
  if temp_limit "$temp"; then
    echo "STOP junction=${temp}C threshold=90C at $now" >> "$run_dir/guard.log"; rc=124
  elif vram_limit "$used"; then
    echo "STOP used_vram=${used} threshold=33200000000 at $now" >> "$run_dir/guard.log"; rc=123
  fi
  if (( rc != 0 )); then
    podman stop --time 10 "$name" >/dev/null 2>&1 || true
    for _ in {1..12}; do
      state=$(podman inspect --format '{{.State.Status}}' "$name" 2>/dev/null || echo unknown)
      [[ $state == exited ]] && break
      sleep 1
    done
    [[ $state == exited ]] || podman kill "$name" >/dev/null 2>&1 || true
    break
  fi
done
wait "$log_pid" 2>/dev/null || true
container_rc=$(podman inspect --format '{{.State.ExitCode}}' "$name" 2>/dev/null || echo 125)
if (( rc == 0 )); then rc=$container_rc; fi
printf '%s\n' "$rc" > "$run_dir/exit.status"
printf 'run_exit=%s container_exit=%s end_time=%s\n' "$rc" "$container_rc" "$(date -u +%FT%TZ)" >> "$run_dir/environment.txt"
podman rm "$name" >/dev/null 2>&1 || true
exit "$rc"
