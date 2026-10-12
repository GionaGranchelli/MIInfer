#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
[[ -f "$repo/CMakeLists.txt" ]] || { echo "invalid repository root: $repo" >&2; exit 2; }
temp_warning() { awk -v t="$1" 'BEGIN { exit !(t >= 85.0) }'; }
temp_limit() { awk -v t="$1" 'BEGIN { exit !(t >= 90.0) }'; }
thermal_action() { awk -v t="$1" 'BEGIN { print (t >= 90 ? "stop" : t >= 85 ? "inspect" : "continue") }'; }
vram_limit() { awk -v b="$1" 'BEGIN { exit !(b >= 33200000000) }'; }
valid_mode() { case "$1" in native|agent|prefix64k|decode) return 0 ;; *) return 1 ;; esac; }
if [[ ${1:-} == --self-check ]]; then
  ! temp_warning 84.9 && temp_warning 85 && ! temp_limit 89.9 && temp_limit 90 \
    && [[ $(thermal_action 84.9) == continue && $(thermal_action 85) == inspect && $(thermal_action 90) == stop ]] \
    && ! vram_limit 33199999999 && vram_limit 33200000000 \
    && valid_mode native && valid_mode agent && valid_mode prefix64k && valid_mode decode \
    && ! valid_mode unknown
  echo 'guard self-check: PASS'
  exit 0
fi
run_dir=${1:?usage: run-guarded.sh CAPTURE_DIR}
mode=${2:-native}
valid_mode "$mode" || { echo "unknown workload mode: $mode (expected native, agent, prefix64k, or decode)" >&2; exit 2; }
model=${M31_MODEL:-/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf}
image=localhost/miinfer-dev:rocm-7.2.1
expected_model=7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169
name=m31-accel-0001-$(basename "$run_dir")
rc=0
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
[[ -z $(git -C "$repo" status --porcelain) ]] || { echo 'refusing dirty source worktree' >&2; exit 7; }
case "$mode" in
  native) binaries=(build/m31-accel-0001/miinfer-m31-accel-0001-agent64k) ;;
  agent) binaries=(build/m31-accel-0001/miinfer build/m31-agent64k-host/miinfer-m31-agent-token-count) ;;
  prefix64k) binaries=(build/m31-accel-0001/miinfer-prefix-state-reuse-bench) ;;
  decode) binaries=(build/m31-accel-0001/miinfer-m31-accel-0001-decode-bench) ;;
esac
for binary in "${binaries[@]}"; do
  [[ -x "$repo/$binary" ]] || { echo "required built executable missing: $binary" >&2; exit 8; }
done
printf 'source_sha=%s\nmodel_sha=%s\nimage=%s\nimage_id=%s\nrocr_visible_devices=0\nstart_junction_c=%s\nstart_vram_used_bytes=%s\nthermal_warning_c=85\nthermal_stop_c=90\nfree_vram_floor_bytes=1073741824\n' \
  "$source_sha" "$model_sha" "$image" "$(podman image inspect --format '{{.Id}}' "$image")" "$temp" "$used" > "$run_dir/environment.txt"
for binary in "${binaries[@]}"; do
  printf 'binary_%s_sha256=%s\n' "$(basename "$binary")" "$(sha256sum "$repo/$binary" | cut -d' ' -f1)" >> "$run_dir/environment.txt"
done
printf '%s %s\n' "$(date -u +%FT%TZ)" "$pre" > "$run_dir/telemetry.log"

case "$mode" in
  native)
    command=(/workspace/MIInfer/build/m31-accel-0001/miinfer-m31-accel-0001-agent64k
      "/models/$model_file" /captures/session.miinfer)
    ;;
  agent)
    command=(bash /workspace/MIInfer/experiments/m31-accel-0001/run-agent-server.sh)
    ;;
  prefix64k)
    if [[ -n ${M31_PREFIX_FIRST_TOKEN_IDS:-} || -n ${M31_PREFIX_SECOND_TOKEN_IDS:-} ]]; then
      [[ -n ${M31_PREFIX_FIRST_TOKEN_IDS:-} && -n ${M31_PREFIX_SECOND_TOKEN_IDS:-}
        && ${M31_PREFIX_FIRST_TOKEN_IDS:-} != */* && ${M31_PREFIX_SECOND_TOKEN_IDS:-} != */*
        && -f "$run_dir/${M31_PREFIX_FIRST_TOKEN_IDS:-}" && -f "$run_dir/${M31_PREFIX_SECOND_TOKEN_IDS:-}" ]] \
        || { echo 'both production token-ID artifacts must exist inside CAPTURE_DIR' >&2; exit 2; }
      command=(/workspace/MIInfer/build/m31-accel-0001/miinfer-prefix-state-reuse-bench
        --request-pair "/captures/$M31_PREFIX_FIRST_TOKEN_IDS" "/captures/$M31_PREFIX_SECOND_TOKEN_IDS"
        "/models/$model_file")
    else
      command=(/workspace/MIInfer/build/m31-accel-0001/miinfer-prefix-state-reuse-bench
        --64k-only "/models/$model_file")
    fi
    ;;
  decode)
    command=(/workspace/MIInfer/build/m31-accel-0001/miinfer-m31-accel-0001-decode-bench
      "/models/$model_file" "${M31_DECODE_PROMPT_TOKENS:-65536}"
      "${M31_DECODE_TOKENS:-128}" "${M31_DECODE_ITERATIONS:-5}" /captures/decode.json)
    if [[ -n ${M31_DECODE_TOKEN_IDS:-} ]]; then
      [[ $M31_DECODE_TOKEN_IDS != */* && -f "$run_dir/$M31_DECODE_TOKEN_IDS" ]] \
        || { echo 'production token-ID artifact must exist inside CAPTURE_DIR' >&2; exit 2; }
      command+=(--prompt-token-ids "/captures/$M31_DECODE_TOKEN_IDS")
    fi
    ;;
  *) echo "unknown workload mode: $mode (expected native, agent, prefix64k, or decode)" >&2; exit 2 ;;
esac
if ! podman run --detach --name "$name" --device /dev/kfd --device /dev/dri \
  --security-opt=label=disable --group-add=keep-groups -e ROCR_VISIBLE_DEVICES=0 \
  -e "M31_MODEL_FILE=$model_file" -e "M31_MODEL_SHA256=$model_sha" \
  -e "M31_SOURCE_SHA=$source_sha" \
  -v "$repo:/workspace/MIInfer:Z" -v "$(dirname "$model"):/models:ro" \
  -v "$run_dir:/captures:Z" "$image" "${command[@]}" > "$run_dir/container.id"; then
  echo 125 > "$run_dir/exit.status"
  exit 125
fi
abort_guard() {
  local signal="$1"
  echo "ABORT signal=$signal at $(date -u +%FT%TZ)" >> "$run_dir/guard.log"
  rc=125
  podman stop --time 10 "$name" >/dev/null 2>&1 || true
}
trap 'abort_guard HUP' HUP
trap 'abort_guard INT' INT
trap 'abort_guard TERM' TERM
podman logs --follow "$name" >> "$run_dir/runner.log" 2>&1 &
log_pid=$!
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
  elif [[ $(thermal_action "$temp") == stop ]]; then
    echo "STOP junction=${temp}C threshold=90C at $now" >> "$run_dir/guard.log"; rc=124
  elif [[ $(thermal_action "$temp") == inspect ]]; then
    echo "STOP junction=${temp}C warning_threshold=85C; operator inspection and a fresh capture are required" >> "$run_dir/guard.log"; rc=122
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
