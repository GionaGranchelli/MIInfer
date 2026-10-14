#!/usr/bin/env bash
set -euo pipefail

name=${1:?run name required}
candidate=${2:-0}
tokens=${3:-2048}
runtime_limit=${4:-240}
root=/tmp/m31-q8-lds-gpu0
evidence=$root/evidence
mkdir -p "$evidence"
log=$evidence/$name.log
temps=$evidence/$name.temps
: > "$temps"

args=(
  --rm --name "$name"
  --security-opt label=disable --group-add keep-groups
  --device=/dev/kfd --device=/dev/dri
  -v "$root/build/mi50-release:/build:ro,Z"
  -v /home/machinist/models:/models:ro,Z
  -v /home/machinist/m31-perf-recovery-0001/build/mi50-release:/prompts:ro,Z
  -v "$evidence:/evidence:Z"
  -w /build -e ROCR_VISIBLE_DEVICES=0
)
if [[ "$candidate" == 1 ]]; then
  args+=(-e MIINFER_M31_Q8_4Q_LDS_REUSE=1)
fi

timeout --signal=TERM --kill-after=5s "${runtime_limit}s" podman run "${args[@]}" \
  localhost/miinfer-dev:rocm-7.2.1 \
  /build/miinfer-m31-fp16-full-model /models/Qwen3.8-27B-Q4_K_M.gguf \
  --tokens "$tokens" --prompt-file /prompts/m31-64k-context.txt \
  --q8-only --logits-out "/evidence/$name.f32" >"$log" 2>&1 &
run_pid=$!
warned=0
while kill -0 "$run_pid" 2>/dev/null; do
  junction=$(rocm-smi -d 0 --showtemp 2>/dev/null \
    | sed -n 's/.*junction.*: \([0-9.]*\).*/\1/pI' | head -n 1)
  if [[ -n "$junction" ]]; then
    printf '%s\n' "$junction" >>"$temps"
    if (( ${junction%%.*} >= 80 )) && (( warned == 0 )); then
      printf 'warning_%s_c=%s\n' "$name" "$junction"
      warned=1
    fi
    if (( ${junction%%.*} >= 85 )); then
      printf 'hard_stop_%s_c=%s\n' "$name" "$junction"
      podman stop --time 2 "$name" >/dev/null 2>&1 || true
      break
    fi
  fi
  sleep 2
done
set +e
wait "$run_pid"
status=$?
set -e
printf 'run=%s exit=%s max_junction_c=%s\n' "$name" "$status" \
  "$(sort -nr "$temps" | head -n 1)"
cat "$log"
if [[ -f "$evidence/$name.f32" ]]; then
  sha256sum "$evidence/$name.f32"
fi
exit "$status"
