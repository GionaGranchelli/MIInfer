#!/usr/bin/env python3
"""Run at most two guarded same-binary E2-0003 control/candidate pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


MODEL_SHA256 = "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
PROMPT_TOKENS = 1024
OUTPUT_TOKENS = 32
CONTEXT_TOKENS = 1280
SAMPLER_CONFIG = {
    "temperature": "0",
    "top_p": "1",
    "top_k": "1",
    "repetition_penalty": "1",
    "frequency_penalty": "0",
    "presence_penalty": "0",
    "stop_token_ids": "disabled",
    "repeat_last_n": "256",
    "reset_state_before": "true",
    "use_hip_graph": "true",
    "seed": "42",
    "sampler": "greedy_argmax",
}
EXPECTED_BDF = "0000:06:00.0"
WARN_C = 80.0
STOP_C = 85.0
ROCM_FLAGS = ("--showbus", "--showuniqueid", "--showproductname", "--showclocks",
              "--showpower", "--showtemp", "--showmemuse", "--showmeminfo", "vram",
              "--showuse", "--showpids")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_capture(command: list[str], timeout: int = 30) -> dict:
    try:
        result = subprocess.run(command, text=True, capture_output=True,
                                timeout=timeout, check=False)
        return {"command": command, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "returncode": None, "stdout": "", "stderr": str(error)}


def parse_rocm(raw: str, gpu_index: int) -> dict:
    body = "\n".join(line for line in raw.splitlines()
                     if re.search(rf"\bGPU\[{gpu_index}\]", line))

    def number(pattern: str):
        match = re.search(pattern, body, re.I)
        return float(match.group(1)) if match else None

    bus = re.search(r"PCI Bus:\s*([^\s]+)", body)
    unique = re.search(r"Unique ID:\s*([^\s]+)", body)
    product = re.search(r"Card Series:\s*(.+)", body)
    return {
        "pci_bdf": bus.group(1) if bus else None,
        "unique_id": unique.group(1) if unique else None,
        "product": product.group(1).strip() if product else None,
        "junction_c": number(r"Temperature \(Sensor junction\) \(C\):\s*([0-9.]+)"),
        "edge_c": number(r"Temperature \(Sensor edge\) \(C\):\s*([0-9.]+)"),
        "sclk_mhz": number(r"sclk clock level:\s*\d+:\s*\(([0-9.]+)Mhz\)"),
        "mclk_mhz": number(r"mclk clock level:\s*\d+:\s*\(([0-9.]+)Mhz\)"),
        "power_w": number(r"Current Socket Graphics Package Power \(W\):\s*([0-9.]+)"),
        "vram_used_bytes": (int(match.group(1)) if (match := re.search(
            r"VRAM Total Used Memory \(B\):\s*(\d+)", body, re.I)) else None),
        "gpu_use_percent": number(r"GPU use \(%\):\s*([0-9.]+)"),
        "kfd_idle": "No KFD PIDs currently running" in raw,
    }


def parse_rocm_map(raw: str) -> dict[int, str]:
    result = {}
    for line in raw.splitlines():
        index = re.search(r"\bGPU\[(\d+)\]", line)
        bus = re.search(r"PCI Bus:\s*([^\s]+)", line, re.I)
        if index and bus:
            result[int(index.group(1))] = bus.group(1)
    return result


def parse_metrics(metrics: str) -> tuple[dict[str, int], dict[str, str], dict]:
    events: dict[str, int] = {}
    properties: dict[str, str] = {}
    result = None
    for line in metrics.splitlines():
        event = re.search(r"\bevent=([^\s]+).*?\bepoch_ms=(\d+)", line)
        if event:
            events[event.group(1)] = int(event.group(2))
        if line.startswith("result="):
            result = json.loads(line[len("result="):])
        elif "=" in line and not line.startswith("event="):
            key, value = line.split("=", 1)
            if key.replace("_", "").isalnum():
                properties[key] = value
    return events, properties, result


def nearest_phase_memory(events: dict[str, int], samples: list[dict], event_name: str) -> dict:
    target = events.get(event_name)
    if target is None:
        return {"event": event_name, "vram_used_bytes": None, "sample_delta_ms": None}
    candidates = [sample for sample in samples
                  if sample.get("returncode") == 0
                  and sample.get("parsed", {}).get("vram_used_bytes") is not None]
    if not candidates:
        return {"event": event_name, "vram_used_bytes": None, "sample_delta_ms": None}
    closest = min(candidates, key=lambda sample: abs(sample["epoch_ms"] - target))
    delta = abs(closest["epoch_ms"] - target)
    if delta > 1500:
        return {"event": event_name, "vram_used_bytes": None, "sample_delta_ms": delta}
    return {"event": event_name,
            "vram_used_bytes": closest["parsed"]["vram_used_bytes"],
            "sample_delta_ms": delta}


def make_prompt(count: int = PROMPT_TOKENS, seed: int = 77) -> list[int]:
    tokens = []
    for _ in range(count):
        seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
        tokens.append(seed % 151643 + 1)
    return tokens


def prompt_fingerprint(tokens: list[int]) -> str:
    value = 14695981039346656037
    for token in tokens:
        for shift in (0, 8, 16, 24):
            value = ((value ^ ((token >> shift) & 0xFF)) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def rocm_command(gpu_index: int) -> list[str]:
    return ["rocm-smi", "-d", str(gpu_index), *ROCM_FLAGS]


def capture_state(directory: Path, label: str, gpu_index: int) -> dict:
    sample = run_capture(rocm_command(gpu_index), timeout=10)
    raw = sample["stdout"] + sample["stderr"]
    (directory / f"{label}.rocm-smi.txt").write_text(raw)
    parsed = parse_rocm(raw, gpu_index)
    parsed["returncode"] = sample["returncode"]
    return parsed


def telemetry_sampler(path: Path, gpu_index: int, stop: threading.Event, start_ns: int):
    with path.open("x") as output:
        while not stop.is_set():
            tick = time.monotonic()
            sample = run_capture(rocm_command(gpu_index), timeout=5)
            raw = sample["stdout"] + sample["stderr"]
            record = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "epoch_ms": time.time_ns() / 1_000_000,
                "elapsed_ms": (time.monotonic_ns() - start_ns) / 1_000_000,
                "returncode": sample["returncode"],
                "parsed": parse_rocm(raw, gpu_index),
                "raw": raw,
            }
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
            output.flush()
            stop.wait(max(0.0, 1.0 - (time.monotonic() - tick)))


def summarize_telemetry(path: Path) -> dict:
    samples = [json.loads(line) for line in path.read_text().splitlines() if line]
    parsed = [item["parsed"] for item in samples if item.get("returncode") == 0]

    def values(field):
        return [row[field] for row in parsed if isinstance(row.get(field), (int, float))]

    return {
        "sample_count": len(samples),
        "successful_sample_count": len(parsed),
        "failed_sample_count": sum(item.get("returncode") != 0 for item in samples),
        "peak_junction_c": max(values("junction_c"), default=None),
        "peak_vram_used_bytes": max(values("vram_used_bytes"), default=None),
        "sclk_mhz_observed": sorted(set(values("sclk_mhz"))),
        "mclk_mhz_observed": sorted(set(values("mclk_mhz"))),
        "power_w_observed": sorted(set(values("power_w"))),
        "gpu_use_percent_observed": sorted(set(values("gpu_use_percent"))),
        "missing_junction_samples": sum(row.get("junction_c") is None for row in parsed),
        "missing_fields": [field for field in ("junction_c", "sclk_mhz", "mclk_mhz",
                                                  "power_w", "vram_used_bytes", "gpu_use_percent")
                           if not values(field)],
        "raw_samples_file": path.name,
    }


def container_command(args, route: str, pair: int, output: Path) -> list[str]:
    runtime = args.container_runtime
    command = [runtime, "run", "--rm", "--pull=never", "--network=none", "--userns=keep-id",
               "--device=/dev/kfd", "--device=/dev/dri", "--group-add", "keep-groups",
               "--security-opt=label=disable",
               "-v", f"{args.binary.resolve()}:/bench/run:ro,Z",
               "-v", f"{args.model.resolve()}:/model.gguf:ro,Z",
               "-v", f"{output.resolve()}:/results:Z",
               "--env", f"M31_EXPECTED_GPU_BDF={args.expected_bdf}",
               "--env", "MIINFER_HIP_GRAPH=1",
               "--env", f"MIINFER_EXPERIMENTAL_MMQ_GATEUP_ONLY={'1' if route == 'mmq_only' else '0'}",
               args.image, "/bench/run", "/model.gguf", "8K",
               "--generate", str(OUTPUT_TOKENS), "--prompt-tokens", str(PROMPT_TOKENS),
               "--iterations", "1", "--warmup", "0",
               "--output", f"/results/{route}-pair-{pair}.metrics"]
    return command


def call(args, out: Path, route: str, pair: int, order: int) -> dict:
    if sha256_file(args.binary) != args.binary_hash:
        raise RuntimeError("benchmark binary changed after preflight")
    if sha256_file(args.model) != args.model_hash:
        raise RuntimeError("model changed after preflight")
    label = f"{route}-pair-{pair}"
    before = capture_state(out / "snapshots", f"{label}-before", args.gpu_index)
    if (before["returncode"] != 0 or before["pci_bdf"] != args.expected_bdf
            or before["junction_c"] is None or before["junction_c"] >= WARN_C
            or before["gpu_use_percent"] is None or before["gpu_use_percent"] > 1
            or not before["kfd_idle"]):
        raise RuntimeError(f"GPU identity/idle/thermal preflight failed before {label}: {before}")
    telemetry_path = out / "telemetry" / f"{label}.jsonl"
    stop = threading.Event()
    start_ns = time.monotonic_ns()
    sampler = threading.Thread(target=telemetry_sampler,
                               args=(telemetry_path, args.gpu_index, stop, start_ns), daemon=True)
    sampler.start()
    guard_log = out / f"{label}.guard.log"
    env = os.environ.copy()
    env.update({"M31_GPU_INDEX": str(args.gpu_index), "M31_GPU_PCI_BDF": args.expected_bdf,
                "M31_WARN_JUNCTION_C": str(WARN_C), "M31_MAX_JUNCTION_C": str(STOP_C)})
    argv = [str(args.guard), str(args.expected_seconds), str(guard_log),
            *container_command(args, route, pair, out)]
    started = time.monotonic()
    try:
        completed = subprocess.run(argv, env=env, text=True, capture_output=True, check=False)
    except OSError as error:
        completed = SimpleProcessResult(127, "", str(error))
    finally:
        stop.set()
        sampler.join(timeout=8)
    if sampler.is_alive():
        raise RuntimeError(f"telemetry sampler did not stop for {label}")
    wall_ms = (time.monotonic() - started) * 1000
    after = capture_state(out / "snapshots", f"{label}-after", args.gpu_index)
    metrics_path = out / f"{route}-pair-{pair}.metrics"
    metrics = metrics_path.read_text() if metrics_path.exists() else ""
    events, properties, result = parse_metrics(metrics)
    guard_text = guard_log.read_text() if guard_log.exists() else ""
    telemetry = summarize_telemetry(telemetry_path)
    samples = [json.loads(line) for line in telemetry_path.read_text().splitlines() if line]
    generated = result.get("token_ids", []) if result else []
    clean = "event=cleanup_complete" in guard_text and "THERMAL_GUARD_BEGIN" in guard_text
    abort = any(marker in guard_text for marker in
                ("THERMAL_GUARD_ABORT", "WATCHDOG_ABORT", "cleanup_failed", "THERMAL_GUARD_ERROR"))
    phase_memory = {name: nearest_phase_memory(events, samples, name)
                    for name in ("MODEL_ALLOC_READY", "PREFILL_END", "DECODE_END")}
    record = {
        "pair": pair, "order": order, "route": route,
        "requested_environment": {
            "MIINFER_EXPERIMENTAL_MMQ_GATEUP_ONLY": "1" if route == "mmq_only" else "0",
            "MIINFER_HIP_GRAPH": "1",
        },
        "start_state": before, "end_state": after,
        "result": result, "effective_config": properties,
        "generated_token_ids": generated,
        "timings": {
            "model_load_ms": interval(events, "MODEL_LOAD_BEGIN", "MODEL_DATA_READY"),
            "model_alloc_ms": interval(events, "MODEL_ALLOC_BEGIN", "MODEL_ALLOC_READY"),
            "prefill_ms": result.get("prefill_ms") if result else None,
            "decode_ms": result.get("decode_ms") if result else None,
            "prefill_tok_s": result.get("prefill_tok_s") if result else None,
            "decode_tok_s": result.get("decode_tok_s") if result else None,
            "guarded_wall_ms": wall_ms,
        },
        "phase_sampled_vram": phase_memory,
        "guard": {"exit_code": completed.returncode,
                  "clean_process_group": clean, "thermal_or_watchdog_abort": abort,
                  "stdout": completed.stdout, "stderr": completed.stderr,
                  "raw_log": guard_log.name},
        "telemetry": telemetry,
        "raw_metrics": metrics_path.name if metrics_path.exists() else None,
    }
    (out / f"{label}.record.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def interval(events: dict[str, int], start: str, end: str):
    left, right = events.get(start), events.get(end)
    return right - left if left is not None and right is not None and right >= left else None


class SimpleProcessResult:
    def __init__(self, returncode: int, stdout: str, stderr: str):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def valid_call(record: dict) -> bool:
    result = record["result"] or {}
    config = record["effective_config"]
    return (record["guard"]["exit_code"] == 0 and record["guard"]["clean_process_group"]
            and not record["guard"]["thermal_or_watchdog_abort"]
            and result.get("tokens_valid") == "PASS"
            and len(record["generated_token_ids"]) == OUTPUT_TOKENS
            and config.get("prompt_tokens") == str(PROMPT_TOKENS)
            and config.get("generate_tokens") == str(OUTPUT_TOKENS)
            and config.get("context_capacity_tokens") == str(CONTEXT_TOKENS)
            and config.get("prompt_fingerprint_fnv1a64") == record["expected_prompt_fingerprint"]
            and sampler_config_valid(config)
            and record["telemetry"]["successful_sample_count"] > 0
            and record["telemetry"]["failed_sample_count"] == 0
            and record["telemetry"]["missing_junction_samples"] == 0
            and not record["telemetry"]["missing_fields"])


def sampler_config_valid(config: dict[str, str]) -> bool:
    return all(config.get(key) == value for key, value in SAMPLER_CONFIG.items())


def wait_cool(args, out: Path, label: str, baseline_c: float,
              baseline_vram: float | None) -> dict:
    deadline = time.monotonic() + args.cooldown_timeout_seconds
    samples = []
    while time.monotonic() < deadline:
        state = capture_state(out / "snapshots", f"cooldown-{label}-{len(samples):03d}", args.gpu_index)
        samples.append(state)
        vram_ok = (baseline_vram is None or state["vram_used_bytes"] is None
                   or state["vram_used_bytes"] <= baseline_vram + 512 * 1024 * 1024)
        if (state["returncode"] == 0 and state["pci_bdf"] == args.expected_bdf
                and state["junction_c"] is not None
                and state["junction_c"] < WARN_C
                and state["junction_c"] <= baseline_c + 2.0
                and state["gpu_use_percent"] is not None and state["gpu_use_percent"] <= 1
                and state["kfd_idle"] and vram_ok):
            result = {"ready": True, "samples": samples}
            (out / f"cooldown-{label}.json").write_text(json.dumps(result, indent=2) + "\n")
            return result
        time.sleep(10)
    result = {"ready": False, "samples": samples}
    (out / f"cooldown-{label}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def medians(records: list[dict], route: str) -> dict:
    selected = [record for record in records if record["route"] == route and record["result"]]
    result = {}
    for field in ("prefill_tok_s", "decode_tok_s"):
        values = [float(record["result"][field]) for record in selected
                  if record["result"].get(field) is not None]
        result[field] = {"samples": values,
                         "descriptive_median": statistics.median(values) if values else None}
    peaks = [record["telemetry"]["peak_vram_used_bytes"] for record in selected
             if record["telemetry"]["peak_vram_used_bytes"] is not None]
    result["peak_sampled_vram_bytes"] = {
        "samples": peaks, "descriptive_median": statistics.median(peaks) if peaks else None}
    requests = [int(record["result"]["model_weights_bytes"]) for record in selected
                if record["result"].get("model_weights_bytes") is not None]
    result["weight_allocation_request_bytes"] = {
        "samples": requests,
        "descriptive_median": statistics.median(requests) if requests else None}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True,
                        help="same clean-source E2-0003 binary for both routes")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--build-flags", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--expected-image-digest", required=True)
    parser.add_argument("--guard", type=Path, default=Path(__file__).with_name("run-m31-exploratory.sh"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-model-sha", default=MODEL_SHA256)
    parser.add_argument("--expected-bdf", default=EXPECTED_BDF)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--pairs", type=int, choices=(1, 2), default=1)
    parser.add_argument("--expected-seconds", type=int, default=240)
    parser.add_argument("--cooldown-timeout-seconds", type=int, default=900)
    parser.add_argument("--container-runtime", choices=("podman",), default="podman")
    args = parser.parse_args()

    try:
        for path in (args.model, args.binary, args.source_manifest, args.guard):
            if not path.is_file():
                raise ValueError(f"required file is absent: {path}")
        if not os.access(args.binary, os.X_OK) or not os.access(args.guard, os.X_OK):
            raise ValueError("benchmark binary and thermal guard must be executable")
        if args.output_dir.exists():
            raise ValueError(f"refusing to overwrite output directory: {args.output_dir}")
        if args.expected_seconds < 60 or args.expected_seconds > 600:
            raise ValueError("expected-seconds must be in [60, 600]")
        model_hash = sha256_file(args.model)
        binary_hash = sha256_file(args.binary)
        args.binary_hash = binary_hash
        args.model_hash = model_hash
        if model_hash != args.expected_model_sha:
            raise ValueError(f"model SHA mismatch: {model_hash}")
        image = run_capture([args.container_runtime, "image", "inspect", args.image,
                             "--format", "{{.Digest}}"], timeout=30)
        image_digest = image["stdout"].strip()
        if image["returncode"] != 0 or image_digest != args.expected_image_digest:
            raise ValueError(f"container digest mismatch: {image_digest!r}")
        map_probe = run_capture(["rocm-smi", "--showbus", "--showproductname", "--showuniqueid"], timeout=15)
        gpu_map = parse_rocm_map(map_probe["stdout"] + map_probe["stderr"])
        if map_probe["returncode"] != 0 or gpu_map != {args.gpu_index: args.expected_bdf}:
            raise ValueError(f"expected only selected GPU index/BDF, found {gpu_map}")
        rocminfo = run_capture(["rocminfo"], timeout=30)
        gfx906_count = len(re.findall(r"^\s*Name:\s*gfx906\s*$", rocminfo["stdout"], re.M))
        if rocminfo["returncode"] != 0 or gfx906_count != 1:
            raise ValueError(f"expected exactly one gfx906 agent, found {gfx906_count}")

        args.output_dir.mkdir(parents=True)
        out = args.output_dir.resolve()
        (out / "snapshots").mkdir()
        (out / "telemetry").mkdir()
        prompt = make_prompt()
        prompt_path = out / "prompt.ids"
        prompt_path.write_text("".join(f"{token}\n" for token in prompt))
        prompt_hash = sha256_file(prompt_path)
        fingerprint = prompt_fingerprint(prompt)
        (out / "prompt.ids.sha256").write_text(f"{prompt_hash}  prompt.ids\n")
        initial = capture_state(out / "snapshots", "initial", args.gpu_index)
        if (initial["returncode"] != 0 or initial["pci_bdf"] != args.expected_bdf
                or initial["junction_c"] is None or initial["junction_c"] >= WARN_C
                or initial["gpu_use_percent"] is None or initial["gpu_use_percent"] > 1
                or not initial["kfd_idle"]):
            raise ValueError(f"GPU is not identifiable, cool, and idle: {initial}")
        environment = {
            "uname": run_capture(["uname", "-a"], timeout=10),
            "rocm_smi_version": run_capture(["rocm-smi", "--version"], timeout=10),
            "hipconfig": run_capture(["hipconfig", "--full"], timeout=20),
            "rocminfo": rocminfo,
            "gpu_map": gpu_map,
            "container_inspect": image,
        }
        for name, item in environment.items():
            if name == "gpu_map":
                continue
            (out / f"environment-{name}.txt").write_text(item["stdout"] + item["stderr"])
        (out / "environment.json").write_text(json.dumps(
            {"gpu_map": gpu_map,
             "commands": {key: {"command": value["command"],
                                "returncode": value["returncode"]}
                          for key, value in environment.items() if key != "gpu_map"}},
            indent=2) + "\n")
        shutil.copy2(args.source_manifest, out / "source-manifest.sha256")
        source_identity = {
            "source_sha": args.source_sha,
            "benchmark": "e2-0003-checkpointed-call",
            "source_manifest_sha256": sha256_file(args.source_manifest),
            "binary_sha256": binary_hash,
            "model_sha256": model_hash,
            "prompt_ids_sha256": prompt_hash,
            "prompt_tokens": PROMPT_TOKENS,
            "prompt_fingerprint_fnv1a64": fingerprint,
            "sampler_config": SAMPLER_CONFIG,
            "container_image": args.image,
            "container_digest": image_digest,
            "build_flags": args.build_flags,
            "host": {"hostname": platform.node(), "kernel": platform.release()},
            "gpu": {"index": args.gpu_index, "pci_bdf": args.expected_bdf,
                    "unique_id": initial["unique_id"], "product": initial["product"]},
        }
        shutil.copy2(args.source_manifest, out / args.source_manifest.name)
        (out / "source-identity.json").write_text(json.dumps(source_identity, indent=2) + "\n")
        baseline_c = initial["junction_c"]
        baseline_vram = initial["vram_used_bytes"]
        records = []
        pair_parity = {}
        for pair in range(1, args.pairs + 1):
            order = (("control", "mmq_only") if pair == 1 else ("mmq_only", "control"))
            for position, route in enumerate(order, start=1):
                if records:
                    cool = wait_cool(args, out, f"before-{route}-pair-{pair}",
                                     baseline_c, baseline_vram)
                    if not cool["ready"]:
                        print("PAIR_STOP: GPU did not return to comparable idle state", flush=True)
                        break
                print(f"PAIR {pair} CALL {position} START route={route}", flush=True)
                record = call(args, out, route, pair, position)
                record["expected_prompt_fingerprint"] = fingerprint
                (out / f"{route}-pair-{pair}.record.json").write_text(
                    json.dumps(record, indent=2) + "\n")
                records.append(record)
                if not valid_call(record):
                    print(f"PAIR_STOP: call gate failed route={route} pair={pair}", flush=True)
                    break
            pair_records = [record for record in records if record["pair"] == pair]
            if len(pair_records) != 2 or not all(valid_call(record) for record in pair_records):
                break
            left, right = pair_records[0], pair_records[1]
            pair_parity[str(pair)] = left["generated_token_ids"] == right["generated_token_ids"]
            if left["effective_config"].get("prompt_fingerprint_fnv1a64") != right["effective_config"].get("prompt_fingerprint_fnv1a64"):
                pair_parity[str(pair)] = False

        control = medians(records, "control")
        candidate = medians(records, "mmq_only")
        control_prefill = control["prefill_tok_s"]["descriptive_median"]
        candidate_prefill = candidate["prefill_tok_s"]["descriptive_median"]
        control_decode = control["decode_tok_s"]["descriptive_median"]
        candidate_decode = candidate["decode_tok_s"]["descriptive_median"]
        memory_saved = None
        control_vram = control["peak_sampled_vram_bytes"]["descriptive_median"]
        candidate_vram = candidate["peak_sampled_vram_bytes"]["descriptive_median"]
        if control_vram is not None and candidate_vram is not None:
            memory_saved = control_vram - candidate_vram
        control_weight_bytes = control["weight_allocation_request_bytes"]["descriptive_median"]
        candidate_weight_bytes = candidate["weight_allocation_request_bytes"]["descriptive_median"]
        allocation_saved = (control_weight_bytes - candidate_weight_bytes
                            if control_weight_bytes is not None and candidate_weight_bytes is not None
                            else None)
        valid = len(records) >= 2 and all(valid_call(record) for record in records)
        gates = {
            "same_binary": all(record["guard"]["exit_code"] == 0 for record in records),
            "all_pairs_token_parity": bool(pair_parity) and all(pair_parity.values()),
            "sampled_vram_saving_at_least_6gb": memory_saved >= 6_000_000_000 if memory_saved is not None else None,
            "weight_allocation_requests_reduced": allocation_saved > 0 if allocation_saved is not None else None,
            "prefill_at_least_95_percent": candidate_prefill / control_prefill >= 0.95
                if control_prefill and candidate_prefill is not None else None,
            "decode_at_least_85_percent": candidate_decode / control_decode >= 0.85
                if control_decode and candidate_decode is not None else None,
            "decode_above_historical_llama_20_56": candidate_decode > 20.56 if candidate_decode is not None else None,
            "thermal_and_operation_reference": "OPERATION_REFERENCE_NOT_RUN_BY_THIS_HARNESS",
            "hip_graph": "REQUESTED_BY_BENCHMARK; SUCCESSFUL_RUN_REQUIRED",
        }
        decision = "BLOCKED_OR_INCONCLUSIVE"
        if valid and gates["all_pairs_token_parity"] and all(
                gates[key] is True for key in ("sampled_vram_saving_at_least_6gb",
                                                "prefill_at_least_95_percent",
                                                "decode_at_least_85_percent",
                                                "decode_above_historical_llama_20_56")):
            decision = "CANDIDATE_A_SELECTION_GATES_PASS_OPERATION_REFERENCE_STILL_REQUIRED"
        verdict = {
            "schema_version": "1.0.0", "experiment": "E2-0003",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_identity": source_identity,
            "workload": {"prompt_tokens": PROMPT_TOKENS, "generated_tokens": OUTPUT_TOKENS,
                         "context_capacity_tokens": CONTEXT_TOKENS, "cold_prefill": True,
                         "warmups": 0, "repetitions_per_call": 1,
                         "prompt": "deterministic e2_0003_prompt.hpp seed=77; fingerprint recorded per call",
                         "sampler_config": SAMPLER_CONFIG,
                         "miinfer_path": "same binary; HIP Graph enabled; only MMQ Gate/Up env differs"},
            "guard": {"warning_junction_c": WARN_C, "stop_junction_c": STOP_C,
                      "expected_seconds_per_call": args.expected_seconds,
                      "sample_interval_seconds": 1.0, "power_policy_changed": False},
            "correctness": {"pair_generated_id_parity": pair_parity,
                            "whole_model_numerical_equivalence": "NOT_ESTABLISHED",
                            "operation_reference": "NOT_RUN"},
            "verdicts": {"harness": "HARNESS_VALIDATED" if valid else "HARNESS_PARTIAL",
                         "candidate_a_gates": gates, "decision": decision},
            "analysis": {"paired_run_count": len(pair_parity), "sampled_memory_saving_bytes": memory_saved,
                         "weight_allocation_request_saving_bytes": allocation_saved,
                         "descriptive_metrics": {"control": control, "mmq_only": candidate}},
            "runs": records,
        }
        (out / "benchmark.json").write_text(json.dumps(verdict, indent=2) + "\n")
        print(f"RESULT harness={verdict['verdicts']['harness']} decision={decision} "
              f"pairs={len(pair_parity)}", flush=True)
        return 0 if valid else 1
    except Exception as error:
        print(f"E2-0003 BLOCKED: {error}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
