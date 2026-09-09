#!/usr/bin/env python3
"""Record MIInfer runtime-only prefill/decode measurements."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import platform
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(command):
    try:
        return subprocess.run(command, capture_output=True, text=True, check=False).stdout
    except OSError:
        return "UNAVAILABLE\n"


def run_case(binary, model, prompt_words, mode, max_tokens, output, timeout_seconds):
    prompt = "hello " * prompt_words
    env = os.environ.copy()
    if mode == "experimental":
        env["MIINFER_PREFILL_LAYER_MAJOR"] = "1"
    suffix = f"{mode}-words{prompt_words}-max{max_tokens}"
    command = [str(binary), "run", str(model)]
    if len(prompt) > 100_000:
        prompt_file = output / f"{suffix}.prompt.txt"
        prompt_file.write_text(prompt)
        command.extend(["--prompt-file", str(prompt_file)])
    else:
        command.extend(["--prompt", prompt])
    command.extend(["--max-tokens", str(max_tokens)])
    began = time.monotonic()
    result = subprocess.run(
        command,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=env, timeout=timeout_seconds, check=False)
    elapsed_ms = (time.monotonic() - began) * 1000.0
    (output / f"{suffix}.stdout").write_text(result.stdout)
    (output / f"{suffix}.stderr").write_text(result.stderr)
    prefill = re.search(r"Prefill Tokens:.*?\(([0-9.]+) ms, ([0-9.]+) tok/s\)", result.stderr)
    decode = re.search(r"Decode Tokens:.*?\(([0-9.]+) ms, ([0-9.]+) tok/s\)", result.stderr)
    ttft = re.search(r"First Token TTFT:\s+([0-9.]+) ms", result.stderr)
    total = re.search(r"Total Latency:\s+([0-9.]+) ms", result.stderr)
    prompt_match = re.search(r"Prompt tokens:\s+(\d+)", result.stderr)
    generated_match = re.search(r"Decode Tokens:\s+(\d+)\s+tokens", result.stderr)
    peak_vram = re.search(r"device_peak_allocated_bytes=(\d+)", result.stderr)
    return {
        "mode": mode,
        "benchmark_kind": "pp" if max_tokens == 1 else "tg",
        "max_tokens": max_tokens,
        "prompt_words": prompt_words,
        "prompt_tokens": None if prompt_match is None else int(prompt_match.group(1)),
        "generated_tokens": None if generated_match is None else int(generated_match.group(1)),
        "prefill_ms": None if prefill is None else float(prefill.group(1)),
        "prefill_tok_s": None if prefill is None else float(prefill.group(2)),
        "decode_ms": None if decode is None else float(decode.group(1)),
        "decode_tok_s": None if decode is None else float(decode.group(2)),
        "first_decode_token_ms": None if ttft is None else float(ttft.group(1)),
        "ttft_ms": None if prefill is None or ttft is None
            else float(prefill.group(1)) + float(ttft.group(1)),
        "steady_decode_tokens": None if max_tokens == 1 or generated_match is None
            else max(int(generated_match.group(1)) - 1, 0),
        "steady_decode_ms": None if max_tokens == 1 or decode is None or ttft is None
            else max(float(decode.group(1)) - float(ttft.group(1)), 0.0),
        "total_ms": None if total is None else float(total.group(1)),
        "peak_vram_bytes": None if peak_vram is None else int(peak_vram.group(1)),
        "wall_ms": elapsed_ms,
        "returncode": result.returncode,
    }


def add_steady_rate(case):
    tokens = case["steady_decode_tokens"]
    milliseconds = case["steady_decode_ms"]
    case["steady_decode_tok_s"] = (1000.0 * tokens / milliseconds
                                    if tokens is not None and milliseconds else None)
    return case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/m18-runtime"))
    parser.add_argument("--prompts", default="8,128,512")
    parser.add_argument("--modes", default="default,experimental")
    parser.add_argument("--max-tokens", type=int, default=1,
                        help="1 for PP-only/first-token runs; use 64 or 128 for TG")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="per-case timeout in seconds")
    args = parser.parse_args()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    output = args.output / f"{stamp}-{os.getpid()}"
    output.mkdir(parents=True)
    model_sha256 = sha256(args.model)
    tracked_environment = (
        "HIP_VISIBLE_DEVICES", "HSA_OVERRIDE_GFX_VERSION", "ROCM_PATH",
        "MIINFER_HIP_GRAPH", "MIINFER_PREFILL_LAYER_MAJOR",
        "MIINFER_PREFILL_DENSE_PROJECTIONS", "MIINFER_PREFILL_DENSE_QKV",
        "MIINFER_PREFILL_DENSE_FFN_DOWN")
    metadata = {
        "binary": str(args.binary.resolve()),
        "model": str(args.model.resolve()),
        "model_sha256": model_sha256,
        "git_commit": command_output(["git", "rev-parse", "HEAD"]).strip(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "environment": {key: os.environ[key] for key in tracked_environment if key in os.environ},
        "hardware_before": command_output(["rocm-smi"]),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    cases = []
    for mode in args.modes.split(","):
        for words in (int(value) for value in args.prompts.split(",")):
            print(f"running mode={mode} prompt_words={words}", flush=True)
            cases.append(add_steady_rate(
                run_case(args.binary, args.model, words, mode, args.max_tokens, output,
                         args.timeout)))
    (output / "summary.json").write_text(json.dumps({
        **metadata,
        "hardware_after": command_output(["rocm-smi"]),
        "cases": cases,
    }, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
