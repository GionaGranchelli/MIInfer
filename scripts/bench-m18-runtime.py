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


def run_case(binary, model, prompt_words, mode, output):
    prompt = "hello " * prompt_words
    env = os.environ.copy()
    if mode == "experimental":
        env["MIINFER_PREFILL_LAYER_MAJOR"] = "1"
    began = time.monotonic()
    result = subprocess.run(
        [str(binary), "run", str(model), "--prompt", prompt, "--max-tokens", "1"],
        capture_output=True, text=True, env=env, timeout=900, check=False)
    elapsed_ms = (time.monotonic() - began) * 1000.0
    (output / f"{mode}-words{prompt_words}.stdout").write_text(result.stdout)
    (output / f"{mode}-words{prompt_words}.stderr").write_text(result.stderr)
    prefill = re.search(r"Prefill Tokens:.*?\(([0-9.]+) ms, ([0-9.]+) tok/s\)", result.stderr)
    decode = re.search(r"Decode Tokens:.*?\(([0-9.]+) ms, ([0-9.]+) tok/s\)", result.stderr)
    ttft = re.search(r"First Token TTFT:\s+([0-9.]+) ms", result.stderr)
    total = re.search(r"Total Latency:\s+([0-9.]+) ms", result.stderr)
    prompt_match = re.search(r"Prompt tokens:\s+(\d+)", result.stderr)
    peak_vram = re.search(r"device_peak_allocated_bytes=(\d+)", result.stderr)
    return {
        "mode": mode,
        "prompt_words": prompt_words,
        "prompt_tokens": None if prompt_match is None else int(prompt_match.group(1)),
        "generated_tokens": 1 if decode else 0,
        "prefill_ms": None if prefill is None else float(prefill.group(1)),
        "prefill_tok_s": None if prefill is None else float(prefill.group(2)),
        "decode_ms": None if decode is None else float(decode.group(1)),
        "decode_tok_s": None if decode is None else float(decode.group(2)),
        "first_decode_token_ms": None if ttft is None else float(ttft.group(1)),
        "ttft_ms": None if prefill is None or ttft is None
            else float(prefill.group(1)) + float(ttft.group(1)),
        "total_ms": None if total is None else float(total.group(1)),
        "peak_vram_bytes": None if peak_vram is None else int(peak_vram.group(1)),
        "wall_ms": elapsed_ms,
        "returncode": result.returncode,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/m18-runtime"))
    parser.add_argument("--prompts", default="8,128,512")
    parser.add_argument("--modes", default="default,experimental")
    args = parser.parse_args()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    output = args.output / f"{stamp}-{os.getpid()}"
    output.mkdir(parents=True)
    model_sha256 = sha256(args.model)
    metadata = {
        "binary": str(args.binary.resolve()),
        "model": str(args.model.resolve()),
        "model_sha256": model_sha256,
        "git_commit": command_output(["git", "rev-parse", "HEAD"]).strip(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "environment": {key: os.environ[key] for key in (
            "HIP_VISIBLE_DEVICES", "HSA_OVERRIDE_GFX_VERSION", "ROCM_PATH",
            "MIINFER_HIP_GRAPH") if key in os.environ},
        "hardware_before": command_output(["rocm-smi"]),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    cases = []
    for mode in args.modes.split(","):
        for words in (int(value) for value in args.prompts.split(",")):
            print(f"running mode={mode} prompt_words={words}", flush=True)
            cases.append(run_case(args.binary, args.model, words, mode, output))
    (output / "summary.json").write_text(json.dumps({
        **metadata,
        "hardware_after": command_output(["rocm-smi"]),
        "cases": cases,
    }, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
