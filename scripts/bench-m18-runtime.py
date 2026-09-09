#!/usr/bin/env python3
"""Record MIInfer runtime-only prefill/decode measurements."""
import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path


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
    return {
        "mode": mode,
        "prompt_words": prompt_words,
        "prompt_tokens": None if prompt_match is None else int(prompt_match.group(1)),
        "generated_tokens": 1 if decode else 0,
        "prefill_ms": None if prefill is None else float(prefill.group(1)),
        "prefill_tok_s": None if prefill is None else float(prefill.group(2)),
        "decode_ms": None if decode is None else float(decode.group(1)),
        "decode_tok_s": None if decode is None else float(decode.group(2)),
        "ttft_ms": None if ttft is None else float(ttft.group(1)),
        "total_ms": None if total is None else float(total.group(1)),
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
    cases = []
    for mode in args.modes.split(","):
        for words in (int(value) for value in args.prompts.split(",")):
            print(f"running mode={mode} prompt_words={words}", flush=True)
            cases.append(run_case(args.binary, args.model, words, mode, output))
    (output / "summary.json").write_text(json.dumps({
        "binary": str(args.binary.resolve()),
        "model": str(args.model.resolve()),
        "cases": cases,
    }, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
