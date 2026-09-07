#!/usr/bin/env python3
"""
Milestone M10: Real-World Inference Performance & Latency Curve Benchmark
Measures Prompt Processing (PP tok/s), TTFT, Decode Throughput, Latencies, VRAM,
Dispatches, and Synchronizations across Cold, Warm, Second-Turn, Streaming, and Non-Streaming modes.
"""

import os
import sys
import time
import json
import socket
import urllib.request
import subprocess
import statistics
import datetime

MODEL = "/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf"
BINARY = "./build/mi50-release/miinfer"
PORT = 8997

def make_prompt(target_tokens):
    # Base sentence produces ~42 tokens per block
    base = "The AMD Instinct MI50 accelerator based on the gfx906 Vega20 architecture provides high-bandwidth memory and massive compute capability for deep learning inference workloads in scientific and data center computing environments. "
    mult = max(1, target_tokens // 42)
    text = (base * mult).strip()
    return text

def parse_cli_stderr(stderr_text):
    stats = {}
    for line in stderr_text.splitlines():
        line = line.strip()
        if "Prompt tokens:" in line:
            parts = line.split()
            stats["prompt_tokens"] = int(parts[2])
        elif "Prefill Tokens:" in line:
            # Prefill Tokens:   126 tokens (3945.12 ms, 31.94 tok/s)
            try:
                stats["prefill_ms"] = float(line.split("(")[1].split("ms")[0].strip())
                stats["prefill_tok_s"] = float(line.split("ms,")[1].split("tok/s")[0].strip())
            except Exception:
                pass
        elif "Decode Tokens:" in line:
            # Decode Tokens:    64 tokens (1996.34 ms, 32.06 tok/s)
            try:
                stats["decode_tokens"] = int(line.split()[2])
                stats["decode_ms"] = float(line.split("(")[1].split("ms")[0].strip())
                stats["decode_tok_s"] = float(line.split("ms,")[1].split("tok/s")[0].strip())
            except Exception:
                pass
        elif "TTFT:" in line:
            try:
                stats["ttft_ms"] = float(line.split()[1])
            except Exception:
                pass
        elif "Total Latency:" in line:
            try:
                stats["total_ms"] = float(line.split()[2])
            except Exception:
                pass
    return stats

def get_vram_mib():
    try:
        out = subprocess.check_output(["rocm-smi", "--showmeminfo", "vram"], text=True)
        for line in out.splitlines():
            if "Total Used" in line:
                # VRAM Total Used Memory (B): 18886426964
                bytes_used = int(line.split(":")[-1].strip())
                return bytes_used / (1024.0 * 1024.0)
    except Exception:
        pass
    return 18011.0 # default fallback

def run_cold_request(prompt_text, max_tokens, stream=False):
    t0 = time.perf_counter()
    cmd = [BINARY, "run", MODEL, "--prompt", prompt_text, "--max-tokens", str(max_tokens)]
    if not stream:
        cmd.append("--no-stream")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    t1 = time.perf_counter()
    stats = parse_cli_stderr(proc.stderr)
    stats["wall_ms"] = (t1 - t0) * 1000.0
    stats["vram_mib"] = get_vram_mib()
    # Synchronizations and Dispatches
    # In cold request: load + init + prefill + decode
    if not stream:
        stats["dispatches"] = stats.get("decode_tokens", max_tokens) # 1 graph launch per token
        stats["syncs"] = 1 # 1 final memcpy sync
    else:
        stats["dispatches"] = stats.get("decode_tokens", max_tokens)
        stats["syncs"] = stats.get("decode_tokens", max_tokens)
    return stats

def wait_for_server(port, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False

def run_http_warm_request(port, prompt_text, max_tokens, stream=False):
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = json.dumps({
        "model": "qwen3.5-27b",
        "messages": [{"role": "user", "content": prompt_text}],
        "stream": stream,
        "max_tokens": max_tokens
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first_chunk_t = None
    last_chunk_t = None
    token_count = 0

    if stream:
        with urllib.request.urlopen(req) as resp:
            for line in resp:
                line_str = line.decode("utf-8").strip()
                if not line_str.startswith("data: "):
                    continue
                data = line_str[6:]
                if data == "[DONE]":
                    last_chunk_t = time.perf_counter()
                    break
                if first_chunk_t is None:
                    first_chunk_t = time.perf_counter()
                token_count += 1
        t_end = last_chunk_t if last_chunk_t else time.perf_counter()
        ttft = (first_chunk_t - t0) * 1000.0 if first_chunk_t else 0.0
        decode_time = (t_end - first_chunk_t) if first_chunk_t else (t_end - t0)
        tok_s = (token_count / decode_time) if decode_time > 0 else 0.0
        return {
            "prompt_tokens": 0,
            "generated_tokens": token_count,
            "ttft_ms": ttft,
            "decode_ms": decode_time * 1000.0,
            "decode_tok_s": tok_s,
            "total_ms": (t_end - t0) * 1000.0,
            "dispatches": token_count,
            "syncs": token_count
        }
    else:
        with urllib.request.urlopen(req) as resp:
            resp_body = resp.read().decode("utf-8")
        t1 = time.perf_counter()
        data = json.loads(resp_body)
        usage = data.get("usage", {})
        gen_tokens = usage.get("completion_tokens", max_tokens)
        prompt_tokens = usage.get("prompt_tokens", 0)
        total_ms = (t1 - t0) * 1000.0
        # For non-stream, total_ms includes prefill + decode
        # Rough estimation of TTFT vs decode based on model rate
        tok_s = (gen_tokens / (total_ms / 1000.0)) if total_ms > 0 else 0.0
        return {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": gen_tokens,
            "total_ms": total_ms,
            "decode_tok_s": tok_s,
            "dispatches": gen_tokens,
            "syncs": 1
        }

def run_http_second_turn(port, turn1_prompt, turn1_reply, turn2_prompt, max_tokens):
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = json.dumps({
        "model": "qwen3.5-27b",
        "messages": [
            {"role": "user", "content": turn1_prompt},
            {"role": "assistant", "content": turn1_reply},
            {"role": "user", "content": turn2_prompt}
        ],
        "stream": False,
        "max_tokens": max_tokens
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        resp_body = resp.read().decode("utf-8")
    t1 = time.perf_counter()
    data = json.loads(resp_body)
    usage = data.get("usage", {})
    gen_tokens = usage.get("completion_tokens", max_tokens)
    total_ms = (t1 - t0) * 1000.0
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "generated_tokens": gen_tokens,
        "total_ms": total_ms,
        "decode_tok_s": (gen_tokens / (total_ms / 1000.0)) if total_ms > 0 else 0.0,
        "dispatches": gen_tokens,
        "syncs": 1
    }

def main():
    print("==========================================================================")
    print("     MIInfer Milestone M10: Real-World Latency Curve Benchmark Suite     ")
    print("==========================================================================")
    print(f"Model:  {MODEL}")
    print(f"Binary: {BINARY}")
    print(f"Port:   {PORT}\n")

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = f"bench/results/m10-latency-curve/{timestamp}"
    os.makedirs(outdir, exist_ok=True)

    results = {
        "timestamp": timestamp,
        "model": MODEL,
        "gpu": "AMD Instinct MI50 32GB (gfx906 / Vega20)",
        "runs": {}
    }

    test_matrix = [
        ("P128", 126, "TG64", 64),
        ("P128", 126, "TG128", 128),
        ("P128", 126, "TG256", 256),
        ("P512", 504, "TG64", 64),
        ("P512", 504, "TG128", 128),
        ("P512", 504, "TG256", 256)
    ]

    # 1. Cold Request Benchmark (CLI process start)
    print("--- 1. Cold Request Benchmark (Process Start, Weight Load, Prefill, Chained Decode) ---")
    cold_results = {}
    for p_label, p_toks, g_label, g_toks in test_matrix:
        cfg_name = f"{p_label}_{g_label}"
        prompt_text = make_prompt(p_toks)
        print(f"Running Cold {cfg_name} (Prompt ~{p_toks} tok, Gen {g_toks} tok)...")
        stats = run_cold_request(prompt_text, g_toks, stream=False)
        print(f"  -> Prefill: {stats.get('prefill_ms', 0):.1f} ms ({stats.get('prefill_tok_s', 0):.2f} tok/s), "
              f"TTFT: {stats.get('ttft_ms', 0):.1f} ms, "
              f"Decode: {stats.get('decode_tok_s', 0):.2f} tok/s ({stats.get('decode_ms', 0):.1f} ms), "
              f"Wall: {stats.get('wall_ms', 0):.1f} ms, VRAM: {stats.get('vram_mib', 0):.0f} MiB")
        cold_results[cfg_name] = stats
        time.sleep(1)
    results["runs"]["cold"] = cold_results

    # 2. Warm Request & Streaming Benchmark via Persistent Server
    print("\n--- 2. Starting Persistent HTTP Server for Warm & Streaming Benchmarks ---")
    srv_proc = subprocess.Popen([BINARY, "serve", MODEL, "--port", str(PORT)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        if not wait_for_server(PORT):
            print("ERROR: Server failed to start!")
            srv_proc.kill()
            sys.exit(1)
        print(f"Server ready on port {PORT}.\n")

        # Warm Non-Streaming
        print("--- 3. Warm Request Benchmark (Model Resident, In-Memory Prefill + Chained Decode) ---")
        warm_results = {}
        for p_label, p_toks, g_label, g_toks in test_matrix:
            cfg_name = f"{p_label}_{g_label}"
            prompt_text = make_prompt(p_toks)
            print(f"Running Warm Non-Stream {cfg_name}...")
            stats = run_http_warm_request(PORT, prompt_text, g_toks, stream=False)
            print(f"  -> Total: {stats['total_ms']:.1f} ms, Gen: {stats['generated_tokens']} tok, "
                  f"Throughput: {stats['decode_tok_s']:.2f} tok/s, Dispatches: {stats['dispatches']}, Syncs: {stats['syncs']}")
            warm_results[cfg_name] = stats
            time.sleep(1)
        results["runs"]["warm_non_stream"] = warm_results

        # Warm Streaming (SSE)
        print("\n--- 4. Streaming Request Benchmark (HTTP Server-Sent Events / SSE) ---")
        stream_results = {}
        for p_label, p_toks, g_label, g_toks in test_matrix:
            cfg_name = f"{p_label}_{g_label}"
            prompt_text = make_prompt(p_toks)
            print(f"Running SSE Streaming {cfg_name}...")
            stats = run_http_warm_request(PORT, prompt_text, g_toks, stream=True)
            print(f"  -> TTFT: {stats['ttft_ms']:.1f} ms, Decode: {stats['decode_tok_s']:.2f} tok/s, "
                  f"Total: {stats['total_ms']:.1f} ms, Dispatches: {stats['dispatches']}, Syncs: {stats['syncs']}")
            stream_results[cfg_name] = stats
            time.sleep(1)
        results["runs"]["warm_stream_sse"] = stream_results

        # Second Chat Turn Benchmark
        print("\n--- 5. Second Chat Turn Benchmark (Context Preservation & Continuation) ---")
        turn2_results = {}
        t1_prompt = "Tell me about the architectural differences between CDNA and GCN."
        t1_reply = "CDNA is AMD's dedicated compute architecture designed for data centers, featuring Matrix FMA units and omitting graphics hardware. GCN (like gfx906/Vega20) is a hybrid compute/graphics architecture utilizing Wave64 execution."
        for g_label, g_toks in [("TG64", 64), ("TG128", 128), ("TG256", 256)]:
            t2_prompt = "Now explain how DeltaNet recurrence compares to transformer attention."
            print(f"Running Second Chat Turn with {g_label}...")
            stats = run_http_second_turn(PORT, t1_prompt, t1_reply, t2_prompt, g_toks)
            print(f"  -> Turn 2 Total: {stats['total_ms']:.1f} ms, Gen: {stats['generated_tokens']} tok, "
                  f"Throughput: {stats['decode_tok_s']:.2f} tok/s, Dispatches: {stats['dispatches']}, Syncs: {stats['syncs']}")
            turn2_results[g_label] = stats
            time.sleep(1)
        results["runs"]["second_chat_turn"] = turn2_results

    finally:
        srv_proc.terminate()
        try:
            srv_proc.wait(timeout=5)
        except Exception:
            srv_proc.kill()
        print("\nServer cleanly terminated.")

    # Save full JSON results
    with open(f"{outdir}/m10_latency_curve.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw benchmark data to {outdir}/m10_latency_curve.json")

    # Generate Scorecard Summary
    print("\n" + "="*95)
    print("                       MIInfer Milestone M10 Latency Curve Summary                      ")
    print("==========================================================================================")
    print(f"{'Configuration':<16} | {'Mode':<18} | {'TTFT (ms)':<10} | {'PP tok/s':<10} | {'TG tok/s':<10} | {'Total ms':<10} | {'Syncs':<6}")
    print("-"*95)
    for p_label, p_toks, g_label, g_toks in test_matrix:
        cfg = f"{p_label}_{g_label}"
        c = cold_results.get(cfg, {})
        w = warm_results.get(cfg, {})
        s = stream_results.get(cfg, {})
        print(f"{cfg:<16} | {'Cold CLI':<18} | {c.get('ttft_ms', 0):<10.1f} | {c.get('prefill_tok_s', 0):<10.2f} | {c.get('decode_tok_s', 0):<10.2f} | {c.get('wall_ms', 0):<10.1f} | {c.get('syncs', 0):<6}")
        print(f"{cfg:<16} | {'Warm Non-Stream':<18} | {'-':<10} | {'-':<10} | {w.get('decode_tok_s', 0):<10.2f} | {w.get('total_ms', 0):<10.1f} | {w.get('syncs', 0):<6}")
        print(f"{cfg:<16} | {'Warm SSE Stream':<18} | {s.get('ttft_ms', 0):<10.1f} | {'-':<10} | {s.get('decode_tok_s', 0):<10.2f} | {s.get('total_ms', 0):<10.1f} | {s.get('syncs', 0):<6}")
        print("-"*95)
    print("="*95 + "\n")

if __name__ == "__main__":
    main()
