#!/usr/bin/env python3
"""M16-C: one parent owns server lifecycle and all durable evidence."""
import argparse, concurrent.futures, hashlib, json, os, socket, subprocess, sys, time, urllib.request
from pathlib import Path

PROMPT = "Reply with exactly one word: MI50."
LEVELS = (1, 2, 4, 8, 16)

def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as out:
        json.dump(value, out, indent=2); out.write("\n"); out.flush(); os.fsync(out.fileno())
    temporary.replace(path)
    with path.open(encoding="utf-8") as source: json.load(source)

def command(argv, cwd):
    return subprocess.run(argv, cwd=cwd, text=True, capture_output=True, check=False)

def http_get(url):
    began = time.monotonic()
    with urllib.request.urlopen(url, timeout=1) as reply:
        return reply.status, reply.read(), time.monotonic() - began

def control_probe(url):
    status, _, latency = http_get(url)
    return {"status": status, "latency_s": latency}

def chat(port, payload):
    body = json.dumps(payload).encode()
    wire = (b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: "
            + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body)
    began, first, response = time.monotonic(), None, b""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=10) as client:
            client.sendall(wire); client.settimeout(180)
            while block := client.recv(8192):
                response += block
                if first is None and b"data: {" in response:
                    first = time.monotonic()
        return {"status": response.split(b"\r\n", 1)[0].decode("ascii", "replace"),
                "ttft_s": None if first is None else first - began, "latency_s": time.monotonic() - began,
                "bytes": len(response), "sse_chunks": sum(x.startswith(b"data: {") for x in response.splitlines()),
                "hash": hashlib.sha256(response).hexdigest()}
    except OSError as error:
        return {"status": "transport-error", "latency_s": time.monotonic() - began, "error": str(error), "bytes": 0, "sse_chunks": 0, "hash": ""}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("miinfer", type=Path); parser.add_argument("model", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("results/m16c")); parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--levels", default="1,2,4,8,16"); parser.add_argument("--stream", choices=("true", "false"), default="true")
    args = parser.parse_args(); root = Path.cwd(); binary, model = args.miinfer.resolve(), args.model.resolve()
    if not binary.is_file() or not model.is_file(): raise SystemExit("binary or model does not exist")
    sha = command(["git", "rev-parse", "--short", "HEAD"], root).stdout.strip()
    artifact = (root / args.output_root / f"{time.strftime('%Y%m%d-%H%M%S')}-{sha}").resolve(); artifact.mkdir(parents=True)
    with socket.socket() as probe: probe.bind(("127.0.0.1", 0)); port = probe.getsockname()[1]
    argv = [str(binary), "serve", str(model), "--host", "127.0.0.1", "--port", str(port)]
    metadata = {"git_sha": sha, "dirty": bool(command(["git", "status", "--porcelain"], root).stdout), "binary": str(binary),
                "model": str(model), "server_argv": argv, "benchmark_argv": sys.argv, "port": port, "hostname": socket.gethostname(),
                "started_utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), "version": command([str(binary), "--version"], root).stdout,
                "rocm_smi": command(["rocm-smi"], root).stdout}
    atomic_json(artifact / "metadata.json", metadata)
    out = (artifact / "server.stdout.log").open("w", encoding="utf-8"); err = (artifact / "server.stderr.log").open("w", encoding="utf-8")
    server = subprocess.Popen(argv, cwd=root, stdout=out, stderr=err)
    metadata["server_pid"] = server.pid; atomic_json(artifact / "metadata.json", metadata); base = f"http://127.0.0.1:{port}"
    result = {"artifact": str(artifact), "levels": [], "failure": None}
    try:
        for _ in range(90):
            if server.poll() is not None: raise RuntimeError(f"server exited before readiness: {server.returncode}")
            try:
                if http_get(base + "/readyz")[0] == 200: break
            except OSError: time.sleep(1)
        else: raise RuntimeError("server readiness timeout")
        (artifact / "metrics-before.txt").write_bytes(http_get(base + "/metrics")[1])
        payload = {"messages": [{"role": "user", "content": PROMPT}], "stream": args.stream == "true", "max_tokens": 64}
        if " 200 " not in chat(port, payload)["status"]: raise RuntimeError("warmup failed")
        for clients in (int(value) for value in args.levels.split(",")):
            trials = []
            for trial in range(args.trials):
                began = time.monotonic()
                with concurrent.futures.ThreadPoolExecutor(max_workers=clients) as pool:
                    futures = [pool.submit(chat, port, payload) for _ in range(clients)]
                    control = {path: control_probe(base + path) for path in ("/healthz", "/readyz", "/metrics", "/v1/models")} if clients >= 8 else {}
                    requests = [future.result() for future in futures]
                trials.append({"trial": trial + 1, "wall_s": time.monotonic() - began, "requests": requests, "control": control})
            result["levels"].append({"clients": clients, "trials": trials})
        (artifact / "metrics-after.txt").write_bytes(http_get(base + "/metrics")[1])
    except Exception as failure:
        result["failure"] = repr(failure)
    finally:
        try:
            atomic_json(artifact / "raw-results.json", result)
        finally:
            server.terminate()
            try: server.wait(timeout=30)
            except subprocess.TimeoutExpired: server.kill(); server.wait(timeout=10); result["failure"] = result["failure"] or "unclean server shutdown"
            out.close(); err.close()
    flat = [request for level in result["levels"] for trial in level["trials"] for request in trial["requests"]]
    atomic_json(artifact / "summary.json", {"accepted": sum(" 200 " in r["status"] for r in flat), "rejected": sum(" 503 " in r["status"] for r in flat), "failure": result["failure"]})
    print(artifact)
    return 1 if result["failure"] else 0

if __name__ == "__main__": raise SystemExit(main())
