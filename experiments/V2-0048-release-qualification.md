# V2-0048 — Release candidate and real-world qualification

## Decision

**RELEASE BLOCKED.** The package installs and passes its basic CLI, API, agent,
and 20-request stability checks, but two default-generation probes produced an
empty visible answer, and a subsequent authenticated 1,024-token API request
aborted the packaged server with an HSA memory-aperture fault. Do not publish
this candidate as v0.2.0.

## Immutable candidate identity

- Canonical `main` source SHA: `0032bb59cbf8c3d23ccd0eaeb5970004408503d0`
  (V2-0047 merge commit).
- Clean detached source worktree; Release; `MIINFER_TARGET_ARCH=gfx906` and
  `MIINFER_HIP_ARCHITECTURE=gfx906`.
- MI50 / gfx906; GCC 16.2.1; HIP Clang 20.0.0; ROCm HIP 7.1.52802;
  CMake 4.3.0.
- Embedded identity: version `0.2.0`, commit `0032bb59cbf8`, dirty `false`,
  Release, target `gfx906`.
- Single CPack artifact:
  `miinfer-0.2.0-gfx906-Linux.tar.gz`, 958,832 bytes,
  SHA-256 `2eb4fc2e8dc6625c0524b8cb4f3c6f2a1ba7f8d7052a3a9e0f8d7b1b32f37efb`.
  Local path at qualification:
  `/tmp/miinfer-v2-0048-build/miinfer-0.2.0-gfx906-Linux.tar.gz`.
- Installed from that archive into `/tmp/miinfer-v2-0048-install`; dynamic
  dependencies resolve from the host ROCm/runtime libraries, not a source or
  build tree.
- Existing `v0.2.0` tag was not modified. No RC tag was created because this
  candidate did not pass qualification.

The archive manifest contains only the two runtime executables, installer,
license, README, and three public documents. The default test-enabled build
also compiles release-test/helper executables (including
`miinfer-m12-gdn-oracle` and `miinfer-q4-q8-dot4-probe`); they are absent from
the archive and are not runtime dependencies. Benchmarks and research tools
remain opt-in. This is not a packaging blocker.

## Passing checks

- Build and CPack completed once from the clean canonical SHA. The archive hash
  remained unchanged through qualification.
- CTest: 26/26 passed, including 14 GPU-required tests and the package archive
  smoke gate.
- The documented archive install flow passed. Installed `--version`, `--help`,
  `doctor --model`, `models`, and `inspect` reported the expected production
  build, gfx906/MI50 compatibility, and Qwen3.8-27B model details.
- The three documented `run` input forms (positional, `--prompt`, stdin) each
  generated 48 tokens successfully with coherent visible output and no protocol
  text.
- Default localhost server startup passed `/healthz`, `/readyz`, `/v1/models`,
  and the bundled Web UI returned HTTP 200. Unauthenticated localhost access
  worked. Unauthenticated non-loopback startup was rejected. With an API-key
  file, non-loopback serving started, rejected an unauthenticated API request
  with 401, and accepted authenticated requests; the test key was not logged.
- Repository-supported streamed OpenAI/tool-call agent driver passed all three
  16K-context logical turns. Generation totals were 1,434 / 1,113 / 1,093
  tokens; visible answers were 1,777 / 4,019 / 1,390 characters. Reused-prefix
  counts were 7,274 / 8,992 / 11,830 tokens. TTFT was 22.98 / 67.31 / 85.44 s.
  All generation and visible-answer gates passed; VRAM stayed at 27.194 GB.
- Bounded stability campaign: 20/20 sequential API requests passed across
  three retained multi-turn sessions. Each session included an 8,929-token
  request capped at TG128; all three TG128 requests generated exactly 128
  tokens. All sessions recorded prefix hits (17/20 requests total); aggregate
  reuse was 44,925 / 53,994 / 53,994 tokens per session. VRAM was
  27.105 GB at start and 27.194 GB at end/high-water (+89 MB). The raw
  per-request record is [`stability-20.json`](../results/v2-0048/stability-20.json).
- Following the later GPU fault, installed device-info still detected gfx906;
  a fresh default localhost server reached readiness and then shut down cleanly.

## Release blockers

### Default generation can return empty visible content

On the authenticated 16K server, a normal short greeting request with no
`max_tokens` override passed (HTTP 200, 13 completion tokens, 32 visible
characters). A normal C++ implementation request with no override also
returned HTTP 200, but its content was empty after using the API default of 256
completion tokens. In the five-turn `miinfer chat` run using its 512-token
default, the visible response lengths were 663, 93, 0, 677, and 0 characters;
three turns exhausted the full 512-token budget with partial or blank output.
Prefix reuse worked (4,591–6,034 tokens), but does not remedy missing output.

This is consistent with the terminal/API reasoning filter consuming the
available generation budget before visible content; the hidden-token count was
not separately captured. The larger-budget agent workload did return visible
answers, so the default-budget limitation is workload-dependent, but ordinary
code/review requests reproduced it.

### Packaged server aborted on a larger-budget request

After the 20-request campaign, the same short C++ request was sent with an
explicit `max_tokens=1024`. The client received
`RemoteDisconnected`. Server log request 63 reached `prefill_started`, then
ROCm reported `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`; the packaged server
terminated with signal 6 (exit status 134). No `/dev/kfd` owner remained.
Installed device-info and a clean server restart passed afterward, but that
does not excuse a GPU fault/server crash during an ordinary API request.

Root cause is **UNKNOWN**. The default-content failure and the GPU fault need a
narrow corrective investigation before release; no runtime or kernel changes
were made in V2-0048.

## Scope and final gate

No performance benchmarking or kernel changes were performed. The 20-request
campaign itself completed without API failures or measured VRAM growth; the
additional API request afterward exposed the crash blocker. The candidate hash
and source SHA above remain the tested identities. V2-0048 does not pass the
release definition of done until both visible-output behavior at ordinary
defaults and server stability are qualified.
