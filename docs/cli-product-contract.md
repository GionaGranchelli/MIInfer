# MIInfer CLI product contract

V2-0046 turns the existing MI50-specific engineering CLI into one discoverable
user interface. `doctor`, `models`, `inspect`, `run`, `chat`, `serve`, and
`config` are public commands; the frozen V2-0045 production runtime remains the
only default inference profile. The CLI owns configuration, model lookup,
input/output presentation, and actionable failures, not kernel or execution-plan
selection.

The default profile is the V2-0045 qualified M28/Prefill V2 configuration.
Internal environment selectors stay available as development controls but are
omitted from public help and quick-start documentation. User configuration is
limited to model directory/default model, host, port, context, and API-key file.
Explicit paths take precedence over configured names; model-name lookup must
report ambiguity rather than pick a file arbitrarily.

Default output is concise and human-readable, `--verbose` adds diagnostics, and
`--json` is supported for inspection-style commands. The serving context limit
must be described honestly: benchmarked prefill coverage is through 8192 tokens;
the existing MIInfer-only serving qualification exercised 16K context. Larger
contexts remain available only with an explicit warning and successful runtime
allocation, not as a performance or quality claim.

V2-0046 must not change the qualified numerical path, kernels, or benchmark
results. Any CLI entry-point refactor must call the same Prefill V2 production
model and sampling contract already used by the qualified server. Product UX
tests cover help, parsing, config precedence, model matching, package layout,
and a real MI50 first-run smoke; the latter is required before promotion.

## V2-0046 verification — 2026-09-30

- The release `miinfer` target builds and all 25 release CTest checks pass on
  the MI50, including the GPU correctness suite.
- The clean-config first-run script passes against Qwen3.8-27B-Q4_K_M: doctor,
  model discovery and inspection, all three `run` prompt forms, stale-selector
  isolation, default-model multi-turn chat with nonzero prefix reuse, server
  readiness, and a successful OpenAI-compatible completion.
- `test-package.sh` passes on a release-layout archive created from the current
  CMake install manifest; the install script, CLI help, config, model listing,
  device probe, and shipped documentation are checked.
- The standard CPack `package` target is not currently buildable with this host
  toolchain: its `preinstall` builds the full benchmark target graph, where GCC
  16 fails parsing a libstdc++ `<format>` attribute in
  `bench/v2_0045_qualification_bench.cpp`. This is independent of the CLI and
  inference runtime; direct CMake install and archive-install validation pass.
- No performance benchmark was rerun. The inference kernels and qualified
  production execution path are unchanged; the only generation API addition
  is an opt-in seed, and the default path does not reseed the runtime.
