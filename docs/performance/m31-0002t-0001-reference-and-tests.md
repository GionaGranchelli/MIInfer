# M31-0002T-0001 — Reference comparison and regression tests

Status: input bytes and major configuration audited; sampler parity fails;
logit-level agreement is unavailable. Read-only investigation; no files were
changed by the reference audit.

## Reference and model identity

The recorded llama.cpp reference is commit
`73a43d1f69345aee8bb186ef4b3172cef892f2e5`, built for HIP/gfx906 with ROCm
7.2.1, Release optimization, no CUDA, HIP graphs enabled, and all model layers
requested for GPU offload. Its helper uses one sequence, context size 1,280,
512-token batch/microbatch, FP16 K/V, automatic flash-attention selection, and
only the greedy sampler.

Both measured calls use model SHA
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169` and the
same direct 1,024-token ID vector/fingerprint. Tokenizer identity, BOS policy,
and chat templates are not exercised by this synthetic-input comparison.
The GGUF is Q4_K_M as a mixed format: the recorded reference loader reports
866 tensors (456 F32, 1 Q8_0, 294 Q4_K, 48 Q5_K, 67 Q6_K). Both runtimes parse
the same bytes, but their layout conversion and quantized arithmetic are not
shown equivalent by the token trace.

The 64 main transformer layers match the runtime topology; GGUF's 65th block
is an auxiliary next-token-prediction layer, not an obvious main-layer count
mismatch. The measured token mismatch is one run only; logits and numerical
margins were not captured. Additionally, the MIInfer binary hash is recorded
without a source-tree SHA binding, so current-tree source analysis cannot
fully establish exact binary provenance.

## Regression coverage and status

Previously reported M31-0004/M31-0005 host coverage includes checkpoint
fingerprinting, invalid-boundary rejection, cache-residency bookkeeping,
and the reusable-context allocation test. M31-0005's poisoned future-KV
attention-boundary test is GPU-required and has not run. Neither milestone's
GPU integration cases establish the current full-model parity result.

For this task, a clean HIP-disabled CMake configure succeeded. A broad build
did not complete because the existing CLI target includes
`hip/hip_fp16.h` even with HIP disabled. Seven available host-only CTests
passed: host-only, openai-api-host, m12-gdn-oracle, model-loader-host,
context-space-host, device-kv-shard-placement-host, and
reusable-context-allocation. The shared prompt test compiled and ran with
`g++ -std=c++17 -Wall -Wextra -Werror`. This is a focused host validation, not
the full project gate. Existing notes also record the HIP configure blocker:
the installed ROCm 6.2 linker cannot load `libxml2.so.2`.

## Minimum follow-up tests

1. Add/retain a host-side parity contract that explicitly configures MIInfer
   repetition/presence/frequency penalties to match the reference sampler and
   checks greedy behavior on repeated-token and near-tied synthetic logits.
2. Add test-only forced-prefix/logit capture to both harnesses; assert the same
   first six generated IDs are consumed before comparing step-7 distributions.
3. If aligned raw logits differ materially, compare the MIInfer graph path with
   eager decode at the same forced prefix, then add the smallest relevant
   recurrent/KV boundary regression. Keep M31-0005's poisoned-future-slot
   test separate and run it only on GPU.
4. For memory, add a CPU GGUF tensor inventory and per-allocation layout ledger;
   no GPU kernel or quantization change is justified yet.

Minimal hardware check after a test binary is buildable: one guarded Z840 MI50
(`0000:06:00.0`, select the gfx906 agent explicitly), same image/model/prompt,
1,280 context, at most eight outputs, no profiler or repetition, with the
existing fast thermal/process guard. The previous guard stopped a real worker
within 310 ms of the threshold, but that is termination evidence, not parity
validation. Do not start this call until instrumentation/build and guard
checks are satisfied.
