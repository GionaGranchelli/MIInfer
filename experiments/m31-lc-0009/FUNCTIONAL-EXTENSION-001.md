# M31-LC-0009 functional-output extension 001

## Registration

This addendum is registered before collecting any output beyond the existing 64-token cap. It extends output length only. It does not modify `corpus.jsonl`, its prompt IDs or acceptance text, the 64-token captures, numeric thresholds, or the exact-token production gate.

## Fixed run plan

- Keep the frozen model, tokenizer, six prompt-ID files, CPU/GPU source and image identities, greedy argmax sampler, stop policy, and context profile from [`PROTOCOL.md`](PROTOCOL.md).
- For each of the six prompts, generate at most **2,048 new tokens** from a fresh prompt state. The runner's 8K context profile accommodates the longest 2,048-token prompt plus this output allowance.
- Run three independent fresh contexts for each of five configurations: pinned CPU llama.cpp; fused GPU eager; fused GPU HIP graph; aligned-MMQ GPU eager; aligned-MMQ GPU HIP graph. Keep both required aligned-MMQ selectors enabled. Run GPU configurations serially on the Z840.
- Keep this as a separate functional run. Do not supply the CPU forced history; each configuration follows its own greedy trajectory from the same frozen prompt. Retain each repeat's generated token IDs, decoded output, runner log, command, and pinned identity hashes on Lake. Do not replace or overwrite the original 64-token captures.
- Decode outputs using the frozen tokenizer identity in `PROTOCOL.md`. Preserve complete decoded output, including reasoning text and final answer, so omitted or contradictory claims remain reviewable.
- Do not increase the cap after inspecting any result. If a required final answer or acceptance condition is not reached within 2,048 tokens, record that configuration as `NOT_REACHED_WITHIN_CAP`; do not call it a pass.

## Frozen acceptance checklist

Evaluate the final answer against every clause below for each repeat and configuration. Mark each clause `PASS`, `FAIL`, or `NOT_REACHED_WITHIN_CAP`, and quote the smallest supporting or contradicting output span. A semantic claim not clearly supported by the captured output is not a pass.

| Frozen prompt | Required clauses |
| --- | --- |
| `ordinary-64-01` | Explain that surface ice slows heat loss and leaves liquid water below. Make no unsupported exact temperature claim. |
| `coding-64-01` | Give `range(len(xs))` or an equivalent correct expression; it covers every valid index and produces an empty range for an empty list. |
| `reasoning-512-01` | State exactly `A=West, B=East, C=North`; no conflicting assignment. |
| `ordinary-512-01` | Keep Tuesday continuation, Thursday staffing review, and Saturday unchanged; cite both increased evening attendance and evening visitor use (language-learning books and/or public computers); invent no safety issue. |
| `coding-2048-01` | Use `if key not in updates` or an equivalent membership check and explain that absence differs from a present key whose value is `None`. |
| `reasoning-2048-01` | Show `H+R=36` and `H=3R`, and state held `27`, returned `9`. |

Report results per repeat; a prompt/configuration passes functional acceptance only when all three repeats satisfy every clause. This result is separate from teacher-forced numerical metrics, repeat determinism, and the production exact-token gate.

## Disposition

This addendum makes the frozen checks reachable and auditable; it is not evidence that any model or route passes them. Keep qualification blocked until the registered runs and Contract O references are complete and reviewed. Any production-gate change still requires separate explicit approval.
