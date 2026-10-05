# M30-0001 Exact-Prefix Continuation Contract

Status: implementation baseline

## Base

`M30_0001_BASE_SHA=b4bcf10186abe3b2f78f7e56343c7c54f67a3e99`

## Scope

An exact prefix checkpoint may continue a prompt only when the model identity,
quantization, state-layout version, and every prefix token match. Any mismatch,
shorter prompt, invalid checkpoint, or reset falls back to the existing full
prefill path.

The reusable state has two owners:

- GDN recurrent state and convolution history are captured by `ReusableContext`.
- GQA KV state remains in the single active `PrefillV2Model` KV ownership. It is
  not duplicated in VRAM for the in-process checkpoint. Disk sessions already
  serialize and restore both families.

The checkpoint also retains the final normalized hidden vector for the cached
boundary. This is required for an exact zero-suffix continuation: generation
must obtain prompt-boundary logits without replaying or mutating the prefix.

## Required observability

Every continuation reports `reuse_hit`, `prefix_tokens_reused`,
`suffix_tokens_executed`, `prefix_tokens_replayed`, and
`checkpoint_position`. The exact-hit path executes only the suffix; a zero
suffix executes no prefill work.

## Explicit non-goals

No fuzzy or longest-prefix matching, multi-checkpoint cache, copy-on-write,
rollback, tail replay, or global cache is introduced by M30-0001.
