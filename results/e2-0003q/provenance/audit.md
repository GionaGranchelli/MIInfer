# E2-0003Q source and sampler audit

## Recovered source

The transferred source bundle was `e2-0003-source-20261010.tar.gz`, SHA-256
`e7898821f5cc2fae0cf73d7a5a919ce0e49d30e94510018f33b64877b7024a5e`.
It contains the original checkpointed-call benchmark and prompt helper, whose
SHA-256 values are respectively `e80404982d8b80db9cb29fb72b323801c70a6d8a7ef918759b7728aed82ea482`
and `896189773dd6fb6b9c5588d8d955583621712127a8675232757b558dce066a7b`.
The archived `CMakeLists.txt` hash is
`877b6acb951f16ac0121b3ebd0dd6f83a95f86553415dcfed9ad94722e84d5ea`.
Exact copies of the recovered benchmark and helper are kept beside this audit.

The archived CMake file declares `miinfer-m31-0002-checkpointed-call`; the
pushed clean base `e75dfcf8ececab5c032cb233f8aeed478268a1ba` does not contain
that source or target. Thus the old binary was not reproducible from that
clean commit alone. The recovered archive includes source changes from the
preserved M31 overlay; this qualification uses the recovered benchmark adapted
to the clean base API and does not import those unrelated core-source changes.

The old build evidence also names a Gate/Up operation-reference executable,
but the recovered archive CMake file has no corresponding target. Its source
was independently preserved as
`results/e2-0003/gateup-operation-reference.cpp` (SHA-256
`04f9a4f211820933b70c049802e57da04e2dc97c676bf7ea1d2668b9287c139b`). The
same source is now declared as an explicit target in this worktree.

The pinned image contains ROCm 7.2.1 hipBLAS headers and `libhipblas.so`, but
omits the CMake package config required by this repository. The qualification
uses the committed `hipblasConfig.cmake` shim to expose those exact installed
paths as the `roc::hipblas` imported target; it does not substitute a library
or alter repository build logic.

The old evidence package's `results/e2-0003/SHA256SUMS` is tracked in the
pushed base, but eight referenced `.log` artifacts were omitted from Git by the
ignore rule, and the tracked `ssh-preflight.log` did not match its manifest.
All nine exact files were recovered from the preserved dirty checkout and
copied into this clean worktree without modifying the original checkout. The
full manifest now verifies. These files must be force-added so a fresh checkout
also contains the complete evidence package.

## Sampler settings

The recovered old benchmark explicitly set temperature `0`, top-p `1`, top-k
`1`, seed `42`, reset-before-call, HIP Graph enabled, and disabled stop IDs.
It did not set repetition, frequency, or presence penalties. The API defaults
were repetition penalty `1.15`, frequency penalty `0`, and presence penalty
`0`. Therefore the old E2 run used repetition penalty `1.15`.

The corrected E1/E3 reference used repetition penalty `1.0`. Its generated
token sequence diverges from the E2 sequence after the first six tokens; the
implicit repetition penalty is a concrete configuration difference and a
plausible cause, but the old run alone does not prove causality. Its historical
28.593/25.521 tok/s pair must not be treated as the explicit-1.0 qualification
baseline. A new same-source short A/B is required.

The qualification benchmark explicitly records temperature `0`, top-p `1`,
top-k `1`, repetition/frequency/presence penalties `1/0/0`, empty stop IDs,
repeat-last-N `256`, seed `42`, greedy argmax sampling, reset-before-call, and
HIP Graph enabled. The benchmark is adapted only to the clean API and renamed;
the recovered originals above remain unchanged for provenance.
