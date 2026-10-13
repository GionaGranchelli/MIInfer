# EXP-0399 — M31 Q8/Q8 two-query shared attention

**Status:** REJECTED

**Baseline:** `88fd63eb0cf8b765f37e5f52168c80783df62e7b`
**Candidate commit:** `fc96884`

## Result

The isolated Q8/Q8 attention output matched the existing path exactly at
`P=8192, S=512` (`max_abs=0`, `RMSE=0`, cosine `1.0`) on MI50 BDF
`0000:88:00.0`. The compiler reported 45 VGPRs, 60 SGPRs, 16,640 bytes LDS,
zero spills, and five waves/SIMD.

Five interleaved samples gave medians of `167.882 ms` for the production
single-query mapping and `267.341 ms` for the two-adjacent-query mapping. The
candidate was `1.59x` slower and is rejected. Correctness and resource use were
not the limiting issue; the additional LDS staging/barriers and mapping cost
outweighed the expected reuse. No production dispatch was changed.

## Next direction

The model has 24 Q heads grouped six per KV head. The production CTA already
places adjacent Q heads for one token together, and each pair shares the same
KV head. EXP-0400 tests reuse along that existing CTA/GQA relationship while
keeping the original grid and wave mapping.
