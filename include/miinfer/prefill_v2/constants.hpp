#pragma once

#include <cstddef>
#include <cstdint>

namespace miinfer::prefill_v2 {

// Qwen3.8-27B architectural dimensions specialized for AMD Instinct MI50 (gfx906 / Wave64)
constexpr std::size_t kHidden = 5120;
constexpr std::size_t kInner = 6144;
constexpr std::size_t kFfnInner = 17408;
constexpr std::size_t kState = 128;
constexpr std::size_t kKHeads = 16;
constexpr std::size_t kVHeads = 48;
constexpr std::size_t kChannels = 10240; // 16*128 + 16*128 + 48*128
constexpr std::size_t kConvKernel = 4;
constexpr std::size_t kRecurrentTimeStepRank = 48;
constexpr float kRmsNormEpsilon = 1.0e-6F;

// Internal recurrent chunk width for chunkwise Gated DeltaNet (C = 64)
constexpr std::uint32_t kGdnChunkSize = 64;

// Maximum supported prefill batch / macro-tile for V2 layer slice
constexpr std::size_t kMaxPrefillBatch = 512;

} // namespace miinfer::prefill_v2
