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

// GQA Attention parameters for Qwen3.8-27B
constexpr std::size_t kQHeads = 24;
constexpr std::size_t kKvHeads = 4;
constexpr std::size_t kHeadDim = 256;
constexpr std::size_t kQDim = kQHeads * kHeadDim;       // 6144
constexpr std::size_t kKvDim = kKvHeads * kHeadDim;     // 1024
constexpr std::size_t kQFullDim = 2 * kQDim;            // 12288 (Query + Gate)
constexpr float kRopeTheta = 1000000.0F;
constexpr std::size_t kDefaultCacheCapacity = 32768;

// Topology block configuration
constexpr std::size_t kRecurrentPerBlock = 3;
constexpr std::size_t kAttentionPerBlock = 1;
constexpr std::size_t kLayersPerBlock = 4;

// Native full-model macro tile size for long sequences
constexpr std::uint32_t kPrefillV2MacroTile = 512;

// Maximum physical prefill batch width executed in one kernel launch (Macro Tile = 512)
constexpr std::size_t kMaxPrefillBatch = 512;

} // namespace miinfer::prefill_v2
