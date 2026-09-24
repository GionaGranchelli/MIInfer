#pragma once

#include "miinfer/prefill_v2/constants.hpp"

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace miinfer::prefill_v2 {

// Non-owning view of recurrent-layer device state.
// Makes incoming and outgoing state evolution explicit.
struct RecurrentLayerState {
    float* d_state = nullptr;        // Recurrent state tensor: [kVHeads, kState, kState] (48 * 128 * 128 = 786,432 floats)
    float* d_conv_history = nullptr; // Conv1d history buffer: [kConvKernel, kChannels] (4 * 10240 = 40,960 floats)
    std::uint32_t position = 0;      // Sequence start position

    static constexpr std::size_t kStateElements = kVHeads * kState * kState;
    static constexpr std::size_t kConvHistoryElements = kConvKernel * kChannels;
    static constexpr std::size_t kStateBytes = kStateElements * sizeof(float);
    static constexpr std::size_t kConvHistoryBytes = kConvHistoryElements * sizeof(float);
};

// Owning container for recurrent-layer GPU state.
// Used by callers, tests, and execution plans to manage state allocation.
class RecurrentLayerStateStorage {
public:
    RecurrentLayerStateStorage();
    ~RecurrentLayerStateStorage();
    RecurrentLayerStateStorage(const RecurrentLayerStateStorage&) = delete;
    RecurrentLayerStateStorage& operator=(const RecurrentLayerStateStorage&) = delete;
    RecurrentLayerStateStorage(RecurrentLayerStateStorage&& other) noexcept;
    RecurrentLayerStateStorage& operator=(RecurrentLayerStateStorage&& other) noexcept;

    void reset();
    void upload(std::span<const float> host_state, std::span<const float> host_history, std::uint32_t pos);
    void download(std::vector<float>& host_state, std::vector<float>& host_history) const;

    [[nodiscard]] RecurrentLayerState view() const noexcept {
        return RecurrentLayerState{d_state_, d_conv_history_, position_};
    }

    void set_position(std::uint32_t pos) noexcept { position_ = pos; }
    [[nodiscard]] std::uint32_t position() const noexcept { return position_; }

private:
    float* d_state_ = nullptr;
    float* d_conv_history_ = nullptr;
    std::uint32_t position_ = 0;
};

} // namespace miinfer::prefill_v2
