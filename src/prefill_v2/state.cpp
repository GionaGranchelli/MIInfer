#include "miinfer/prefill_v2/state.hpp"
#include "miinfer/hip_check.hpp"

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <utility>

namespace miinfer::prefill_v2 {

RecurrentLayerStateStorage::RecurrentLayerStateStorage() {
    MIINFER_HIP_CHECK(hipMalloc(&d_state_, RecurrentLayerState::kStateBytes));
    MIINFER_HIP_CHECK(hipMalloc(&d_conv_history_, RecurrentLayerState::kConvHistoryBytes));
    reset();
}

RecurrentLayerStateStorage::~RecurrentLayerStateStorage() {
    if (d_state_ != nullptr) {
        (void)hipFree(d_state_);
        d_state_ = nullptr;
    }
    if (d_conv_history_ != nullptr) {
        (void)hipFree(d_conv_history_);
        d_conv_history_ = nullptr;
    }
}

RecurrentLayerStateStorage::RecurrentLayerStateStorage(RecurrentLayerStateStorage&& other) noexcept
    : d_state_(std::exchange(other.d_state_, nullptr)),
      d_conv_history_(std::exchange(other.d_conv_history_, nullptr)),
      position_(other.position_) {}

RecurrentLayerStateStorage& RecurrentLayerStateStorage::operator=(RecurrentLayerStateStorage&& other) noexcept {
    if (this != &other) {
        if (d_state_ != nullptr) (void)hipFree(d_state_);
        if (d_conv_history_ != nullptr) (void)hipFree(d_conv_history_);
        d_state_ = std::exchange(other.d_state_, nullptr);
        d_conv_history_ = std::exchange(other.d_conv_history_, nullptr);
        position_ = other.position_;
    }
    return *this;
}

void RecurrentLayerStateStorage::reset() {
    if (d_state_ != nullptr) {
        MIINFER_HIP_CHECK(hipMemset(d_state_, 0, RecurrentLayerState::kStateBytes));
    }
    if (d_conv_history_ != nullptr) {
        MIINFER_HIP_CHECK(hipMemset(d_conv_history_, 0, RecurrentLayerState::kConvHistoryBytes));
    }
    position_ = 0;
}

void RecurrentLayerStateStorage::upload(
    std::span<const float> host_state,
    std::span<const float> host_history,
    std::uint32_t pos) {
    if (host_state.size() != RecurrentLayerState::kStateElements) {
        throw std::runtime_error("RecurrentLayerStateStorage: invalid state elements count");
    }
    if (host_history.size() != RecurrentLayerState::kConvHistoryElements) {
        throw std::runtime_error("RecurrentLayerStateStorage: invalid history elements count");
    }
    MIINFER_HIP_CHECK(hipMemcpy(d_state_, host_state.data(),
                                RecurrentLayerState::kStateBytes, hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_conv_history_, host_history.data(),
                                RecurrentLayerState::kConvHistoryBytes, hipMemcpyHostToDevice));
    position_ = pos;
}

void RecurrentLayerStateStorage::download(
    std::vector<float>& host_state,
    std::vector<float>& host_history) const {
    host_state.resize(RecurrentLayerState::kStateElements);
    host_history.resize(RecurrentLayerState::kConvHistoryElements);
    MIINFER_HIP_CHECK(hipMemcpy(host_state.data(), d_state_,
                                RecurrentLayerState::kStateBytes, hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(host_history.data(), d_conv_history_,
                                RecurrentLayerState::kConvHistoryBytes, hipMemcpyDeviceToHost));
}

} // namespace miinfer::prefill_v2
