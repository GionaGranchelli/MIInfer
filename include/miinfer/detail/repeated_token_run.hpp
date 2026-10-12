#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace miinfer::detail {

class RepeatedTokenRunDetector {
public:
    static constexpr std::size_t kThreshold = 4;

    bool observe(std::uint32_t token) noexcept {
        if (seen_ && token == previous_) {
            ++run_length_;
        } else {
            previous_ = token;
            run_length_ = 1;
            seen_ = true;
        }
        return run_length_ >= kThreshold;
    }

private:
    std::uint32_t previous_ = 0;
    std::size_t run_length_ = 0;
    bool seen_ = false;
};

inline bool has_repeated_token_run(std::span<const std::uint32_t> tokens) noexcept {
    RepeatedTokenRunDetector detector;
    bool detected = false;
    for (const auto token : tokens) detected = detector.observe(token) || detected;
    return detected;
}

}  // namespace miinfer::detail
