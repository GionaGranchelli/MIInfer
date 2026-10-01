#pragma once

#include <algorithm>
#include <cstddef>
#include <optional>

namespace miinfer {

constexpr std::size_t kDefaultMaxOutputTokens = 4096;

enum class GenerationStopReason {
    kStopToken,
    kOutputLimit,
    kCancelled,
};

constexpr std::optional<std::size_t> resolve_output_token_limit(
    std::optional<std::size_t> requested,
    std::size_t prompt_tokens,
    std::size_t context_tokens) noexcept {
    if (context_tokens == 0 || prompt_tokens >= context_tokens) return std::nullopt;
    const std::size_t available = context_tokens - prompt_tokens;
    if (requested) {
        if (*requested == 0 || *requested > available) return std::nullopt;
        return requested;
    }
    return std::min(kDefaultMaxOutputTokens, available);
}

} // namespace miinfer
