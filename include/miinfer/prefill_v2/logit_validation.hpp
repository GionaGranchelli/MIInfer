#pragma once

#include <cmath>
#include <stdexcept>

namespace miinfer::prefill_v2::detail {

inline void require_finite_logit(float value) {
    if (!std::isfinite(value)) throw std::runtime_error("non-finite logits");
}

} // namespace miinfer::prefill_v2::detail
