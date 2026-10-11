#include "miinfer/prefill_v2/experimental_options.hpp"

#include <stdexcept>

int main() {
    using miinfer::prefill_v2::mmq_gateup_only_from_env;
    using miinfer::prefill_v2::mmq_gateup_quantized_sum_from_env;
    if (mmq_gateup_only_from_env(nullptr)) return 1;
    if (mmq_gateup_only_from_env("0")) return 2;
    if (!mmq_gateup_only_from_env("1")) return 3;
    if (mmq_gateup_quantized_sum_from_env(nullptr)) return 5;
    if (mmq_gateup_quantized_sum_from_env("0")) return 6;
    if (!mmq_gateup_quantized_sum_from_env("1")) return 7;
    bool rejected_existing = false;
    try {
        (void)mmq_gateup_only_from_env("true");
    } catch (const std::invalid_argument&) {
        rejected_existing = true;
    }
    if (!rejected_existing) return 4;
    bool rejected = false;
    try {
        (void)mmq_gateup_quantized_sum_from_env("true");
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    return rejected ? 0 : 8;
}
