#include "miinfer/prefill_v2/experimental_options.hpp"

#include <stdexcept>

int main() {
    using miinfer::prefill_v2::mmq_gateup_only_from_env;
    if (mmq_gateup_only_from_env(nullptr)) return 1;
    if (mmq_gateup_only_from_env("0")) return 2;
    if (!mmq_gateup_only_from_env("1")) return 3;
    try {
        (void)mmq_gateup_only_from_env("true");
    } catch (const std::invalid_argument&) {
        return 0;
    }
    return 4;
}
