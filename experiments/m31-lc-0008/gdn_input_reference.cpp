#include "miinfer/m12_gdn_chunk.hpp"
#include "miinfer/prefill_v2/constants.hpp"
#include "miinfer/prefill_v2/state.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kTokens = 512;

void check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}

std::vector<float> read_f32(const std::filesystem::path& path, std::size_t count) {
    std::vector<float> values(count);
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + path.string());
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(count * sizeof(float)));
    if (!input || input.peek() != std::ifstream::traits_type::eof())
        throw std::runtime_error("wrong byte size for " + path.string());
    return values;
}

float* upload(const std::vector<float>& values) {
    float* device = nullptr;
    check(hipMalloc(&device, values.size() * sizeof(float)));
    check(hipMemcpy(device, values.data(), values.size() * sizeof(float), hipMemcpyHostToDevice));
    return device;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    using namespace miinfer::prefill_v2;
    const std::filesystem::path dir(argv[1]);
    const std::string prefix = "cpuinput.cpu.tile-0.";
    auto query = read_f32(dir / (prefix + "q_conv_predelta-0.full.f32"), kTokens * kKHeads * kState);
    auto key = read_f32(dir / (prefix + "k_conv_predelta-0.full.f32"), kTokens * kKHeads * kState);
    auto value = read_f32(dir / (prefix + "v_conv_predelta-0.full.f32"), kTokens * kVHeads * kState);
    auto beta = read_f32(dir / (prefix + "beta_sigmoid-0.full.f32"), kTokens * kVHeads);
    auto gate = read_f32(dir / (prefix + "gate-0.full.f32"), kTokens * kVHeads);
    std::vector<float> decay(gate.size());
    for (std::size_t i = 0; i < gate.size(); ++i) decay[i] = std::exp(gate[i]);

    float *d_query = upload(query), *d_key = upload(key), *d_value = upload(value);
    float *d_beta = upload(beta), *d_decay = upload(decay), *d_state = nullptr, *d_output = nullptr;
    check(hipMalloc(&d_state, RecurrentLayerState::kStateBytes));
    check(hipMemset(d_state, 0, RecurrentLayerState::kStateBytes));
    check(hipMalloc(&d_output, kTokens * kVHeads * kState * sizeof(float)));
    miinfer::launch_mx_gdn_chunk(d_query, d_key, d_value, d_beta, d_decay, d_state, d_output,
                                 kTokens, kKHeads, kVHeads, kState);
    check(hipDeviceSynchronize());

    std::vector<float> state(RecurrentLayerState::kStateElements);
    check(hipMemcpy(state.data(), d_state, RecurrentLayerState::kStateBytes, hipMemcpyDeviceToHost));
    const auto output_path = dir / "gdn.gpu-from-cpu-tile0.state-token-512.f32";
    if (std::filesystem::exists(output_path)) throw std::runtime_error("refusing to overwrite " + output_path.string());
    std::ofstream output(output_path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(state.data()),
                 static_cast<std::streamsize>(state.size() * sizeof(float)));
    if (!output) throw std::runtime_error("failed writing " + output_path.string());

    check(hipFree(d_query)); check(hipFree(d_key)); check(hipFree(d_value));
    check(hipFree(d_beta)); check(hipFree(d_decay)); check(hipFree(d_state)); check(hipFree(d_output));
}
