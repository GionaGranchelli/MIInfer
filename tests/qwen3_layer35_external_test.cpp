#include "miinfer/qwen3_gpu_layer.hpp"
#include "miinfer/q4_q8_gemv.hpp"
#include "miinfer/qwen3_primitives.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"
#include "miinfer/hip_check.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct Metrics {
    float max_abs = 0.0F;
    float mean_abs = 0.0F;
    float rmse = 0.0F;
    float max_rel = 0.0F;
    std::size_t max_index = 0;
    float actual_at_max = 0.0F;
    float expected_at_max = 0.0F;
    bool finite = true;
};

Metrics metrics(const std::vector<float>& actual, const std::vector<float>& expected);

struct Q8ReplayBlock {
    std::uint16_t d_bits = 0;
    std::int16_t sum = 0;
    std::int8_t qs[32]{};
};

std::uint16_t half_bits(float value) {
    const _Float16 half = static_cast<_Float16>(value);
    std::uint16_t bits = 0;
    static_assert(sizeof(half) == sizeof(bits));
    std::memcpy(&bits, &half, sizeof(bits));
    return bits;
}

float round_to_nearest_even(float value) {
    const float lower = std::floor(value);
    const float fraction = value - lower;
    if (fraction < 0.5F) return lower;
    if (fraction > 0.5F) return lower + 1.0F;
    const auto lower_integer = static_cast<long long>(lower);
    return (lower_integer % 2 == 0) ? lower : lower + 1.0F;
}

std::vector<Q8ReplayBlock> replay_q8_contract(
    const std::vector<float>& input,
    bool pinned_avx_contract) {
    if (input.empty() || input.size() % 32 != 0) {
        throw std::runtime_error("invalid layer-35 Q8 replay input");
    }
    std::vector<Q8ReplayBlock> output(input.size() / 32);
    for (std::size_t block = 0; block < output.size(); ++block) {
        const std::size_t base = block * 32;
        float amax = 0.0F;
        for (int index = 0; index < 32; ++index) {
            amax = std::max(amax, std::fabs(input[base + index]));
        }
        const float scale = amax / 127.0F;
        const float inverse = pinned_avx_contract
            ? (amax == 0.0F ? 0.0F : 127.0F / amax)
            : (scale == 0.0F ? 0.0F : 1.0F / scale);
        output[block].d_bits = half_bits(scale);
        for (int index = 0; index < 32; ++index) {
            const float scaled = input[base + index] * inverse;
            const float rounded = pinned_avx_contract
                ? round_to_nearest_even(scaled)
                : std::round(scaled);
            const int quantized = std::clamp(static_cast<int>(rounded), -127, 127);
            output[block].qs[index] = static_cast<std::int8_t>(quantized);
            output[block].sum = static_cast<std::int16_t>(output[block].sum + quantized);
        }
    }
    return output;
}

void report_q8_contract_replay(const std::vector<float>& input) {
    const auto miinfer = replay_q8_contract(input, false);
    const auto pinned = replay_q8_contract(input, true);
    std::size_t different_blocks = 0;
    std::size_t different_lanes = 0;
    std::size_t first_block = 0;
    int first_lane = 0;
    bool found_first = false;
    std::uint16_t max_scale_delta = 0;
    for (std::size_t block = 0; block < miinfer.size(); ++block) {
        const auto scale_delta = static_cast<std::uint16_t>(
            std::abs(static_cast<int>(miinfer[block].d_bits)
                     - static_cast<int>(pinned[block].d_bits)));
        max_scale_delta = std::max(max_scale_delta, scale_delta);
        bool block_differs = false;
        for (int lane = 0; lane < 32; ++lane) {
            if (miinfer[block].qs[lane] != pinned[block].qs[lane]) {
                ++different_lanes;
                block_differs = true;
                if (!found_first) {
                    first_block = block;
                    first_lane = lane;
                    found_first = true;
                }
            }
        }
        if (block_differs) ++different_blocks;
    }
    std::cout << "layer-35 ffn_norm Q8 contract replay (MIInfer vs pinned x86 AVX):\n"
              << "  blocks=" << miinfer.size()
              << " different_blocks=" << different_blocks
              << " different_lanes=" << different_lanes
              << " max_scale_bits_delta=" << max_scale_delta;
    if (found_first) {
        std::cout << " first_difference=block " << first_block
                  << " lane " << first_lane
                  << " miinfer_q=" << static_cast<int>(miinfer[first_block].qs[first_lane])
                  << " pinned_q=" << static_cast<int>(pinned[first_block].qs[first_lane]);
    }
    std::cout << '\n';
}

enum class AccumulationContract {
    miinfer,
    combined_scale,
    double_precision,
    four_float_accumulators,
};

float replay_projection_row(
    const miinfer::Qwen3TensorView& tensor,
    const std::vector<Q8ReplayBlock>& q8,
    std::size_t row,
    std::size_t columns,
    AccumulationContract contract) {
    const auto* weights = reinterpret_cast<const miinfer::Q4_0HostBlock*>(tensor.data());
    const std::size_t blocks_per_row = columns / 32;
    const auto block_value = [&](std::size_t block) {
        const auto& weight = weights[row * blocks_per_row + block];
        const auto& activation = q8[block];
        int dot = 0;
        for (int index = 0; index < 32; ++index) {
            const auto packed = weight.qs[index < 16 ? index : index - 16];
            const int nibble = index < 16 ? packed & 0x0f : (packed >> 4) & 0x0f;
            dot += (nibble - 8) * static_cast<int>(activation.qs[index]);
        }
        const float weight_scale = miinfer::fp16_bits_to_float(weight.d_bits);
        const float activation_scale = miinfer::fp16_bits_to_float(activation.d_bits);
        return std::pair{dot, std::pair{weight_scale, activation_scale}};
    };
    if (contract == AccumulationContract::double_precision) {
        double sum = 0.0;
        for (std::size_t block = 0; block < blocks_per_row; ++block) {
            const auto [dot, scales] = block_value(block);
            sum += static_cast<double>(dot) * scales.first * scales.second;
        }
        return static_cast<float>(sum);
    }
    if (contract == AccumulationContract::four_float_accumulators) {
        std::array<float, 4> sums{};
        for (std::size_t block = 0; block < blocks_per_row; ++block) {
            const auto [dot, scales] = block_value(block);
            sums[block % sums.size()] += static_cast<float>(dot) * scales.first * scales.second;
        }
        return ((sums[0] + sums[1]) + (sums[2] + sums[3]));
    }
    float sum = 0.0F;
    for (std::size_t block = 0; block < blocks_per_row; ++block) {
        const auto [dot, scales] = block_value(block);
        if (contract == AccumulationContract::combined_scale) {
            sum += static_cast<float>(dot) * (scales.first * scales.second);
        } else {
            sum += static_cast<float>(dot) * scales.first * scales.second;
        }
    }
    return sum;
}

std::vector<float> replay_projection(
    const miinfer::Qwen3TensorView& tensor,
    const std::vector<Q8ReplayBlock>& q8,
    std::size_t rows,
    std::size_t columns,
    AccumulationContract contract) {
    std::vector<float> output(rows);
    for (std::size_t row = 0; row < rows; ++row) {
        output[row] = replay_projection_row(tensor, q8, row, columns, contract);
    }
    return output;
}

std::span<const float> f32_tensor(
    const miinfer::Qwen3TensorView& tensor,
    std::size_t elements) {
    if (tensor.type() != miinfer::GgufTensorType::f32
        || tensor.bytes() != elements * sizeof(float)) {
        throw std::runtime_error("unexpected F32 tensor: " + tensor.name());
    }
    return {reinterpret_cast<const float*>(tensor.data()), elements};
}

enum class RmsReduction {
    float_sequential,
    double_sequential,
    pinned_ggml,
    four_float,
};

struct RmsVariant {
    std::vector<float> rms;
    std::vector<float> norm;
    double sum = 0.0;
    float mean = 0.0F;
    float root = 0.0F;
    float inverse = 0.0F;
};

std::vector<float> host_swiglu(
    const std::vector<float>& gate,
    const std::vector<float>& up);

RmsVariant replay_rms_norm(
    const std::vector<float>& input,
    std::span<const float> weights,
    float epsilon,
    RmsReduction reduction) {
    if (input.empty() || input.size() != weights.size()) {
        throw std::runtime_error("RMSNorm replay size mismatch");
    }
    RmsVariant result;
    result.rms.resize(input.size());
    result.norm.resize(input.size());
    if (reduction == RmsReduction::float_sequential) {
        float sum = 0.0F;
        for (const float value : input) sum += value * value;
        result.sum = sum;
    } else if (reduction == RmsReduction::four_float) {
        std::array<float, 4> sums{};
        for (std::size_t index = 0; index < input.size(); ++index) {
            sums[index % sums.size()] += input[index] * input[index];
        }
        result.sum = (sums[0] + sums[1]) + (sums[2] + sums[3]);
    } else {
        double sum = 0.0;
        for (const float value : input) {
            // ggml's pinned CPU path uses: (ggml_float)(x[i] * x[i]).
            // Keep the product in float for the pinned variant; the current
            // MIInfer path widens before multiplication.
            const double product = reduction == RmsReduction::pinned_ggml
                ? static_cast<double>(value * value)
                : static_cast<double>(value) * value;
            sum += product;
        }
        result.sum = sum;
    }
    result.mean = static_cast<float>(result.sum / input.size());
    result.root = std::sqrt(result.mean + epsilon);
    result.inverse = 1.0F / result.root;
    for (std::size_t index = 0; index < input.size(); ++index) {
        result.rms[index] = input[index] * result.inverse;
        result.norm[index] = result.rms[index] * weights[index];
    }
    return result;
}

struct FfnTailReplay {
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> swiglu;
    std::vector<float> down;
    std::vector<float> layer_output;
};

FfnTailReplay replay_ffn_tail(
    const miinfer::Qwen3Model& model,
    const std::vector<float>& ffn_input,
    const std::vector<float>& ffn_norm,
    const std::vector<float>& expected_layer_input) {
    const auto& layer = model.layers().at(35);
    const auto q8 = replay_q8_contract(ffn_norm, false);
    FfnTailReplay result{
        replay_projection(layer.gate, q8, ffn_norm.size() * 3, ffn_norm.size(),
                          AccumulationContract::miinfer),
        replay_projection(layer.up, q8, ffn_norm.size() * 3, ffn_norm.size(),
                          AccumulationContract::miinfer),
        {},
        {},
        {},
    };
    result.swiglu = host_swiglu(result.gate, result.up);
    const auto down_q8 = replay_q8_contract(result.swiglu, false);
    result.down = replay_projection(layer.down, down_q8, ffn_input.size(),
                                    result.swiglu.size(), AccumulationContract::miinfer);
    result.layer_output.resize(ffn_input.size());
    for (std::size_t index = 0; index < result.layer_output.size(); ++index) {
        result.layer_output[index] = expected_layer_input[index] + result.down[index];
    }
    return result;
}

void report_pre_ffn_contract(
    const miinfer::Qwen3Model& model,
    const std::vector<float>& input,
    const std::vector<std::vector<float>>& external) {
    const auto& layer = model.layers().at(35);
    const auto external_o = [&] {
        std::vector<float> result(external[11].size());
        for (std::size_t index = 0; index < result.size(); ++index) {
            result[index] = external[11][index] - input[index];
        }
        return result;
    }();
    const auto attention_q8 = replay_q8_contract(external[10], false);
    const auto replayed_o = replay_projection(
        layer.output, attention_q8, external_o.size(), external[10].size(),
        AccumulationContract::miinfer);
    const auto o_error = metrics(replayed_o, external_o);
    std::vector<float> residual_from_external_o(input.size());
    std::vector<float> residual_from_host_o(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        residual_from_external_o[index] = external_o[index] + input[index];
        residual_from_host_o[index] = replayed_o[index] + input[index];
    }
    const auto residual_external = metrics(residual_from_external_o, external[11]);
    const auto residual_host = metrics(residual_from_host_o, external[11]);
    std::cout << "layer-35 external-conditioned O/residual replay:\n"
              << "  O replay vs derived external O: max_abs=" << o_error.max_abs
              << " mean_abs=" << o_error.mean_abs
              << " rmse=" << o_error.rmse
              << " max_rel=" << o_error.max_rel << '\n'
              << "  external O + external input vs external ffn_input: max_abs="
              << residual_external.max_abs << " mean_abs=" << residual_external.mean_abs
              << "\n"
              << "  replayed O + external input vs external ffn_input: max_abs="
              << residual_host.max_abs << " mean_abs=" << residual_host.mean_abs << '\n';

    const auto ffn_weights = f32_tensor(layer.ffn_norm, input.size());
    const std::array reductions{
        std::pair{"float-sequential", RmsReduction::float_sequential},
        std::pair{"double-sequential", RmsReduction::double_sequential},
        std::pair{"pinned-ggml", RmsReduction::pinned_ggml},
        std::pair{"four-float", RmsReduction::four_float},
    };
    std::cout << "layer-35 pre-FFN RMSNorm candidates (external ffn_input):\n";
    for (const auto& [name, reduction] : reductions) {
        const auto rms = replay_rms_norm(external[11], ffn_weights, 1.0e-6F, reduction);
        const auto norm_error = metrics(rms.norm, external[12]);
        const auto tail = replay_ffn_tail(model, external[11], rms.norm, external[11]);
        const auto gate_error = metrics(tail.gate, external[13]);
        const auto up_error = metrics(tail.up, external[14]);
        const auto swiglu_error = metrics(tail.swiglu, external[15]);
        const auto down_error = metrics(tail.down, external[16]);
        const auto layer_error = metrics(tail.layer_output, external[17]);
        std::cout << "  " << name
                  << ": sum=" << rms.sum
                  << " mean=" << rms.mean
                  << " root=" << rms.root
                  << " inverse=" << rms.inverse
                  << " norm_max_abs=" << norm_error.max_abs
                  << " gate_max_abs=" << gate_error.max_abs
                  << " up_max_abs=" << up_error.max_abs
                  << " swiglu_max_abs=" << swiglu_error.max_abs
                  << " down_max_abs=" << down_error.max_abs
                  << " layer_max_abs=" << layer_error.max_abs
                  << " layer_mean_abs=" << layer_error.mean_abs
                  << " layer_rmse=" << layer_error.rmse << '\n';
    }
}

std::vector<float> expand_gqa(
    const std::vector<float>& values,
    std::size_t query_heads,
    std::size_t kv_heads,
    std::size_t head_dim) {
    if (values.size() != kv_heads * head_dim || query_heads % kv_heads != 0) {
        throw std::runtime_error("invalid GQA expansion dimensions");
    }
    std::vector<float> expanded(query_heads * head_dim);
    const std::size_t groups = query_heads / kv_heads;
    for (std::size_t query_head = 0; query_head < query_heads; ++query_head) {
        const std::size_t kv_head = query_head / groups;
        std::copy_n(values.data() + kv_head * head_dim, head_dim,
                    expanded.data() + query_head * head_dim);
    }
    return expanded;
}

std::vector<float> fp16_roundtrip(const std::vector<float>& values) {
    std::vector<float> result(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        result[index] = miinfer::fp16_bits_to_float(half_bits(values[index]));
    }
    return result;
}

std::vector<miinfer::Q8ExactBlock> quantize_gpu_q8_exact(
    std::span<const float> input,
    bool f16_input) {
    if (input.empty() || input.size() % miinfer::kQ8_1BlockSize != 0) {
        throw std::runtime_error("invalid GPU Q8 boundary input");
    }
    void* input_device = nullptr;
    void* input_half = nullptr;
    void* output_device = nullptr;
    const auto bytes = input.size() * sizeof(float);
    const auto block_count = input.size() / miinfer::kQ8_1BlockSize;
    MIINFER_HIP_CHECK(hipMalloc(&input_device, bytes));
    MIINFER_HIP_CHECK(hipMalloc(&output_device,
                                block_count * sizeof(miinfer::Q8ExactBlock)));
    MIINFER_HIP_CHECK(hipMemcpy(input_device, input.data(), bytes, hipMemcpyHostToDevice));
    if (f16_input) {
        MIINFER_HIP_CHECK(hipMalloc(&input_half, input.size() * sizeof(__half)));
        miinfer::launch_qwen3_f32_to_f16(
            static_cast<const float*>(input_device), static_cast<__half*>(input_half),
            static_cast<std::uint32_t>(input.size()));
        miinfer::launch_q8_exact_quantize(
            static_cast<const __half*>(input_half),
            static_cast<miinfer::Q8ExactBlock*>(output_device),
            static_cast<int>(input.size()));
    } else {
        miinfer::launch_q8_exact_quantize_f32(
            static_cast<const float*>(input_device),
            static_cast<miinfer::Q8ExactBlock*>(output_device),
            static_cast<int>(input.size()));
    }
    std::vector<miinfer::Q8ExactBlock> result(block_count);
    MIINFER_HIP_CHECK(hipMemcpy(result.data(), output_device,
                                result.size() * sizeof(miinfer::Q8ExactBlock),
                                hipMemcpyDeviceToHost));
    if (input_half != nullptr) MIINFER_HIP_CHECK(hipFree(input_half));
    MIINFER_HIP_CHECK(hipFree(output_device));
    MIINFER_HIP_CHECK(hipFree(input_device));
    return result;
}

struct Q8BoundaryDiff {
    std::size_t different_scale_blocks = 0;
    std::size_t different_sum_blocks = 0;
    std::size_t different_lane_values = 0;
    std::size_t first_block = 0;
    int first_lane = -1;
    std::uint16_t max_scale_bits_delta = 0;
    int max_sum_delta = 0;
};

Q8BoundaryDiff compare_q8_blocks(
    const std::vector<miinfer::Q8ExactBlock>& actual,
    const std::vector<Q8ReplayBlock>& expected) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("Q8 boundary block count mismatch");
    }
    Q8BoundaryDiff result;
    bool found_first = false;
    for (std::size_t block = 0; block < actual.size(); ++block) {
        std::uint16_t actual_d_bits = 0;
        std::memcpy(&actual_d_bits, &actual[block].d, sizeof(actual_d_bits));
        const auto scale_delta = static_cast<std::uint16_t>(
            std::abs(static_cast<int>(actual_d_bits)
                     - static_cast<int>(expected[block].d_bits)));
        result.max_scale_bits_delta = std::max(result.max_scale_bits_delta, scale_delta);
        if (actual_d_bits != expected[block].d_bits) ++result.different_scale_blocks;
        const int sum_delta = static_cast<int>(actual[block].sum)
            - static_cast<int>(expected[block].sum);
        result.max_sum_delta = std::max(result.max_sum_delta, std::abs(sum_delta));
        if (sum_delta != 0) ++result.different_sum_blocks;
        bool block_differs = actual_d_bits != expected[block].d_bits
            || sum_delta != 0;
        for (int lane = 0; lane < miinfer::kQ8_1BlockSize; ++lane) {
            if (actual[block].qs[lane] != expected[block].qs[lane]) {
                ++result.different_lane_values;
                block_differs = true;
                if (!found_first) {
                    result.first_block = block;
                    result.first_lane = lane;
                    found_first = true;
                }
            }
        }
        if (block_differs && !found_first) {
            result.first_block = block;
            found_first = true;
        }
    }
    return result;
}

std::vector<float> dequantize_q8_exact(
    const std::vector<miinfer::Q8ExactBlock>& blocks) {
    std::vector<float> result(blocks.size() * miinfer::kQ8_1BlockSize);
    for (std::size_t block = 0; block < blocks.size(); ++block) {
        std::uint16_t d_bits = 0;
        std::memcpy(&d_bits, &blocks[block].d, sizeof(d_bits));
        const float scale = miinfer::fp16_bits_to_float(d_bits);
        for (int lane = 0; lane < miinfer::kQ8_1BlockSize; ++lane) {
            result[block * miinfer::kQ8_1BlockSize + lane] =
                scale * static_cast<float>(blocks[block].qs[lane]);
        }
    }
    return result;
}

struct AttentionPathReplay {
    std::vector<float> v;
    std::vector<float> attention_output;
    std::vector<float> ffn_input;
    std::vector<float> ffn_norm;
    std::vector<float> layer_output;
};

AttentionPathReplay replay_attention_path(
    const miinfer::Qwen3Model& model,
    const std::vector<float>& layer_input,
    const std::vector<float>& attn_norm,
    bool materialize_attention_f16 = false) {
    const auto& config = model.config();
    const auto& layer = model.layers().at(35);
    const auto v_q8 = replay_q8_contract(attn_norm, false);
    AttentionPathReplay result;
    result.v = replay_projection(layer.v, v_q8,
                                  config.kv_heads * config.head_dim,
                                  config.hidden_size, AccumulationContract::miinfer);
    result.attention_output = expand_gqa(
        result.v, config.attention_heads, config.kv_heads, config.head_dim);
    if (materialize_attention_f16) {
        result.attention_output = fp16_roundtrip(result.attention_output);
    }
    const auto o_q8 = replay_q8_contract(result.attention_output, false);
    const auto o = replay_projection(layer.output, o_q8, config.hidden_size,
                                     config.hidden_size, AccumulationContract::miinfer);
    result.ffn_input.resize(layer_input.size());
    for (std::size_t index = 0; index < layer_input.size(); ++index) {
        result.ffn_input[index] = layer_input[index] + o[index];
    }
    const auto ffn_weights = f32_tensor(layer.ffn_norm, layer_input.size());
    result.ffn_norm = replay_rms_norm(
        result.ffn_input, ffn_weights, 1.0e-6F, RmsReduction::double_sequential).norm;
    const auto tail = replay_ffn_tail(model, result.ffn_input, result.ffn_norm,
                                      result.ffn_input);
    result.layer_output = tail.layer_output;
    return result;
}

void report_attention_v_contract(
    const miinfer::Qwen3Model& model,
    const std::vector<float>& input,
    const std::vector<std::vector<float>>& external,
    const miinfer::Qwen3LayerTrace& host) {
    const auto& config = model.config();
    const auto external_expanded_v = expand_gqa(
        external[5], config.attention_heads, config.kv_heads, config.head_dim);
    const auto external_expanded_v_f16 = fp16_roundtrip(external_expanded_v);
    const auto host_expanded_v = expand_gqa(
        host.v_projection, config.attention_heads, config.kv_heads, config.head_dim);
    const auto external_identity = metrics(external_expanded_v, external[10]);
    const auto external_f16_identity = metrics(external_expanded_v_f16, external[10]);
    const auto host_identity = metrics(host_expanded_v, host.attention_output);
    std::cout << "layer-35 position-zero GQA attention identity:\n"
              << "  external expanded V vs external attention_output: max_abs="
              << external_identity.max_abs << " mean_abs=" << external_identity.mean_abs
              << " rmse=" << external_identity.rmse << '\n'
              << "  FP16(external expanded V) vs external attention_output: max_abs="
              << external_f16_identity.max_abs << " mean_abs=" << external_f16_identity.mean_abs
              << " rmse=" << external_f16_identity.rmse << '\n'
              << "  host expanded V vs host attention_output: max_abs="
              << host_identity.max_abs << " mean_abs=" << host_identity.mean_abs
              << " rmse=" << host_identity.rmse << '\n';

    const auto& layer = model.layers().at(35);
    const auto external_v_q8 = replay_q8_contract(external[0], false);
    const auto replay_external_v = replay_projection(
        layer.v, external_v_q8, config.kv_heads * config.head_dim,
        config.hidden_size, AccumulationContract::miinfer);
    const auto replay_external_expanded = expand_gqa(
        replay_external_v, config.attention_heads, config.kv_heads, config.head_dim);
    const auto external_v_error = metrics(replay_external_v, external[5]);
    const auto external_attention_error = metrics(replay_external_expanded, external[10]);

    const auto host_v_q8 = replay_q8_contract(host.attn_norm, false);
    const auto replay_host_v = replay_projection(
        layer.v, host_v_q8, config.kv_heads * config.head_dim,
        config.hidden_size, AccumulationContract::miinfer);
    const auto replay_host_expanded = expand_gqa(
        replay_host_v, config.attention_heads, config.kv_heads, config.head_dim);
    const auto host_v_error = metrics(replay_host_v, external[5]);
    const auto host_attention_error = metrics(replay_host_expanded, external[10]);
    const auto replay_external_expanded_f16 = fp16_roundtrip(replay_external_expanded);
    const auto external_replay_f16_attention_error = metrics(
        replay_external_expanded_f16, external[10]);
    std::cout << "layer-35 V replay and GQA expansion:\n"
              << "  external attn_norm -> host V: V_max_abs=" << external_v_error.max_abs
              << " V_mean_abs=" << external_v_error.mean_abs
              << " expanded_attention_max_abs=" << external_attention_error.max_abs
              << " expanded_attention_mean_abs=" << external_attention_error.mean_abs
              << " f16_expanded_attention_max_abs=" << external_replay_f16_attention_error.max_abs
              << " f16_expanded_attention_mean_abs=" << external_replay_f16_attention_error.mean_abs
              << '\n'
              << "  host attn_norm -> host V: V_max_abs=" << host_v_error.max_abs
              << " V_mean_abs=" << host_v_error.mean_abs
              << " expanded_attention_max_abs=" << host_attention_error.max_abs
              << " expanded_attention_mean_abs=" << host_attention_error.mean_abs << '\n';

    const auto attn_weights = f32_tensor(layer.attention_norm, input.size());
    const std::array reductions{
        std::pair{"float-sequential", RmsReduction::float_sequential},
        std::pair{"double-sequential", RmsReduction::double_sequential},
        std::pair{"pinned-ggml", RmsReduction::pinned_ggml},
        std::pair{"four-float", RmsReduction::four_float},
    };
    std::cout << "layer-35 attention RMSNorm candidates (external layer input):\n";
    for (const auto& [name, reduction] : reductions) {
        const auto rms = replay_rms_norm(input, attn_weights, 1.0e-6F, reduction);
        const auto norm_error = metrics(rms.norm, external[0]);
        const auto path = replay_attention_path(model, input, rms.norm);
        const auto path_f16 = replay_attention_path(model, input, rms.norm, true);
        const auto v_error = metrics(path.v, external[5]);
        const auto attention_error = metrics(path.attention_output, external[10]);
        const auto attention_f16_error = metrics(path_f16.attention_output, external[10]);
        const auto ffn_input_error = metrics(path.ffn_input, external[11]);
        const auto ffn_norm_error = metrics(path.ffn_norm, external[12]);
        const auto layer_error = metrics(path.layer_output, external[17]);
        const auto ffn_input_f16_error = metrics(path_f16.ffn_input, external[11]);
        const auto layer_f16_error = metrics(path_f16.layer_output, external[17]);
        std::cout << "  " << name
                  << ": sum=" << rms.sum
                  << " mean=" << rms.mean
                  << " root=" << rms.root
                  << " inverse=" << rms.inverse
                  << " attn_norm_max_abs=" << norm_error.max_abs
                  << " V_max_abs=" << v_error.max_abs
                  << " attention_output_max_abs=" << attention_error.max_abs
                  << " attention_output_f16_max_abs=" << attention_f16_error.max_abs
                  << " ffn_input_max_abs=" << ffn_input_error.max_abs
                  << " ffn_input_f16_max_abs=" << ffn_input_f16_error.max_abs
                  << " ffn_norm_max_abs=" << ffn_norm_error.max_abs
                  << " layer_max_abs=" << layer_error.max_abs
                  << " layer_f16_max_abs=" << layer_f16_error.max_abs
                  << " layer_mean_abs=" << layer_error.mean_abs
                  << " layer_rmse=" << layer_error.rmse << '\n';
    }
}

void report_v_precision_contract(
    const miinfer::Qwen3Model& model,
    const miinfer::Qwen3GpuPlan& plan,
    const std::vector<float>& layer_input,
    const std::vector<std::vector<float>>& external) {
    using Precision = miinfer::Qwen3ProjectionPrecision;
    const auto& config = model.config();
    const std::array<std::pair<const char*, Precision>, 4> variants{
        std::pair{"F16->Q8Exact->F16", Precision::f16_input_q8_f16_output},
        std::pair{"F32->Q8Exact->F16", Precision::f32_input_q8_f16_output},
        std::pair{"F16->Q8Exact->F32", Precision::f16_input_q8_f32_output},
        std::pair{"F32->Q8Exact->F32", Precision::f32_input_q8_f32_output},
    };

    const auto& layer = model.layers().at(35);
    const auto external_q8 = replay_q8_contract(external[0], false);
    const auto host_external_v = replay_projection(
        layer.v, external_q8, config.kv_heads * config.head_dim,
        config.hidden_size, AccumulationContract::miinfer);
    const auto host_v_error = metrics(host_external_v, external[5]);
    std::cout << "M4-B18 layer-35 V precision/cause replay (external attn_norm input):\n"
              << "  host external-conditioned V: max_abs=" << host_v_error.max_abs
              << " mean_abs=" << host_v_error.mean_abs
              << " rmse=" << host_v_error.rmse
              << " max_rel=" << host_v_error.max_rel << '\n';

    std::cout << "  GPU V policy matrix (all policies use Q8Exact metadata):\n";
    for (const auto& [name, precision] : variants) {
        const auto v = miinfer::execute_qwen3_projection_gpu_probe(
            plan, 35, miinfer::Qwen3Projection::v, external[0], precision).output;
        const auto v_error = metrics(v, external[5]);
        const auto expanded = expand_gqa(
            v, config.attention_heads, config.kv_heads, config.head_dim);
        // The external position-zero attention checkpoint is exactly the
        // FP16-materialized expanded V.  Apply that same boundary before the
        // existing GPU downstream replay so this probe changes only V policy.
        const auto attention = fp16_roundtrip(expanded);
        const auto downstream = miinfer::execute_qwen3_layer_gpu_attention_override(
            plan, 35, layer_input, attention, 0);
        const auto attention_error = metrics(downstream.attention_output, external[10]);
        const auto layer_error = metrics(downstream.layer_output, external[17]);
        std::cout << "    " << name
                  << ": V_max_abs=" << v_error.max_abs
                  << " V_mean_abs=" << v_error.mean_abs
                  << " V_rmse=" << v_error.rmse
                  << " V_max_rel=" << v_error.max_rel
                  << " attention_max_abs=" << attention_error.max_abs
                  << " layer_max_abs=" << layer_error.max_abs
                  << " layer_mean_abs=" << layer_error.mean_abs
                  << " layer_rmse=" << layer_error.rmse << '\n';
    }
}

void report_v_attention_q8_boundary(
    const miinfer::Qwen3Model& model,
    const miinfer::Qwen3GpuPlan& plan,
    const std::vector<float>& layer_input,
    const std::vector<std::vector<float>>& external) {
    using Precision = miinfer::Qwen3ProjectionPrecision;
    const auto& config = model.config();
    const std::array<std::pair<const char*, Precision>, 4> variants{
        std::pair{"F16->Q8Exact->F16", Precision::f16_input_q8_f16_output},
        std::pair{"F32->Q8Exact->F16", Precision::f32_input_q8_f16_output},
        std::pair{"F16->Q8Exact->F32", Precision::f16_input_q8_f32_output},
        std::pair{"F32->Q8Exact->F32", Precision::f32_input_q8_f32_output},
    };

    const auto reference_q8 = replay_q8_contract(external[10], false);
    const auto gpu_reference_q8 = quantize_gpu_q8_exact(external[10], true);
    const auto reference_q8_diff = compare_q8_blocks(gpu_reference_q8, reference_q8);
    const auto reference_dequant = dequantize_q8_exact(gpu_reference_q8);
    const auto reference_dequant_error = metrics(reference_dequant, external[10]);
    std::cout << "M4-B19 layer-35 attention FP16->Q8Exact->O boundary:\n"
              << "  GPU quantization of external attention vs host Q8 contract:"
              << " different_scale_blocks=" << reference_q8_diff.different_scale_blocks
              << " different_sum_blocks=" << reference_q8_diff.different_sum_blocks
              << " different_lane_values=" << reference_q8_diff.different_lane_values
              << " max_scale_bits_delta=" << reference_q8_diff.max_scale_bits_delta
              << " max_sum_delta=" << reference_q8_diff.max_sum_delta << '\n'
              << "  external attention dequantized from GPU Q8: max_abs="
              << reference_dequant_error.max_abs
              << " mean_abs=" << reference_dequant_error.mean_abs
              << " rmse=" << reference_dequant_error.rmse << '\n';

    std::vector<float> external_o(external[11].size());
    for (std::size_t index = 0; index < external_o.size(); ++index) {
        external_o[index] = external[11][index] - layer_input[index];
    }

    const auto external_control = miinfer::execute_qwen3_layer_gpu_attention_override(
        plan, 35, layer_input, external[10], 0);
    std::vector<float> external_control_o(external_control.ffn_input.size());
    for (std::size_t index = 0; index < external_control_o.size(); ++index) {
        external_control_o[index] = external_control.ffn_input[index] - layer_input[index];
    }
    const auto control_o_error = metrics(external_control_o, external_o);
    const auto control_layer_error = metrics(external_control.layer_output, external[17]);
    std::cout << "  external attention GPU control: O_max_abs=" << control_o_error.max_abs
              << " layer_max_abs=" << control_layer_error.max_abs
              << " layer_mean_abs=" << control_layer_error.mean_abs
              << " layer_rmse=" << control_layer_error.rmse << '\n';

    std::cout << "  candidate attention/Q8/O boundary (against external attention/Q8):\n";
    for (const auto& [name, precision] : variants) {
        const auto v = miinfer::execute_qwen3_projection_gpu_probe(
            plan, 35, miinfer::Qwen3Projection::v, external[0], precision).output;
        const auto attention = fp16_roundtrip(expand_gqa(
            v, config.attention_heads, config.kv_heads, config.head_dim));
        const auto attention_error = metrics(attention, external[10]);
        const auto q8 = quantize_gpu_q8_exact(attention, true);
        const auto q8_diff = compare_q8_blocks(q8, reference_q8);
        const auto dequantized = dequantize_q8_exact(q8);
        const auto dequant_error = metrics(dequantized, external[10]);
        const auto downstream = miinfer::execute_qwen3_layer_gpu_attention_override(
            plan, 35, layer_input, attention, 0);
        std::vector<float> candidate_o(downstream.ffn_input.size());
        for (std::size_t index = 0; index < candidate_o.size(); ++index) {
            candidate_o[index] = downstream.ffn_input[index] - layer_input[index];
        }
        const auto o_error = metrics(candidate_o, external_o);
        const auto layer_error = metrics(downstream.layer_output, external[17]);
        std::cout << "    " << name
                  << ": attention_max_abs=" << attention_error.max_abs
                  << " q8_scale_blocks=" << q8_diff.different_scale_blocks
                  << " q8_sum_blocks=" << q8_diff.different_sum_blocks
                  << " q8_lanes=" << q8_diff.different_lane_values
                  << " q8_max_scale_bits_delta=" << q8_diff.max_scale_bits_delta
                  << " q8_max_sum_delta=" << q8_diff.max_sum_delta
                  << " dequant_max_abs=" << dequant_error.max_abs
                  << " O_max_abs=" << o_error.max_abs
                  << " layer_max_abs=" << layer_error.max_abs
                  << " layer_mean_abs=" << layer_error.mean_abs
                  << " layer_rmse=" << layer_error.rmse;
        if (q8_diff.first_lane >= 0) {
            std::cout << " first_q8_difference=block " << q8_diff.first_block
                      << " lane " << q8_diff.first_lane;
        } else if (q8_diff.different_scale_blocks != 0
                   || q8_diff.different_sum_blocks != 0) {
            std::cout << " first_q8_difference=block " << q8_diff.first_block
                      << " metadata";
        }
        std::cout << '\n';
    }
}

void report_host_projection_contracts(
    const miinfer::Qwen3Model& model,
    const std::vector<float>& ffn_norm,
    const std::vector<std::vector<float>>& external,
    const miinfer::Qwen3LayerTrace& host) {
    const auto current_q8 = replay_q8_contract(ffn_norm, false);
    const auto pinned_q8 = replay_q8_contract(ffn_norm, true);
    const auto gate_error = metrics(host.gate, external[13]);
    const auto up_error = metrics(host.up, external[14]);
    const std::array rows{std::size_t{5607}, gate_error.max_index, up_error.max_index};
    const auto& layer = model.layers().at(35);
    const std::array cases{
        std::pair{"Gate", &layer.gate},
        std::pair{"Up", &layer.up},
    };
    const std::array contracts{
        std::pair{"current-sequential", AccumulationContract::miinfer},
        std::pair{"combined-scale", AccumulationContract::combined_scale},
        std::pair{"double-sequential", AccumulationContract::double_precision},
        std::pair{"four-float", AccumulationContract::four_float_accumulators},
    };
    std::cout << "layer-35 Gate/Up projections with exact external ffn_norm input:\n";
    for (const auto& [name, tensor] : cases) {
        const auto& expected = name == std::string_view{"Gate"} ? external[13] : external[14];
        const auto current = replay_projection(
            *tensor, current_q8, expected.size(), 4096, AccumulationContract::miinfer);
        const auto pinned = replay_projection(
            *tensor, pinned_q8, expected.size(), 4096, AccumulationContract::miinfer);
        const auto current_result = metrics(current, expected);
        const auto pinned_result = metrics(pinned, expected);
        std::cout << "  " << name
                  << ": current_q8 max_abs=" << current_result.max_abs
                  << " mean_abs=" << current_result.mean_abs
                  << " rmse=" << current_result.rmse
                  << " pinned_q8 max_abs=" << pinned_result.max_abs
                  << " mean_abs=" << pinned_result.mean_abs
                  << " rmse=" << pinned_result.rmse << '\n';
    }
    std::cout << "layer-35 Gate/Up critical-row accumulation replay:\n";
    for (const auto& [name, tensor] : cases) {
        const auto& expected = name == std::string_view{"Gate"} ? external[13] : external[14];
        std::cout << "  " << name << ":\n";
        for (const auto row : rows) {
            std::cout << "    row=" << row << " external=" << expected[row];
            for (const auto& [contract_name, contract] : contracts) {
                const float current = replay_projection_row(
                    *tensor, current_q8, row, 4096, contract);
                const float pinned = replay_projection_row(
                    *tensor, pinned_q8, row, 4096, contract);
                std::cout << " " << contract_name
                          << "_value=" << current
                          << "_err=" << std::fabs(current - expected[row])
                          << " host_err=" << std::fabs((host.*(name == std::string_view{"Gate"}
                              ? &miinfer::Qwen3LayerTrace::gate
                              : &miinfer::Qwen3LayerTrace::up))[row] - expected[row])
                          << " pinned_q8_err=" << std::fabs(pinned - expected[row]);
            }
            std::cout << '\n';
        }
    }
}

std::vector<float> read_f32(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open layer-35 trace: " + path.string());
    file.seekg(0, std::ios::end);
    const auto size = file.tellg();
    if (size < 0 || size % static_cast<std::streamoff>(sizeof(float)) != 0) {
        throw std::runtime_error("invalid F32 layer-35 trace: " + path.string());
    }
    file.seekg(0, std::ios::beg);
    std::vector<float> values(static_cast<std::size_t>(size) / sizeof(float));
    file.read(reinterpret_cast<char*>(values.data()), size);
    if (!file) throw std::runtime_error("short layer-35 trace: " + path.string());
    return values;
}

Metrics metrics(const std::vector<float>& actual, const std::vector<float>& expected) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("layer-35 checkpoint size mismatch");
    }
    Metrics result;
    double sum_abs = 0.0;
    double sum_squared = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index]) || !std::isfinite(expected[index])) {
            result.finite = false;
        }
        const float error = std::fabs(actual[index] - expected[index]);
        sum_abs += error;
        sum_squared += static_cast<double>(error) * error;
        if (error > result.max_abs) {
            result.max_abs = error;
            result.max_index = index;
            result.actual_at_max = actual[index];
            result.expected_at_max = expected[index];
        }
        result.max_rel = std::max(result.max_rel,
            error / std::max(1.0F, std::fabs(expected[index])));
    }
    if (!actual.empty()) {
        result.mean_abs = static_cast<float>(sum_abs / actual.size());
        result.rmse = static_cast<float>(std::sqrt(sum_squared / actual.size()));
    }
    return result;
}

std::vector<float> host_swiglu(
    const std::vector<float>& gate,
    const std::vector<float>& up) {
    if (gate.size() != up.size()) {
        throw std::runtime_error("layer-35 Gate/Up size mismatch");
    }
    std::vector<float> output(gate.size());
    miinfer::silu_mul_reference(gate, up, output);
    return output;
}

struct Checkpoint {
    const char* name;
    const char* file;
    std::vector<float> miinfer::Qwen3LayerTrace::*field;
    float tolerance;
};

constexpr std::array kCheckpoints{
    Checkpoint{"attn_norm", "attn-norm.f32", &miinfer::Qwen3LayerTrace::attn_norm, 5.0e-2F},
    Checkpoint{"q_projection", "q-projection.f32", &miinfer::Qwen3LayerTrace::q_projection, 5.0e-2F},
    Checkpoint{"q_reshape", "q-reshape.f32", &miinfer::Qwen3LayerTrace::q_reshape, 5.0e-2F},
    Checkpoint{"q_normed", "q-normed.f32", &miinfer::Qwen3LayerTrace::q_normed, 5.0e-2F},
    Checkpoint{"q_rope", "q-rope.f32", &miinfer::Qwen3LayerTrace::q_rope, 5.0e-2F},
    Checkpoint{"v_projection", "v-projection.f32", &miinfer::Qwen3LayerTrace::v_projection, 5.0e-2F},
    Checkpoint{"k_projection", "k-projection.f32", &miinfer::Qwen3LayerTrace::k_projection, 5.0e-2F},
    Checkpoint{"k_reshape", "k-reshape.f32", &miinfer::Qwen3LayerTrace::k_reshape, 5.0e-2F},
    Checkpoint{"k_normed", "k-normed.f32", &miinfer::Qwen3LayerTrace::k_normed, 5.0e-2F},
    Checkpoint{"k_rope", "k-rope.f32", &miinfer::Qwen3LayerTrace::k_rope, 5.0e-2F},
    Checkpoint{"attention_output", "attention-output.f32", &miinfer::Qwen3LayerTrace::attention_output, 5.0e-2F},
    Checkpoint{"ffn_input", "ffn-input.f32", &miinfer::Qwen3LayerTrace::ffn_input, 5.0e-2F},
    Checkpoint{"ffn_norm", "ffn-norm.f32", &miinfer::Qwen3LayerTrace::ffn_norm, 5.0e-2F},
    Checkpoint{"gate", "gate.f32", &miinfer::Qwen3LayerTrace::gate, 5.0e-2F},
    Checkpoint{"up", "up.f32", &miinfer::Qwen3LayerTrace::up, 5.0e-2F},
    Checkpoint{"swiglu", "swiglu.f32", &miinfer::Qwen3LayerTrace::swiglu, 5.0e-2F},
    Checkpoint{"ffn_output", "ffn-output.f32", &miinfer::Qwen3LayerTrace::ffn_output, 5.0e-2F},
    Checkpoint{"layer_output", "layer-output.f32", &miinfer::Qwen3LayerTrace::layer_output, 5.0e-2F},
};

bool report(const char* authority, const miinfer::Qwen3LayerTrace& actual,
            const std::vector<std::vector<float>>& external) {
    std::cout << authority << " vs external layer-35 trace:\n";
    bool passed = true;
    bool previous_passed = true;
    for (const auto& checkpoint : kCheckpoints) {
        const auto result = metrics(actual.*(checkpoint.field),
                                    external[&checkpoint - kCheckpoints.data()]);
        const bool checkpoint_pass = result.finite && result.max_abs <= checkpoint.tolerance;
        std::cout << "  " << checkpoint.name << ": "
                  << (checkpoint_pass ? "PASS" : "FAIL")
                  << " max_abs=" << result.max_abs
                  << " mean_abs=" << result.mean_abs
                  << " rmse=" << result.rmse
                  << " max_rel=" << result.max_rel
                  << " max_index=" << result.max_index
                  << " actual=" << result.actual_at_max
                  << " external=" << result.expected_at_max << '\n';
        if (!checkpoint_pass && previous_passed) {
            std::cout << "  FIRST DIVERGENCE: " << checkpoint.name << '\n';
        }
        previous_passed = previous_passed && checkpoint_pass;
        passed = passed && checkpoint_pass;
    }
    return passed;
}

void report_pair(const char* label, const miinfer::Qwen3LayerTrace& actual,
                 const miinfer::Qwen3LayerTrace& expected) {
    std::cout << label << ":\n";
    for (const auto& checkpoint : kCheckpoints) {
        const auto result = metrics(actual.*(checkpoint.field), expected.*(checkpoint.field));
        std::cout << "  " << checkpoint.name << " max_abs=" << result.max_abs
                  << " mean_abs=" << result.mean_abs
                  << " max_rel=" << result.max_rel << '\n';
    }
}

int run(const std::filesystem::path& model_path,
        const std::filesystem::path& trace_path) {
    std::vector<std::vector<float>> external;
    external.reserve(kCheckpoints.size());
    for (const auto& checkpoint : kCheckpoints) {
        external.push_back(read_f32(trace_path / checkpoint.file));
    }
    const auto input = read_f32(trace_path / "layer-input.f32");
    const auto terminal = read_f32(trace_path / "layer-output.f32");
    const auto final_norm_input = read_f32(trace_path / "final-norm-input.f32");
    const auto terminal_boundary = metrics(final_norm_input, terminal);
    const bool terminal_boundary_pass = terminal_boundary.max_abs == 0.0F;
    std::cout << "terminal layer-output -> final-norm-input boundary: "
              << (terminal_boundary_pass ? "PASS" : "FAIL")
              << " max_abs=" << terminal_boundary.max_abs << '\n';

    const auto model = miinfer::Qwen3Model::load(model_path.string());
    const auto plan = miinfer::Qwen3GpuPlan::build(model);
    const auto host = miinfer::execute_qwen3_layer_host_teacher_forced(model, 35, input);
    const auto gpu = miinfer::execute_qwen3_layer_gpu_teacher_forced(plan, 35, input);
    report_pre_ffn_contract(model, input, external);
    report_attention_v_contract(model, input, external, host);
    report_v_precision_contract(model, plan, input, external);
    report_v_attention_q8_boundary(model, plan, input, external);
    report_q8_contract_replay(external[12]);
    report_host_projection_contracts(model, external[12], external, host);

    bool passed = terminal_boundary_pass;
    passed = report("HOST", host, external) && passed;
    passed = report("MI50 GPU", gpu, external) && passed;
    report_pair("GPU vs host", gpu, host);

    // Gate and Up each pass the standalone 0.05 diagnostic bound, but their
    // small independent errors can be amplified by the multiplicative
    // SwiGLU.  Keep the external vectors fixed and substitute one host
    // projection at a time to identify the causal contributor.
    const auto host_gate_external_up = host_swiglu(host.gate, external[14]);
    const auto external_gate_host_up = host_swiglu(external[13], host.up);
    const auto host_gate_host_up = host_swiglu(host.gate, host.up);
    const auto external_gate_external_up = host_swiglu(external[13], external[14]);
    const std::array hybrids{
        std::pair{"external Gate + external Up", &external_gate_external_up},
        std::pair{"host Gate + external Up", &host_gate_external_up},
        std::pair{"external Gate + host Up", &external_gate_host_up},
        std::pair{"host Gate + host Up", &host_gate_host_up},
    };
    const auto host_swiglu_error = metrics(host.swiglu, external[15]);
    std::cout << "host hybrid SwiGLU attribution (layer 35):\n";
    for (const auto& [label, values] : hybrids) {
        const auto result = metrics(*values, external[15]);
        std::cout << "  " << label
                  << ": max_abs=" << result.max_abs
                  << " mean_abs=" << result.mean_abs
                  << " rmse=" << result.rmse
                  << " max_rel=" << result.max_rel
                  << " max_index=" << result.max_index
                  << " actual=" << result.actual_at_max
                  << " external=" << result.expected_at_max << '\n';
    }
    const auto swiglu_index = host_swiglu_error.max_index;
    std::cout << "host Gate/Up errors at host-vs-external SwiGLU worst index "
              << swiglu_index << ":\n"
              << "  Gate: abs=" << std::fabs(host.gate[swiglu_index] - external[13][swiglu_index])
              << " host=" << host.gate[swiglu_index]
              << " external=" << external[13][swiglu_index] << '\n'
              << "  Up: abs=" << std::fabs(host.up[swiglu_index] - external[14][swiglu_index])
              << " host=" << host.up[swiglu_index]
              << " external=" << external[14][swiglu_index] << '\n';

    // Hybrid injections make the terminal-layer diagnosis causal.  The
    // inputs are independent external checkpoints; only the named GPU stage
    // is allowed to compute a new value.
    using Precision = miinfer::Qwen3ProjectionPrecision;
    const auto external_gate_up_swiglu = miinfer::execute_qwen3_ffn_gpu_probe(
        plan, 35, external[11], external[12], external[13], external[14], {}, {},
        Precision::f32_input_q8_f32_output);
    const auto external_swiglu_down = miinfer::execute_qwen3_ffn_gpu_probe(
        plan, 35, external[11], external[12], external[13], external[14],
        external[15], {}, Precision::f32_input_q8_f32_output);
    const auto external_residual = miinfer::execute_qwen3_ffn_gpu_probe(
        plan, 35, external[11], external[12], external[13], external[14],
        external[15], external[16], Precision::f32_input_q8_f32_output);
    const auto swiglu_hybrid = metrics(external_gate_up_swiglu.swiglu, external[15]);
    const auto down_hybrid = metrics(external_swiglu_down.ffn_output, external[16]);
    const auto residual_hybrid = metrics(external_residual.layer_output, external[17]);
    std::cout << "hybrid external-injection diagnostics:\n"
              << "  GPU SwiGLU (external gate/up): max_abs=" << swiglu_hybrid.max_abs
              << " mean_abs=" << swiglu_hybrid.mean_abs
              << " max_rel=" << swiglu_hybrid.max_rel << '\n'
              << "  GPU Down (external SwiGLU): max_abs=" << down_hybrid.max_abs
              << " mean_abs=" << down_hybrid.mean_abs
              << " max_rel=" << down_hybrid.max_rel << '\n'
              << "  GPU residual (external Down): max_abs=" << residual_hybrid.max_abs
              << " mean_abs=" << residual_hybrid.mean_abs
              << " max_rel=" << residual_hybrid.max_rel << '\n';
    std::cout << "M4-B19 external layer-35 comparison: "
              << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: miinfer-qwen3-layer35-external-test MODEL LAYER35_TRACE\n";
        return 2;
    }
    try {
        return run(argv[1], argv[2]);
    } catch (const std::exception& error) {
        std::cerr << "M4-B19 external layer-35 test error: " << error.what() << '\n';
        return 1;
    }
}
