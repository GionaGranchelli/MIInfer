#include "miinfer/gguf.hpp"
#include "miinfer/qwen3_primitives.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using miinfer::GgufFile;
using miinfer::GgufTensor;
using miinfer::GgufTensorType;

constexpr std::size_t kHidden = 5120;
constexpr std::size_t kInner = 6144;
constexpr std::size_t kFfnInner = 17408;
constexpr std::size_t kState = 128;
constexpr std::size_t kKHeads = 16;
constexpr std::size_t kVHeads = 48;
constexpr std::size_t kChannels = 10240;
constexpr float kEpsilon = 1.0e-6F;

struct Q4K {
    std::uint16_t d;
    std::uint16_t dmin;
    std::uint8_t scales[12];
    std::uint8_t qs[128];
};

struct Q5K {
    std::uint16_t d;
    std::uint16_t dmin;
    std::uint8_t scales[12];
    std::uint8_t qh[32];
    std::uint8_t qs[128];
};

struct Q6K {
    std::uint8_t ql[128];
    std::uint8_t qh[64];
    std::int8_t scales[16];
    std::uint16_t d;
};

struct Q8K {
    float d;
    std::int8_t qs[256];
    std::int16_t bsums[16];
};

static_assert(sizeof(Q4K) == 144);
static_assert(sizeof(Q5K) == 176);
static_assert(sizeof(Q6K) == 210);
static_assert(sizeof(Q8K) == 292);

struct Metrics {
    float max_abs = 0.0F;
    float rmse = 0.0F;
    std::size_t index = 0;
};

const GgufTensor& tensor(const GgufFile& model, std::string_view name) {
    for (const auto& candidate : model.tensors()) {
        if (candidate.name == name) return candidate;
    }
    throw std::runtime_error("missing tensor: " + std::string(name));
}

std::vector<float> read_f32(const std::filesystem::path& path, std::size_t count) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open fixture: " + path.string());
    std::vector<float> result(count);
    input.read(reinterpret_cast<char*>(result.data()),
               static_cast<std::streamsize>(count * sizeof(float)));
    if (input.gcount() != static_cast<std::streamsize>(count * sizeof(float))) {
        throw std::runtime_error("short fixture: " + path.string());
    }
    return result;
}

std::vector<std::uint32_t> read_tokens(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open token fixture: " + path.string());
    std::vector<std::uint32_t> result;
    std::uint32_t token = 0;
    while (input >> token) result.push_back(token);
    return result;
}

std::filesystem::path checkpoint(const std::filesystem::path& fixture, std::size_t position,
                                 std::string_view name) {
    const auto prefix = std::to_string(position) + "-" + std::string(name) + "-";
    for (const auto& entry : std::filesystem::directory_iterator(fixture / "tensors")) {
        if (entry.path().filename().string().starts_with(prefix)) return entry.path();
    }
    throw std::runtime_error("missing checkpoint: " + prefix);
}

Metrics compare(std::span<const float> actual, std::span<const float> expected) {
    if (actual.size() != expected.size() || actual.empty()) {
        throw std::runtime_error("comparison size mismatch");
    }
    double squared = 0.0;
    Metrics result;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
            throw std::runtime_error("non-finite comparison value");
        }
        const float delta = std::fabs(actual[i] - expected[i]);
        if (delta > result.max_abs) {
            result.max_abs = delta;
            result.index = i;
        }
        squared += static_cast<double>(delta) * delta;
    }
    result.rmse = static_cast<float>(std::sqrt(squared / actual.size()));
    return result;
}

void require_match(std::string_view label, const Metrics& error, float tolerance) {
    if (error.max_abs > tolerance) {
        throw std::runtime_error(std::string(label) + " exceeds tolerance: "
                                 + std::to_string(error.max_abs));
    }
}

float half(std::uint16_t bits) noexcept {
    return miinfer::fp16_bits_to_float(bits);
}

void scale_min(const std::uint8_t* q, std::size_t index, std::uint8_t& scale,
               std::uint8_t& minimum) {
    if (index < 4) {
        scale = q[index] & 63U;
        minimum = q[index + 4] & 63U;
    } else {
        scale = static_cast<std::uint8_t>((q[index + 4] & 0x0fU)
                                           | ((q[index - 4] >> 6U) << 4U));
        minimum = static_cast<std::uint8_t>((q[index + 4] >> 4U)
                                             | ((q[index] >> 6U) << 4U));
    }
}

void dequant_q4(const Q4K& block, std::span<float, 256> output) {
    const float d = half(block.d);
    const float minimum = half(block.dmin);
    std::size_t q_offset = 0;
    for (std::size_t group = 0; group < 8; group += 2) {
        std::uint8_t s0 = 0, m0 = 0, s1 = 0, m1 = 0;
        scale_min(block.scales, group, s0, m0);
        scale_min(block.scales, group + 1, s1, m1);
        for (std::size_t i = 0; i < 32; ++i) {
            output[group * 32 + i] = d * static_cast<float>(s0)
                * static_cast<float>(block.qs[q_offset + i] & 0x0fU)
                - minimum * static_cast<float>(m0);
            output[group * 32 + 32 + i] = d * static_cast<float>(s1)
                * static_cast<float>(block.qs[q_offset + i] >> 4U)
                - minimum * static_cast<float>(m1);
        }
        q_offset += 32;
    }
}

void dequant_q5(const Q5K& block, std::span<float, 256> output) {
    const float d = half(block.d);
    const float minimum = half(block.dmin);
    std::size_t q_offset = 0;
    std::uint8_t high0 = 1;
    std::uint8_t high1 = 2;
    for (std::size_t group = 0; group < 8; group += 2) {
        std::uint8_t s0 = 0, m0 = 0, s1 = 0, m1 = 0;
        scale_min(block.scales, group, s0, m0);
        scale_min(block.scales, group + 1, s1, m1);
        for (std::size_t i = 0; i < 32; ++i) {
            output[group * 32 + i] = d * static_cast<float>(s0)
                * static_cast<float>((block.qs[q_offset + i] & 0x0fU)
                    + ((block.qh[i] & high0) != 0 ? 16 : 0))
                - minimum * static_cast<float>(m0);
            output[group * 32 + 32 + i] = d * static_cast<float>(s1)
                * static_cast<float>((block.qs[q_offset + i] >> 4U)
                    + ((block.qh[i] & high1) != 0 ? 16 : 0))
                - minimum * static_cast<float>(m1);
        }
        q_offset += 32;
        high0 = static_cast<std::uint8_t>(high0 << 2U);
        high1 = static_cast<std::uint8_t>(high1 << 2U);
    }
}

void dequant_q6(const Q6K& block, std::span<float, 256> output) {
    const float d = half(block.d);
    for (std::size_t base = 0; base < 256; base += 128) {
        const auto* ql = block.ql + base / 2;
        const auto* qh = block.qh + base / 4;
        const auto* scales = block.scales + base / 16;
        for (std::size_t i = 0; i < 32; ++i) {
            const std::size_t scale = i / 16;
            const auto q0 = static_cast<std::int8_t>((ql[i] & 0x0fU)
                | (((qh[i] >> 0U) & 3U) << 4U)) - 32;
            const auto q1 = static_cast<std::int8_t>((ql[i + 32] & 0x0fU)
                | (((qh[i] >> 2U) & 3U) << 4U)) - 32;
            const auto q2 = static_cast<std::int8_t>((ql[i] >> 4U)
                | (((qh[i] >> 4U) & 3U) << 4U)) - 32;
            const auto q3 = static_cast<std::int8_t>((ql[i + 32] >> 4U)
                | (((qh[i] >> 6U) & 3U) << 4U)) - 32;
            output[base + i] = d * scales[scale] * q0;
            output[base + i + 32] = d * scales[scale + 2] * q1;
            output[base + i + 64] = d * scales[scale + 4] * q2;
            output[base + i + 96] = d * scales[scale + 6] * q3;
        }
    }
}

std::vector<Q8K> quantize_q8(std::span<const float> input) {
    if (input.empty() || input.size() % 256 != 0) throw std::runtime_error("invalid Q8_K input");
    std::vector<Q8K> result(input.size() / 256);
    for (std::size_t block = 0; block < result.size(); ++block) {
        auto& q = result[block];
        const auto source = input.subspan(block * 256, 256);
        float maximum = 0.0F;
        for (const float value : source) maximum = std::max(maximum, std::fabs(value));
        q.d = maximum == 0.0F ? 0.0F : maximum / 127.0F;
        for (std::size_t i = 0; i < 256; ++i) {
            const float scaled = q.d == 0.0F ? 0.0F : source[i] / q.d;
            q.qs[i] = static_cast<std::int8_t>(std::clamp(
                static_cast<int>(std::nearbyint(scaled)), -127, 127));
        }
        for (std::size_t group = 0; group < 16; ++group) {
            for (std::size_t i = 0; i < 16; ++i) q.bsums[group] += q.qs[group * 16 + i];
        }
    }
    return result;
}

std::vector<float> gemv(const GgufTensor& weight, std::span<const float> input,
                        std::size_t rows, std::size_t columns) {
    if (weight.dimensions != std::vector<std::uint64_t>{columns, rows}
        || input.size() != columns) {
        throw std::runtime_error("unexpected GEMV shape for " + weight.name);
    }
    const std::size_t blocks = columns / 256;
    const auto input_q8 = quantize_q8(input);
    std::vector<float> result(rows);
    std::array<float, 256> values{};
    for (std::size_t row = 0; row < rows; ++row) {
        float sum = 0.0F;
        for (std::size_t block = 0; block < blocks; ++block) {
            const auto* raw = weight.data + (row * blocks + block) *
                (weight.type == GgufTensorType::q4_k ? sizeof(Q4K) :
                 weight.type == GgufTensorType::q5_k ? sizeof(Q5K) : sizeof(Q6K));
            if (weight.type == GgufTensorType::q4_k) {
                const auto& q4 = *reinterpret_cast<const Q4K*>(raw);
                dequant_q4(q4, values);
                for (std::size_t i = 0; i < values.size(); ++i) {
                    sum += values[i] * static_cast<float>(input_q8[block].qs[i]) * input_q8[block].d;
                }
                continue;
            } else if (weight.type == GgufTensorType::q5_k) {
                const auto& q5 = *reinterpret_cast<const Q5K*>(raw);
                dequant_q5(q5, values);
                for (std::size_t i = 0; i < values.size(); ++i) {
                    sum += values[i] * static_cast<float>(input_q8[block].qs[i]) * input_q8[block].d;
                }
                continue;
            } else if (weight.type == GgufTensorType::q6_k) {
                const auto& q6 = *reinterpret_cast<const Q6K*>(raw);
                dequant_q6(q6, values);
                for (std::size_t i = 0; i < values.size(); ++i) {
                    sum += values[i] * static_cast<float>(input_q8[block].qs[i]) * input_q8[block].d;
                }
                continue;
            } else {
                throw std::runtime_error("unsupported quantized GEMV: " + weight.name);
            }
            for (std::size_t i = 0; i < values.size(); ++i) {
                sum += values[i] * input[block * values.size() + i];
            }
        }
        result[row] = sum;
    }
    return result;
}

std::vector<float> f32_gemv(const GgufTensor& weight, std::span<const float> input,
                            std::size_t rows, std::size_t columns) {
    if (weight.type != GgufTensorType::f32
        || weight.dimensions != std::vector<std::uint64_t>{columns, rows}) {
        throw std::runtime_error("unexpected F32 GEMV shape for " + weight.name);
    }
    const auto* source = reinterpret_cast<const float*>(weight.data);
    std::vector<float> result(rows);
    for (std::size_t row = 0; row < rows; ++row) {
        float sum = 0.0F;
        for (std::size_t i = 0; i < columns; ++i) sum += source[row * columns + i] * input[i];
        result[row] = sum;
    }
    return result;
}

std::vector<float> f32_values(const GgufTensor& weight, std::size_t count) {
    if (weight.type != GgufTensorType::f32 || weight.byte_size != count * sizeof(float)) {
        throw std::runtime_error("unexpected F32 vector: " + weight.name);
    }
    return {reinterpret_cast<const float*>(weight.data),
            reinterpret_cast<const float*>(weight.data) + count};
}

void l2_norm(std::span<float> values, std::size_t count) {
    for (std::size_t base = 0; base < values.size(); base += count) {
        double sum = 0.0;
        for (std::size_t i = 0; i < count; ++i) sum += static_cast<double>(values[base + i]) * values[base + i];
        const float scale = 1.0F / std::max(std::sqrt(static_cast<float>(sum)), kEpsilon);
        for (std::size_t i = 0; i < count; ++i) values[base + i] *= scale;
    }
}

void silu(std::span<float> values) {
    for (float& value : values) value = value / (1.0F + std::exp(-value));
}

void rms_rows(std::span<const float> input, std::span<const float> weights,
              std::span<float> output, std::size_t width) {
    if (input.size() != output.size() || weights.size() != width || input.size() % width != 0) {
        throw std::runtime_error("invalid RMS row shape");
    }
    for (std::size_t base = 0; base < input.size(); base += width) {
        double sum = 0.0;
        for (std::size_t i = 0; i < width; ++i) sum += static_cast<double>(input[base + i]) * input[base + i];
        const float inverse = 1.0F / std::sqrt(static_cast<float>(sum / width) + kEpsilon);
        for (std::size_t i = 0; i < width; ++i) output[base + i] = input[base + i] * inverse * weights[i];
    }
}

std::vector<float> embedding(const GgufTensor& weight, std::size_t token) {
    if (weight.type != GgufTensorType::q4_k
        || weight.dimensions != std::vector<std::uint64_t>{kHidden, 248320}) {
        throw std::runtime_error("unexpected embedding tensor");
    }
    const auto* row = reinterpret_cast<const Q4K*>(weight.data) + token * (kHidden / 256);
    std::vector<float> result(kHidden);
    std::array<float, 256> values{};
    for (std::size_t block = 0; block < kHidden / 256; ++block) {
        dequant_q4(row[block], values);
        std::copy(values.begin(), values.end(), result.begin() + block * 256);
    }
    return result;
}

std::vector<float> conv_output(std::span<const float> current,
                               const std::vector<std::array<float, kChannels>>& history,
                               const GgufTensor& kernel) {
    if (current.size() != kChannels || kernel.type != GgufTensorType::f32
        || kernel.dimensions != std::vector<std::uint64_t>{4, kChannels}) {
        throw std::runtime_error("unexpected convolution shape");
    }
    const auto* c = reinterpret_cast<const float*>(kernel.data);
    std::vector<float> result(kChannels);
    for (std::size_t channel = 0; channel < kChannels; ++channel) {
        result[channel] = current[channel] * c[channel * 4 + 3];
        for (std::size_t i = 1; i < 4; ++i) {
            if (history.size() >= i) result[channel] += history[history.size() - i][channel]
                * c[channel * 4 + (3 - i)];
        }
    }
    return result;
}

std::vector<float> recurrent(std::span<const float> q, std::span<const float> k,
                             std::span<const float> v, std::span<const float> gate,
                             std::span<const float> beta, std::span<const float> state_in,
                             std::vector<float>& state_out) {
    if (q.size() != kKHeads * kState || k.size() != kKHeads * kState
        || v.size() != kVHeads * kState || gate.size() != kVHeads
        || beta.size() != kVHeads || state_in.size() != kVHeads * kState * kState) {
        throw std::runtime_error("invalid recurrent shapes");
    }
    state_out.assign(state_in.begin(), state_in.end());
    std::vector<float> output(kVHeads * kState);
    for (std::size_t h = 0; h < kVHeads; ++h) {
        const std::size_t kh = h % kKHeads;
        const float decay = std::exp(gate[h]);
        const float beta_value = beta[h];
        float* state = state_out.data() + h * kState * kState;
        for (std::size_t j = 0; j < kState; ++j) {
            for (std::size_t i = 0; i < kState; ++i) state[j * kState + i] *= decay;
        }
        std::array<float, kState> delta{};
        for (std::size_t j = 0; j < kState; ++j) {
            float dot = 0.0F;
            for (std::size_t i = 0; i < kState; ++i) dot += state[j * kState + i] * k[kh * kState + i];
            delta[j] = (v[h * kState + j] - dot) * beta_value;
        }
        for (std::size_t j = 0; j < kState; ++j) {
            for (std::size_t i = 0; i < kState; ++i) state[j * kState + i] += delta[j] * k[kh * kState + i];
        }
        for (std::size_t j = 0; j < kState; ++j) {
            float dot = 0.0F;
            for (std::size_t i = 0; i < kState; ++i) dot += state[j * kState + i] * q[kh * kState + i];
            output[h * kState + j] = dot / std::sqrt(static_cast<float>(kState));
        }
    }
    return output;
}

std::vector<float> layer_path(const GgufFile& model, const std::filesystem::path& fixture,
                              std::size_t position, std::span<const float> input_norm,
                              std::span<const float> state, std::vector<float>& state_out,
                              bool full_tail) {
    const auto& qkv_weight = tensor(model, "blk.0.attn_qkv.weight");
    const auto& gate_weight = tensor(model, "blk.0.attn_gate.weight");
    const auto& beta_weight = tensor(model, "blk.0.ssm_beta.weight");
    const auto& alpha_weight = tensor(model, "blk.0.ssm_alpha.weight");
    const auto& dt_weight = tensor(model, "blk.0.ssm_dt.bias");
    const auto& a_weight = tensor(model, "blk.0.ssm_a");
    const auto& ssm_norm = tensor(model, "blk.0.ssm_norm.weight");
    const auto& ssm_out = tensor(model, "blk.0.ssm_out.weight");

    auto qkv = gemv(qkv_weight, input_norm, kChannels, kHidden);
    auto z = gemv(gate_weight, input_norm, kInner, kHidden);
    auto beta_raw = f32_gemv(beta_weight, input_norm, kVHeads, kHidden);
    auto alpha_raw = f32_gemv(alpha_weight, input_norm, kVHeads, kHidden);
    const auto dt = f32_values(dt_weight, kVHeads);
    const auto a = f32_values(a_weight, kVHeads);
    std::vector<float> beta(kVHeads), gate(kVHeads);
    for (std::size_t h = 0; h < kVHeads; ++h) {
        beta[h] = 1.0F / (1.0F + std::exp(-beta_raw[h]));
        const float x = alpha_raw[h] + dt[h];
        const float softplus = x <= 20.0F ? std::log1p(std::exp(x)) : x;
        gate[h] = softplus * a[h];
    }

    auto raw = read_f32(fixture / "tensors" / (std::to_string(position) + "-conv_output_raw-0-2.f32"), kChannels);
    silu(raw);
    const std::size_t q_size = kKHeads * kState;
    std::vector<float> q(raw.begin(), raw.begin() + q_size);
    std::vector<float> k(raw.begin() + q_size, raw.begin() + 2 * q_size);
    std::vector<float> v(raw.begin() + 2 * q_size, raw.end());
    l2_norm(q, kState);
    l2_norm(k, kState);
    auto output = recurrent(q, k, v, gate, beta, state, state_out);
    const auto expected_output = read_f32(checkpoint(fixture, position, "attn_output-0"), kInner);
    const auto output_error = compare(output, expected_output);
    require_match("recurrent output", output_error, 1.0e-4F);
    std::cout << "position=" << position << " recurrent_output max_abs=" << output_error.max_abs
              << " rmse=" << output_error.rmse << " index=" << output_error.index << '\n';

    std::vector<float> gated(output.size());
    std::vector<float> norm_weights = f32_values(ssm_norm, kState);
    rms_rows(output, norm_weights, gated, kState);
    for (std::size_t i = 0; i < gated.size(); ++i) gated[i] *= z[i] / (1.0F + std::exp(-z[i]));
    const auto expected_final = read_f32(checkpoint(fixture, position, "final_output-0"), kInner);
    const auto final_error = compare(gated, expected_final);
    require_match("gated output", final_error, 1.0e-4F);
    std::cout << "position=" << position << " gated_output max_abs=" << final_error.max_abs
              << " rmse=" << final_error.rmse << " index=" << final_error.index << '\n';

    if (full_tail) {
        auto projected = gemv(ssm_out, gated, kHidden, kInner);
        const auto embed = embedding(tensor(model, "token_embd.weight"), 14556);
        std::vector<float> residual(kHidden);
        for (std::size_t i = 0; i < kHidden; ++i) residual[i] = embed[i] + projected[i];
        const auto expected_residual = read_f32(checkpoint(fixture, 0, "attn_residual-0"), kHidden);
        const auto residual_error = compare(residual, expected_residual);
        require_match("attention residual", residual_error, 1.0e-3F);
        std::cout << "position=0 attn_residual max_abs=" << residual_error.max_abs
                  << " rmse=" << residual_error.rmse << " index=" << residual_error.index << '\n';

        auto post_norm_weights = f32_values(tensor(model, "blk.0.post_attention_norm.weight"), kHidden);
        std::vector<float> post_norm(kHidden);
        rms_rows(residual, post_norm_weights, post_norm, kHidden);
        const auto expected_post_norm = read_f32(checkpoint(fixture, 0, "attn_post_norm-0"), kHidden);
        const auto post_error = compare(post_norm, expected_post_norm);
        require_match("post-attention norm", post_error, 1.0e-3F);
        std::cout << "position=0 post_attention_norm max_abs=" << post_error.max_abs
                  << " rmse=" << post_error.rmse << " index=" << post_error.index << '\n';

        auto ffn_gate = gemv(tensor(model, "blk.0.ffn_gate.weight"), post_norm, kFfnInner, kHidden);
        auto ffn_up = gemv(tensor(model, "blk.0.ffn_up.weight"), post_norm, kFfnInner, kHidden);
        for (std::size_t i = 0; i < kFfnInner; ++i) ffn_gate[i] = ffn_gate[i] / (1.0F + std::exp(-ffn_gate[i])) * ffn_up[i];
        auto ffn = gemv(tensor(model, "blk.0.ffn_down.weight"), ffn_gate, kHidden, kFfnInner);
        const auto expected_ffn = read_f32(checkpoint(fixture, 0, "ffn_out-0"), kHidden);
        const auto ffn_error = compare(ffn, expected_ffn);
        require_match("FFN output", ffn_error, 1.0e-3F);
        std::cout << "position=0 ffn_output max_abs=" << ffn_error.max_abs
                  << " rmse=" << ffn_error.rmse << " index=" << ffn_error.index << '\n';
        for (std::size_t i = 0; i < kHidden; ++i) ffn[i] += residual[i];
        const auto expected_lout = read_f32(checkpoint(fixture, 0, "l_out-0"), kHidden);
        const auto lout_error = compare(ffn, expected_lout);
        require_match("layer output", lout_error, 1.0e-3F);
        std::cout << "position=0 layer_output max_abs=" << lout_error.max_abs
                  << " rmse=" << lout_error.rmse << " index=" << lout_error.index << '\n';
    }
    return output;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: miinfer-m6a3-qwen35-layer MODEL.gguf FIXTURE_DIR\n";
        return 2;
    }
    try {
        const auto model = GgufFile::open(argv[1]);
        const std::filesystem::path fixture = argv[2];
        const auto positions = std::array<std::size_t, 5>{0, 1, 2, 4, 8};
        std::vector<std::array<float, kChannels>> qkv_history;
        std::vector<float> state_out;
        const auto generated = read_tokens(fixture / "generated_tokens.txt");
        if (generated.size() < 8) throw std::runtime_error("fixture has fewer than 8 generated tokens");

        for (std::size_t index = 0; index < positions.size(); ++index) {
            const auto position = positions[index];
            const auto input = read_f32(checkpoint(fixture, position, "model_input_embed"), kHidden);
            const auto norm_expected = read_f32(checkpoint(fixture, position, "attn_norm-0"), kHidden);
            const auto token = index == 0 ? 14556U : static_cast<std::size_t>(generated[position - 1]);
            const auto embed = embedding(tensor(*model, "token_embd.weight"), token);
            const auto embed_error = compare(embed, input);
            require_match("embedding", embed_error, 1.0e-6F);
            std::cout << "position=" << position << " embedding max_abs=" << embed_error.max_abs
                      << " rmse=" << embed_error.rmse << " index=" << embed_error.index << '\n';
            const auto norm_weights = f32_values(tensor(*model, "blk.0.attn_norm.weight"), kHidden);
            std::vector<float> normalized(kHidden);
            miinfer::rms_norm_reference(input, norm_weights, normalized, kEpsilon);
            const auto norm_error = compare(normalized, norm_expected);
            require_match("attention norm", norm_error, 1.0e-6F);
            std::cout << "position=" << position << " attn_norm max_abs=" << norm_error.max_abs
                      << " rmse=" << norm_error.rmse << " index=" << norm_error.index << '\n';

            if (position <= 2) {
                auto qkv = gemv(tensor(*model, "blk.0.attn_qkv.weight"), norm_expected, kChannels, kHidden);
                const auto raw = conv_output(qkv, qkv_history, tensor(*model, "blk.0.ssm_conv1d.weight"));
                const auto expected_raw = read_f32(checkpoint(fixture, position, "conv_output_raw-0"), kChannels);
                const auto conv_error = compare(raw, expected_raw);
                require_match("convolution output", conv_error, 1.0e-3F);
                std::cout << "position=" << position << " conv_output max_abs=" << conv_error.max_abs
                          << " rmse=" << conv_error.rmse << " index=" << conv_error.index << '\n';
                const auto expected_qkv = read_f32(
                    checkpoint(fixture, position, "linear_attn_qkv_mixed-0"), kChannels);
                const auto qkv_error = compare(qkv, expected_qkv);
                require_match("QKV projection", qkv_error, 1.0e-3F);
                std::cout << "position=" << position << " qkv_projection max_abs=" << qkv_error.max_abs
                          << " rmse=" << qkv_error.rmse << " index=" << qkv_error.index << '\n';
                std::array<float, kChannels> saved{};
                std::copy(qkv.begin(), qkv.end(), saved.begin());
                qkv_history.push_back(saved);
            }

            const auto state_expected = read_f32(checkpoint(fixture, position, "state_predelta-0"), kVHeads * kState * kState);
            const bool full_tail = position == 0;
            layer_path(*model, fixture, position, norm_expected, state_expected, state_out, full_tail);
            if (position < 2) {
                const auto next_state = read_f32(checkpoint(fixture, position + 1, "state_predelta-0"), kVHeads * kState * kState);
                if (!state_out.empty()) {
                    const auto state_error = compare(state_out, next_state);
                    require_match("recurrent state", state_error, 1.0e-4F);
                    std::cout << "position=" << position << " next_state max_abs=" << state_error.max_abs
                              << " rmse=" << state_error.rmse << " index=" << state_error.index << '\n';
                }
            }
        }
        std::cout << "M6-A3 recurrent layer reference harness complete\n";
    } catch (const std::exception& error) {
        std::cerr << "M6-A3 failed: " << error.what() << '\n';
        return 1;
    }
}
