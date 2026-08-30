#include "miinfer/qwen3_layer.hpp"

#include "miinfer/qwen3_primitives.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>

namespace miinfer {

namespace {

constexpr std::size_t kQ4BlockBytes = sizeof(Q4_0HostBlock);
constexpr std::size_t kQBlockSize = 32;

struct HostQ8Block {
    std::uint16_t d_bits = 0;
    std::uint16_t s_bits = 0;
    std::int8_t qs[32]{};
};

static_assert(sizeof(HostQ8Block) == 36);

std::span<const float> f32_tensor(const Qwen3TensorView& tensor, std::size_t elements) {
    if (tensor.type() != GgufTensorType::f32 || tensor.bytes() != elements * sizeof(float)) {
        throw std::runtime_error("unexpected F32 tensor size: " + tensor.name());
    }
    return {reinterpret_cast<const float*>(tensor.data()), elements};
}

std::vector<float> multiply_weight(
    std::span<const float> input,
    std::span<const float> weight) {
    if (input.size() != weight.size()) throw std::invalid_argument("elementwise size mismatch");
    std::vector<float> output(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) output[i] = input[i] * weight[i];
    return output;
}

void rms_normalize_only(
    std::span<const float> input,
    std::span<float> output,
    float epsilon) {
    if (input.empty() || input.size() != output.size()) {
        throw std::invalid_argument("RMSNorm size mismatch");
    }
    double sum = 0.0;
    for (const float value : input) sum += static_cast<double>(value) * value;
    const float inverse_rms = 1.0F / std::sqrt(static_cast<float>(sum / input.size()) + epsilon);
    for (std::size_t i = 0; i < input.size(); ++i) output[i] = input[i] * inverse_rms;
}

std::vector<HostQ8Block> quantize_q8_1_float(std::span<const float> input) {
    if (input.empty() || input.size() % kQBlockSize != 0) {
        throw std::invalid_argument("Q8_1 input length must be a non-zero multiple of 32");
    }
    std::vector<HostQ8Block> output(input.size() / kQBlockSize);
    for (std::size_t block_index = 0; block_index < output.size(); ++block_index) {
        const std::size_t base = block_index * kQBlockSize;
        float amax = 0.0F;
        for (int i = 0; i < static_cast<int>(kQBlockSize); ++i) amax = std::max(amax, std::fabs(input[base + i]));
        const float scale = amax / 127.0F;
        const float inverse = scale == 0.0F ? 0.0F : 1.0F / scale;
        auto& destination = output[block_index];
        destination.d_bits = 0;
        destination.s_bits = 0;
        int sum = 0;
        for (int i = 0; i < static_cast<int>(kQBlockSize); ++i) {
            const int q = std::clamp(static_cast<int>(std::round(input[base + i] * inverse)), -127, 127);
            destination.qs[i] = static_cast<std::int8_t>(q);
            sum += q;
        }
        // The canonical Q8_1 metadata is stored as FP16.  Reuse the public
        // host conversion only after quantizing in FP32, matching the model
        // input contract without pulling HIP headers into host builds.
        auto float_to_half_bits = [](float value) {
            // This helper is intentionally local to the correctness oracle;
            // the model's stored FP16 conversion is exposed in the primitive
            // API in the opposite direction, so use the portable compiler
            // conversion through a tiny IEEE-754 implementation here.
            std::uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            const std::uint32_t sign = (bits >> 16) & 0x8000U;
            const std::uint32_t exponent = (bits >> 23) & 0xffU;
            const std::uint32_t mantissa = bits & 0x7fffffU;
            if (exponent == 0xffU) return static_cast<std::uint16_t>(sign | 0x7c00U | (mantissa >> 13));
            const int half_exponent = static_cast<int>(exponent) - 127 + 15;
            if (half_exponent <= 0) return static_cast<std::uint16_t>(sign);
            if (half_exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00U);
            return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(half_exponent) << 10)
                                               | (mantissa >> 13));
        };
        destination.d_bits = float_to_half_bits(scale);
        destination.s_bits = float_to_half_bits(static_cast<float>(sum) * scale);
    }
    return output;
}

std::vector<float> q4_q8_matvec(
    const Qwen3TensorView& tensor,
    std::span<const float> input,
    std::size_t rows,
    std::size_t columns) {
    if (tensor.type() != GgufTensorType::q4_0 || columns % kQBlockSize != 0
        || input.size() != columns
        || tensor.bytes() != rows * (columns / kQBlockSize) * kQ4BlockBytes) {
        throw std::runtime_error("invalid Q4_0 projection contract: " + tensor.name());
    }
    const auto q8 = quantize_q8_1_float(input);
    const auto* weights = reinterpret_cast<const Q4_0HostBlock*>(tensor.data());
    const std::size_t blocks_per_row = columns / kQBlockSize;
    std::vector<float> output(rows, 0.0F);
    for (std::size_t row = 0; row < rows; ++row) {
        float sum = 0.0F;
        for (std::size_t block = 0; block < blocks_per_row; ++block) {
            const auto& weight = weights[row * blocks_per_row + block];
            const auto& activation = q8[block];
            const float wd = fp16_bits_to_float(weight.d_bits);
            const float ad = fp16_bits_to_float(activation.d_bits);
            for (int i = 0; i < static_cast<int>(kQBlockSize); ++i) {
                const auto packed = weight.qs[i < 16 ? i : i - 16];
                const int q4 = i < 16 ? packed & 0x0f : (packed >> 4) & 0x0f;
                sum += static_cast<float>(q4 - 8) * wd
                       * static_cast<float>(activation.qs[i]) * ad;
            }
        }
        output[row] = sum;
    }
    return output;
}

std::vector<float> repeat_kv_for_gqa(std::span<const float> values, std::size_t heads_per_kv) {
    constexpr std::size_t kHeadDim = 128;
    if (values.size() % kHeadDim != 0 || heads_per_kv == 0) throw std::invalid_argument("invalid GQA input");
    const std::size_t kv_heads = values.size() / kHeadDim;
    std::vector<float> output(values.size() * heads_per_kv);
    for (std::size_t kv = 0; kv < kv_heads; ++kv) {
        for (std::size_t group = 0; group < heads_per_kv; ++group) {
            std::copy_n(values.data() + kv * kHeadDim, kHeadDim,
                        output.data() + (kv * heads_per_kv + group) * kHeadDim);
        }
    }
    return output;
}

void add_in_place(std::span<float> destination, std::span<const float> source) {
    if (destination.size() != source.size()) throw std::invalid_argument("residual size mismatch");
    for (std::size_t i = 0; i < destination.size(); ++i) destination[i] += source[i];
}

}  // namespace

Qwen3LayerTrace execute_qwen3_layer0_host(
    const Qwen3Model& model,
    std::uint32_t token,
    std::size_t position) {
    if (position != 0) {
        throw std::invalid_argument("layer-0 host fixture currently supports position zero only");
    }
    const auto& config = model.config();
    if (token >= config.vocab_size) throw std::invalid_argument("token ID is outside the vocabulary");
    if (model.layers().empty()) throw std::runtime_error("model has no layers");
    const auto& layer = model.layers().front();
    const auto hidden = static_cast<std::size_t>(config.hidden_size);
    const auto intermediate = static_cast<std::size_t>(config.intermediate_size);
    const auto heads = static_cast<std::size_t>(config.attention_heads);
    const auto kv_heads = static_cast<std::size_t>(config.kv_heads);
    const auto head_dim = static_cast<std::size_t>(config.head_dim);
    if (heads * head_dim != hidden || kv_heads * head_dim != 1024 || heads % kv_heads != 0) {
        throw std::runtime_error("unsupported Qwen3 head geometry");
    }

    Qwen3LayerTrace trace;
    trace.embedding.resize(hidden);
    q4_0_embedding_reference(
        {model.token_embeddings().data(), model.token_embeddings().bytes()},
        config.vocab_size, hidden, token, trace.embedding);

    const auto attn_norm_weight = f32_tensor(layer.attention_norm, hidden);
    std::vector<float> attn_rms(hidden);
    rms_normalize_only(trace.embedding, attn_rms, config.rms_epsilon);
    trace.attn_rms = attn_rms;
    trace.attn_norm = multiply_weight(attn_rms, attn_norm_weight);

    trace.q_projection = q4_q8_matvec(layer.q, trace.attn_norm, hidden, hidden);
    trace.q_reshape = trace.q_projection;
    std::vector<float> q_rms(trace.q_projection.size());
    const auto q_norm_weight = f32_tensor(layer.q_norm, head_dim);
    for (std::size_t head = 0; head < heads; ++head) {
        rms_normalize_only(
            {trace.q_projection.data() + head * head_dim, head_dim},
            {q_rms.data() + head * head_dim, head_dim},
            config.rms_epsilon);
    }
    trace.q_rms = q_rms;
    trace.q_normed.resize(hidden);
    for (std::size_t head = 0; head < heads; ++head) {
        const auto normalized = multiply_weight(
            {q_rms.data() + head * head_dim, head_dim}, q_norm_weight);
        std::copy(normalized.begin(), normalized.end(), trace.q_normed.begin() + head * head_dim);
    }
    trace.q_rope.resize(hidden);
    rope_qwen3_reference(trace.q_normed, trace.q_rope, heads, head_dim, position, config.rope_theta);

    trace.v_projection = q4_q8_matvec(layer.v, trace.attn_norm, kv_heads * head_dim, hidden);
    trace.v_reshape = trace.v_projection;
    trace.k_projection = q4_q8_matvec(layer.k, trace.attn_norm, kv_heads * head_dim, hidden);
    trace.k_reshape = trace.k_projection;
    std::vector<float> k_rms(trace.k_projection.size());
    const auto k_norm_weight = f32_tensor(layer.k_norm, head_dim);
    for (std::size_t head = 0; head < kv_heads; ++head) {
        rms_normalize_only(
            {trace.k_projection.data() + head * head_dim, head_dim},
            {k_rms.data() + head * head_dim, head_dim},
            config.rms_epsilon);
    }
    trace.k_rms = k_rms;
    trace.k_normed.resize(kv_heads * head_dim);
    for (std::size_t head = 0; head < kv_heads; ++head) {
        const auto normalized = multiply_weight(
            {k_rms.data() + head * head_dim, head_dim}, k_norm_weight);
        std::copy(normalized.begin(), normalized.end(), trace.k_normed.begin() + head * head_dim);
    }
    trace.k_rope.resize(kv_heads * head_dim);
    rope_qwen3_reference(trace.k_normed, trace.k_rope, kv_heads, head_dim, position, config.rope_theta);
    trace.k_view = trace.k_rope;
    trace.v_view = trace.v_reshape;
    trace.q_view = trace.q_rope;
    trace.q_permuted = trace.q_rope;

    // Position zero with an empty cache has one attention key/value.  GQA
    // therefore expands each KV head across its four query heads; attention
    // probabilities are exactly one and attention output is the expanded V.
    trace.attention_output = repeat_kv_for_gqa(trace.v_view, heads / kv_heads);
    const auto attention_projected = q4_q8_matvec(layer.output, trace.attention_output, hidden, hidden);
    trace.ffn_input = attention_projected;
    add_in_place(trace.ffn_input, trace.embedding);

    const auto ffn_norm_weight = f32_tensor(layer.ffn_norm, hidden);
    trace.ffn_rms.resize(hidden);
    rms_normalize_only(trace.ffn_input, trace.ffn_rms, config.rms_epsilon);
    trace.ffn_norm = multiply_weight(trace.ffn_rms, ffn_norm_weight);
    trace.gate = q4_q8_matvec(layer.gate, trace.ffn_norm, intermediate, hidden);
    trace.up = q4_q8_matvec(layer.up, trace.ffn_norm, intermediate, hidden);
    trace.swiglu.resize(intermediate);
    silu_mul_reference(trace.gate, trace.up, trace.swiglu);
    trace.ffn_output = q4_q8_matvec(layer.down, trace.swiglu, hidden, intermediate);
    trace.layer_output = trace.ffn_input;
    add_in_place(trace.layer_output, trace.ffn_output);
    return trace;
}

}  // namespace miinfer
