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

std::uint16_t float_to_half_bits_rn(float value) noexcept {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000U;
    const std::uint32_t exponent = (bits >> 23) & 0xffU;
    const std::uint32_t mantissa = bits & 0x7fffffU;
    if (exponent == 0xffU) {
        return static_cast<std::uint16_t>(sign | 0x7c00U | (mantissa != 0 ? 0x0200U : 0U));
    }
    const int half_exponent = static_cast<int>(exponent) - 127 + 15;
    if (half_exponent <= 0) {
        if (half_exponent < -10) return static_cast<std::uint16_t>(sign);
        const std::uint32_t significand = mantissa | 0x00800000U;
        const int shift = 14 - half_exponent;
        std::uint32_t rounded = significand >> shift;
        const std::uint32_t remainder = significand & ((1U << shift) - 1U);
        const std::uint32_t halfway = 1U << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (rounded & 1U) != 0)) ++rounded;
        return static_cast<std::uint16_t>(sign | rounded);
    }
    if (half_exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00U);
    std::uint32_t rounded_mantissa = mantissa >> 13;
    const std::uint32_t remainder = mantissa & 0x1fffU;
    if (remainder > 0x1000U || (remainder == 0x1000U && (rounded_mantissa & 1U) != 0)) {
        ++rounded_mantissa;
        if (rounded_mantissa == 0x400U) {
            rounded_mantissa = 0;
            if (half_exponent + 1 >= 31) return static_cast<std::uint16_t>(sign | 0x7c00U);
            return static_cast<std::uint16_t>(sign
                | (static_cast<std::uint32_t>(half_exponent + 1) << 10));
        }
    }
    return static_cast<std::uint16_t>(sign
        | (static_cast<std::uint32_t>(half_exponent) << 10) | rounded_mantissa);
}

std::uint16_t float_to_half_bits_trunc(float value) noexcept {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000U;
    const std::uint32_t exponent = (bits >> 23) & 0xffU;
    const std::uint32_t mantissa = bits & 0x7fffffU;
    if (exponent == 0xffU) return static_cast<std::uint16_t>(sign | 0x7c00U | (mantissa >> 13));
    const int half_exponent = static_cast<int>(exponent) - 127 + 15;
    if (half_exponent <= 0) return static_cast<std::uint16_t>(sign);
    if (half_exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00U);
    return static_cast<std::uint16_t>(sign
        | (static_cast<std::uint32_t>(half_exponent) << 10) | (mantissa >> 13));
}

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

std::vector<HostQ8Block> quantize_q8_1_float(
    std::span<const float> input, bool reference_fp16_rounding) {
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
        // The normal host forward path retains its historical truncating
        // conversion for comparison continuity.  Teacher-forced replay sets
        // this flag to reproduce GGML's round-to-nearest FP16 scale contract.
        destination.d_bits = reference_fp16_rounding ? float_to_half_bits_rn(scale)
                                                     : float_to_half_bits_trunc(scale);
        destination.s_bits = reference_fp16_rounding
            ? float_to_half_bits_rn(static_cast<float>(sum) * scale)
            : float_to_half_bits_trunc(static_cast<float>(sum) * scale);
    }
    return output;
}

std::vector<float> q4_q8_matvec(
    const Qwen3TensorView& tensor,
    std::span<const float> input,
    std::size_t rows,
    std::size_t columns,
    bool reference_fp16_rounding = false) {
    if (tensor.type() != GgufTensorType::q4_0 || columns % kQBlockSize != 0
        || input.size() != columns
        || tensor.bytes() != rows * (columns / kQBlockSize) * kQ4BlockBytes) {
        throw std::runtime_error("invalid Q4_0 projection contract: " + tensor.name());
    }
    const auto q8 = quantize_q8_1_float(input, reference_fp16_rounding);
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
            int block_dot = 0;
            for (int i = 0; i < static_cast<int>(kQBlockSize); ++i) {
                const auto packed = weight.qs[i < 16 ? i : i - 16];
                const int q4 = i < 16 ? packed & 0x0f : (packed >> 4) & 0x0f;
                block_dot += (q4 - 8) * static_cast<int>(activation.qs[i]);
            }
            // Match the pinned reference CPU Q4_0 x Q8_0 path: integer
            // accumulation is completed for the block before the stored
            // FP16 scales are applied.
            sum += static_cast<float>(block_dot) * wd * ad;
        }
        output[row] = sum;
    }
    return output;
}

std::vector<float> q4_f32_matvec(
    const Qwen3TensorView& tensor,
    std::span<const float> input,
    std::size_t rows,
    std::size_t columns) {
    if (tensor.type() != GgufTensorType::q4_0 || columns % kQBlockSize != 0
        || input.size() != columns
        || tensor.bytes() != rows * (columns / kQBlockSize) * kQ4BlockBytes) {
        throw std::runtime_error("invalid Q4_0 reference projection contract: " + tensor.name());
    }
    const auto* weights = reinterpret_cast<const Q4_0HostBlock*>(tensor.data());
    const std::size_t blocks_per_row = columns / kQBlockSize;
    std::vector<float> output(rows, 0.0F);
    for (std::size_t row = 0; row < rows; ++row) {
        float sum = 0.0F;
        for (std::size_t block = 0; block < blocks_per_row; ++block) {
            const auto& weight = weights[row * blocks_per_row + block];
            const float scale = fp16_bits_to_float(weight.d_bits);
            for (std::size_t index = 0; index < kQBlockSize; ++index) {
                const auto packed = weight.qs[index < 16 ? index : index - 16];
                const int nibble = index < 16 ? packed & 0x0f : (packed >> 4) & 0x0f;
                sum += scale * static_cast<float>(nibble - 8)
                       * input[block * kQBlockSize + index];
            }
        }
        output[row] = sum;
    }
    return output;
}

void add_in_place(std::span<float> destination, std::span<const float> source) {
    if (destination.size() != source.size()) throw std::invalid_argument("residual size mismatch");
    for (std::size_t i = 0; i < destination.size(); ++i) destination[i] += source[i];
}

}  // namespace

Qwen3Layer0KvCache::Qwen3Layer0KvCache(
    std::size_t kv_heads, std::size_t head_dim, std::size_t capacity)
    : kv_heads_(kv_heads), head_dim_(head_dim), capacity_(capacity),
      keys_(kv_heads * capacity * head_dim, 0.0F),
      values_(kv_heads * capacity * head_dim, 0.0F) {
    if (kv_heads == 0 || head_dim == 0 || capacity == 0) {
        throw std::invalid_argument("invalid KV-cache dimensions");
    }
}

void Qwen3Layer0KvCache::reset() noexcept {
    length_ = 0;
    std::fill(keys_.begin(), keys_.end(), 0.0F);
    std::fill(values_.begin(), values_.end(), 0.0F);
}

void Qwen3Layer0KvCache::append(
    std::size_t position, std::span<const float> keys, std::span<const float> values) {
    if (position != length_ || position >= capacity_
        || keys.size() != kv_heads_ * head_dim_
        || values.size() != kv_heads_ * head_dim_) {
        throw std::invalid_argument("invalid KV-cache append");
    }
    for (std::size_t head = 0; head < kv_heads_; ++head) {
        const auto destination = (head * capacity_ + position) * head_dim_;
        std::copy_n(keys.data() + head * head_dim_, head_dim_, keys_.data() + destination);
        std::copy_n(values.data() + head * head_dim_, head_dim_, values_.data() + destination);
    }
    ++length_;
}

Qwen3LayerTrace qwen3_layer_host_impl(
    const Qwen3Model& model,
    std::size_t layer_index,
    std::span<const float> input,
    std::size_t position,
    Qwen3Layer0KvCache& cache,
    bool use_exact_reference_projection,
    bool reference_fp16_rounding = false) {
    const auto& config = model.config();
    if (model.layers().empty() || layer_index >= model.layers().size()) {
        throw std::runtime_error("invalid Qwen3 layer index");
    }
    const auto hidden = static_cast<std::size_t>(config.hidden_size);
    const auto intermediate = static_cast<std::size_t>(config.intermediate_size);
    const auto heads = static_cast<std::size_t>(config.attention_heads);
    const auto kv_heads = static_cast<std::size_t>(config.kv_heads);
    const auto head_dim = static_cast<std::size_t>(config.head_dim);
    if (heads * head_dim != hidden || kv_heads * head_dim != 1024 || heads % kv_heads != 0
        || input.size() != hidden
        || cache.kv_heads() != kv_heads || cache.head_dim() != head_dim
        || position != cache.length() || position >= cache.capacity()) {
        throw std::runtime_error("unsupported Qwen3 head geometry");
    }

    Qwen3LayerTrace trace;
    trace.embedding.assign(input.begin(), input.end());
    const auto& layer = model.layers()[layer_index];
    const auto project = [use_exact_reference_projection, reference_fp16_rounding](
        const Qwen3TensorView& tensor, std::span<const float> values,
        std::size_t rows, std::size_t columns) {
        return use_exact_reference_projection
            ? q4_f32_matvec(tensor, values, rows, columns)
            : q4_q8_matvec(tensor, values, rows, columns, reference_fp16_rounding);
    };

    const auto attn_norm_weight = f32_tensor(layer.attention_norm, hidden);
    std::vector<float> attn_rms(hidden);
    rms_normalize_only(trace.embedding, attn_rms, config.rms_epsilon);
    trace.attn_rms = attn_rms;
    trace.attn_norm = multiply_weight(attn_rms, attn_norm_weight);

    trace.q_projection = project(layer.q, trace.attn_norm, hidden, hidden);
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

    trace.v_projection = project(layer.v, trace.attn_norm, kv_heads * head_dim, hidden);
    trace.v_reshape = trace.v_projection;
    trace.k_projection = project(layer.k, trace.attn_norm, kv_heads * head_dim, hidden);
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

    // Cache post-RoPE K and unmodified V before attention.  The explicit
    // layout is [kv_head][position][head_dim], with position as the middle
    // stride and length tracking the valid causal prefix.
    cache.append(position, trace.k_rope, trace.v_view);

    trace.attention_scores.resize(heads * cache.length());
    trace.attention_probabilities.resize(heads * cache.length());
    trace.attention_output.assign(hidden, 0.0F);
    for (std::size_t head = 0; head < heads; ++head) {
        const std::size_t q_base = head * head_dim;
        const std::size_t score_base = head * cache.length();
        for (std::size_t cached_position = 0; cached_position < cache.length(); ++cached_position) {
            const std::size_t cache_offset =
                ((head / (heads / kv_heads)) * cache.capacity() + cached_position) * head_dim;
            float score = 0.0F;
            for (std::size_t i = 0; i < head_dim; ++i) {
                score += trace.q_rope[q_base + i] * cache.keys()[cache_offset + i];
            }
            trace.attention_scores[score_base + cached_position] =
                score / std::sqrt(static_cast<float>(head_dim));
        }
        softmax_reference(
            {trace.attention_scores.data() + score_base, cache.length()},
            {trace.attention_probabilities.data() + score_base, cache.length()});
        for (std::size_t i = 0; i < head_dim; ++i) {
            float value = 0.0F;
            for (std::size_t cached_position = 0; cached_position < cache.length(); ++cached_position) {
                const std::size_t cache_offset =
                    ((head / (heads / kv_heads)) * cache.capacity() + cached_position) * head_dim;
                value += trace.attention_probabilities[score_base + cached_position]
                         * cache.values()[cache_offset + i];
            }
            trace.attention_output[q_base + i] = value;
        }
    }
    const auto attention_projected = project(layer.output, trace.attention_output, hidden, hidden);
    trace.ffn_input = attention_projected;
    add_in_place(trace.ffn_input, trace.embedding);

    const auto ffn_norm_weight = f32_tensor(layer.ffn_norm, hidden);
    trace.ffn_rms.resize(hidden);
    rms_normalize_only(trace.ffn_input, trace.ffn_rms, config.rms_epsilon);
    trace.ffn_norm = multiply_weight(trace.ffn_rms, ffn_norm_weight);
    trace.gate = project(layer.gate, trace.ffn_norm, intermediate, hidden);
    trace.up = project(layer.up, trace.ffn_norm, intermediate, hidden);
    trace.swiglu.resize(intermediate);
    silu_mul_reference(trace.gate, trace.up, trace.swiglu);
    trace.ffn_output = project(layer.down, trace.swiglu, hidden, intermediate);
    trace.layer_output = trace.ffn_input;
    add_in_place(trace.layer_output, trace.ffn_output);
    return trace;
}

Qwen3LayerTrace execute_qwen3_layer0_host(
    const Qwen3Model& model,
    std::uint32_t token,
    std::size_t position) {
    if (position != 0) {
        throw std::invalid_argument("standalone layer-0 execution only supports position zero");
    }
    const auto& config = model.config();
    Qwen3Layer0KvCache cache(config.kv_heads, config.head_dim, 1);
    return execute_qwen3_layer0_host(model, token, position, cache);
}

Qwen3LayerTrace execute_qwen3_layer0_host(
    const Qwen3Model& model,
    std::uint32_t token,
    std::size_t position,
    Qwen3Layer0KvCache& cache) {
    const auto& config = model.config();
    if (token >= config.vocab_size) throw std::invalid_argument("token ID is outside the vocabulary");
    std::vector<float> embedding(config.hidden_size);
    q4_0_embedding_reference(
        {model.token_embeddings().data(), model.token_embeddings().bytes()},
        config.vocab_size, config.hidden_size, token, embedding);
    return qwen3_layer_host_impl(model, 0, embedding, position, cache, false);
}

Qwen3ForwardTrace execute_qwen3_forward_host(
    const Qwen3Model& model,
    std::uint32_t token,
    std::size_t position) {
    const auto& config = model.config();
    if (position != 0) {
        throw std::invalid_argument("full host forward currently supports position zero only");
    }
    if (token >= config.vocab_size) {
        throw std::invalid_argument("token ID is outside the vocabulary");
    }
    if (model.layers().size() != config.layer_count) {
        throw std::runtime_error("model layer inventory does not match configuration");
    }

    Qwen3ForwardTrace forward;
    forward.embedding.resize(config.hidden_size);
    q4_0_embedding_reference(
        {model.token_embeddings().data(), model.token_embeddings().bytes()},
        config.vocab_size, config.hidden_size, token, forward.embedding);

    std::vector<Qwen3Layer0KvCache> caches;
    caches.reserve(model.layers().size());
    for (std::size_t layer = 0; layer < model.layers().size(); ++layer) {
        caches.emplace_back(config.kv_heads, config.head_dim, 1);
    }

    std::vector<float> hidden = forward.embedding;
    forward.layer_outputs.reserve(model.layers().size());
    for (std::size_t layer = 0; layer < model.layers().size(); ++layer) {
        // The pinned CPU reference selects the Q4_0 x Q8_0 quantized dot
        // path for these projections.  q4_q8_matvec mirrors that arithmetic
        // (including integer block accumulation) while retaining the same
        // Q8 block values used by the production GPU contract.
        const auto layer_trace = qwen3_layer_host_impl(
            model, layer, hidden, position, caches[layer], false);
        hidden = layer_trace.layer_output;
        forward.layer_outputs.push_back(hidden);
    }

    const auto final_norm_weight = f32_tensor(model.final_norm(), config.hidden_size);
    forward.final_norm.resize(config.hidden_size);
    rms_norm_reference(hidden, final_norm_weight, forward.final_norm, config.rms_epsilon);
    forward.logits.resize(config.vocab_size);
    q6_k_q8_k_gemv_reference(
        {model.output().data(), model.output().bytes()},
        forward.final_norm,
        forward.logits,
        config.vocab_size,
        config.hidden_size);
    return forward;
}

Qwen3LayerTrace execute_qwen3_layer_host_teacher_forced(
    const Qwen3Model& model,
    std::size_t layer_index,
    std::span<const float> input,
    std::size_t position) {
    const auto& config = model.config();
    if (position != 0) {
        throw std::invalid_argument("teacher-forced replay currently supports position zero only");
    }
    if (input.size() != config.hidden_size || layer_index >= model.layers().size()) {
        throw std::invalid_argument("invalid teacher-forced host layer input/index");
    }
    Qwen3Layer0KvCache cache(config.kv_heads, config.head_dim, 1);
    return qwen3_layer_host_impl(model, layer_index, input, position, cache, false, true);
}

}  // namespace miinfer
