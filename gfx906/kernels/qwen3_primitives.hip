#include "miinfer/qwen3_gpu_primitives.hpp"

#include "miinfer/hip_check.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx906__)
#error "Qwen3 primitives are intentionally gfx906-only"
#endif

namespace miinfer {

namespace {

__device__ __forceinline__ float half_to_float(__half value) {
    return __half2float(value);
}

__device__ __forceinline__ float q4_value(const std::byte* block, int index) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(block);
    const float scale = half_to_float(*reinterpret_cast<const __half*>(bytes));
    const std::uint8_t packed = bytes[2 + (index < 16 ? index : index - 16)];
    const int nibble = index < 16 ? packed & 0x0f : packed >> 4;
    return scale * static_cast<float>(nibble - 8);
}

__device__ __forceinline__ int q6_value(const Q6KDeviceBlock& block, int index) {
    const int group = index / 128;
    const int lane = index % 128;
    const int quarter = lane / 32;
    const int in_quarter = lane % 32;
    const int low_index = group * 64 + in_quarter
                          + (quarter == 1 || quarter == 3 ? 32 : 0);
    const int low = quarter < 2 ? block.ql[low_index] & 0x0f : block.ql[low_index] >> 4;
    const int high = (block.qh[group * 32 + in_quarter] >> (2 * quarter)) & 0x03;
    return (low | (high << 4)) - 32;
}

__device__ __forceinline__ void q4_k_scale_min(
    const Q4KDeviceBlock& block, int group, int& scale, int& minimum) {
    if (group < 4) {
        scale = block.scales[group] & 63;
        minimum = block.scales[group + 4] & 63;
    } else {
        scale = (block.scales[group + 4] & 0x0f)
                | ((block.scales[group - 4] >> 6) << 4);
        minimum = (block.scales[group + 4] >> 4)
                  | ((block.scales[group] >> 6) << 4);
    }
}

__device__ __forceinline__ float q4_k_value(const Q4KDeviceBlock& block, int index) {
    const int group = index / 32;
    const int q_index = (index / 64) * 32 + index % 32;
    const std::uint8_t packed = block.qs[q_index];
    const int quantized = (group & 1) == 0 ? packed & 0x0f : packed >> 4;
    int scale = 0;
    int minimum = 0;
    q4_k_scale_min(block, group, scale, minimum);
    return half_to_float(block.d) * static_cast<float>(scale * quantized)
           - half_to_float(block.dmin) * static_cast<float>(minimum);
}

__device__ __forceinline__ void q5_k_scale_min(
    const Q5KDeviceBlock& block, int group, int& scale, int& minimum) {
    if (group < 4) {
        scale = block.scales[group] & 63;
        minimum = block.scales[group + 4] & 63;
    } else {
        scale = (block.scales[group + 4] & 0x0f)
                | ((block.scales[group - 4] >> 6) << 4);
        minimum = (block.scales[group + 4] >> 4)
                  | ((block.scales[group] >> 6) << 4);
    }
}

__global__ void qwen3_q4_embedding_kernel(
    const std::byte* weights,
    std::uint32_t token,
    std::uint32_t vocabulary,
    std::uint32_t hidden_size,
    float* output) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= hidden_size || token >= vocabulary) return;
    const std::uint32_t blocks_per_row = hidden_size / 32;
    const std::uint32_t block_index = token * blocks_per_row + index / 32;
    output[index] = q4_value(weights + static_cast<std::size_t>(block_index) * 18, index % 32);
}

__global__ void qwen35_q4_k_embedding_kernel(
    const Q4KDeviceBlock* weights,
    std::uint32_t token,
    std::uint32_t vocabulary,
    std::uint32_t hidden_size,
    float* output) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= hidden_size || token >= vocabulary) return;
    const std::uint32_t blocks_per_row = hidden_size / 256;
    const auto& block = weights[static_cast<std::size_t>(token) * blocks_per_row + index / 256];
    output[index] = q4_k_value(block, static_cast<int>(index % 256));
}

__global__ void qwen3_rms_norm_kernel(
    const float* input,
    const float* weights,
    float* output,
    std::uint32_t elements,
    float epsilon) {
    __shared__ float partials[256];
    float sum = 0.0F;
    for (std::uint32_t index = threadIdx.x; index < elements; index += blockDim.x) {
        sum += input[index] * input[index];
    }
    partials[threadIdx.x] = sum;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) partials[threadIdx.x] += partials[threadIdx.x + stride];
        __syncthreads();
    }
    const float inverse_rms = rsqrtf(partials[0] / static_cast<float>(elements) + epsilon);
    for (std::uint32_t index = threadIdx.x; index < elements; index += blockDim.x) {
        output[index] = input[index] * inverse_rms * weights[index];
    }
}

__global__ void qwen3_rms_normalize_kernel(
    const float* input, float* output, std::uint32_t elements, float epsilon) {
    __shared__ float partials[256];
    float sum = 0.0F;
    for (std::uint32_t index = threadIdx.x; index < elements; index += blockDim.x) {
        sum += input[index] * input[index];
    }
    partials[threadIdx.x] = sum;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) partials[threadIdx.x] += partials[threadIdx.x + stride];
        __syncthreads();
    }
    const float inverse_rms = rsqrtf(partials[0] / static_cast<float>(elements) + epsilon);
    for (std::uint32_t index = threadIdx.x; index < elements; index += blockDim.x) {
        output[index] = input[index] * inverse_rms;
    }
}

__global__ void qwen3_elementwise_mul_kernel(
    const float* left, const float* right, float* output, std::uint32_t elements) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = left[index] * right[index];
}

__global__ void qwen3_head_rms_normalize_kernel(
    const float* input, float* output, std::uint32_t heads,
    std::uint32_t head_dim, float epsilon) {
    const std::uint32_t head = blockIdx.x;
    if (head >= heads) return;
    __shared__ float partials[256];
    float sum = 0.0F;
    const std::uint32_t base = head * head_dim;
    for (std::uint32_t index = threadIdx.x; index < head_dim; index += blockDim.x) {
        const float value = input[base + index];
        sum += value * value;
    }
    partials[threadIdx.x] = sum;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) partials[threadIdx.x] += partials[threadIdx.x + stride];
        __syncthreads();
    }
    const float inverse_rms = rsqrtf(partials[0] / static_cast<float>(head_dim) + epsilon);
    for (std::uint32_t index = threadIdx.x; index < head_dim; index += blockDim.x) {
        output[base + index] = input[base + index] * inverse_rms;
    }
}

__global__ void qwen3_head_mul_kernel(
    const float* input, const float* weights, float* output,
    std::uint32_t heads, std::uint32_t head_dim) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= heads * head_dim) return;
    output[index] = input[index] * weights[index % head_dim];
}

__global__ void qwen35_split_q_gate_kernel(
    const float* input, float* query, float* gate,
    std::uint32_t heads, std::uint32_t head_dim) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t elements = heads * head_dim;
    if (index >= elements) return;
    const std::uint32_t head = index / head_dim;
    const std::uint32_t local = index % head_dim;
    const std::uint32_t source = head * (2 * head_dim) + local;
    query[index] = input[source];
    gate[index] = input[source + head_dim];
}

__global__ void qwen35_rope_sections_kernel(
    const float* input, float* output, std::uint32_t heads,
    std::uint32_t head_dim, std::uint32_t position, float theta) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t elements = heads * head_dim;
    if (index >= elements) return;
    const std::uint32_t local = index % head_dim;
    const std::uint32_t base = (index / head_dim) * head_dim;
    if (local >= 64) {
        output[index] = input[index];
        return;
    }
    if (local >= 32) return;

    float theta_t = static_cast<float>(position);
    float theta_h = static_cast<float>(position);
    float theta_w = static_cast<float>(position);
    constexpr float theta_e = 0.0F;
    constexpr int sections[] = {11, 11, 10, 0};
    constexpr int section_count = 32;
    const float theta_scale = powf(theta, -2.0F / 64.0F);
    for (std::uint32_t pair = 0; pair < local; ++pair) {
        const int sector = static_cast<int>(pair % section_count);
        float angle = theta_t;
        if (sector % 3 == 1 && sector < 3 * sections[1]) angle = theta_h;
        else if (sector % 3 == 2 && sector < 3 * sections[2]) angle = theta_w;
        else if (sector % 3 == 0 && sector < 3 * sections[0]) angle = theta_t;
        else angle = theta_e;
        (void) angle;
        theta_t *= theta_scale;
        theta_h *= theta_scale;
        theta_w *= theta_scale;
    }
    const int sector = static_cast<int>(local % section_count);
    float angle = theta_t;
    if (sector % 3 == 1 && sector < 3 * sections[1]) angle = theta_h;
    else if (sector % 3 == 2 && sector < 3 * sections[2]) angle = theta_w;
    else if (sector % 3 == 0 && sector < 3 * sections[0]) angle = theta_t;
    else angle = theta_e;
    const float cosine = cosf(angle);
    const float sine = sinf(angle);
    const float x0 = input[base + local];
    const float x1 = input[base + local + 32];
    output[base + local] = x0 * cosine - x1 * sine;
    output[base + local + 32] = x0 * sine + x1 * cosine;
}

__global__ void qwen35_sigmoid_mul_kernel(
    const float* input, const float* gate, float* output, std::uint32_t elements) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = input[index] / (1.0F + expf(-gate[index]));
}

__global__ void qwen35_deltanet_state_update_kernel(
    const float* query, const float* key, const float* value,
    const float* beta, const float* decay, float* state, float* output,
    std::uint32_t key_heads, std::uint32_t value_heads, std::uint32_t state_size) {
    const std::uint32_t head = blockIdx.x;
    const std::uint32_t row = threadIdx.x;
    if (head >= value_heads || row >= state_size || key_heads == 0) return;

    const std::uint32_t key_head = head % key_heads;
    const std::size_t state_base =
        (static_cast<std::size_t>(head) * state_size + row) * state_size;
    const std::size_t key_base = static_cast<std::size_t>(key_head) * state_size;
    const float beta_value = beta[head];
    const float decay_value = decay[head];
    float* state_row = state + state_base;

    float key_dot = 0.0F;
    for (std::uint32_t column = 0; column < state_size; ++column) {
        state_row[column] *= decay_value;
        key_dot += state_row[column] * key[key_base + column];
    }
    const float delta = (value[static_cast<std::size_t>(head) * state_size + row]
                         - key_dot) * beta_value;
    float query_dot = 0.0F;
    const std::size_t query_base = static_cast<std::size_t>(key_head) * state_size;
    for (std::uint32_t column = 0; column < state_size; ++column) {
        state_row[column] += delta * key[key_base + column];
        query_dot += state_row[column] * query[query_base + column];
    }
    output[static_cast<std::size_t>(head) * state_size + row] =
        query_dot * rsqrtf(static_cast<float>(state_size));
}

__global__ void qwen35_deltanet_state_update_no_decay_store_kernel(
    const float* query, const float* key, const float* value,
    const float* beta, const float* decay, float* state, float* output,
    std::uint32_t key_heads, std::uint32_t value_heads, std::uint32_t state_size) {
    const std::uint32_t head = blockIdx.x;
    const std::uint32_t row = threadIdx.x;
    if (head >= value_heads || row >= state_size || key_heads == 0) return;

    const std::uint32_t key_head = head % key_heads;
    const std::size_t state_base =
        (static_cast<std::size_t>(head) * state_size + row) * state_size;
    const std::size_t key_base = static_cast<std::size_t>(key_head) * state_size;
    const float beta_value = beta[head];
    const float decay_value = decay[head];
    float* state_row = state + state_base;

    float key_dot = 0.0F;
    for (std::uint32_t column = 0; column < state_size; ++column) {
        const float decayed = state_row[column] * decay_value;
        key_dot += decayed * key[key_base + column];
    }
    const float delta = (value[static_cast<std::size_t>(head) * state_size + row]
                         - key_dot) * beta_value;
    float query_dot = 0.0F;
    const std::size_t query_base = static_cast<std::size_t>(key_head) * state_size;
    for (std::uint32_t column = 0; column < state_size; ++column) {
        const float updated = state_row[column] * decay_value
            + delta * key[key_base + column];
        state_row[column] = updated;
        query_dot += updated * query[query_base + column];
    }
    output[static_cast<std::size_t>(head) * state_size + row] =
        query_dot * rsqrtf(static_cast<float>(state_size));
}

__global__ void qwen35_deltanet_state_update_transposed_kernel(
    const float* query, const float* key, const float* value,
    const float* beta, const float* decay, float* state, float* output,
    std::uint32_t key_heads, std::uint32_t value_heads, std::uint32_t state_size) {
    const std::uint32_t head = blockIdx.x;
    const std::uint32_t row = threadIdx.x;
    if (head >= value_heads || row >= state_size || key_heads == 0) return;

    const std::uint32_t key_head = head % key_heads;
    const std::size_t state_base =
        static_cast<std::size_t>(head) * state_size * state_size;
    const std::size_t key_base = static_cast<std::size_t>(key_head) * state_size;
    const float beta_value = beta[head];
    const float decay_value = decay[head];

    float key_dot = 0.0F;
    for (std::uint32_t column = 0; column < state_size; ++column) {
        const std::size_t index = state_base
            + static_cast<std::size_t>(column) * state_size + row;
        const float decayed = state[index] * decay_value;
        state[index] = decayed;
        key_dot += decayed * key[key_base + column];
    }
    const float delta = (value[static_cast<std::size_t>(head) * state_size + row]
                         - key_dot) * beta_value;
    float query_dot = 0.0F;
    for (std::uint32_t column = 0; column < state_size; ++column) {
        const std::size_t index = state_base
            + static_cast<std::size_t>(column) * state_size + row;
        const float updated = state[index] + delta * key[key_base + column];
        state[index] = updated;
        query_dot += updated * query[key_base + column];
    }
    output[static_cast<std::size_t>(head) * state_size + row] =
        query_dot * rsqrtf(static_cast<float>(state_size));
}

__global__ void qwen35_conv_silu_split_kernel(
    const float* current_qkv, const float* conv_weights, float* history,
    float* query, float* key, float* value, std::uint32_t position,
    std::uint32_t history_capacity, std::uint32_t channels,
    std::uint32_t conv_kernel) {
    const std::uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels || history_capacity == 0 || conv_kernel == 0) return;

    float result = current_qkv[channel] *
        conv_weights[static_cast<std::size_t>(channel) * conv_kernel + conv_kernel - 1];
    for (std::uint32_t lag = 1; lag < conv_kernel; ++lag) {
        if (position < lag) continue;
        const std::uint32_t slot = (position - lag) % history_capacity;
        result += history[static_cast<std::size_t>(slot) * channels + channel]
            * conv_weights[static_cast<std::size_t>(channel) * conv_kernel
                           + conv_kernel - 1 - lag];
    }
    history[static_cast<std::size_t>(position % history_capacity) * channels + channel]
        = current_qkv[channel];
    const float activated = result / (1.0F + expf(-result));
    const std::uint32_t q_size = channels / 5;
    const std::uint32_t k_size = q_size;
    if (channel < q_size) query[channel] = activated;
    else if (channel < q_size + k_size) key[channel - q_size] = activated;
    else value[channel - q_size - k_size] = activated;
}

__global__ void qwen35_head_l2_normalize_kernel(
    const float* input, float* output, std::uint32_t heads, std::uint32_t head_dim) {
    const std::uint32_t head = blockIdx.x;
    if (head >= heads) return;
    __shared__ float partials[128];
    float sum = 0.0F;
    const std::size_t base = static_cast<std::size_t>(head) * head_dim;
    for (std::uint32_t index = threadIdx.x; index < head_dim; index += blockDim.x) {
        const float value = input[base + index];
        sum += value * value;
    }
    partials[threadIdx.x] = sum;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) partials[threadIdx.x] += partials[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(fmaxf(partials[0], 1.0e-6F));
    for (std::uint32_t index = threadIdx.x; index < head_dim; index += blockDim.x) {
        output[base + index] = input[base + index] * scale;
    }
}

__global__ void qwen3_rope_kernel(
    const float* input, float* output, std::uint32_t heads,
    std::uint32_t head_dim, std::uint32_t position, float theta) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= heads * head_dim) return;
    const std::uint32_t local_index = index % head_dim;
    const std::uint32_t half_dim = head_dim / 2;
    const std::uint32_t pair = local_index < half_dim ? local_index : local_index - half_dim;
    const std::uint32_t head_base = (index / head_dim) * head_dim;
    const float frequency = powf(theta, -2.0F * static_cast<float>(pair) / static_cast<float>(head_dim));
    const float angle = static_cast<float>(position) * frequency;
    const float c = cosf(angle);
    const float s = sinf(angle);
    const float x0 = input[head_base + pair];
    const float x1 = input[head_base + half_dim + pair];
    if (local_index < half_dim) {
        output[head_base + pair] = x0 * c - x1 * s;
    } else {
        output[head_base + half_dim + pair] = x0 * s + x1 * c;
    }
}

__global__ void qwen3_single_token_attention_kernel(
    const float* q, const float* k, const float* v, float* output,
    float* scores, float* probabilities, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim, float scale) {
    const std::uint32_t head = blockIdx.x;
    if (head >= query_heads || kv_heads == 0 || query_heads % kv_heads != 0) return;
    const std::uint32_t q_base = head * head_dim;
    const std::uint32_t kv_base = (head / (query_heads / kv_heads)) * head_dim;
    float score = 0.0F;
    for (std::uint32_t index = 0; index < head_dim; ++index) {
        score += q[q_base + index] * k[kv_base + index];
    }
    scores[head] = score * scale;
    // With one key/value in the position-zero fixture, stabilized softmax is
    // exactly one.  This kernel is intentionally limited to that fixture;
    // multi-position causal attention belongs to the KV-cache slice.
    probabilities[head] = 1.0F;
    for (std::uint32_t index = 0; index < head_dim; ++index) {
        output[q_base + index] = v[kv_base + index];
    }
}

__global__ void qwen3_cached_attention_kernel(
    const float* q, const float* key_cache, const float* value_cache,
    std::uint32_t cache_length, std::uint32_t cache_capacity, float* output,
    float* scores, float* probabilities, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim, float scale) {
    const std::uint32_t head = blockIdx.x;
    if (head >= query_heads || kv_heads == 0 || query_heads % kv_heads != 0
        || cache_length == 0 || threadIdx.x != 0) return;
    const std::uint32_t q_base = head * head_dim;
    const std::uint32_t kv_head = head / (query_heads / kv_heads);
    const std::uint32_t score_base = head * cache_length;
    float maximum = -INFINITY;
    for (std::uint32_t position = 0; position < cache_length; ++position) {
        const std::size_t cache_base =
            (static_cast<std::size_t>(kv_head) * cache_capacity + position) * head_dim;
        float score = 0.0F;
        for (std::uint32_t index = 0; index < head_dim; ++index) {
            score += q[q_base + index] * key_cache[cache_base + index];
        }
        score *= scale;
        scores[score_base + position] = score;
        maximum = fmaxf(maximum, score);
    }
    float denominator = 0.0F;
    for (std::uint32_t position = 0; position < cache_length; ++position) {
        const float probability = expf(scores[score_base + position] - maximum);
        probabilities[score_base + position] = probability;
        denominator += probability;
    }
    for (std::uint32_t index = 0; index < head_dim; ++index) {
        float value = 0.0F;
        for (std::uint32_t position = 0; position < cache_length; ++position) {
            const std::size_t cache_base =
                (static_cast<std::size_t>(kv_head) * cache_capacity + position) * head_dim;
            value += (probabilities[score_base + position] / denominator)
                     * value_cache[cache_base + index];
        }
        output[q_base + index] = value;
    }
    for (std::uint32_t position = 0; position < cache_length; ++position) {
        probabilities[score_base + position] /= denominator;
    }
}

__global__ void qwen3_cached_attention_parallel_kernel(
    const float* q, const float* key_cache, const float* value_cache,
    std::uint32_t cache_length, std::uint32_t cache_capacity, float* output,
    float* scores, float* probabilities, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim, float scale) {
    const std::uint32_t head = blockIdx.x;
    const std::uint32_t lane = threadIdx.x;
    if (head >= query_heads || kv_heads == 0 || query_heads % kv_heads != 0
        || cache_length == 0 || head_dim == 0) return;

    __shared__ float partials[256];
    __shared__ float maximum;
    __shared__ float denominator;
    const std::uint32_t q_base = head * head_dim;
    const std::uint32_t kv_head = head / (query_heads / kv_heads);
    const std::uint32_t score_base = head * cache_length;

    if (lane == 0) maximum = -INFINITY;
    __syncthreads();
    for (std::uint32_t position = 0; position < cache_length; ++position) {
        const std::size_t cache_base =
            (static_cast<std::size_t>(kv_head) * cache_capacity + position) * head_dim;
        float sum = 0.0F;
        for (std::uint32_t index = lane; index < head_dim; index += blockDim.x) {
            sum += q[q_base + index] * key_cache[cache_base + index];
        }
        partials[lane] = sum;
        __syncthreads();
        for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (lane < stride) partials[lane] += partials[lane + stride];
            __syncthreads();
        }
        if (lane == 0) {
            const float score = partials[0] * scale;
            scores[score_base + position] = score;
            maximum = fmaxf(maximum, score);
        }
        __syncthreads();
    }

    float exp_sum = 0.0F;
    for (std::uint32_t position = lane; position < cache_length; position += blockDim.x) {
        const float probability = expf(scores[score_base + position] - maximum);
        probabilities[score_base + position] = probability;
        exp_sum += probability;
    }
    partials[lane] = exp_sum;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (lane < stride) partials[lane] += partials[lane + stride];
        __syncthreads();
    }
    if (lane == 0) denominator = partials[0];
    __syncthreads();

    for (std::uint32_t index = lane; index < head_dim; index += blockDim.x) {
        float value = 0.0F;
        for (std::uint32_t position = 0; position < cache_length; ++position) {
            const std::size_t cache_base =
                (static_cast<std::size_t>(kv_head) * cache_capacity + position) * head_dim;
            value += (probabilities[score_base + position] / denominator)
                     * value_cache[cache_base + index];
        }
        output[q_base + index] = value;
    }
    __syncthreads();
    for (std::uint32_t position = lane; position < cache_length; position += blockDim.x) {
        probabilities[score_base + position] /= denominator;
    }
}

__device__ __forceinline__ float qwen3_wave_sum(float value) {
    for (std::uint32_t offset = 32; offset > 0; offset >>= 1) {
        value += __shfl_down(value, offset, 64);
    }
    return value;
}

// C12b diagnostic candidate. Four Wave64s partition the history dimension;
// each wave computes scores and value partials for positions wave, wave+4,
// ... . This deliberately changes reduction order and is therefore not a
// production path until the full autoregressive contract is demonstrated.
__global__ void qwen3_cached_attention_history_parallel_kernel(
    const float* q, const float* key_cache, const float* value_cache,
    std::uint32_t cache_length, std::uint32_t cache_capacity, float* output,
    float* scores, float* probabilities, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim, float scale) {
    const std::uint32_t head = blockIdx.x;
    const std::uint32_t lane = threadIdx.x;
    const std::uint32_t wave = lane / 64;
    const std::uint32_t wave_lane = lane % 64;
    if (head >= query_heads || kv_heads == 0 || query_heads % kv_heads != 0
        || cache_length == 0 || head_dim == 0 || head_dim > 256 || wave >= 4) return;

    __shared__ float local_maxima[4];
    __shared__ float local_denominators[4];
    __shared__ float value_partials[4 * 256];
    const std::uint32_t q_base = head * head_dim;
    const std::uint32_t kv_head = head / (query_heads / kv_heads);
    const std::uint32_t score_base = head * cache_length;

    float local_maximum = -INFINITY;
    for (std::uint32_t position = wave; position < cache_length; position += 4) {
        const std::size_t cache_base =
            (static_cast<std::size_t>(kv_head) * cache_capacity + position) * head_dim;
        float score = 0.0F;
        for (std::uint32_t index = wave_lane; index < head_dim; index += 64) {
            score += q[q_base + index] * key_cache[cache_base + index];
        }
        score = qwen3_wave_sum(score) * scale;
        if (wave_lane == 0) {
            scores[score_base + position] = score;
            local_maximum = fmaxf(local_maximum, score);
        }
    }
    if (wave_lane == 0) local_maxima[wave] = local_maximum;
    __syncthreads();
    if (lane == 0) {
        float maximum = local_maxima[0];
        for (std::uint32_t index = 1; index < 4; ++index) {
            maximum = fmaxf(maximum, local_maxima[index]);
        }
        local_maxima[0] = maximum;
    }
    __syncthreads();
    const float maximum = local_maxima[0];

    float local_denominator = 0.0F;
    for (std::uint32_t position = wave; position < cache_length; position += 4) {
        const float probability = expf(scores[score_base + position] - maximum);
        probabilities[score_base + position] = probability;
        local_denominator += probability;
    }
    local_denominator = qwen3_wave_sum(local_denominator);
    if (wave_lane == 0) local_denominators[wave] = local_denominator;
    __syncthreads();
    if (lane == 0) {
        float denominator = local_denominators[0];
        for (std::uint32_t index = 1; index < 4; ++index) {
            denominator += local_denominators[index];
        }
        local_denominators[0] = denominator;
    }
    __syncthreads();
    const float denominator = local_denominators[0];

    for (std::uint32_t index = wave_lane; index < head_dim; index += 64) {
        float value = 0.0F;
        for (std::uint32_t position = wave; position < cache_length; position += 4) {
            const std::size_t cache_base =
                (static_cast<std::size_t>(kv_head) * cache_capacity + position) * head_dim;
            value += (probabilities[score_base + position] / denominator)
                     * value_cache[cache_base + index];
        }
        value_partials[wave * 256 + index] = value;
    }
    __syncthreads();
    const std::uint32_t output_waves = (head_dim + 63) / 64;
    if (wave < output_waves) {
        for (std::uint32_t index = wave_lane; index < head_dim; index += 64) {
            float value = 0.0F;
            for (std::uint32_t partial = 0; partial < 4; ++partial) {
                value += value_partials[partial * 256 + index];
            }
            output[q_base + index] = value;
        }
    }
    for (std::uint32_t position = wave; position < cache_length; position += 4) {
        probabilities[score_base + position] /= denominator;
    }
}

__global__ void qwen3_kv_cache_store_kernel(
    const float* keys, const float* values, float* key_cache, float* value_cache,
    std::uint32_t position, std::uint32_t cache_capacity,
    std::uint32_t kv_heads, std::uint32_t head_dim) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t elements = kv_heads * head_dim;
    if (index >= elements || cache_capacity == 0 || position >= cache_capacity) return;
    const std::uint32_t head = index / head_dim;
    const std::uint32_t dimension = index % head_dim;
    const std::size_t cache_base =
        (static_cast<std::size_t>(head) * cache_capacity + position) * head_dim;
    key_cache[cache_base + dimension] = keys[index];
    value_cache[cache_base + dimension] = values[index];
}

__global__ void qwen3_silu_mul_kernel(
    const float* gate, const float* up, float* output, std::uint32_t elements) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        const float value = gate[index];
        output[index] = (value / (1.0F + expf(-value))) * up[index];
    }
}

__global__ void qwen3_add_kernel(
    const float* left, const float* right, float* output, std::uint32_t elements) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = left[index] + right[index];
}

__global__ void qwen3_f32_to_f16_kernel(
    const float* input, __half* output, std::uint32_t elements) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = __float2half_rn(input[index]);
}

__global__ void qwen3_f16_to_f32_kernel(
    const __half* input, float* output, std::uint32_t elements) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = __half2float(input[index]);
}

__global__ void qwen3_q6_k_gemv_kernel(
    const Q6KDeviceBlock* weights,
    const float* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns) {
    const std::uint32_t row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float partials[256];
    float sum = 0.0F;
    const std::uint32_t blocks_per_row = columns / 256;
    const auto* row_weights = weights + static_cast<std::size_t>(row) * blocks_per_row;
    for (std::uint32_t block = threadIdx.x; block < blocks_per_row; block += blockDim.x) {
        const auto& quant = row_weights[block];
        const float block_scale = half_to_float(quant.d);
        const std::uint32_t input_offset = block * 256;
        for (int index = 0; index < 256; ++index) {
            const float value = block_scale
                                * static_cast<float>(quant.scales[index / 16])
                                * static_cast<float>(q6_value(quant, index));
            sum += value * input[input_offset + index];
        }
    }
    partials[threadIdx.x] = sum;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) partials[threadIdx.x] += partials[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) output[row] = partials[0];
}

__global__ void qwen3_q5_k_q8_k_gemv_kernel(
    const Q5KDeviceBlock* weights, const Q8KDeviceBlock* input,
    float* output, std::uint32_t rows, std::uint32_t columns) {
    const std::uint32_t row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float partials[256];
    float sum = 0.0F;
    const std::uint32_t blocks_per_row = columns / 256;
    const auto* row_weights = weights + static_cast<std::size_t>(row) * blocks_per_row;
    for (std::uint32_t block = threadIdx.x; block < blocks_per_row; block += blockDim.x) {
        const auto& quant = row_weights[block];
        const auto& activation = input[block];
        int32_t partials[8] = {};
        int scales[8];
        int minimums[8];
        for (int group = 0; group < 8; ++group) {
            q5_k_scale_min(quant, group, scales[group], minimums[group]);
        }
        for (int pair = 0; pair < 4; ++pair) {
            const int qh_shift = 2 * pair;
            const int q_base = pair * 32;
            for (int index = 0; index < 32; ++index) {
                const std::uint8_t packed = quant.ql[q_base + index];
                const int high = quant.qh[index] >> qh_shift;
                const int element = pair * 64 + index;
                partials[index % 8] += scales[pair * 2] * ((packed & 0x0f) + ((high & 1) << 4))
                                       * static_cast<int>(activation.qs[element]);
                partials[index % 8] += scales[pair * 2 + 1] * ((packed >> 4) + (((high >> 1) & 1) << 4))
                                       * static_cast<int>(activation.qs[element + 32]);
            }
        }
        int sumi = 0;
        for (int group = 0; group < 8; ++group) {
            sumi += (activation.bsums[group * 2] + activation.bsums[group * 2 + 1])
                    * minimums[group];
        }
        const float d = half_to_float(quant.d) * activation.d;
        const float dmin = half_to_float(quant.dmin) * activation.d;
        float contribution = -dmin * static_cast<float>(sumi);
        for (const int32_t partial : partials) {
            contribution += d * static_cast<float>(partial);
        }
        sum += contribution;
    }
    partials[threadIdx.x] = sum;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) partials[threadIdx.x] += partials[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) output[row] = partials[0];
}

__global__ void qwen35_f32_gemv_kernel(
    const float* weights, const float* input, float* output,
    std::uint32_t rows, std::uint32_t columns) {
    const std::uint32_t row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float partials[256];
    float sum = 0.0F;
    for (std::uint32_t column = threadIdx.x; column < columns; column += blockDim.x) {
        sum += weights[static_cast<std::size_t>(row) * columns + column] * input[column];
    }
    partials[threadIdx.x] = sum;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) partials[threadIdx.x] += partials[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) output[row] = partials[0];
}

__global__ void qwen35_prepare_beta_decay_kernel(
    const float* beta_raw, const float* alpha_raw, const float* dt,
    const float* a, float* beta, float* decay, std::uint32_t elements) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    beta[index] = 1.0F / (1.0F + expf(-beta_raw[index]));
    const float x = alpha_raw[index] + dt[index];
    const float softplus = x <= 20.0F ? log1pf(expf(x)) : x;
    decay[index] = expf(softplus * a[index]);
}

__global__ void qwen3_q8_k_quantize_kernel(
    const float* input, Q8KDeviceBlock* output, std::uint32_t elements) {
    const std::uint32_t block = blockIdx.x;
    const std::uint32_t base = block * 256;
    if (base >= elements) return;
    __shared__ float maxima[256];
    const std::uint32_t lane = threadIdx.x;
    maxima[lane] = fabsf(input[base + lane]);
    __syncthreads();
    for (std::uint32_t stride = 128; stride > 0; stride /= 2) {
        if (lane < stride) maxima[lane] = fmaxf(maxima[lane], maxima[lane + stride]);
        __syncthreads();
    }
    const float maximum = maxima[0];
    output[block].d = maximum == 0.0F ? 0.0F : maximum / 127.0F;
    const float inverse = maximum == 0.0F ? 0.0F : 127.0F / maximum;
    int quantized = static_cast<int>(rintf(input[base + lane] * inverse));
    quantized = max(-127, min(127, quantized));
    output[block].qs[lane] = static_cast<std::int8_t>(quantized);
    __syncthreads();
    if (lane < 16) {
        int sum = 0;
        for (int index = 0; index < 16; ++index) sum += output[block].qs[lane * 16 + index];
        output[block].bsums[lane] = static_cast<std::int16_t>(sum);
    }
}

__global__ void qwen3_q6_k_q8_k_gemv_kernel(
    const Q6KDeviceBlock* weights,
    const Q8KDeviceBlock* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns) {
    const std::uint32_t row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float partials[256];
    const std::uint32_t lane = threadIdx.x;
    const std::uint32_t blocks_per_row = columns / 256;
    const auto* row_weights = weights + static_cast<std::size_t>(row) * blocks_per_row;
    float sum = 0.0F;
    for (std::uint32_t block = 0; block < blocks_per_row; ++block) {
        const auto& quant = row_weights[block];
        const int q = q6_value(quant, lane);
        const float scale = half_to_float(quant.d) * input[block].d
                            * static_cast<float>(quant.scales[lane / 16]);
        sum += scale * static_cast<float>(q) * static_cast<float>(input[block].qs[lane]);
    }
    partials[lane] = sum;
    __syncthreads();
    for (std::uint32_t stride = 128; stride > 0; stride /= 2) {
        if (lane < stride) partials[lane] += partials[lane + stride];
        __syncthreads();
    }
    if (lane == 0) output[row] = partials[0];
}

__global__ void qwen3_q6_k_q8_k_gemv_dot4_kernel(
    const Q6KDeviceBlock* weights,
    const Q8KDeviceBlock* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns) {
    const std::uint32_t row = blockIdx.x;
    const std::uint32_t lane = threadIdx.x;
    if (row >= rows) return;
    __shared__ float partials[64];
    const std::uint32_t blocks_per_row = columns / 256;
    const auto* row_weights = weights + static_cast<std::size_t>(row) * blocks_per_row;
    float sum = 0.0F;
    for (std::uint32_t block = 0; block < blocks_per_row; ++block) {
        const auto& quant = row_weights[block];
        const auto& activation = input[block];
        const std::uint32_t base = lane * 4;
        int packed_q6 = 0;
        int packed_q8 = 0;
#pragma unroll
        for (std::uint32_t index = 0; index < 4; ++index) {
            packed_q6 |= (q6_value(quant, static_cast<int>(base + index)) & 0xff)
                         << (8 * index);
            packed_q8 |= (static_cast<int>(activation.qs[base + index]) & 0xff)
                         << (8 * index);
        }
        const float scale = half_to_float(quant.d) * activation.d
                            * static_cast<float>(quant.scales[base / 16]);
        sum += scale * static_cast<float>(
            __builtin_amdgcn_sdot4(packed_q6, packed_q8, 0, false));
    }
    partials[lane] = sum;
    __syncthreads();
    for (std::uint32_t stride = 32; stride > 0; stride /= 2) {
        if (lane < stride) partials[lane] += partials[lane + stride];
        __syncthreads();
    }
    if (lane == 0) output[row] = partials[0];
}

__global__ void qwen3_q4_k_q8_k_gemv_kernel(
    const Q4KDeviceBlock* weights,
    const Q8KDeviceBlock* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns) {
    const std::uint32_t row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float partials[256];
    constexpr std::uint32_t metadata_tile_blocks = 32;
    __shared__ int tile_scales[metadata_tile_blocks][8];
    __shared__ int tile_minimums[metadata_tile_blocks][8];
    __shared__ float tile_d[metadata_tile_blocks];
    __shared__ float tile_dmin[metadata_tile_blocks];
    const std::uint32_t lane = threadIdx.x;
    const std::uint32_t blocks_per_row = columns / 256;
    const auto* row_weights = weights + static_cast<std::size_t>(row) * blocks_per_row;
    float sum = 0.0F;
    for (std::uint32_t tile_base = 0; tile_base < blocks_per_row;
         tile_base += metadata_tile_blocks) {
        const std::uint32_t tile_count = min(metadata_tile_blocks, blocks_per_row - tile_base);
        if (lane < tile_count) {
            const auto& quant = row_weights[tile_base + lane];
            tile_d[lane] = half_to_float(quant.d);
            tile_dmin[lane] = half_to_float(quant.dmin);
            for (std::uint32_t group = 0; group < 8; ++group) {
                q4_k_scale_min(quant, static_cast<int>(group), tile_scales[lane][group],
                               tile_minimums[lane][group]);
            }
        }
        __syncthreads();
        const std::uint32_t group = lane / 32;
        const std::uint32_t quant_index = (lane / 64) * 32 + lane % 32;
        for (std::uint32_t block = 0; block < tile_count; ++block) {
            const auto& quant = row_weights[tile_base + block];
            const std::uint8_t packed = quant.qs[quant_index];
            const int quantized = (group & 1) == 0 ? packed & 0x0f : packed >> 4;
            const float value = tile_d[block]
                                * static_cast<float>(tile_scales[block][group] * quantized)
                                - tile_dmin[block]
                                * static_cast<float>(tile_minimums[block][group]);
            sum += value * static_cast<float>(input[tile_base + block].qs[lane])
                   * input[tile_base + block].d;
        }
        if (tile_base + tile_count < blocks_per_row) __syncthreads();
    }
    partials[lane] = sum;
    __syncthreads();
    for (std::uint32_t stride = 128; stride > 0; stride /= 2) {
        if (lane < stride) partials[lane] += partials[lane + stride];
        __syncthreads();
    }
    if (lane == 0) output[row] = partials[0];
}

__device__ __forceinline__ int qwen3_dot4_i8(int lhs, int rhs, int accumulator) {
    return __builtin_amdgcn_sdot4(lhs, rhs, accumulator, false);
}

__device__ __forceinline__ std::uint32_t qwen3_load_b2(
    const std::uint8_t* values, int index) {
    const auto* values16 = reinterpret_cast<const std::uint16_t*>(values);
    return static_cast<std::uint32_t>(values16[2 * index])
           | (static_cast<std::uint32_t>(values16[2 * index + 1]) << 16);
}

__device__ __forceinline__ int qwen3_q6_mmvq_word(
    std::uint32_t low, std::uint32_t high, int nibble_shift) {
    std::uint32_t packed = 0;
#pragma unroll
    for (int byte = 0; byte < 4; ++byte) {
        const std::uint32_t low_nibble = (low >> (4 * nibble_shift + 8 * byte)) & 0x0fU;
        const std::uint32_t high_bits =
            ((high >> (4 * nibble_shift + 8 * byte)) << 4) & 0x30U;
        const std::uint32_t q = low_nibble | high_bits;
        packed |= ((q - 32U) & 0xffU) << (8 * byte);
    }
    return static_cast<int>(packed);
}

__device__ __forceinline__ float qwen3_q6_k_q8_1_mmvq_dot(
    const Q6KDeviceBlock& quant, const Q8_1Block* input, int iqs) {
    constexpr int qi6_half = 16;
    constexpr int qi6_quarter = 8;
    constexpr int qi8_1 = 8;
    const int q8_offset = 4 * (iqs / qi6_half) + (iqs % qi6_half) / qi8_1;
    const int scale_offset = qi6_quarter * (iqs / qi6_half)
                             + (iqs % qi6_half) / 4;
    const int high_shift = 2 * ((iqs % qi6_half) / qi6_quarter);
    const std::uint32_t low = qwen3_load_b2(quant.ql, iqs);
    const std::uint32_t high = qwen3_load_b2(
        quant.qh, qi6_quarter * (iqs / qi6_half) + iqs % qi6_quarter) >> high_shift;
    float sum = 0.0F;
#pragma unroll
    for (int part = 0; part < 2; ++part) {
        const auto& q8 = input[q8_offset + 2 * part];
        const int q6 = qwen3_q6_mmvq_word(low, high, part);
        const int q8_values = static_cast<int>(*reinterpret_cast<const std::uint32_t*>(
            q8.qs + 4 * (iqs % qi8_1)));
        const float d8 = half_to_float(q8.d);
        sum += d8 * static_cast<float>(qwen3_dot4_i8(q6, q8_values, 0)
                                       * quant.scales[scale_offset + 4 * part]);
    }
    return half_to_float(quant.d) * sum;
}

__global__ void qwen3_q6_k_q8_1_mmvq_kernel(
    const Q6KDeviceBlock* weights,
    const Q8_1Block* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns) {
    const std::uint32_t row = blockIdx.x;
    const std::uint32_t tid = threadIdx.x;
    if (row >= rows) return;
    __shared__ float partials[128];
    const std::uint32_t blocks_per_row = columns / 256;
    const auto* row_weights = weights + static_cast<std::size_t>(row) * blocks_per_row;
    float sum = 0.0F;
    const int iqs = static_cast<int>(tid % 32);
    for (std::uint32_t block = tid / 32; block < blocks_per_row; block += 4) {
        sum += qwen3_q6_k_q8_1_mmvq_dot(row_weights[block], input + block * 8, iqs);
    }
    partials[tid] = sum;
    __syncthreads();
    if (tid < 64) partials[tid] += partials[tid + 64];
    __syncthreads();
    for (std::uint32_t stride = 32; stride > 0; stride /= 2) {
        if (tid < stride) partials[tid] += partials[tid + stride];
        __syncthreads();
    }
    if (tid == 0) output[row] = partials[0];
}

__device__ __forceinline__ float qwen3_q5_k_q8_1_mmvq_dot(
    const Q5KDeviceBlock& quant, const Q8_1Block* input, int iqs) {
    // Q5_K's MMVQ VDR is 2: each logical lane handles two Q8_1 blocks.
    const int q8_offset = 2 * ((iqs / 2) / 4);
    const int ql_offset = 16 * q8_offset + 4 * ((iqs / 2) % 4);
    const auto* ql = reinterpret_cast<const int*>(quant.ql + ql_offset);
    const auto* qh = reinterpret_cast<const int*>(quant.qh + 4 * ((iqs / 2) % 4));
    const int vl[2] = {ql[0], ql[4]};
    const int vh[2] = {qh[0] >> q8_offset, qh[4] >> q8_offset};

    int scales[2];
    int minimums[2];
    const int group = 2 * (iqs / 8);
    q5_k_scale_min(quant, group, scales[0], minimums[0]);
    q5_k_scale_min(quant, group + 1, scales[1], minimums[1]);

    float sum_d = 0.0F;
    float sum_min = 0.0F;
#pragma unroll
    for (int part = 0; part < 2; ++part) {
        const auto& q8 = input[q8_offset + part];
        const auto* q8_words = reinterpret_cast<const int*>(q8.qs)
                               + ((iqs / 2) % 4);
        const int q8_0 = q8_words[0];
        const int q8_1 = q8_words[4];
        const int low_0 = (vl[0] >> (4 * part)) & 0x0F0F0F0F;
        const int low_1 = (vl[1] >> (4 * part)) & 0x0F0F0F0F;
        const int high_0 = ((vh[0] >> part) << 4) & 0x10101010;
        const int high_1 = ((vh[1] >> part) << 4) & 0x10101010;
        const int q0 = low_0 | high_0;
        const int q1 = low_1 | high_1;
        const int dot = qwen3_dot4_i8(q0, q8_0, qwen3_dot4_i8(q1, q8_1, 0));
        const int q8_sum = qwen3_dot4_i8(0x01010101, q8_0,
                                         qwen3_dot4_i8(0x01010101, q8_1, 0));
        const float d8 = half_to_float(q8.d);
        sum_d += d8 * static_cast<float>(dot * scales[part]);
        sum_min += d8 * static_cast<float>(q8_sum * minimums[part]);
    }
    return half_to_float(quant.d) * sum_d - half_to_float(quant.dmin) * sum_min;
}

__global__ void qwen3_q5_k_q8_1_mmvq_kernel(
    const Q5KDeviceBlock* weights,
    const Q8_1Block* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns) {
    const std::uint32_t row = blockIdx.x;
    const std::uint32_t tid = threadIdx.x;
    if (row >= rows) return;
    __shared__ float partials[128];
    const std::uint32_t blocks_per_row = columns / 256;
    const auto* row_weights = weights + static_cast<std::size_t>(row) * blocks_per_row;
    float sum = 0.0F;
    const int iqs = 2 * static_cast<int>(tid % 16);
    for (std::uint32_t block = tid / 16; block < blocks_per_row; block += 8) {
        sum += qwen3_q5_k_q8_1_mmvq_dot(row_weights[block], input + block * 8, iqs);
    }
    partials[tid] = sum;
    __syncthreads();
    if (tid < 64) partials[tid] += partials[tid + 64];
    __syncthreads();
    for (std::uint32_t stride = 32; stride > 0; stride /= 2) {
        if (tid < stride) partials[tid] += partials[tid + stride];
        __syncthreads();
    }
    if (tid == 0) output[row] = partials[0];
}

__device__ __forceinline__ float qwen3_q4_k_q8_1_mmvq_dot(
    const Q4KDeviceBlock& quant, const Q8_1Block* input, int iqs) {
    const int q8_offset = 2 * ((iqs / 2) / 4);
    const int ql_offset = 16 * q8_offset + 4 * ((iqs / 2) % 4);
    const auto* ql = reinterpret_cast<const int*>(quant.qs + ql_offset);
    const int vl[2] = {ql[0], ql[4]};

    int scales[2];
    int minimums[2];
    const int group = 2 * (iqs / 8);
    q4_k_scale_min(quant, group, scales[0], minimums[0]);
    q4_k_scale_min(quant, group + 1, scales[1], minimums[1]);

    float sum_d = 0.0F;
    float sum_min = 0.0F;
#pragma unroll
    for (int part = 0; part < 2; ++part) {
        const auto& q8 = input[q8_offset + part];
        const auto* q8_words = reinterpret_cast<const int*>(q8.qs)
                               + ((iqs / 2) % 4);
        const int q0 = (vl[0] >> (4 * part)) & 0x0F0F0F0F;
        const int q1 = (vl[1] >> (4 * part)) & 0x0F0F0F0F;
        const int dot = qwen3_dot4_i8(q0, q8_words[0],
                                      qwen3_dot4_i8(q1, q8_words[4], 0));
        const int q8_sum = qwen3_dot4_i8(0x01010101, q8_words[0],
                                         qwen3_dot4_i8(0x01010101, q8_words[4], 0));
        const float d8 = half_to_float(q8.d);
        sum_d += d8 * static_cast<float>(dot * scales[part]);
        sum_min += d8 * static_cast<float>(q8_sum * minimums[part]);
    }
    return half_to_float(quant.d) * sum_d - half_to_float(quant.dmin) * sum_min;
}

__global__ void qwen3_q4_k_q8_1_mmvq_kernel(
    const Q4KDeviceBlock* weights,
    const Q8_1Block* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns) {
    const std::uint32_t row = blockIdx.x;
    const std::uint32_t tid = threadIdx.x;
    if (row >= rows) return;
    __shared__ float partials[128];
    const std::uint32_t blocks_per_row = columns / 256;
    const auto* row_weights = weights + static_cast<std::size_t>(row) * blocks_per_row;
    float sum = 0.0F;
    const int iqs = 2 * static_cast<int>(tid % 16);
    for (std::uint32_t block = tid / 16; block < blocks_per_row; block += 8) {
        sum += qwen3_q4_k_q8_1_mmvq_dot(row_weights[block], input + block * 8, iqs);
    }
    partials[tid] = sum;
    __syncthreads();
    if (tid < 64) partials[tid] += partials[tid + 64];
    __syncthreads();
    for (std::uint32_t stride = 32; stride > 0; stride /= 2) {
        if (tid < stride) partials[tid] += partials[tid + stride];
        __syncthreads();
    }
    if (tid == 0) output[row] = partials[0];
}

__global__ void qwen3_q4_k_q8_k_gemv_dot4_kernel(
    const Q4KDeviceBlock* weights,
    const Q8KDeviceBlock* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns) {
    const std::uint32_t row = blockIdx.x;
    const std::uint32_t lane = threadIdx.x;
    if (row >= rows || lane >= 64) return;
    __shared__ float partials[64];
    const std::uint32_t blocks_per_row = columns / 256;
    const auto* row_weights = weights + static_cast<std::size_t>(row) * blocks_per_row;
    const std::uint32_t element = lane * 4;
    const std::uint32_t group = element / 32;
    const std::uint32_t in_group = element % 32;
    const std::uint32_t q_offset = (group / 2) * 32 + in_group;
    const bool high_nibble = (group & 1) != 0;
    const std::uint32_t scale_group = group;
    float sum = 0.0F;
    for (std::uint32_t block = 0; block < blocks_per_row; ++block) {
        const auto& quant = row_weights[block];
        const std::uint32_t packed_q =
            *reinterpret_cast<const std::uint32_t*>(quant.qs + q_offset);
        const std::uint32_t q = high_nibble
            ? (packed_q >> 4) & 0x0f0f0f0f
            : packed_q & 0x0f0f0f0f;
        const std::uint32_t x =
            *reinterpret_cast<const std::uint32_t*>(input[block].qs + element);
        const int q_dot_x = qwen3_dot4_i8(static_cast<int>(q), static_cast<int>(x), 0);
        const int x_sum = qwen3_dot4_i8(0x01010101, static_cast<int>(x), 0);
        int scale = 0;
        int minimum = 0;
        q4_k_scale_min(quant, static_cast<int>(scale_group), scale, minimum);
        sum += input[block].d * (half_to_float(quant.d) * static_cast<float>(scale * q_dot_x)
                                 - half_to_float(quant.dmin)
                                       * static_cast<float>(minimum * x_sum));
    }
    partials[lane] = sum;
    __syncthreads();
    for (std::uint32_t stride = 32; stride > 0; stride /= 2) {
        if (lane < stride) partials[lane] += partials[lane + stride];
        __syncthreads();
    }
    if (lane == 0) output[row] = partials[0];
}

__device__ __forceinline__ bool qwen3_argmax_wins(
    float candidate_value, std::uint32_t candidate_index,
    float best_value, std::uint32_t best_index) {
    return candidate_value > best_value
        || (candidate_value == best_value && candidate_index < best_index);
}

__global__ void qwen3_argmax_kernel(
    const float* input, std::uint32_t* output, std::uint32_t elements) {
    __shared__ float values[256];
    __shared__ std::uint32_t indices[256];

    float best_value = -INFINITY;
    std::uint32_t best_index = elements;
    for (std::uint32_t index = threadIdx.x; index < elements; index += blockDim.x) {
        const float value = input[index];
        if (qwen3_argmax_wins(value, index, best_value, best_index)) {
            best_value = value;
            best_index = index;
        }
    }
    values[threadIdx.x] = best_value;
    indices[threadIdx.x] = best_index;
    __syncthreads();

    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride
            && qwen3_argmax_wins(values[threadIdx.x + stride], indices[threadIdx.x + stride],
                                 values[threadIdx.x], indices[threadIdx.x])) {
            values[threadIdx.x] = values[threadIdx.x + stride];
            indices[threadIdx.x] = indices[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) output[0] = indices[0];
}

}  // namespace

void launch_qwen3_q4_embedding(
    const std::byte* weights,
    std::uint32_t token,
    std::uint32_t vocabulary,
    std::uint32_t hidden_size,
    float* output,
    hipStream_t stream) {
    if (weights == nullptr || output == nullptr || vocabulary == 0 || hidden_size == 0
        || hidden_size % 32 != 0 || token >= vocabulary) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen3_q4_embedding_kernel,
                       dim3((hidden_size + 255) / 256), dim3(256), 0, stream,
                       weights, token, vocabulary, hidden_size, output);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen35_q4_k_embedding(
    const Q4KDeviceBlock* weights,
    std::uint32_t token,
    std::uint32_t vocabulary,
    std::uint32_t hidden_size,
    float* output,
    hipStream_t stream) {
    if (weights == nullptr || output == nullptr || vocabulary == 0 || hidden_size == 0
        || hidden_size % 256 != 0 || token >= vocabulary) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen35_q4_k_embedding_kernel,
                       dim3((hidden_size + 255) / 256), dim3(256), 0, stream,
                       weights, token, vocabulary, hidden_size, output);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen3_rms_norm(
    const float* input,
    const float* weights,
    float* output,
    std::uint32_t elements,
    float epsilon,
    hipStream_t stream) {
    if (input == nullptr || weights == nullptr || output == nullptr || elements == 0
        || elements > 256U * 1024U || epsilon <= 0.0F) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen3_rms_norm_kernel, dim3(1), dim3(256), 0, stream,
                       input, weights, output, elements, epsilon);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen3_rms_normalize(
    const float* input, float* output, std::uint32_t elements,
    float epsilon, hipStream_t stream) {
    if (input == nullptr || output == nullptr || elements == 0
        || elements > 256U * 1024U || epsilon <= 0.0F) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen3_rms_normalize_kernel, dim3(1), dim3(256), 0, stream,
                       input, output, elements, epsilon);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen3_elementwise_mul(
    const float* left, const float* right, float* output,
    std::uint32_t elements, hipStream_t stream) {
    if (left == nullptr || right == nullptr || output == nullptr || elements == 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen3_elementwise_mul_kernel,
                       dim3((elements + 255) / 256), dim3(256), 0, stream,
                       left, right, output, elements);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen3_head_rms_normalize(
    const float* input, float* output, std::uint32_t heads,
    std::uint32_t head_dim, float epsilon, hipStream_t stream) {
    if (input == nullptr || output == nullptr || heads == 0 || head_dim == 0
        || head_dim > 256 || epsilon <= 0.0F) MIINFER_HIP_CHECK(hipErrorInvalidValue);
    const unsigned threads = head_dim > 128 ? 256 : 128;
    hipLaunchKernelGGL(qwen3_head_rms_normalize_kernel, dim3(heads), dim3(threads), 0, stream,
                       input, output, heads, head_dim, epsilon);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen3_head_mul(
    const float* input, const float* weights, float* output,
    std::uint32_t heads, std::uint32_t head_dim, hipStream_t stream) {
    if (input == nullptr || weights == nullptr || output == nullptr || heads == 0 || head_dim == 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen3_head_mul_kernel,
                       dim3((heads * head_dim + 255) / 256), dim3(256), 0, stream,
                       input, weights, output, heads, head_dim);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen35_split_q_gate(
    const float* input, float* query, float* gate,
    std::uint32_t heads, std::uint32_t head_dim, hipStream_t stream) {
    if (input == nullptr || query == nullptr || gate == nullptr || heads == 0 || head_dim == 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen35_split_q_gate_kernel,
                       dim3((heads * head_dim + 255) / 256), dim3(256), 0, stream,
                       input, query, gate, heads, head_dim);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen35_rope_sections(
    const float* input, float* output, std::uint32_t heads,
    std::uint32_t head_dim, std::uint32_t position, float theta, hipStream_t stream) {
    if (input == nullptr || output == nullptr || heads == 0 || head_dim < 64
        || theta <= 0.0F) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen35_rope_sections_kernel,
                       dim3((heads * head_dim + 255) / 256), dim3(256), 0, stream,
                       input, output, heads, head_dim, position, theta);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen35_sigmoid_mul(
    const float* input, const float* gate, float* output,
    std::uint32_t elements, hipStream_t stream) {
    if (input == nullptr || gate == nullptr || output == nullptr || elements == 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen35_sigmoid_mul_kernel,
                       dim3((elements + 255) / 256), dim3(256), 0, stream,
                       input, gate, output, elements);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen35_deltanet_state_update(
    const float* query, const float* key, const float* value,
    const float* beta, const float* decay, float* state, float* output,
    std::uint32_t key_heads, std::uint32_t value_heads,
    std::uint32_t state_size, hipStream_t stream) {
    if (query == nullptr || key == nullptr || value == nullptr || beta == nullptr
        || decay == nullptr || state == nullptr || output == nullptr
        || key_heads == 0 || value_heads == 0 || state_size == 0
        || state_size > 1024 || value_heads % key_heads != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen35_deltanet_state_update_kernel,
                       dim3(value_heads), dim3(state_size), 0, stream,
                       query, key, value, beta, decay, state, output,
                       key_heads, value_heads, state_size);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen35_deltanet_state_update_no_decay_store(
    const float* query, const float* key, const float* value,
    const float* beta, const float* decay, float* state, float* output,
    std::uint32_t key_heads, std::uint32_t value_heads,
    std::uint32_t state_size, hipStream_t stream) {
    if (query == nullptr || key == nullptr || value == nullptr || beta == nullptr
        || decay == nullptr || state == nullptr || output == nullptr
        || key_heads == 0 || value_heads == 0 || state_size == 0
        || state_size > 1024 || value_heads % key_heads != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen35_deltanet_state_update_no_decay_store_kernel,
                       dim3(value_heads), dim3(state_size), 0, stream,
                       query, key, value, beta, decay, state, output,
                       key_heads, value_heads, state_size);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen35_deltanet_state_update_transposed(
    const float* query, const float* key, const float* value,
    const float* beta, const float* decay, float* state, float* output,
    std::uint32_t key_heads, std::uint32_t value_heads,
    std::uint32_t state_size, hipStream_t stream) {
    if (query == nullptr || key == nullptr || value == nullptr || beta == nullptr
        || decay == nullptr || state == nullptr || output == nullptr
        || key_heads == 0 || value_heads == 0 || state_size == 0
        || state_size > 1024 || value_heads % key_heads != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen35_deltanet_state_update_transposed_kernel,
                       dim3(value_heads), dim3(state_size), 0, stream,
                       query, key, value, beta, decay, state, output,
                       key_heads, value_heads, state_size);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen35_conv_silu_split(
    const float* current_qkv, const float* conv_weights, float* history,
    float* query, float* key, float* value, std::uint32_t position,
    std::uint32_t history_capacity, std::uint32_t channels,
    std::uint32_t conv_kernel, hipStream_t stream) {
    if (current_qkv == nullptr || conv_weights == nullptr || history == nullptr
        || query == nullptr || key == nullptr || value == nullptr
        || history_capacity == 0 || channels == 0 || channels % 5 != 0
        || conv_kernel == 0 || conv_kernel > history_capacity) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen35_conv_silu_split_kernel,
                       dim3((channels + 255) / 256), dim3(256), 0, stream,
                       current_qkv, conv_weights, history, query, key, value,
                       position, history_capacity, channels, conv_kernel);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen35_head_l2_normalize(
    const float* input, float* output, std::uint32_t heads,
    std::uint32_t head_dim, hipStream_t stream) {
    if (input == nullptr || output == nullptr || heads == 0 || head_dim == 0
        || head_dim > 128) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen35_head_l2_normalize_kernel,
                       dim3(heads), dim3(128), 0, stream,
                       input, output, heads, head_dim);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen3_rope(
    const float* input, float* output, std::uint32_t heads,
    std::uint32_t head_dim, std::uint32_t position, float theta, hipStream_t stream) {
    if (input == nullptr || output == nullptr || heads == 0 || head_dim == 0
        || head_dim % 2 != 0 || theta <= 0.0F) MIINFER_HIP_CHECK(hipErrorInvalidValue);
    hipLaunchKernelGGL(qwen3_rope_kernel,
                       dim3((heads * head_dim + 255) / 256), dim3(256), 0, stream,
                       input, output, heads, head_dim, position, theta);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen3_single_token_attention(
    const float* q, const float* k, const float* v, float* output,
    float* scores, float* probabilities, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim, float scale, hipStream_t stream) {
    if (q == nullptr || k == nullptr || v == nullptr || output == nullptr
        || scores == nullptr || probabilities == nullptr || query_heads == 0
        || kv_heads == 0 || head_dim == 0 || query_heads % kv_heads != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen3_single_token_attention_kernel, dim3(query_heads), dim3(1), 0, stream,
                       q, k, v, output, scores, probabilities, query_heads, kv_heads, head_dim, scale);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen3_cached_attention(
    const float* q, const float* key_cache, const float* value_cache,
    std::uint32_t cache_length, std::uint32_t cache_capacity, float* output,
    float* scores, float* probabilities, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim, float scale, hipStream_t stream) {
    if (q == nullptr || key_cache == nullptr || value_cache == nullptr || output == nullptr
        || scores == nullptr || probabilities == nullptr || cache_length == 0
        || cache_length > cache_capacity || query_heads == 0 || kv_heads == 0
        || head_dim == 0 || query_heads % kv_heads != 0 || !std::isfinite(scale)) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen3_cached_attention_kernel, dim3(query_heads), dim3(1), 0, stream,
                       q, key_cache, value_cache, cache_length, cache_capacity, output,
                       scores, probabilities, query_heads, kv_heads, head_dim, scale);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen3_cached_attention_parallel(
    const float* q, const float* key_cache, const float* value_cache,
    std::uint32_t cache_length, std::uint32_t cache_capacity, float* output,
    float* scores, float* probabilities, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim, float scale, hipStream_t stream) {
    if (q == nullptr || key_cache == nullptr || value_cache == nullptr || output == nullptr
        || scores == nullptr || probabilities == nullptr || cache_length == 0
        || cache_length > cache_capacity || query_heads == 0 || kv_heads == 0
        || head_dim == 0 || head_dim > 256 || query_heads % kv_heads != 0
        || !std::isfinite(scale)) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen3_cached_attention_parallel_kernel,
                       dim3(query_heads), dim3(256), 0, stream,
                       q, key_cache, value_cache, cache_length, cache_capacity, output,
                       scores, probabilities, query_heads, kv_heads, head_dim, scale);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen3_cached_attention_history_parallel(
    const float* q, const float* key_cache, const float* value_cache,
    std::uint32_t cache_length, std::uint32_t cache_capacity, float* output,
    float* scores, float* probabilities, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim, float scale, hipStream_t stream) {
    if (q == nullptr || key_cache == nullptr || value_cache == nullptr || output == nullptr
        || scores == nullptr || probabilities == nullptr || cache_length == 0
        || cache_length > cache_capacity || query_heads == 0 || kv_heads == 0
        || head_dim == 0 || head_dim > 256 || query_heads % kv_heads != 0
        || !std::isfinite(scale)) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen3_cached_attention_history_parallel_kernel,
                       dim3(query_heads), dim3(256), 0, stream,
                       q, key_cache, value_cache, cache_length, cache_capacity, output,
                       scores, probabilities, query_heads, kv_heads, head_dim, scale);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen3_kv_cache_store(
    const float* keys, const float* values, float* key_cache, float* value_cache,
    std::uint32_t position, std::uint32_t cache_capacity,
    std::uint32_t kv_heads, std::uint32_t head_dim, hipStream_t stream) {
    if (keys == nullptr || values == nullptr || key_cache == nullptr || value_cache == nullptr
        || cache_capacity == 0 || position >= cache_capacity || kv_heads == 0 || head_dim == 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    const auto elements = kv_heads * head_dim;
    hipLaunchKernelGGL(qwen3_kv_cache_store_kernel,
                       dim3((elements + 255) / 256), dim3(256), 0, stream,
                       keys, values, key_cache, value_cache, position, cache_capacity,
                       kv_heads, head_dim);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen3_silu_mul(
    const float* gate, const float* up, float* output,
    std::uint32_t elements, hipStream_t stream) {
    if (gate == nullptr || up == nullptr || output == nullptr || elements == 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen3_silu_mul_kernel,
                       dim3((elements + 255) / 256), dim3(256), 0, stream,
                       gate, up, output, elements);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen3_add(
    const float* left, const float* right, float* output,
    std::uint32_t elements, hipStream_t stream) {
    if (left == nullptr || right == nullptr || output == nullptr || elements == 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen3_add_kernel,
                       dim3((elements + 255) / 256), dim3(256), 0, stream,
                       left, right, output, elements);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen3_f32_to_f16(
    const float* input, __half* output, std::uint32_t elements, hipStream_t stream) {
    if (input == nullptr || output == nullptr || elements == 0) MIINFER_HIP_CHECK(hipErrorInvalidValue);
    hipLaunchKernelGGL(qwen3_f32_to_f16_kernel,
                       dim3((elements + 255) / 256), dim3(256), 0, stream,
                       input, output, elements);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen3_f16_to_f32(
    const __half* input, float* output, std::uint32_t elements, hipStream_t stream) {
    if (input == nullptr || output == nullptr || elements == 0) MIINFER_HIP_CHECK(hipErrorInvalidValue);
    hipLaunchKernelGGL(qwen3_f16_to_f32_kernel,
                       dim3((elements + 255) / 256), dim3(256), 0, stream,
                       input, output, elements);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen3_q6_k_gemv(
    const Q6KDeviceBlock* weights,
    const float* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns,
    hipStream_t stream) {
    if (weights == nullptr || input == nullptr || output == nullptr || rows == 0 || columns == 0
        || columns % 256 != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen3_q6_k_gemv_kernel, dim3(rows), dim3(256), 0, stream,
                       weights, input, output, rows, columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen3_q8_k_quantize(
    const float* input, Q8KDeviceBlock* output, std::uint32_t elements, hipStream_t stream) {
    if (input == nullptr || output == nullptr || elements == 0 || elements % 256 != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen3_q8_k_quantize_kernel,
                       dim3(elements / 256), dim3(256), 0, stream,
                       input, output, elements);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen3_q6_k_q8_k_gemv(
    const Q6KDeviceBlock* weights,
    const Q8KDeviceBlock* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns,
    hipStream_t stream) {
    if (weights == nullptr || input == nullptr || output == nullptr || rows == 0 || columns == 0
        || columns % 256 != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen3_q6_k_q8_k_gemv_kernel, dim3(rows), dim3(256), 0, stream,
                       weights, input, output, rows, columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen3_q6_k_q8_k_gemv_dot4(
    const Q6KDeviceBlock* weights,
    const Q8KDeviceBlock* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns,
    hipStream_t stream) {
    if (weights == nullptr || input == nullptr || output == nullptr || rows == 0 || columns == 0
        || columns % 256 != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen3_q6_k_q8_k_gemv_dot4_kernel, dim3(rows), dim3(64), 0, stream,
                       weights, input, output, rows, columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen3_q6_k_q8_1_mmvq(
    const Q6KDeviceBlock* weights,
    const Q8_1Block* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns,
    hipStream_t stream) {
    if (weights == nullptr || input == nullptr || output == nullptr || rows == 0 || columns == 0
        || columns % 256 != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen3_q6_k_q8_1_mmvq_kernel, dim3(rows), dim3(128), 0, stream,
                       weights, input, output, rows, columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen3_q5_k_q8_1_mmvq(
    const Q5KDeviceBlock* weights,
    const Q8_1Block* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns,
    hipStream_t stream) {
    if (weights == nullptr || input == nullptr || output == nullptr || rows == 0 || columns == 0
        || columns % 256 != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen3_q5_k_q8_1_mmvq_kernel, dim3(rows), dim3(128), 0, stream,
                       weights, input, output, rows, columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen3_q4_k_q8_1_mmvq(
    const Q4KDeviceBlock* weights,
    const Q8_1Block* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns,
    hipStream_t stream) {
    if (weights == nullptr || input == nullptr || output == nullptr || rows == 0 || columns == 0
        || columns % 256 != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen3_q4_k_q8_1_mmvq_kernel, dim3(rows), dim3(128), 0, stream,
                       weights, input, output, rows, columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen3_q4_k_q8_k_gemv(
    const Q4KDeviceBlock* weights,
    const Q8KDeviceBlock* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns,
    hipStream_t stream) {
    if (weights == nullptr || input == nullptr || output == nullptr || rows == 0 || columns == 0
        || columns % 256 != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen3_q4_k_q8_k_gemv_kernel, dim3(rows), dim3(256), 0, stream,
                       weights, input, output, rows, columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen3_q4_k_q8_k_gemv_dot4(
    const Q4KDeviceBlock* weights,
    const Q8KDeviceBlock* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns,
    hipStream_t stream) {
    if (weights == nullptr || input == nullptr || output == nullptr || rows == 0 || columns == 0
        || columns % 256 != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen3_q4_k_q8_k_gemv_dot4_kernel, dim3(rows), dim3(64), 0, stream,
                       weights, input, output, rows, columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen3_q5_k_q8_k_gemv(
    const Q5KDeviceBlock* weights, const Q8KDeviceBlock* input,
    float* output, std::uint32_t rows, std::uint32_t columns, hipStream_t stream) {
    if (weights == nullptr || input == nullptr || output == nullptr || rows == 0
        || columns == 0 || columns % 256 != 0) MIINFER_HIP_CHECK(hipErrorInvalidValue);
    hipLaunchKernelGGL(qwen3_q5_k_q8_k_gemv_kernel, dim3(rows), dim3(256), 0, stream,
                       weights, input, output, rows, columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen35_f32_gemv(
    const float* weights, const float* input, float* output,
    std::uint32_t rows, std::uint32_t columns, hipStream_t stream) {
    if (weights == nullptr || input == nullptr || output == nullptr || rows == 0 || columns == 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen35_f32_gemv_kernel, dim3(rows), dim3(256), 0, stream,
                       weights, input, output, rows, columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen35_prepare_beta_decay(
    const float* beta_raw, const float* alpha_raw, const float* dt, const float* a,
    float* beta, float* decay, std::uint32_t elements, hipStream_t stream) {
    if (beta_raw == nullptr || alpha_raw == nullptr || dt == nullptr || a == nullptr
        || beta == nullptr || decay == nullptr || elements == 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen35_prepare_beta_decay_kernel,
                       dim3((elements + 255) / 256), dim3(256), 0, stream,
                       beta_raw, alpha_raw, dt, a, beta, decay, elements);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen3_argmax(
    const float* input, std::uint32_t* output, std::uint32_t elements, hipStream_t stream) {
    if (input == nullptr || output == nullptr || elements == 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen3_argmax_kernel, dim3(1), dim3(256), 0, stream,
                       input, output, elements);
    MIINFER_HIP_CHECK(hipGetLastError());
}

}  // namespace miinfer
