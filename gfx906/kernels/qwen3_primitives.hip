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
    __shared__ float partials[128];
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
        || head_dim > 128 || epsilon <= 0.0F) MIINFER_HIP_CHECK(hipErrorInvalidValue);
    hipLaunchKernelGGL(qwen3_head_rms_normalize_kernel, dim3(heads), dim3(128), 0, stream,
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

}  // namespace miinfer
