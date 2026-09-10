#include "miinfer/qwen3_gpu_primitives.hpp"
#include "miinfer/q4_q8_gemv.hpp"

#include "miinfer/hip_check.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx906__)
#error "Qwen3 primitives are intentionally gfx906-only"
#endif

namespace miinfer {

namespace {

__device__ __forceinline__ float half_to_float(__half value) {
    return __half2float(value);
}

__device__ __forceinline__ float qwen3_wave_sum(float value) {
    for (std::uint32_t offset = 32; offset > 0; offset >>= 1) {
        value += __shfl_down(value, offset, 64);
    }
    return value;
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

struct Q4KMetadata {
    __half d;
    __half dmin;
    std::uint8_t scales[12];
};

static_assert(sizeof(Q4KMetadata) == 16);

struct Q4KDecodedMetadata {
    __half d;
    __half dmin;
    std::uint8_t scales[8];
    std::uint8_t minimums[8];
};

static_assert(sizeof(Q4KDecodedMetadata) == 20);

__device__ __forceinline__ void q4_k_scale_min(
    const std::uint8_t* scales, int group, int& scale, int& minimum) {
    if (group < 4) {
        scale = scales[group] & 63;
        minimum = scales[group + 4] & 63;
    } else {
        scale = (scales[group + 4] & 0x0f) | ((scales[group - 4] >> 6) << 4);
        minimum = (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);
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

__global__ void qwen35_q4_k_embedding_device_token_kernel(
    const Q4KDeviceBlock* weights,
    const std::uint32_t* token_ptr,
    std::uint32_t vocabulary,
    std::uint32_t hidden_size,
    float* output) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t token = *token_ptr;
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

__global__ void qwen3_rms_norm_vec4_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    float* __restrict__ output,
    std::uint32_t elements,
    float epsilon) {

    __shared__ float s_wave_sums[4];
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t wave_id = tid / 64;
    const std::uint32_t lane_id = tid % 64;

    const float4* in_vec = reinterpret_cast<const float4*>(input);
    const float4* w_vec = reinterpret_cast<const float4*>(weights);
    float4* out_vec = reinterpret_cast<float4*>(output);

    float4 in_val[5];
    float sum = 0.0f;

    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        const std::uint32_t idx = i * 256 + tid;
        in_val[i] = in_vec[idx];
        sum += in_val[i].x * in_val[i].x;
        sum += in_val[i].y * in_val[i].y;
        sum += in_val[i].z * in_val[i].z;
        sum += in_val[i].w * in_val[i].w;
    }

    const float wave_sum_val = qwen3_wave_sum(sum);
    if (lane_id == 0) {
        s_wave_sums[wave_id] = wave_sum_val;
    }
    __syncthreads();

    const float total_sum = s_wave_sums[0] + s_wave_sums[1] + s_wave_sums[2] + s_wave_sums[3];
    const float inv_rms = rsqrtf(total_sum / static_cast<float>(elements) + epsilon);

    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        const std::uint32_t idx = i * 256 + tid;
        const float4 w = w_vec[idx];
        float4 out;
        out.x = in_val[i].x * inv_rms * w.x;
        out.y = in_val[i].y * inv_rms * w.y;
        out.z = in_val[i].z * inv_rms * w.z;
        out.w = in_val[i].w * inv_rms * w.w;
        out_vec[idx] = out;
    }
}

__global__ void qwen3_rms_norm_batch_vec4_kernel(
    const float* input, const float* weights, float* output,
    std::uint32_t elements, float epsilon) {
    __shared__ float wave_sums[4];
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t token = blockIdx.x;
    const std::size_t base = static_cast<std::size_t>(token) * elements;
    const float4* in = reinterpret_cast<const float4*>(input + base);
    const float4* weight = reinterpret_cast<const float4*>(weights);
    float4* out = reinterpret_cast<float4*>(output + base);
    const std::uint32_t vec_count = elements / 4;
    float sum = 0.0F;
    for (std::uint32_t index = tid; index < vec_count; index += blockDim.x) {
        const float4 value = in[index];
        sum += value.x * value.x + value.y * value.y + value.z * value.z + value.w * value.w;
    }
    const float wave_sum = qwen3_wave_sum(sum);
    if (tid % 64 == 0) wave_sums[tid / 64] = wave_sum;
    __syncthreads();
    const float inv_rms = rsqrtf((wave_sums[0] + wave_sums[1] + wave_sums[2] + wave_sums[3])
                                 / static_cast<float>(elements) + epsilon);
    for (std::uint32_t index = tid; index < vec_count; index += blockDim.x) {
        const float4 value = in[index];
        const float4 scale = weight[index];
        out[index] = {value.x * inv_rms * scale.x, value.y * inv_rms * scale.y,
                      value.z * inv_rms * scale.z, value.w * inv_rms * scale.w};
    }
}

__global__ void m23_q8_1_mmq_quantize_kernel(
    const float* input, miinfer::M23Q8_1MmqBlock* output, std::uint32_t elements) {
    const std::uint32_t token = blockIdx.y;
    const std::uint32_t block = blockIdx.x;
    const std::uint32_t lane = threadIdx.x;
    const std::size_t base = static_cast<std::size_t>(token) * elements + block * 128;
    __shared__ float maxima[128];
    __shared__ float sums[128];
    __shared__ int q_sums[128];
    const float value = input[base + lane];
    maxima[lane] = fabsf(value);
    sums[lane] = value;
    __syncthreads();
    const std::uint32_t group = lane / 32;
    const std::uint32_t group_lane = lane % 32;
    for (std::uint32_t stride = 16; stride > 0; stride /= 2) {
        if (group_lane < stride) {
            maxima[lane] = fmaxf(maxima[lane], maxima[lane + stride]);
            sums[lane] += sums[lane + stride];
        }
        __syncthreads();
    }
    const float max_abs = maxima[group * 32];
    const float d = max_abs == 0.0F ? 0.0F : max_abs / 127.0F;
    const int quantized = d == 0.0F ? 0 : static_cast<int>(rintf(value / d));
    const int clamped = max(-127, min(127, quantized));
    q_sums[lane] = clamped;
    __syncthreads();
    for (std::uint32_t stride = 16; stride > 0; stride /= 2) {
        if (group_lane < stride) q_sums[lane] += q_sums[lane + stride];
        __syncthreads();
    }
    auto& dst = output[static_cast<std::size_t>(block) * gridDim.y + token];
    dst.qs[lane] = static_cast<std::int8_t>(clamped);
    if (group_lane == 0) {
        dst.d[group] = d;
        dst.s[group] = sums[group * 32];
        dst.qsum_scaled[group] = __half2float(__float2half_rn(d)) * static_cast<float>(q_sums[group * 32]);
    }
}

__device__ __forceinline__ float m23_q4_k_q8_1_dot(
    const Q4KDeviceBlock& quant, const miinfer::M23Q8_1MmqBlock* input,
    std::uint32_t token_count, std::uint32_t token, std::uint32_t weight_block, int iqs) {
    const int q8_offset = 2 * ((iqs / 2) / 4);
    const int ql_offset = 16 * q8_offset + 4 * ((iqs / 2) % 4);
    const auto* ql = reinterpret_cast<const int*>(quant.qs + ql_offset);
    const int vl[2] = {ql[0], ql[4]};
    const int group = 2 * (iqs / 8);
    int scales[2], minimums[2];
    q4_k_scale_min(quant.scales, group, scales[0], minimums[0]);
    q4_k_scale_min(quant.scales, group + 1, scales[1], minimums[1]);
    float sum_d = 0.0F, sum_min = 0.0F;
#pragma unroll
    for (int part = 0; part < 2; ++part) {
        const int q8_index = q8_offset + part;
        const auto& q8 = input[static_cast<std::size_t>(weight_block * 2 + q8_index / 4)
                               * token_count + token];
        const int q8_word = 4 * ((iqs / 2) % 4);
        const auto* q8_words = reinterpret_cast<const int*>(q8.qs + (q8_index % 4) * 32)
                               + q8_word / 4;
        const int q0 = (vl[0] >> (4 * part)) & 0x0F0F0F0F;
        const int q1 = (vl[1] >> (4 * part)) & 0x0F0F0F0F;
        const int dot = __builtin_amdgcn_sdot4(
            q0, q8_words[0], __builtin_amdgcn_sdot4(q1, q8_words[4], 0, false), false);
        const int q8_sum = __builtin_amdgcn_sdot4(
            0x01010101, q8_words[0],
            __builtin_amdgcn_sdot4(0x01010101, q8_words[4], 0, false), false);
        const float d8 = half_to_float(__float2half_rn(q8.d[q8_index % 4]));
        sum_d += d8 * static_cast<float>(dot * scales[part]);
        sum_min += d8 * static_cast<float>(q8_sum * minimums[part]);
    }
    return half_to_float(quant.d) * sum_d - half_to_float(quant.dmin) * sum_min;
}

__global__ void m23_q4_k_q8_1_mmq_kernel(
    const Q4KDeviceBlock* weights, const miinfer::M23Q8_1MmqBlock* input,
    float* output, std::uint32_t rows, std::uint32_t columns, std::uint32_t token_count) {
    const std::uint32_t row_base = blockIdx.x * 64 + threadIdx.y * 16;
    const std::uint32_t token_base = blockIdx.y * 128 + threadIdx.x;
    const std::uint32_t blocks_per_row = columns / 256;
    float accum[16][2] = {};
#pragma unroll
    for (int row_offset = 0; row_offset < 16; ++row_offset) {
        const std::uint32_t row = row_base + row_offset;
        if (row >= rows) continue;
        const auto* row_weights = weights + static_cast<std::size_t>(row) * blocks_per_row;
#pragma unroll 1
        for (std::uint32_t block = 0; block < blocks_per_row; ++block) {
#pragma unroll
            for (int iqs = 0; iqs < 32; iqs += 2) {
                if (token_base < token_count) {
                    accum[row_offset][0] += m23_q4_k_q8_1_dot(
                        row_weights[block], input, token_count, token_base, block, iqs);
                }
                if (token_base + 64 < token_count) {
                    accum[row_offset][1] += m23_q4_k_q8_1_dot(
                        row_weights[block], input, token_count, token_base + 64, block, iqs);
                }
            }
        }
        if (token_base < token_count) output[static_cast<std::size_t>(token_base) * rows + row] = accum[row_offset][0];
        if (token_base + 64 < token_count) output[static_cast<std::size_t>(token_base + 64) * rows + row] = accum[row_offset][1];
    }
}

template <bool EmitQ8>
__global__ void qwen3_fused_add_rms_norm_vec4_kernel(
    const float* __restrict__ residual_in,
    const float* __restrict__ projection,
    const float* __restrict__ weights,
    float* __restrict__ residual_out,
    float* __restrict__ normalized_out,
    Q8_1Block* __restrict__ q8_out,
    std::uint32_t elements,
    float epsilon) {

    __shared__ float s_wave_sums[4];
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t wave_id = tid / 64;
    const std::uint32_t lane_id = tid % 64;

    const float4* in_vec = reinterpret_cast<const float4*>(residual_in);
    const float4* proj_vec = reinterpret_cast<const float4*>(projection);
    const float4* w_vec = reinterpret_cast<const float4*>(weights);
    float4* res_out_vec = reinterpret_cast<float4*>(residual_out);
    float4* norm_out_vec = reinterpret_cast<float4*>(normalized_out);

    float4 res[5];
    float sum = 0.0f;

    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        const std::uint32_t idx = i * 256 + tid;
        const float4 a = in_vec[idx];
        const float4 b = proj_vec[idx];
        res[i].x = a.x + b.x;
        res[i].y = a.y + b.y;
        res[i].z = a.z + b.z;
        res[i].w = a.w + b.w;
        if (res_out_vec != nullptr) {
            res_out_vec[idx] = res[i];
        }

        sum += res[i].x * res[i].x;
        sum += res[i].y * res[i].y;
        sum += res[i].z * res[i].z;
        sum += res[i].w * res[i].w;
    }

    const float wave_sum_val = qwen3_wave_sum(sum);
    if (lane_id == 0) {
        s_wave_sums[wave_id] = wave_sum_val;
    }
    __syncthreads();

    const float total_sum = s_wave_sums[0] + s_wave_sums[1] + s_wave_sums[2] + s_wave_sums[3];
    const float inv_rms = rsqrtf(total_sum / static_cast<float>(elements) + epsilon);

    const int group_in_wave = lane_id / 8;
    const int k = lane_id % 8;
    const int group_in_block = wave_id * 8 + group_in_wave;

    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        const std::uint32_t idx = i * 256 + tid;
        const float4 w = w_vec[idx];
        float4 out;
        out.x = res[i].x * inv_rms * w.x;
        out.y = res[i].y * inv_rms * w.y;
        out.z = res[i].z * inv_rms * w.z;
        out.w = res[i].w * inv_rms * w.w;
        if (norm_out_vec != nullptr) {
            norm_out_vec[idx] = out;
        }

        if constexpr (EmitQ8) {
            const int global_block = i * 32 + group_in_block;

            float amax = fmaxf(fmaxf(fabsf(out.x), fabsf(out.y)), fmaxf(fabsf(out.z), fabsf(out.w)));
            #pragma unroll
            for (int offset = 4; offset > 0; offset /= 2) {
                amax = fmaxf(amax, __shfl_xor(amax, offset, 64));
            }

            const float scale = amax / 127.0f;
            const float inv_scale = (scale != 0.0f) ? (1.0f / scale) : 0.0f;

            const int q0 = max(-127, min(127, static_cast<int>(roundf(out.x * inv_scale))));
            const int q1 = max(-127, min(127, static_cast<int>(roundf(out.y * inv_scale))));
            const int q2 = max(-127, min(127, static_cast<int>(roundf(out.z * inv_scale))));
            const int q3 = max(-127, min(127, static_cast<int>(roundf(out.w * inv_scale))));

            int qsum = q0 + q1 + q2 + q3;
            #pragma unroll
            for (int offset = 4; offset > 0; offset /= 2) {
                qsum += __shfl_xor(qsum, offset, 64);
            }

            if (k == 0) {
                q8_out[global_block].d = __float2half_rn(scale);
                q8_out[global_block].s = __float2half_rn(static_cast<float>(qsum) * scale);
            }

            const std::uint32_t packed =
                (static_cast<std::uint32_t>(static_cast<std::uint8_t>(q0))) |
                (static_cast<std::uint32_t>(static_cast<std::uint8_t>(q1)) << 8) |
                (static_cast<std::uint32_t>(static_cast<std::uint8_t>(q2)) << 16) |
                (static_cast<std::uint32_t>(static_cast<std::uint8_t>(q3)) << 24);

            reinterpret_cast<std::uint32_t*>(q8_out[global_block].qs)[k] = packed;
        }
    }
}

__global__ void qwen3_fused_add_rms_norm_kernel(
    const float* __restrict__ residual_in,
    const float* __restrict__ projection,
    const float* __restrict__ weights,
    float* __restrict__ residual_out,
    float* __restrict__ normalized_out,
    std::uint32_t elements,
    float epsilon) {

    __shared__ float partials[256];
    float sum = 0.0F;
    for (std::uint32_t index = threadIdx.x; index < elements; index += blockDim.x) {
        const float val = residual_in[index] + projection[index];
        residual_out[index] = val;
        sum += val * val;
    }
    partials[threadIdx.x] = sum;
    __syncthreads();

    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) partials[threadIdx.x] += partials[threadIdx.x + stride];
        __syncthreads();
    }

    const float inverse_rms = rsqrtf(partials[0] / static_cast<float>(elements) + epsilon);
    for (std::uint32_t index = threadIdx.x; index < elements; index += blockDim.x) {
        normalized_out[index] = residual_out[index] * inverse_rms * weights[index];
    }
}

__global__ void qwen3_fused_add_rms_norm_batch_kernel(
    const float* __restrict__ residual_in,
    const float* __restrict__ projection,
    const float* __restrict__ weights,
    float* __restrict__ residual_out,
    float* __restrict__ normalized_out,
    std::uint32_t elements,
    float epsilon) {
    const std::size_t offset = static_cast<std::size_t>(blockIdx.x) * elements;
    __shared__ float partials[256];
    float sum = 0.0F;
    for (std::uint32_t index = threadIdx.x; index < elements; index += blockDim.x) {
        const float val = residual_in[offset + index] + projection[offset + index];
        residual_out[offset + index] = val;
        sum += val * val;
    }
    partials[threadIdx.x] = sum;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) partials[threadIdx.x] += partials[threadIdx.x + stride];
        __syncthreads();
    }
    const float inverse_rms = rsqrtf(partials[0] / static_cast<float>(elements) + epsilon);
    for (std::uint32_t index = threadIdx.x; index < elements; index += blockDim.x) {
        normalized_out[offset + index] = residual_out[offset + index] * inverse_rms * weights[index];
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

__device__ __forceinline__ void compute_qwen35_rope_pair(
    float x0, float x1, std::uint32_t local, std::uint32_t position, float theta,
    float& out0, float& out1) {
    if (local >= 32) {
        out0 = x0;
        out1 = x1;
        return;
    }
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
    out0 = x0 * cosine - x1 * sine;
    out1 = x0 * sine + x1 * cosine;
}

__global__ void qwen35_fused_q_split_norm_rope_kernel(
    const float* __restrict__ qfull,
    const float* __restrict__ q_norm_weight,
    float* __restrict__ query_rope,
    float* __restrict__ gate,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t position,
    float theta,
    float epsilon) {

    const std::uint32_t head = blockIdx.x;
    const std::uint32_t tid = threadIdx.x;
    if (head >= heads || tid >= head_dim) return;

    __shared__ float s_partials[256];
    __shared__ float s_rot[64];

    const std::uint32_t src_base = head * (head_dim * 2);
    const float q_val = qfull[src_base + tid];
    const float g_val = qfull[src_base + head_dim + tid];
    gate[head * head_dim + tid] = g_val;

    s_partials[tid] = q_val * q_val;
    __syncthreads();

    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) s_partials[tid] += s_partials[tid + stride];
        __syncthreads();
    }

    const float inv_rms = rsqrtf(s_partials[0] / static_cast<float>(head_dim) + epsilon);
    const float norm_val = q_val * inv_rms * q_norm_weight[tid];

    if (tid < 64) {
        s_rot[tid] = norm_val;
    }
    __syncthreads();

    float final_val = norm_val;
    if (tid < 32) {
        float out0, out1;
        compute_qwen35_rope_pair(s_rot[tid], s_rot[tid + 32], tid, position, theta, out0, out1);
        final_val = out0;
        s_rot[tid + 32] = out1;
    }
    __syncthreads();

    if (tid >= 32 && tid < 64) {
        final_val = s_rot[tid];
    }

    query_rope[head * head_dim + tid] = final_val;
}

__global__ void qwen35_fused_k_norm_rope_kv_store_kernel(
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ k_norm_weight,
    float* __restrict__ key_cache,
    float* __restrict__ value_cache,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t position,
    std::uint32_t cache_capacity,
    float theta,
    float epsilon) {

    const std::uint32_t head = blockIdx.x;
    const std::uint32_t tid = threadIdx.x;
    if (head >= heads || tid >= head_dim) return;

    __shared__ float s_partials[256];
    __shared__ float s_rot[64];

    const std::uint32_t base = head * head_dim;
    const float k_val = key[base + tid];
    s_partials[tid] = k_val * k_val;
    __syncthreads();

    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) s_partials[tid] += s_partials[tid + stride];
        __syncthreads();
    }

    const float inv_rms = rsqrtf(s_partials[0] / static_cast<float>(head_dim) + epsilon);
    const float norm_val = k_val * inv_rms * k_norm_weight[tid];

    if (tid < 64) {
        s_rot[tid] = norm_val;
    }
    __syncthreads();

    float final_k = norm_val;
    if (tid < 32) {
        float out0, out1;
        compute_qwen35_rope_pair(s_rot[tid], s_rot[tid + 32], tid, position, theta, out0, out1);
        final_k = out0;
        s_rot[tid + 32] = out1;
    }
    __syncthreads();

    if (tid >= 32 && tid < 64) {
        final_k = s_rot[tid];
    }

    const std::size_t cache_base =
        (static_cast<std::size_t>(head) * cache_capacity + position) * head_dim;
    key_cache[cache_base + tid] = final_k;
    value_cache[cache_base + tid] = value[base + tid];
}

__global__ void qwen35_fused_k_norm_rope_kv_store_f16_kernel(
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ k_norm_weight,
    __half* __restrict__ key_cache,
    __half* __restrict__ value_cache,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t position,
    std::uint32_t cache_capacity,
    float theta,
    float epsilon) {

    const std::uint32_t head = blockIdx.x;
    const std::uint32_t tid = threadIdx.x;
    if (head >= heads || tid >= head_dim) return;

    __shared__ float s_partials[256];
    __shared__ float s_rot[64];

    const std::uint32_t base = head * head_dim;
    const float k_val = key[base + tid];
    s_partials[tid] = k_val * k_val;
    __syncthreads();

    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) s_partials[tid] += s_partials[tid + stride];
        __syncthreads();
    }

    const float inv_rms = rsqrtf(s_partials[0] / static_cast<float>(head_dim) + epsilon);
    const float norm_val = k_val * inv_rms * k_norm_weight[tid];

    if (tid < 64) {
        s_rot[tid] = norm_val;
    }
    __syncthreads();

    float final_k = norm_val;
    if (tid < 32) {
        float out0, out1;
        compute_qwen35_rope_pair(s_rot[tid], s_rot[tid + 32], tid, position, theta, out0, out1);
        final_k = out0;
        s_rot[tid + 32] = out1;
    }
    __syncthreads();

    if (tid >= 32 && tid < 64) {
        final_k = s_rot[tid];
    }

    const std::size_t cache_base =
        (static_cast<std::size_t>(head) * cache_capacity + position) * head_dim;
    key_cache[cache_base + tid] = __float2half_rn(final_k);
    value_cache[cache_base + tid] = __float2half_rn(value[base + tid]);
}

__global__ void qwen35_fused_q_split_norm_rope_batch_kernel(
    const float* __restrict__ qfull,
    const float* __restrict__ q_norm_weight,
    float* __restrict__ query_rope,
    float* __restrict__ gate,
    std::uint32_t token_count,
    std::uint32_t base_position,
    std::uint32_t heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    float theta,
    float epsilon) {
    const std::uint32_t pair = blockIdx.x;
    const std::uint32_t token = pair / heads;
    const std::uint32_t head = pair % heads;
    const std::uint32_t tid = threadIdx.x;
    if (token >= token_count || head >= heads || tid >= head_dim || head_dim != 256
        || kv_heads == 0) return;
    __shared__ float partials[256];
    __shared__ float rotated[64];
    const std::size_t qfull_base =
        static_cast<std::size_t>(token) * ((heads * 2 + kv_heads) * head_dim)
        + head * (2 * head_dim);
    const std::size_t output_base =
        (static_cast<std::size_t>(token) * heads + head) * head_dim;
    const float q_val = qfull[qfull_base + tid];
    gate[output_base + tid] = qfull[qfull_base + head_dim + tid];
    partials[tid] = q_val * q_val;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) partials[tid] += partials[tid + stride];
        __syncthreads();
    }
    const float norm_val = q_val * rsqrtf(partials[0] / static_cast<float>(head_dim) + epsilon)
        * q_norm_weight[tid];
    if (tid < 64) rotated[tid] = norm_val;
    __syncthreads();
    float final_val = norm_val;
    if (tid < 32) {
        float out0, out1;
        compute_qwen35_rope_pair(rotated[tid], rotated[tid + 32], tid,
                                 base_position + token, theta, out0, out1);
        final_val = out0;
        rotated[tid + 32] = out1;
    }
    __syncthreads();
    if (tid >= 32 && tid < 64) final_val = rotated[tid];
    query_rope[output_base + tid] = final_val;
}

__global__ void qwen35_fused_k_norm_rope_kv_store_batch_kernel(
    const float* __restrict__ qfull,
    const float* __restrict__ value,
    const float* __restrict__ k_norm_weight,
    float* __restrict__ key_cache,
    float* __restrict__ value_cache,
    std::uint32_t token_count,
    std::uint32_t base_position,
    std::uint32_t cache_capacity,
    std::uint32_t query_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    float theta,
    float epsilon) {
    const std::uint32_t pair = blockIdx.x;
    const std::uint32_t token = pair / kv_heads;
    const std::uint32_t head = pair % kv_heads;
    const std::uint32_t tid = threadIdx.x;
    if (token >= token_count || head >= kv_heads || tid >= head_dim || head_dim != 256
        || query_heads == 0) return;
    __shared__ float partials[256];
    __shared__ float rotated[64];
    const std::size_t qfull_base = static_cast<std::size_t>(token)
        * ((query_heads * 2 + kv_heads) * head_dim)
        + query_heads * 2 * head_dim + head * head_dim;
    const std::size_t value_base = static_cast<std::size_t>(token) * kv_heads * head_dim
        + head * head_dim;
    const float k_val = qfull[qfull_base + tid];
    partials[tid] = k_val * k_val;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) partials[tid] += partials[tid + stride];
        __syncthreads();
    }
    const float norm_val = k_val * rsqrtf(partials[0] / static_cast<float>(head_dim) + epsilon)
        * k_norm_weight[tid];
    if (tid < 64) rotated[tid] = norm_val;
    __syncthreads();
    float final_val = norm_val;
    if (tid < 32) {
        float out0, out1;
        compute_qwen35_rope_pair(rotated[tid], rotated[tid + 32], tid,
                                 base_position + token, theta, out0, out1);
        final_val = out0;
        rotated[tid + 32] = out1;
    }
    __syncthreads();
    if (tid >= 32 && tid < 64) final_val = rotated[tid];
    const std::size_t cache_base =
        (static_cast<std::size_t>(head) * cache_capacity + base_position + token) * head_dim;
    key_cache[cache_base + tid] = final_val;
    value_cache[cache_base + tid] = value[value_base + tid];
}

__global__ void qwen35_fused_k_norm_rope_kv_store_batch_f16_kernel(
    const float* __restrict__ qfull,
    const float* __restrict__ value,
    const float* __restrict__ k_norm_weight,
    __half* __restrict__ key_cache,
    __half* __restrict__ value_cache,
    std::uint32_t token_count,
    std::uint32_t base_position,
    std::uint32_t cache_capacity,
    std::uint32_t query_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    float theta,
    float epsilon) {
    const std::uint32_t pair = blockIdx.x;
    const std::uint32_t token = pair / kv_heads;
    const std::uint32_t head = pair % kv_heads;
    const std::uint32_t tid = threadIdx.x;
    if (token >= token_count || head >= kv_heads || tid >= head_dim || head_dim != 256
        || query_heads == 0) return;
    __shared__ float partials[256];
    __shared__ float rotated[64];
    const std::size_t qfull_base = static_cast<std::size_t>(token)
        * ((query_heads * 2 + kv_heads) * head_dim)
        + query_heads * 2 * head_dim + head * head_dim;
    const std::size_t value_base = static_cast<std::size_t>(token) * kv_heads * head_dim
        + head * head_dim;
    const float k_val = qfull[qfull_base + tid];
    partials[tid] = k_val * k_val;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) partials[tid] += partials[tid + stride];
        __syncthreads();
    }
    const float norm_val = k_val * rsqrtf(partials[0] / static_cast<float>(head_dim) + epsilon)
        * k_norm_weight[tid];
    if (tid < 64) rotated[tid] = norm_val;
    __syncthreads();
    float final_val = norm_val;
    if (tid < 32) {
        float out0, out1;
        compute_qwen35_rope_pair(rotated[tid], rotated[tid + 32], tid,
                                 base_position + token, theta, out0, out1);
        final_val = out0;
        rotated[tid + 32] = out1;
    }
    __syncthreads();
    if (tid >= 32 && tid < 64) final_val = rotated[tid];
    const std::size_t cache_base =
        (static_cast<std::size_t>(head) * cache_capacity + base_position + token) * head_dim;
    key_cache[cache_base + tid] = __float2half_rn(final_val);
    value_cache[cache_base + tid] = __float2half_rn(value[value_base + tid]);
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

__global__ void qwen35_deltanet_state_update_transposed_no_decay_store_kernel(
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
        key_dot += state[index] * decay_value * key[key_base + column];
    }
    const float delta = (value[static_cast<std::size_t>(head) * state_size + row]
                         - key_dot) * beta_value;
    float query_dot = 0.0F;
    for (std::uint32_t column = 0; column < state_size; ++column) {
        const std::size_t index = state_base
            + static_cast<std::size_t>(column) * state_size + row;
        const float updated = state[index] * decay_value + delta * key[key_base + column];
        state[index] = updated;
        query_dot += updated * query[key_base + column];
    }
    output[static_cast<std::size_t>(head) * state_size + row] =
        query_dot * rsqrtf(static_cast<float>(state_size));
}

__global__ void qwen35_deltanet_state_update_transposed_no_decay_store_lds_inputs_kernel(
    const float* query, const float* key, const float* value,
    const float* beta, const float* decay, float* state, float* output,
    std::uint32_t key_heads, std::uint32_t value_heads, std::uint32_t state_size) {
    const std::uint32_t head = blockIdx.x;
    const std::uint32_t row = threadIdx.x;
    if (head >= value_heads || row >= state_size || key_heads == 0) return;

    extern __shared__ float shared_inputs[];
    const std::uint32_t key_head = head % key_heads;
    const std::size_t key_base = static_cast<std::size_t>(key_head) * state_size;
    float* shared_query = shared_inputs;
    float* shared_key = shared_inputs + state_size;
    shared_query[row] = query[key_base + row];
    shared_key[row] = key[key_base + row];
    __syncthreads();

    const std::size_t state_base =
        static_cast<std::size_t>(head) * state_size * state_size;
    const float beta_value = beta[head];
    const float decay_value = decay[head];
    float key_dot = 0.0F;
    for (std::uint32_t column = 0; column < state_size; ++column) {
        const std::size_t index = state_base
            + static_cast<std::size_t>(column) * state_size + row;
        key_dot += state[index] * decay_value * shared_key[column];
    }
    const float delta = (value[static_cast<std::size_t>(head) * state_size + row]
                         - key_dot) * beta_value;
    float query_dot = 0.0F;
    for (std::uint32_t column = 0; column < state_size; ++column) {
        const std::size_t index = state_base
            + static_cast<std::size_t>(column) * state_size + row;
        const float updated = state[index] * decay_value + delta * shared_key[column];
        state[index] = updated;
        query_dot += updated * shared_query[column];
    }
    output[static_cast<std::size_t>(head) * state_size + row] =
        query_dot * rsqrtf(static_cast<float>(state_size));
}

__global__ void qwen35_deltanet_state_update_transposed_fused_beta_alpha_kernel(
    const float* query, const float* key, const float* value,
    const float* beta_raw, const float* alpha_raw, const float* dt,
    const float* a, float* state, float* output,
    std::uint32_t key_heads, std::uint32_t value_heads, std::uint32_t state_size) {
    const std::uint32_t head = blockIdx.x;
    const std::uint32_t row = threadIdx.x;
    if (head >= value_heads || row >= state_size || key_heads == 0) return;
    extern __shared__ float fused_shared[];
    const std::uint32_t key_head = head % key_heads;
    const std::size_t key_base = static_cast<std::size_t>(key_head) * state_size;
    float* shared_query = fused_shared;
    float* shared_key = fused_shared + state_size;
    shared_query[row] = query[key_base + row];
    shared_key[row] = key[key_base + row];
    if (row == 0) {
        fused_shared[2 * state_size] = 1.0F / (1.0F + expf(-beta_raw[head]));
        const float x = alpha_raw[head] + dt[head];
        const float softplus = x <= 20.0F ? log1pf(expf(x)) : x;
        fused_shared[2 * state_size + 1] = expf(softplus * a[head]);
    }
    __syncthreads();
    const std::size_t state_base = static_cast<std::size_t>(head) * state_size * state_size;
    const float beta_value = fused_shared[2 * state_size];
    const float decay_value = fused_shared[2 * state_size + 1];
    float key_dot = 0.0F;
    for (std::uint32_t column = 0; column < state_size; ++column) {
        key_dot += state[state_base + static_cast<std::size_t>(column) * state_size + row]
                   * decay_value * shared_key[column];
    }
    const float delta = (value[static_cast<std::size_t>(head) * state_size + row] - key_dot)
                        * beta_value;
    float query_dot = 0.0F;
    for (std::uint32_t column = 0; column < state_size; ++column) {
        const std::size_t index = state_base + static_cast<std::size_t>(column) * state_size + row;
        const float updated = state[index] * decay_value + delta * shared_key[column];
        state[index] = updated;
        query_dot += updated * shared_query[column];
    }
    output[static_cast<std::size_t>(head) * state_size + row] =
        query_dot * rsqrtf(static_cast<float>(state_size));
}

__global__ void qwen35_deltanet_state_update_transposed_row_waves_kernel(
    const float* query, const float* key, const float* value,
    const float* beta, const float* decay, float* state, float* output,
    std::uint32_t key_heads, std::uint32_t value_heads, std::uint32_t state_size) {
    const std::uint32_t head = blockIdx.x;
    const std::uint32_t thread = threadIdx.x;
    const std::uint32_t warp = thread / 64;
    const std::uint32_t lane = thread % 64;
    if (head >= value_heads || key_heads == 0) return;

    extern __shared__ float row_wave_shared[];
    const std::uint32_t key_head = head % key_heads;
    const std::size_t key_base = static_cast<std::size_t>(key_head) * state_size;
    if (thread < state_size) {
        row_wave_shared[thread] = query[key_base + thread];
        row_wave_shared[state_size + thread] = key[key_base + thread];
    }
    __syncthreads();
    const std::uint32_t row = blockIdx.y * 4 + warp;
    if (row >= state_size) return;

    const std::size_t state_base =
        static_cast<std::size_t>(head) * state_size * state_size;
    const std::size_t index0 = state_base + static_cast<std::size_t>(lane) * state_size + row;
    const std::size_t index1 = state_base
        + static_cast<std::size_t>(lane + 64) * state_size + row;
    const float decay_value = decay[head];
    const float beta_value = beta[head];
    const float key_dot_partial = decay_value * (
        state[index0] * row_wave_shared[state_size + lane]
        + state[index1] * row_wave_shared[state_size + lane + 64]);
    float key_dot = key_dot_partial;
#pragma unroll
    for (int offset = 32; offset > 0; offset /= 2) {
        key_dot += __shfl_down(key_dot, offset, 64);
    }
    const float delta = (value[static_cast<std::size_t>(head) * state_size + row]
                        - key_dot) * beta_value;
    const float updated0 = state[index0] * decay_value
        + delta * row_wave_shared[state_size + lane];
    const float updated1 = state[index1] * decay_value
        + delta * row_wave_shared[state_size + lane + 64];
    state[index0] = updated0;
    state[index1] = updated1;
    float query_dot = updated0 * row_wave_shared[lane]
        + updated1 * row_wave_shared[lane + 64];
#pragma unroll
    for (int offset = 32; offset > 0; offset /= 2) {
        query_dot += __shfl_down(query_dot, offset, 64);
    }
    if (lane == 0) {
        output[static_cast<std::size_t>(head) * state_size + row] =
            query_dot * rsqrtf(static_cast<float>(state_size));
    }
}

__global__ void qwen35_deltanet_fused_recurrent_core_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ decay,
    const float* __restrict__ ssm_norm,
    const float* __restrict__ gate,
    float* __restrict__ state,
    float* __restrict__ gated_output,
    float* __restrict__ recurrent_output,
    std::uint32_t key_heads,
    std::uint32_t value_heads,
    std::uint32_t state_size,
    float epsilon,
    miinfer::Q8_1Block* __restrict__ gated_output_q8) {
    const std::uint32_t head = blockIdx.x;
    const std::uint32_t row = threadIdx.x;
    if (head >= value_heads || row >= state_size || key_heads == 0) return;

    extern __shared__ float shared_mem[];
    const std::uint32_t key_head = head % key_heads;
    const std::size_t key_base = static_cast<std::size_t>(key_head) * state_size;
    float* shared_query = shared_mem;
    float* shared_key = shared_mem + state_size;
    shared_query[row] = query[key_base + row];
    shared_key[row] = key[key_base + row];
    __syncthreads();

    const std::size_t state_base =
        static_cast<std::size_t>(head) * state_size * state_size;
    const float beta_value = beta[head];
    const float decay_value = decay[head];

    // Register caching of the 128-element state row to eliminate redundant HBM read pass
    float s[128];
    float key_dot = 0.0F;
    #pragma unroll 4
    for (std::uint32_t column = 0; column < 128; ++column) {
        const std::size_t index = state_base
            + static_cast<std::size_t>(column) * 128 + row;
        const float val = state[index];
        s[column] = val;
        key_dot += val * decay_value * shared_key[column];
    }
    const float delta = (value[static_cast<std::size_t>(head) * 128 + row]
                         - key_dot) * beta_value;
    float query_dot = 0.0F;
    #pragma unroll 4
    for (std::uint32_t column = 0; column < 128; ++column) {
        const std::size_t index = state_base
            + static_cast<std::size_t>(column) * 128 + row;
        const float updated = s[column] * decay_value + delta * shared_key[column];
        state[index] = updated;
        query_dot += updated * shared_query[column];
    }
    const float rec_val = query_dot * rsqrtf(128.0F);
    if (recurrent_output != nullptr) {
        recurrent_output[static_cast<std::size_t>(head) * 128 + row] = rec_val;
    }

    __syncthreads();
    // Reuse shared memory for head RMS reduction:
    // shared_partials matches qwen3_head_rms_normalize reduction tree bit-for-bit
    float* shared_partials = shared_mem + 128;
    shared_partials[row] = rec_val * rec_val;
    __syncthreads();

    for (std::uint32_t stride = 64; stride > 0; stride /= 2) {
        if (row < stride) {
            shared_partials[row] += shared_partials[row + stride];
        }
        __syncthreads();
    }

    const float inverse_rms = rsqrtf(shared_partials[0] / 128.0F + epsilon);
    const float head_norm = rec_val * inverse_rms;
    const float head_scaled = head_norm * ssm_norm[row];
    const std::size_t out_idx = static_cast<std::size_t>(head) * 128 + row;
    const float g = gate[out_idx];
    const float silu_g = g / (1.0F + expf(-g));
    const float gated_val = silu_g * head_scaled;
    if (gated_output != nullptr) {
        gated_output[out_idx] = gated_val;
    }
    if (gated_output_q8 != nullptr) {
        const int lane = static_cast<int>(row % 64);
        const int wave_in_block = static_cast<int>(row / 64);
        const int block_in_wave = lane / 32;
        const int in_block = lane % 32;
        const int head_block_idx = wave_in_block * 2 + block_in_wave;
        const int global_block = static_cast<int>(head * (state_size / 32) + head_block_idx);

        float amax = fabsf(gated_val);
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            amax = fmaxf(amax, __shfl_xor(amax, offset, 64));
        }

        const float scale = amax / 127.0f;
        const float inv_scale = (scale != 0.0f) ? (1.0f / scale) : 0.0f;
        const int q_int = max(-127, min(127, static_cast<int>(roundf(gated_val * inv_scale))));

        int sum = q_int;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_xor(sum, offset, 64);
        }

        if (in_block == 0) {
            gated_output_q8[global_block].d = __float2half_rn(scale);
            gated_output_q8[global_block].s = __float2half_rn(static_cast<float>(sum) * scale);
        }
        gated_output_q8[global_block].qs[in_block] = static_cast<std::int8_t>(q_int);
    }
}

__global__ void qwen35_deltanet_fused_recurrent_core_batched4_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ decay,
    const float* __restrict__ ssm_norm,
    const float* __restrict__ gate,
    float* __restrict__ state,
    float* __restrict__ gated_output,
    std::uint32_t key_heads,
    std::uint32_t value_heads,
    std::uint32_t state_size,
    float epsilon,
    miinfer::Q8_1Block* __restrict__ gated_output_q8) {
    constexpr std::uint32_t batch = 4;
    const std::uint32_t head = blockIdx.x;
    const std::uint32_t row = threadIdx.x;
    if (head >= value_heads || row >= state_size || key_heads == 0) return;

    extern __shared__ float shared_mem[];
    float* shared_query = shared_mem;
    float* shared_key = shared_mem + batch * state_size;
    float* shared_partials = shared_mem + 2 * batch * state_size;
    const std::uint32_t key_head = head % key_heads;
    const std::size_t state_base = static_cast<std::size_t>(head) * state_size * state_size;
    float s[128];
    #pragma unroll 4
    for (std::uint32_t column = 0; column < 128; ++column) {
        s[column] = state[state_base + static_cast<std::size_t>(column) * 128 + row];
    }

    for (std::uint32_t token = 0; token < batch; ++token) {
        shared_query[token * state_size + row] =
            query[token * key_heads * state_size + key_head * state_size + row];
        shared_key[token * state_size + row] =
            key[token * key_heads * state_size + key_head * state_size + row];
        __syncthreads();

        const float beta_value = beta[token * value_heads + head];
        const float decay_value = decay[token * value_heads + head];
        float key_dot = 0.0F;
        #pragma unroll 4
        for (std::uint32_t column = 0; column < 128; ++column) {
            key_dot += s[column] * decay_value * shared_key[token * state_size + column];
        }
        const float delta = (value[token * value_heads * state_size
                                    + head * state_size + row] - key_dot) * beta_value;
        float query_dot = 0.0F;
        #pragma unroll 4
        for (std::uint32_t column = 0; column < 128; ++column) {
            const float updated = s[column] * decay_value
                + delta * shared_key[token * state_size + column];
            s[column] = updated;
            query_dot += updated * shared_query[token * state_size + column];
        }

        const float recurrent_value = query_dot * rsqrtf(128.0F);
        shared_partials[row] = recurrent_value * recurrent_value;
        __syncthreads();
        for (std::uint32_t stride = 64; stride > 0; stride /= 2) {
            if (row < stride) shared_partials[row] += shared_partials[row + stride];
            __syncthreads();
        }
        const float inverse_rms = rsqrtf(shared_partials[0] / 128.0F + epsilon);
        const float head_scaled = recurrent_value * inverse_rms * ssm_norm[row];
        const std::size_t output_index = static_cast<std::size_t>(token)
            * value_heads * state_size + static_cast<std::size_t>(head) * state_size + row;
        const float gate_value = gate[output_index];
        gated_output[output_index] =
            (gate_value / (1.0F + expf(-gate_value))) * head_scaled;
        if (gated_output_q8 != nullptr) {
            const int lane = static_cast<int>(row % 64);
            const int wave_in_block = static_cast<int>(row / 64);
            const int block_in_wave = lane / 32;
            const int in_block = lane % 32;
            const int head_block_idx = wave_in_block * 2 + block_in_wave;
            const int token_blocks = static_cast<int>(value_heads * (state_size / 32));
            const int global_block = static_cast<int>(token) * token_blocks
                + static_cast<int>(head) * (state_size / 32) + head_block_idx;
            float amax = fabsf(gated_output[output_index]);
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                amax = fmaxf(amax, __shfl_xor(amax, offset, 64));
            }
            const float scale = amax / 127.0f;
            const float inv_scale = scale != 0.0F ? 1.0F / scale : 0.0F;
            const int q_int = max(-127, min(127,
                static_cast<int>(roundf(gated_output[output_index] * inv_scale))));
            int sum = q_int;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                sum += __shfl_xor(sum, offset, 64);
            }
            if (in_block == 0) {
                gated_output_q8[global_block].d = __float2half_rn(scale);
                gated_output_q8[global_block].s =
                    __float2half_rn(static_cast<float>(sum) * scale);
            }
            gated_output_q8[global_block].qs[in_block] = static_cast<std::int8_t>(q_int);
        }
        __syncthreads();
    }

    #pragma unroll 4
    for (std::uint32_t column = 0; column < 128; ++column) {
        state[state_base + static_cast<std::size_t>(column) * 128 + row] = s[column];
    }
}

__global__ void qwen35_deltanet_prefill_stage_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ decay,
    const float* __restrict__ gate,
    float* __restrict__ query_batch,
    float* __restrict__ key_batch,
    float* __restrict__ value_batch,
    float* __restrict__ beta_batch,
    float* __restrict__ decay_batch,
    float* __restrict__ gate_batch,
    std::uint32_t token_index,
    std::uint32_t key_heads,
    std::uint32_t value_heads,
    std::uint32_t state_size) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t key_elements = key_heads * state_size;
    const std::uint32_t value_elements = value_heads * state_size;
    if (index < key_elements) {
        query_batch[token_index * key_elements + index] = query[index];
        key_batch[token_index * key_elements + index] = key[index];
    }
    if (index < value_elements) {
        value_batch[token_index * value_elements + index] = value[index];
        gate_batch[token_index * value_elements + index] = gate[index];
    }
    if (index < value_heads) {
        beta_batch[token_index * value_heads + index] = beta[index];
        decay_batch[token_index * value_heads + index] = decay[index];
    }
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

__global__ void qwen35_conv_silu_split_batch_kernel(
    const float* qkv, const float* conv_weights, float* history,
    float* query, float* key, float* value, std::uint32_t base_position,
    std::uint32_t token_count, std::uint32_t history_capacity,
    std::uint32_t channels, std::uint32_t conv_kernel) {
    const std::uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) return;
    const std::uint32_t q_size = channels / 5;
    for (std::uint32_t token = 0; token < token_count; ++token) {
        const std::uint32_t position = base_position + token;
        const float current = qkv[static_cast<std::size_t>(token) * channels + channel];
        float result = current * conv_weights[
            static_cast<std::size_t>(channel) * conv_kernel + conv_kernel - 1];
        for (std::uint32_t lag = 1; lag < conv_kernel; ++lag) {
            if (position < lag) continue;
            const std::uint32_t slot = (position - lag) % history_capacity;
            result += history[static_cast<std::size_t>(slot) * channels + channel]
                * conv_weights[static_cast<std::size_t>(channel) * conv_kernel
                               + conv_kernel - 1 - lag];
        }
        history[static_cast<std::size_t>(position % history_capacity) * channels + channel] = current;
        const float activated = result / (1.0F + expf(-result));
        if (channel < q_size) {
            query[static_cast<std::size_t>(token) * q_size + channel] = activated;
        } else if (channel < 2 * q_size) {
            key[static_cast<std::size_t>(token) * q_size + channel - q_size] = activated;
        } else {
            value[static_cast<std::size_t>(token) * (3 * q_size) + channel - 2 * q_size] = activated;
        }
    }
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

__global__ void qwen35_dual_head_l2_normalize_kernel(
    const float* input_a, const float* input_b, float* output_a, float* output_b,
    std::uint32_t heads, std::uint32_t head_dim) {
    const std::uint32_t head = blockIdx.x;
    if (head >= heads) return;
    __shared__ float partials_a[128];
    __shared__ float partials_b[128];
    const std::size_t base = static_cast<std::size_t>(head) * head_dim;
    float sum_a = 0.0F;
    float sum_b = 0.0F;
    for (std::uint32_t index = threadIdx.x; index < head_dim; index += blockDim.x) {
        const float a = input_a[base + index];
        const float b = input_b[base + index];
        sum_a += a * a;
        sum_b += b * b;
    }
    partials_a[threadIdx.x] = sum_a;
    partials_b[threadIdx.x] = sum_b;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partials_a[threadIdx.x] += partials_a[threadIdx.x + stride];
            partials_b[threadIdx.x] += partials_b[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float scale_a = rsqrtf(fmaxf(partials_a[0], 1.0e-6F));
    const float scale_b = rsqrtf(fmaxf(partials_b[0], 1.0e-6F));
    for (std::uint32_t index = threadIdx.x; index < head_dim; index += blockDim.x) {
        output_a[base + index] = input_a[base + index] * scale_a;
        output_b[base + index] = input_b[base + index] * scale_b;
    }
}

__global__ void qwen35_dual_head_l2_normalize_batch_kernel(
    const float* input_a, const float* input_b, float* output_a, float* output_b,
    std::uint32_t heads, std::uint32_t head_dim) {
    const std::uint32_t flat_head = blockIdx.x;
    const std::uint32_t head = flat_head % heads;
    const std::uint32_t token = flat_head / heads;
    const std::size_t base = (static_cast<std::size_t>(token) * heads + head) * head_dim;
    __shared__ float partials_a[128];
    __shared__ float partials_b[128];
    const std::uint32_t lane = threadIdx.x;
    const float a = lane < head_dim ? input_a[base + lane] : 0.0F;
    const float b = lane < head_dim ? input_b[base + lane] : 0.0F;
    partials_a[lane] = a * a;
    partials_b[lane] = b * b;
    __syncthreads();
    for (std::uint32_t stride = 64; stride > 0; stride /= 2) {
        if (lane < stride) {
            partials_a[lane] += partials_a[lane + stride];
            partials_b[lane] += partials_b[lane + stride];
        }
        __syncthreads();
    }
    if (lane < head_dim) {
        output_a[base + lane] = a * rsqrtf(fmaxf(partials_a[0], 1.0e-6F));
        output_b[base + lane] = b * rsqrtf(fmaxf(partials_b[0], 1.0e-6F));
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

constexpr std::uint32_t kMaxAttentionSplits = 64;
constexpr std::uint32_t kMaxAttentionHeads = 24;

__device__ static float s_attn_split_acc[kMaxAttentionSplits * kMaxAttentionHeads * 256];
__device__ static float s_attn_split_max[kMaxAttentionSplits * kMaxAttentionHeads];
__device__ static float s_attn_split_sum[kMaxAttentionSplits * kMaxAttentionHeads];

// Wave64 Barrier-Free Vectorized (float4) Split-K Kernel
// 1 Wave (64 threads) per workgroup: blockDim = 64. Zero LDS, Zero Barriers in inner loop!
__global__ void qwen3_wave64_splitk_stage1_f32_kernel(
    const float* __restrict__ q,
    const float* __restrict__ key_cache,
    const float* __restrict__ value_cache,
    std::uint32_t cache_length,
    std::uint32_t cache_capacity,
    std::uint32_t query_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    float scale,
    std::uint32_t num_splits) {

    const std::uint32_t head = blockIdx.x;
    const std::uint32_t split = blockIdx.y;
    const std::uint32_t lane = threadIdx.x; // 0 .. 63 (Wave64)
    if (head >= query_heads || lane >= 64) return;

    const std::uint32_t kv_head = head / (query_heads / kv_heads);
    const std::uint32_t q_base = head * head_dim;

    const std::uint32_t tokens_per_split = (cache_length + num_splits - 1) / num_splits;
    const std::uint32_t pos_start = split * tokens_per_split;
    const std::uint32_t pos_end = min(pos_start + tokens_per_split, cache_length);

    // Load Q vector once into VGPRs (128-bit vector load)
    const float4 q_vec = *reinterpret_cast<const float4*>(q + q_base + lane * 4);

    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    // Loop over tokens: 100% BARRIER FREE! ZERO LDS!
    for (std::uint32_t pos = pos_start; pos < pos_end; ++pos) {
        const std::size_t cache_base =
            (static_cast<std::size_t>(kv_head) * cache_capacity + pos) * head_dim;

        // 128-bit coalesced load of K
        const float4 k_vec = *reinterpret_cast<const float4*>(key_cache + cache_base + lane * 4);

        float prod = q_vec.x * k_vec.x + q_vec.y * k_vec.y + q_vec.z * k_vec.z + q_vec.w * k_vec.w;
        prod = qwen3_wave_sum(prod);

        float alpha = 1.0f;
        float weight = 0.0f;
        if (lane == 0) {
            const float dot = prod * scale;
            if (dot > running_max) {
                alpha = expf(running_max - dot);
                running_sum = running_sum * alpha + 1.0f;
                running_max = dot;
                weight = 1.0f;
            } else {
                weight = expf(dot - running_max);
                running_sum += weight;
            }
        }

        // Fast in-register wave broadcast (DPP/shfl)
        alpha = __shfl(alpha, 0, 64);
        weight = __shfl(weight, 0, 64);

        // 128-bit coalesced load of V
        const float4 v_vec = *reinterpret_cast<const float4*>(value_cache + cache_base + lane * 4);

        acc.x = acc.x * alpha + weight * v_vec.x;
        acc.y = acc.y * alpha + weight * v_vec.y;
        acc.z = acc.z * alpha + weight * v_vec.z;
        acc.w = acc.w * alpha + weight * v_vec.w;
    }

    const std::size_t split_head_idx = static_cast<std::size_t>(split) * query_heads + head;
    *reinterpret_cast<float4*>(s_attn_split_acc + split_head_idx * head_dim + lane * 4) = acc;

    if (lane == 0) {
        s_attn_split_max[split_head_idx] = running_max;
        s_attn_split_sum[split_head_idx] = running_sum;
    }
}

// Wave64 Barrier-Free Vectorized (half4) Split-K Kernel (FP16 KV Cache)
// 1 Wave (64 threads) per workgroup: blockDim = 64. Zero LDS, Zero Barriers in inner loop!
__global__ void qwen3_wave64_splitk_stage1_f16_kernel(
    const float* __restrict__ q,
    const __half* __restrict__ key_cache,
    const __half* __restrict__ value_cache,
    std::uint32_t cache_length,
    std::uint32_t cache_capacity,
    std::uint32_t query_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    float scale,
    std::uint32_t num_splits) {

    const std::uint32_t head = blockIdx.x;
    const std::uint32_t split = blockIdx.y;
    const std::uint32_t lane = threadIdx.x; // 0 .. 63 (Wave64)
    if (head >= query_heads || lane >= 64) return;

    const std::uint32_t kv_head = head / (query_heads / kv_heads);
    const std::uint32_t q_base = head * head_dim;

    const std::uint32_t tokens_per_split = (cache_length + num_splits - 1) / num_splits;
    const std::uint32_t pos_start = split * tokens_per_split;
    const std::uint32_t pos_end = min(pos_start + tokens_per_split, cache_length);

    // Load Q vector once into VGPRs
    const float4 q_vec = *reinterpret_cast<const float4*>(q + q_base + lane * 4);

    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    // Loop over tokens: 100% BARRIER FREE! ZERO LDS!
    for (std::uint32_t pos = pos_start; pos < pos_end; ++pos) {
        const std::size_t cache_base =
            (static_cast<std::size_t>(kv_head) * cache_capacity + pos) * head_dim;

        // 64-bit coalesced load of 4 halfs
        const auto k_raw = *reinterpret_cast<const uint64_t*>(key_cache + cache_base + lane * 4);
        const auto* k_half = reinterpret_cast<const __half*>(&k_raw);
        const float k0 = __half2float(k_half[0]);
        const float k1 = __half2float(k_half[1]);
        const float k2 = __half2float(k_half[2]);
        const float k3 = __half2float(k_half[3]);

        float prod = q_vec.x * k0 + q_vec.y * k1 + q_vec.z * k2 + q_vec.w * k3;
        prod = qwen3_wave_sum(prod);

        float alpha = 1.0f;
        float weight = 0.0f;
        if (lane == 0) {
            const float dot = prod * scale;
            if (dot > running_max) {
                alpha = expf(running_max - dot);
                running_sum = running_sum * alpha + 1.0f;
                running_max = dot;
                weight = 1.0f;
            } else {
                weight = expf(dot - running_max);
                running_sum += weight;
            }
        }

        // Fast in-register wave broadcast
        alpha = __shfl(alpha, 0, 64);
        weight = __shfl(weight, 0, 64);

        // 64-bit coalesced load of 4 halfs for V
        const auto v_raw = *reinterpret_cast<const uint64_t*>(value_cache + cache_base + lane * 4);
        const auto* v_half = reinterpret_cast<const __half*>(&v_raw);
        const float v0 = __half2float(v_half[0]);
        const float v1 = __half2float(v_half[1]);
        const float v2 = __half2float(v_half[2]);
        const float v3 = __half2float(v_half[3]);

        acc.x = acc.x * alpha + weight * v0;
        acc.y = acc.y * alpha + weight * v1;
        acc.z = acc.z * alpha + weight * v2;
        acc.w = acc.w * alpha + weight * v3;
    }

    const std::size_t split_head_idx = static_cast<std::size_t>(split) * query_heads + head;
    *reinterpret_cast<float4*>(s_attn_split_acc + split_head_idx * head_dim + lane * 4) = acc;

    if (lane == 0) {
        s_attn_split_max[split_head_idx] = running_max;
        s_attn_split_sum[split_head_idx] = running_sum;
    }
}

// Stage 2 reduction across splits
__global__ void qwen3_splitk_stage2_kernel(
    float* __restrict__ output,
    const float* __restrict__ gate,
    float* __restrict__ gated_output,
    std::uint32_t query_heads,
    std::uint32_t head_dim,
    std::uint32_t num_splits,
    miinfer::Q8_1Block* __restrict__ gated_output_q8) {

    const std::uint32_t head = blockIdx.x;
    const std::uint32_t tid = threadIdx.x;
    if (head >= query_heads || tid >= head_dim) return;

    __shared__ float s_global_sum;
    __shared__ float s_alphas[kMaxAttentionSplits];

    if (tid == 0) {
        float g_max = -INFINITY;
        for (std::uint32_t s = 0; s < num_splits; ++s) {
            const float m = s_attn_split_max[s * query_heads + head];
            g_max = fmaxf(g_max, m);
        }
        float g_sum = 0.0F;
        for (std::uint32_t s = 0; s < num_splits; ++s) {
            const float m = s_attn_split_max[s * query_heads + head];
            const float l = s_attn_split_sum[s * query_heads + head];
            const float a = expf(m - g_max);
            s_alphas[s] = a;
            g_sum += l * a;
        }
        s_global_sum = g_sum;
    }
    __syncthreads();

    float total_acc = 0.0F;
    for (std::uint32_t s = 0; s < num_splits; ++s) {
        const float a = s_alphas[s];
        const float partial = s_attn_split_acc[(s * query_heads + head) * head_dim + tid];
        total_acc += partial * a;
    }

    const float out_val = (s_global_sum > 0.0F) ? (total_acc / s_global_sum) : 0.0F;
    const std::uint32_t q_base = head * head_dim;
    if (output != nullptr) {
        output[q_base + tid] = out_val;
    }
    if (gate != nullptr) {
        const float g = gate[q_base + tid];
        const float sig = 1.0F / (1.0F + expf(-g));
        const float gated_val = out_val * sig;
        if (gated_output != nullptr) {
            gated_output[q_base + tid] = gated_val;
        }
        if (gated_output_q8 != nullptr) {
            const int lane = static_cast<int>(tid % 64);
            const int wave_in_block = static_cast<int>(tid / 64);
            const int block_in_wave = lane / 32;
            const int in_block = lane % 32;
            const int head_block_idx = wave_in_block * 2 + block_in_wave;
            const int global_block = static_cast<int>(head * (head_dim / 32) + head_block_idx);

            float amax = fabsf(gated_val);
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                amax = fmaxf(amax, __shfl_xor(amax, offset, 64));
            }

            const float scale = amax / 127.0f;
            const float inv_scale = (scale != 0.0f) ? (1.0f / scale) : 0.0f;
            const int q_int = max(-127, min(127, static_cast<int>(roundf(gated_val * inv_scale))));

            int sum = q_int;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                sum += __shfl_xor(sum, offset, 64);
            }

            if (in_block == 0) {
                gated_output_q8[global_block].d = __float2half_rn(scale);
                gated_output_q8[global_block].s = __float2half_rn(static_cast<float>(sum) * scale);
            }
            gated_output_q8[global_block].qs[in_block] = static_cast<std::int8_t>(q_int);
        }
    }
}

// M23 wide-prefill causal attention.  The cache already contains the complete
// chunk when this launches, so one wave can serve each token/query-head pair
// without dispatching the decode attention kernel once per token.
__global__ void qwen35_tiled_online_attention_batch_f32_kernel(
    const float* __restrict__ q,
    const float* __restrict__ key_cache,
    const float* __restrict__ value_cache,
    const float* __restrict__ gate,
    float* __restrict__ gated_output,
    std::uint32_t token_count,
    std::uint32_t base_position,
    std::uint32_t cache_capacity,
    std::uint32_t query_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    float scale) {
    const std::uint32_t pair = blockIdx.x;
    const std::uint32_t token = pair / query_heads;
    const std::uint32_t head = pair % query_heads;
    const std::uint32_t lane = threadIdx.x;
    if (token >= token_count || head >= query_heads || lane >= 64
        || kv_heads == 0 || query_heads % kv_heads != 0 || head_dim != 256) return;

    const std::uint32_t kv_head = head / (query_heads / kv_heads);
    const std::size_t q_base =
        (static_cast<std::size_t>(token) * query_heads + head) * head_dim;
    const std::uint32_t cache_length = base_position + token + 1;
    if (cache_length > cache_capacity) return;
    const float4 q_vec = *reinterpret_cast<const float4*>(q + q_base + lane * 4);
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    for (std::uint32_t position = 0; position < cache_length; ++position) {
        const std::size_t cache_base =
            (static_cast<std::size_t>(kv_head) * cache_capacity + position) * head_dim;
        const float4 k_vec = *reinterpret_cast<const float4*>(key_cache + cache_base + lane * 4);
        float score = qwen3_wave_sum(
            q_vec.x * k_vec.x + q_vec.y * k_vec.y + q_vec.z * k_vec.z + q_vec.w * k_vec.w);
        score = __shfl(score, 0, 64) * scale;
        float alpha = 1.0f;
        float weight = 0.0f;
        if (lane == 0) {
            if (score > running_max) {
                alpha = expf(running_max - score);
                running_sum = running_sum * alpha + 1.0f;
                running_max = score;
                weight = 1.0f;
            } else {
                weight = expf(score - running_max);
                running_sum += weight;
            }
        }
        alpha = __shfl(alpha, 0, 64);
        weight = __shfl(weight, 0, 64);
        const float4 v_vec = *reinterpret_cast<const float4*>(value_cache + cache_base + lane * 4);
        acc.x = acc.x * alpha + weight * v_vec.x;
        acc.y = acc.y * alpha + weight * v_vec.y;
        acc.z = acc.z * alpha + weight * v_vec.z;
        acc.w = acc.w * alpha + weight * v_vec.w;
    }
    const float denominator = __shfl(running_sum, 0, 64);
    const float4 gate_vec = *reinterpret_cast<const float4*>(gate + q_base + lane * 4);
    const std::size_t output_base = q_base;
    gated_output[output_base + lane * 4 + 0] = (acc.x / denominator)
        * (1.0f / (1.0f + expf(-gate_vec.x)));
    gated_output[output_base + lane * 4 + 1] = (acc.y / denominator)
        * (1.0f / (1.0f + expf(-gate_vec.y)));
    gated_output[output_base + lane * 4 + 2] = (acc.z / denominator)
        * (1.0f / (1.0f + expf(-gate_vec.z)));
    gated_output[output_base + lane * 4 + 3] = (acc.w / denominator)
        * (1.0f / (1.0f + expf(-gate_vec.w)));
}

__global__ void qwen35_tiled_online_attention_batch_f16_kernel(
    const float* __restrict__ q,
    const __half* __restrict__ key_cache,
    const __half* __restrict__ value_cache,
    const float* __restrict__ gate,
    float* __restrict__ gated_output,
    std::uint32_t token_count,
    std::uint32_t base_position,
    std::uint32_t cache_capacity,
    std::uint32_t query_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dim,
    float scale) {
    const std::uint32_t pair = blockIdx.x;
    const std::uint32_t token = pair / query_heads;
    const std::uint32_t head = pair % query_heads;
    const std::uint32_t lane = threadIdx.x;
    if (token >= token_count || head >= query_heads || lane >= 64
        || kv_heads == 0 || query_heads % kv_heads != 0 || head_dim != 256) return;

    const std::uint32_t kv_head = head / (query_heads / kv_heads);
    const std::size_t q_base =
        (static_cast<std::size_t>(token) * query_heads + head) * head_dim;
    const std::uint32_t cache_length = base_position + token + 1;
    if (cache_length > cache_capacity) return;
    const float4 q_vec = *reinterpret_cast<const float4*>(q + q_base + lane * 4);
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    for (std::uint32_t position = 0; position < cache_length; ++position) {
        const std::size_t cache_base =
            (static_cast<std::size_t>(kv_head) * cache_capacity + position) * head_dim;
        const auto k_raw = *reinterpret_cast<const uint64_t*>(key_cache + cache_base + lane * 4);
        const auto* k_half = reinterpret_cast<const __half*>(&k_raw);
        float score = qwen3_wave_sum(
            q_vec.x * __half2float(k_half[0]) + q_vec.y * __half2float(k_half[1])
            + q_vec.z * __half2float(k_half[2]) + q_vec.w * __half2float(k_half[3]));
        score = __shfl(score, 0, 64) * scale;
        float alpha = 1.0f;
        float weight = 0.0f;
        if (lane == 0) {
            if (score > running_max) {
                alpha = expf(running_max - score);
                running_sum = running_sum * alpha + 1.0f;
                running_max = score;
                weight = 1.0f;
            } else {
                weight = expf(score - running_max);
                running_sum += weight;
            }
        }
        alpha = __shfl(alpha, 0, 64);
        weight = __shfl(weight, 0, 64);
        const auto v_raw = *reinterpret_cast<const uint64_t*>(value_cache + cache_base + lane * 4);
        const auto* v_half = reinterpret_cast<const __half*>(&v_raw);
        acc.x = acc.x * alpha + weight * __half2float(v_half[0]);
        acc.y = acc.y * alpha + weight * __half2float(v_half[1]);
        acc.z = acc.z * alpha + weight * __half2float(v_half[2]);
        acc.w = acc.w * alpha + weight * __half2float(v_half[3]);
    }
    const float denominator = __shfl(running_sum, 0, 64);
    const float4 gate_vec = *reinterpret_cast<const float4*>(gate + q_base + lane * 4);
    const std::size_t output_base = q_base;
    gated_output[output_base + lane * 4 + 0] = (acc.x / denominator)
        * (1.0f / (1.0f + expf(-gate_vec.x)));
    gated_output[output_base + lane * 4 + 1] = (acc.y / denominator)
        * (1.0f / (1.0f + expf(-gate_vec.y)));
    gated_output[output_base + lane * 4 + 2] = (acc.z / denominator)
        * (1.0f / (1.0f + expf(-gate_vec.z)));
    gated_output[output_base + lane * 4 + 3] = (acc.w / denominator)
        * (1.0f / (1.0f + expf(-gate_vec.w)));
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

__global__ void qwen3_kv_cache_store_f16_kernel(
    const float* keys, const float* values, __half* key_cache, __half* value_cache,
    std::uint32_t position, std::uint32_t cache_capacity,
    std::uint32_t kv_heads, std::uint32_t head_dim) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t elements = kv_heads * head_dim;
    if (index >= elements || cache_capacity == 0 || position >= cache_capacity) return;
    const std::uint32_t head = index / head_dim;
    const std::uint32_t dimension = index % head_dim;
    const std::size_t cache_base =
        (static_cast<std::size_t>(head) * cache_capacity + position) * head_dim;
    key_cache[cache_base + dimension] = __float2half_rn(keys[index]);
    value_cache[cache_base + dimension] = __float2half_rn(values[index]);
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

__global__ void qwen35_f32_dual_gemv_kernel(
    const float* weights_a, const float* weights_b, const float* input,
    float* output_a, float* output_b, std::uint32_t rows, std::uint32_t columns) {
    const std::uint32_t row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float partials_a[256];
    __shared__ float partials_b[256];
    float sum_a = 0.0F;
    float sum_b = 0.0F;
    for (std::uint32_t column = threadIdx.x; column < columns; column += blockDim.x) {
        const float value = input[column];
        const std::size_t index = static_cast<std::size_t>(row) * columns + column;
        sum_a += weights_a[index] * value;
        sum_b += weights_b[index] * value;
    }
    partials_a[threadIdx.x] = sum_a;
    partials_b[threadIdx.x] = sum_b;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partials_a[threadIdx.x] += partials_a[threadIdx.x + stride];
            partials_b[threadIdx.x] += partials_b[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        output_a[row] = partials_a[0];
        output_b[row] = partials_b[0];
    }
}

__global__ void qwen35_f32_dual_gemm_batch_kernel(
    const float* weights_a, const float* weights_b, const float* input,
    float* output_a, float* output_b, std::uint32_t rows, std::uint32_t columns) {
    const std::uint32_t flat_row = blockIdx.x;
    const std::uint32_t row = flat_row % rows;
    const std::uint32_t token = flat_row / rows;
    __shared__ float partials_a[256];
    __shared__ float partials_b[256];
    float sum_a = 0.0F;
    float sum_b = 0.0F;
    for (std::uint32_t column = threadIdx.x; column < columns; column += blockDim.x) {
        const float value = input[static_cast<std::size_t>(token) * columns + column];
        const std::size_t index = static_cast<std::size_t>(row) * columns + column;
        sum_a += weights_a[index] * value;
        sum_b += weights_b[index] * value;
    }
    partials_a[threadIdx.x] = sum_a;
    partials_b[threadIdx.x] = sum_b;
    __syncthreads();
    for (std::uint32_t stride = 128; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partials_a[threadIdx.x] += partials_a[threadIdx.x + stride];
            partials_b[threadIdx.x] += partials_b[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        output_a[static_cast<std::size_t>(token) * rows + row] = partials_a[0];
        output_b[static_cast<std::size_t>(token) * rows + row] = partials_b[0];
    }
}

__global__ void qwen35_f32_dual_beta_decay_kernel(
    const float* beta_weights, const float* alpha_weights, const float* input,
    const float* dt, const float* a, float* beta, float* decay,
    std::uint32_t rows, std::uint32_t columns) {
    const std::uint32_t row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float partials_beta[256];
    __shared__ float partials_alpha[256];
    float sum_beta = 0.0F;
    float sum_alpha = 0.0F;
    for (std::uint32_t column = threadIdx.x; column < columns; column += blockDim.x) {
        const float value = input[column];
        const std::size_t index = static_cast<std::size_t>(row) * columns + column;
        sum_beta += beta_weights[index] * value;
        sum_alpha += alpha_weights[index] * value;
    }
    partials_beta[threadIdx.x] = sum_beta;
    partials_alpha[threadIdx.x] = sum_alpha;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partials_beta[threadIdx.x] += partials_beta[threadIdx.x + stride];
            partials_alpha[threadIdx.x] += partials_alpha[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float beta_raw = partials_beta[0];
        const float alpha_raw = partials_alpha[0];
        beta[row] = 1.0F / (1.0F + expf(-beta_raw));
        const float x = alpha_raw + dt[row];
        const float softplus = x <= 20.0F ? log1pf(expf(x)) : x;
        decay[row] = expf(softplus * a[row]);
    }
}

__global__ void qwen35_prepare_beta_decay_kernel(
    const float* beta_raw, const float* alpha_raw, const float* dt,
    const float* a, float* beta, float* decay, std::uint32_t elements) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    // Qwen3.5's recurrent beta/alpha projections have 48 heads.  `elements`
    // is token-major for wide prefill, while dt/a remain one head vector.
    constexpr std::uint32_t heads = 48;
    const std::uint32_t head = index % heads;
    beta[index] = 1.0F / (1.0F + expf(-beta_raw[index]));
    const float x = alpha_raw[index] + dt[head];
    const float softplus = x <= 20.0F ? log1pf(expf(x)) : x;
    decay[index] = expf(softplus * a[head]);
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

__device__ __forceinline__ float m23_q6_k_q8_1_dot(
    const Q6KDeviceBlock& quant, const miinfer::M23Q8_1MmqBlock* input,
    std::uint32_t token_count, std::uint32_t token, std::uint32_t weight_block, int iqs) {
    constexpr int qi6_half = 16;
    constexpr int qi6_quarter = 8;
    constexpr int qi8_1 = 8;
    const int q8_offset = 4 * (iqs / qi6_half) + (iqs % qi6_half) / qi8_1;
    const int scale_offset = qi6_quarter * (iqs / qi6_half) + (iqs % qi6_half) / 4;
    const int high_shift = 2 * ((iqs % qi6_half) / qi6_quarter);
    const std::uint32_t low = qwen3_load_b2(quant.ql, iqs);
    const std::uint32_t high = qwen3_load_b2(
        quant.qh, qi6_quarter * (iqs / qi6_half) + iqs % qi6_quarter) >> high_shift;
    float sum = 0.0F;
#pragma unroll
    for (int part = 0; part < 2; ++part) {
        const int q8_index = q8_offset + 2 * part;
        const auto& q8 = input[static_cast<std::size_t>(weight_block * 2 + q8_index / 4)
                               * token_count + token];
        const int q6 = qwen3_q6_mmvq_word(low, high, part);
        const int q8_values = static_cast<int>(*reinterpret_cast<const std::uint32_t*>(
            q8.qs + (q8_index % 4) * 32 + 4 * (iqs % qi8_1)));
        const float d8 = half_to_float(__float2half_rn(q8.d[q8_index % 4]));
        sum += d8 * static_cast<float>(
            __builtin_amdgcn_sdot4(q6, q8_values, 0, false)
            * quant.scales[scale_offset + 4 * part]);
    }
    return half_to_float(quant.d) * sum;
}

__global__ void m23_q6_k_q8_1_mmq_kernel(
    const Q6KDeviceBlock* weights, const miinfer::M23Q8_1MmqBlock* input,
    float* output, std::uint32_t rows, std::uint32_t columns, std::uint32_t token_count) {
    const std::uint32_t row_base = blockIdx.x * 64 + threadIdx.y * 16;
    const std::uint32_t token_base = blockIdx.y * 128 + threadIdx.x;
    const std::uint32_t blocks_per_row = columns / 256;
    float accum[16][2] = {};
#pragma unroll
    for (int row_offset = 0; row_offset < 16; ++row_offset) {
        const std::uint32_t row = row_base + row_offset;
        if (row >= rows) continue;
        const auto* row_weights = weights + static_cast<std::size_t>(row) * blocks_per_row;
#pragma unroll 1
        for (std::uint32_t block = 0; block < blocks_per_row; ++block) {
#pragma unroll
            for (int iqs = 0; iqs < 32; ++iqs) {
                if (token_base < token_count) {
                    accum[row_offset][0] += m23_q6_k_q8_1_dot(
                        row_weights[block], input, token_count, token_base, block, iqs);
                }
                if (token_base + 64 < token_count) {
                    accum[row_offset][1] += m23_q6_k_q8_1_dot(
                        row_weights[block], input, token_count, token_base + 64, block, iqs);
                }
            }
        }
        if (token_base < token_count) output[static_cast<std::size_t>(token_base) * rows + row] = accum[row_offset][0];
        if (token_base + 64 < token_count) output[static_cast<std::size_t>(token_base + 64) * rows + row] = accum[row_offset][1];
    }
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

__device__ __forceinline__ float m23_q5_k_q8_1_dot(
    const Q5KDeviceBlock& quant, const miinfer::M23Q8_1MmqBlock* input,
    std::uint32_t token_count, std::uint32_t token, std::uint32_t weight_block, int iqs) {
    const int q8_offset = 2 * ((iqs / 2) / 4);
    const int ql_offset = 16 * q8_offset + 4 * ((iqs / 2) % 4);
    const auto* ql = reinterpret_cast<const int*>(quant.ql + ql_offset);
    const auto* qh = reinterpret_cast<const int*>(quant.qh + 4 * ((iqs / 2) % 4));
    const int vl[2] = {ql[0], ql[4]};
    const int vh[2] = {qh[0] >> q8_offset, qh[4] >> q8_offset};
    const int group = 2 * (iqs / 8);
    int scales[2], minimums[2];
    q5_k_scale_min(quant, group, scales[0], minimums[0]);
    q5_k_scale_min(quant, group + 1, scales[1], minimums[1]);
    float sum_d = 0.0F, sum_min = 0.0F;
#pragma unroll
    for (int part = 0; part < 2; ++part) {
        const int q8_index = q8_offset + part;
        const auto& q8 = input[static_cast<std::size_t>(weight_block * 2 + q8_index / 4)
                               * token_count + token];
        const auto* q8_words = reinterpret_cast<const int*>(q8.qs + (q8_index % 4) * 32)
                               + ((iqs / 2) % 4);
        const int q0 = ((vl[0] >> (4 * part)) & 0x0F0F0F0F)
                       | (((vh[0] >> part) << 4) & 0x10101010);
        const int q1 = ((vl[1] >> (4 * part)) & 0x0F0F0F0F)
                       | (((vh[1] >> part) << 4) & 0x10101010);
        const int dot = __builtin_amdgcn_sdot4(
            q0, q8_words[0], __builtin_amdgcn_sdot4(q1, q8_words[4], 0, false), false);
        const int q8_sum = __builtin_amdgcn_sdot4(
            0x01010101, q8_words[0],
            __builtin_amdgcn_sdot4(0x01010101, q8_words[4], 0, false), false);
        const float d8 = half_to_float(__float2half_rn(q8.d[q8_index % 4]));
        sum_d += d8 * static_cast<float>(dot * scales[part]);
        sum_min += d8 * static_cast<float>(q8_sum * minimums[part]);
    }
    return half_to_float(quant.d) * sum_d - half_to_float(quant.dmin) * sum_min;
}

__global__ void m23_q5_k_q8_1_mmq_kernel(
    const Q5KDeviceBlock* weights, const miinfer::M23Q8_1MmqBlock* input,
    float* output, std::uint32_t rows, std::uint32_t columns, std::uint32_t token_count) {
    const std::uint32_t row_base = blockIdx.x * 64 + threadIdx.y * 16;
    const std::uint32_t token_base = blockIdx.y * 128 + threadIdx.x;
    const std::uint32_t blocks_per_row = columns / 256;
    float accum[16][2] = {};
#pragma unroll
    for (int row_offset = 0; row_offset < 16; ++row_offset) {
        const std::uint32_t row = row_base + row_offset;
        if (row >= rows) continue;
        const auto* row_weights = weights + static_cast<std::size_t>(row) * blocks_per_row;
#pragma unroll 1
        for (std::uint32_t block = 0; block < blocks_per_row; ++block) {
#pragma unroll
            for (int iqs = 0; iqs < 32; iqs += 2) {
                if (token_base < token_count) accum[row_offset][0] += m23_q5_k_q8_1_dot(
                    row_weights[block], input, token_count, token_base, block, iqs);
                if (token_base + 64 < token_count) accum[row_offset][1] += m23_q5_k_q8_1_dot(
                    row_weights[block], input, token_count, token_base + 64, block, iqs);
            }
        }
        if (token_base < token_count) output[static_cast<std::size_t>(token_base) * rows + row] = accum[row_offset][0];
        if (token_base + 64 < token_count) output[static_cast<std::size_t>(token_base + 64) * rows + row] = accum[row_offset][1];
    }
}

__device__ __forceinline__ float qwen3_q4_k_q8_1_mmvq_dot_impl(
    const Q4KDeviceBlock& quant, __half d, __half dmin, const std::uint8_t* scale_bytes,
    const Q8_1Block* input, int iqs) {
    const int q8_offset = 2 * ((iqs / 2) / 4);
    const int ql_offset = 16 * q8_offset + 4 * ((iqs / 2) % 4);
    const auto* ql = reinterpret_cast<const int*>(quant.qs + ql_offset);
    const int vl[2] = {ql[0], ql[4]};

    int minimums[2];
    const int group = 2 * (iqs / 8);
    int scale_values[2];
    q4_k_scale_min(scale_bytes, group, scale_values[0], minimums[0]);
    q4_k_scale_min(scale_bytes, group + 1, scale_values[1], minimums[1]);

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
        sum_d += d8 * static_cast<float>(dot * scale_values[part]);
        sum_min += d8 * static_cast<float>(q8_sum * minimums[part]);
    }
    return half_to_float(d) * sum_d - half_to_float(dmin) * sum_min;
}

__device__ __forceinline__ float qwen3_q4_k_q8_1_mmvq_dot(
    const Q4KDeviceBlock& quant, const Q8_1Block* input, int iqs) {
    return qwen3_q4_k_q8_1_mmvq_dot_impl(
        quant, quant.d, quant.dmin, quant.scales, input, iqs);
}

__device__ __forceinline__ float qwen3_q4_k_q8_1_mmvq_dot_metadata(
    const Q4KDeviceBlock& quant, const Q4KMetadata& metadata,
    const Q8_1Block* input, int iqs) {
    return qwen3_q4_k_q8_1_mmvq_dot_impl(
        quant, metadata.d, metadata.dmin, metadata.scales, input, iqs);
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

__global__ void qwen3_q4_k_q8_1_mmvq_lds_input_kernel(
    const Q4KDeviceBlock* weights,
    const Q8_1Block* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns) {
    extern __shared__ Q8_1Block input_tile[];
    const std::uint32_t thread = threadIdx.x;
    const std::uint32_t block_count = columns / 32;
    for (std::uint32_t block = thread; block < block_count; block += 256) {
        input_tile[block] = input[block];
    }
    __syncthreads();

    const std::uint32_t row_in_group = thread / 128;
    const std::uint32_t tid = thread % 128;
    const std::uint32_t row = blockIdx.x * 2 + row_in_group;
    if (row >= rows) return;

    __shared__ float partials[2][128];
    const std::uint32_t blocks_per_row = columns / 256;
    const auto* row_weights = weights + static_cast<std::size_t>(row) * blocks_per_row;
    float sum = 0.0F;
    const int iqs = 2 * static_cast<int>(tid % 16);
    for (std::uint32_t block = tid / 16; block < blocks_per_row; block += 8) {
        sum += qwen3_q4_k_q8_1_mmvq_dot(row_weights[block], input_tile + block * 8, iqs);
    }
    partials[row_in_group][tid] = sum;
    __syncthreads();
    if (tid < 64) partials[row_in_group][tid] += partials[row_in_group][tid + 64];
    __syncthreads();
    for (std::uint32_t stride = 32; stride > 0; stride /= 2) {
        if (tid < stride) partials[row_in_group][tid] += partials[row_in_group][tid + stride];
        __syncthreads();
    }
    if (tid == 0) output[row] = partials[row_in_group][0];
}

__global__ void qwen3_q4_k_q8_1_mmvq_lds_metadata_kernel(
    const Q4KDeviceBlock* weights,
    const Q8_1Block* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns) {
    extern __shared__ unsigned char shared[];
    const std::uint32_t thread = threadIdx.x;
    const std::uint32_t row_in_group = thread / 128;
    const std::uint32_t tid = thread % 128;
    const std::uint32_t blocks_per_row = columns / 256;
    const std::uint32_t input_blocks = columns / 32;
    auto* input_tile = reinterpret_cast<Q8_1Block*>(shared);
    auto* metadata_tile = reinterpret_cast<Q4KMetadata*>(
        shared + static_cast<std::size_t>(input_blocks) * sizeof(Q8_1Block));

    for (std::uint32_t block = thread; block < input_blocks; block += 256) {
        input_tile[block] = input[block];
    }
    const std::uint32_t row = blockIdx.x * 2 + row_in_group;
    if (row < rows) {
        const auto* row_weights = weights + static_cast<std::size_t>(row) * blocks_per_row;
        auto* row_metadata = metadata_tile + static_cast<std::size_t>(row_in_group) * blocks_per_row;
        for (std::uint32_t block = tid; block < blocks_per_row; block += 128) {
            row_metadata[block].d = row_weights[block].d;
            row_metadata[block].dmin = row_weights[block].dmin;
            for (std::uint32_t byte = 0; byte < 12; ++byte) {
                row_metadata[block].scales[byte] = row_weights[block].scales[byte];
            }
        }
    }
    __syncthreads();
    if (row >= rows) return;

    __shared__ float partials[2][128];
    const auto* row_weights = weights + static_cast<std::size_t>(row) * blocks_per_row;
    const auto* row_metadata = metadata_tile
        + static_cast<std::size_t>(row_in_group) * blocks_per_row;
    float sum = 0.0F;
    const int iqs = 2 * static_cast<int>(tid % 16);
    for (std::uint32_t block = tid / 16; block < blocks_per_row; block += 8) {
        sum += qwen3_q4_k_q8_1_mmvq_dot_metadata(
            row_weights[block], row_metadata[block], input_tile + block * 8, iqs);
    }
    partials[row_in_group][tid] = sum;
    __syncthreads();
    if (tid < 64) partials[row_in_group][tid] += partials[row_in_group][tid + 64];
    __syncthreads();
    for (std::uint32_t stride = 32; stride > 0; stride /= 2) {
        if (tid < stride) partials[row_in_group][tid] += partials[row_in_group][tid + stride];
        __syncthreads();
    }
    if (tid == 0) output[row] = partials[row_in_group][0];
}

__device__ __forceinline__ float qwen3_q4_k_q8_1_mmvq_dot_decoded_metadata(
    const Q4KDeviceBlock& quant, const Q4KDecodedMetadata& metadata,
    const Q8_1Block* input, int iqs) {
    const int q8_offset = 2 * ((iqs / 2) / 4);
    const int ql_offset = 16 * q8_offset + 4 * ((iqs / 2) % 4);
    const auto* ql = reinterpret_cast<const int*>(quant.qs + ql_offset);
    const int vl[2] = {ql[0], ql[4]};
    const int group = 2 * (iqs / 8);
    const int scale_values[2] = {metadata.scales[group], metadata.scales[group + 1]};
    const int minimum_values[2] = {
        metadata.minimums[group], metadata.minimums[group + 1]};

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
        sum_d += d8 * static_cast<float>(dot * scale_values[part]);
        sum_min += d8 * static_cast<float>(q8_sum * minimum_values[part]);
    }
    return half_to_float(metadata.d) * sum_d - half_to_float(metadata.dmin) * sum_min;
}

__global__ void qwen3_q4_k_q8_1_mmvq_lds_decoded_metadata_kernel(
    const Q4KDeviceBlock* weights,
    const Q8_1Block* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns) {
    extern __shared__ unsigned char shared[];
    const std::uint32_t thread = threadIdx.x;
    const std::uint32_t row_in_group = thread / 128;
    const std::uint32_t tid = thread % 128;
    const std::uint32_t blocks_per_row = columns / 256;
    const std::uint32_t input_blocks = columns / 32;
    auto* input_tile = reinterpret_cast<Q8_1Block*>(shared);
    auto* metadata_tile = reinterpret_cast<Q4KDecodedMetadata*>(
        shared + static_cast<std::size_t>(input_blocks) * sizeof(Q8_1Block));

    for (std::uint32_t block = thread; block < input_blocks; block += 256) {
        input_tile[block] = input[block];
    }
    const std::uint32_t row = blockIdx.x * 2 + row_in_group;
    if (row < rows) {
        const auto* row_weights = weights + static_cast<std::size_t>(row) * blocks_per_row;
        auto* row_metadata = metadata_tile + static_cast<std::size_t>(row_in_group) * blocks_per_row;
        for (std::uint32_t block = tid; block < blocks_per_row; block += 128) {
            row_metadata[block].d = row_weights[block].d;
            row_metadata[block].dmin = row_weights[block].dmin;
            for (int group = 0; group < 8; ++group) {
                int scale = 0;
                int minimum = 0;
                q4_k_scale_min(row_weights[block], group, scale, minimum);
                row_metadata[block].scales[group] = static_cast<std::uint8_t>(scale);
                row_metadata[block].minimums[group] = static_cast<std::uint8_t>(minimum);
            }
        }
    }
    __syncthreads();
    if (row >= rows) return;

    __shared__ float partials[2][128];
    const auto* row_weights = weights + static_cast<std::size_t>(row) * blocks_per_row;
    const auto* row_metadata = metadata_tile
        + static_cast<std::size_t>(row_in_group) * blocks_per_row;
    float sum = 0.0F;
    const int iqs = 2 * static_cast<int>(tid % 16);
    for (std::uint32_t block = tid / 16; block < blocks_per_row; block += 8) {
        sum += qwen3_q4_k_q8_1_mmvq_dot_decoded_metadata(
            row_weights[block], row_metadata[block], input_tile + block * 8, iqs);
    }
    partials[row_in_group][tid] = sum;
    __syncthreads();
    if (tid < 64) partials[row_in_group][tid] += partials[row_in_group][tid + 64];
    __syncthreads();
    for (std::uint32_t stride = 32; stride > 0; stride /= 2) {
        if (tid < stride) partials[row_in_group][tid] += partials[row_in_group][tid + stride];
        __syncthreads();
    }
    if (tid == 0) output[row] = partials[row_in_group][0];
}

__device__ __forceinline__ float qwen3_q4_k_q8_1_mmvq_dot_expanded(
    const Q4KExpandedDeviceBlock& quant, const Q8_1Block* input, int iqs) {
    const int q8_offset = 2 * ((iqs / 2) / 4);
    const int ql_offset = 16 * q8_offset + 4 * ((iqs / 2) % 4);
    const int q8_word_offset = (iqs / 2) % 4;
    const int group = 2 * (iqs / 8);
    float sum_d = 0.0F;
    float sum_min = 0.0F;
#pragma unroll
    for (int part = 0; part < 2; ++part) {
        const auto& q8 = input[q8_offset + part];
        const auto* q8_words = reinterpret_cast<const int*>(q8.qs) + q8_word_offset;
        int q0 = 0;
        int q1 = 0;
#pragma unroll
        for (int byte = 0; byte < 4; ++byte) {
            q0 |= static_cast<int>(quant.qs[2 * (ql_offset + byte) + part]) << (8 * byte);
            q1 |= static_cast<int>(quant.qs[2 * (ql_offset + 16 + byte) + part]) << (8 * byte);
        }
        const int dot = qwen3_dot4_i8(q0, q8_words[0],
                                      qwen3_dot4_i8(q1, q8_words[4], 0));
        const int q8_sum = qwen3_dot4_i8(0x01010101, q8_words[0],
                                         qwen3_dot4_i8(0x01010101, q8_words[4], 0));
        const float d8 = half_to_float(q8.d);
        sum_d += d8 * static_cast<float>(dot * quant.scales[group + part]);
        sum_min += d8 * static_cast<float>(q8_sum * quant.minimums[group + part]);
    }
    return half_to_float(quant.d) * sum_d - half_to_float(quant.dmin) * sum_min;
}

__global__ void qwen3_q4_k_q8_1_mmvq_expanded_kernel(
    const Q4KExpandedDeviceBlock* weights,
    const Q8_1Block* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns) {
    const std::uint32_t thread = threadIdx.x;
    const std::uint32_t row_in_group = thread / 128;
    const std::uint32_t tid = thread % 128;
    const std::uint32_t row = blockIdx.x * 2 + row_in_group;
    if (row >= rows) return;
    __shared__ float partials[2][128];
    const std::uint32_t blocks_per_row = columns / 256;
    const auto* row_weights = weights + static_cast<std::size_t>(row) * blocks_per_row;
    float sum = 0.0F;
    const int iqs = 2 * static_cast<int>(tid % 16);
    for (std::uint32_t block = tid / 16; block < blocks_per_row; block += 8) {
        sum += qwen3_q4_k_q8_1_mmvq_dot_expanded(row_weights[block], input + block * 8, iqs);
    }
    partials[row_in_group][tid] = sum;
    __syncthreads();
    if (tid < 64) partials[row_in_group][tid] += partials[row_in_group][tid + 64];
    __syncthreads();
    for (std::uint32_t stride = 32; stride > 0; stride /= 2) {
        if (tid < stride) partials[row_in_group][tid] += partials[row_in_group][tid + stride];
        __syncthreads();
    }
    if (tid == 0) output[row] = partials[row_in_group][0];
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

__device__ static float s_argmax_partial_vals[256];
__device__ static std::uint32_t s_argmax_partial_idxs[256];

__global__ void qwen3_argmax_stage1_kernel(
    const float* __restrict__ input,
    std::uint32_t elements) {
    __shared__ float s_vals[256];
    __shared__ std::uint32_t s_idxs[256];

    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t total_threads = gridDim.x * blockDim.x;
    const std::uint32_t global_tid = blockIdx.x * blockDim.x + tid;

    float best_val = -INFINITY;
    std::uint32_t best_idx = elements;

    for (std::uint32_t idx = global_tid; idx < elements; idx += total_threads) {
        const float val = input[idx];
        if (qwen3_argmax_wins(val, idx, best_val, best_idx)) {
            best_val = val;
            best_idx = idx;
        }
    }

    s_vals[tid] = best_val;
    s_idxs[tid] = best_idx;
    __syncthreads();

    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride && qwen3_argmax_wins(s_vals[tid + stride], s_idxs[tid + stride], s_vals[tid], s_idxs[tid])) {
            s_vals[tid] = s_vals[tid + stride];
            s_idxs[tid] = s_idxs[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        s_argmax_partial_vals[blockIdx.x] = s_vals[0];
        s_argmax_partial_idxs[blockIdx.x] = s_idxs[0];
    }
}

__global__ void qwen3_argmax_stage2_kernel(
    std::uint32_t* __restrict__ output,
    std::uint32_t num_partials) {
    __shared__ float s_vals[256];
    __shared__ std::uint32_t s_idxs[256];

    const std::uint32_t tid = threadIdx.x;
    float best_val = -INFINITY;
    std::uint32_t best_idx = 0xFFFFFFFF;

    if (tid < num_partials) {
        best_val = s_argmax_partial_vals[tid];
        best_idx = s_argmax_partial_idxs[tid];
    }

    s_vals[tid] = best_val;
    s_idxs[tid] = best_idx;
    __syncthreads();

    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride && qwen3_argmax_wins(s_vals[tid + stride], s_idxs[tid + stride], s_vals[tid], s_idxs[tid])) {
            s_vals[tid] = s_vals[tid + stride];
            s_idxs[tid] = s_idxs[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) output[0] = s_idxs[0];
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

void launch_qwen35_q4_k_embedding_device_token(
    const Q4KDeviceBlock* weights,
    const std::uint32_t* token_ptr,
    std::uint32_t vocabulary,
    std::uint32_t hidden_size,
    float* output,
    hipStream_t stream) {
    if (weights == nullptr || token_ptr == nullptr || output == nullptr || vocabulary == 0 || hidden_size == 0
        || hidden_size % 256 != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen35_q4_k_embedding_device_token_kernel,
                       dim3((hidden_size + 255) / 256), dim3(256), 0, stream,
                       weights, token_ptr, vocabulary, hidden_size, output);
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
    if (elements == 5120 && (reinterpret_cast<std::uintptr_t>(input) % 16 == 0)
        && (reinterpret_cast<std::uintptr_t>(weights) % 16 == 0)
        && (reinterpret_cast<std::uintptr_t>(output) % 16 == 0)) {
        hipLaunchKernelGGL(qwen3_rms_norm_vec4_kernel, dim3(1), dim3(256), 0, stream,
                           input, weights, output, elements, epsilon);
    } else {
        hipLaunchKernelGGL(qwen3_rms_norm_kernel, dim3(1), dim3(256), 0, stream,
                           input, weights, output, elements, epsilon);
    }
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen3_rms_norm_batch(
    const float* input, const float* weights, float* output, std::uint32_t token_count,
    std::uint32_t elements, float epsilon, hipStream_t stream) {
    if (input == nullptr || weights == nullptr || output == nullptr || token_count == 0
        || token_count > 512 || elements == 0 || elements % 1024 != 0 || epsilon <= 0.0F) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen3_rms_norm_batch_vec4_kernel, dim3(token_count), dim3(256), 0, stream,
                       input, weights, output, elements, epsilon);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_m23_q8_1_mmq_quantize(
    const float* input, M23Q8_1MmqBlock* output, std::uint32_t token_count,
    std::uint32_t elements, hipStream_t stream) {
    if (input == nullptr || output == nullptr || token_count == 0 || token_count > 512
        || elements == 0 || elements % 128 != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(m23_q8_1_mmq_quantize_kernel,
                       dim3(elements / 128, token_count), dim3(128), 0, stream,
                       input, output, elements);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_m23_q4_k_q8_1_mmq(
    const Q4KDeviceBlock* weights, const M23Q8_1MmqBlock* input, float* output,
    std::uint32_t rows, std::uint32_t columns, std::uint32_t token_count, hipStream_t stream) {
    if (weights == nullptr || input == nullptr || output == nullptr || rows == 0 || columns == 0
        || columns % 256 != 0 || token_count == 0 || token_count > 512) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(m23_q4_k_q8_1_mmq_kernel,
                       dim3((rows + 63) / 64, (token_count + 127) / 128), dim3(64, 4), 0, stream,
                       weights, input, output, rows, columns, token_count);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_m23_q6_k_q8_1_mmq(
    const Q6KDeviceBlock* weights, const M23Q8_1MmqBlock* input, float* output,
    std::uint32_t rows, std::uint32_t columns, std::uint32_t token_count, hipStream_t stream) {
    if (weights == nullptr || input == nullptr || output == nullptr || rows == 0 || columns == 0
        || columns % 256 != 0 || token_count == 0 || token_count > 512) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(m23_q6_k_q8_1_mmq_kernel,
                       dim3((rows + 63) / 64, (token_count + 127) / 128), dim3(64, 4), 0, stream,
                       weights, input, output, rows, columns, token_count);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_m23_q5_k_q8_1_mmq(
    const Q5KDeviceBlock* weights, const M23Q8_1MmqBlock* input, float* output,
    std::uint32_t rows, std::uint32_t columns, std::uint32_t token_count, hipStream_t stream) {
    if (weights == nullptr || input == nullptr || output == nullptr || rows == 0 || columns == 0
        || columns % 256 != 0 || token_count == 0 || token_count > 512) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(m23_q5_k_q8_1_mmq_kernel,
                       dim3((rows + 63) / 64, (token_count + 127) / 128), dim3(64, 4), 0, stream,
                       weights, input, output, rows, columns, token_count);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen3_fused_add_rms_norm(
    const float* residual_in,
    const float* projection,
    const float* weights,
    float* residual_out,
    float* normalized_out,
    std::uint32_t elements,
    float epsilon,
    hipStream_t stream,
    Q8_1Block* q8_out) {
    if (residual_in == nullptr || projection == nullptr || weights == nullptr
        || residual_out == nullptr || (normalized_out == nullptr && q8_out == nullptr) || elements == 0
        || elements > 256U * 1024U || epsilon <= 0.0F) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    if (elements == 5120 && (reinterpret_cast<std::uintptr_t>(residual_in) % 16 == 0)
        && (reinterpret_cast<std::uintptr_t>(projection) % 16 == 0)
        && (reinterpret_cast<std::uintptr_t>(weights) % 16 == 0)
        && (residual_out == nullptr || reinterpret_cast<std::uintptr_t>(residual_out) % 16 == 0)
        && (normalized_out == nullptr || reinterpret_cast<std::uintptr_t>(normalized_out) % 16 == 0)) {
        if (q8_out != nullptr) {
            hipLaunchKernelGGL(qwen3_fused_add_rms_norm_vec4_kernel<true>, dim3(1), dim3(256), 0, stream,
                               residual_in, projection, weights, residual_out, normalized_out, q8_out, elements, epsilon);
        } else {
            hipLaunchKernelGGL(qwen3_fused_add_rms_norm_vec4_kernel<false>, dim3(1), dim3(256), 0, stream,
                               residual_in, projection, weights, residual_out, normalized_out, nullptr, elements, epsilon);
        }
    } else {
        if (q8_out != nullptr) {
            throw std::runtime_error("launch_qwen3_fused_add_rms_norm with q8_out requires elements == 5120 and 16-byte alignment");
        }
        hipLaunchKernelGGL(qwen3_fused_add_rms_norm_kernel, dim3(1), dim3(256), 0, stream,
                           residual_in, projection, weights, residual_out, normalized_out, elements, epsilon);
    }
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen3_fused_add_rms_norm_batch(
    const float* residual_in, const float* projection, const float* weights,
    float* residual_out, float* normalized_out, std::uint32_t token_count,
    std::uint32_t elements, float epsilon, hipStream_t stream) {
    if (residual_in == nullptr || projection == nullptr || weights == nullptr
        || residual_out == nullptr || normalized_out == nullptr || token_count == 0
        || token_count > 512 || elements == 0 || elements > 256U * 1024U || epsilon <= 0.0F) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen3_fused_add_rms_norm_batch_kernel, dim3(token_count), dim3(256), 0, stream,
                       residual_in, projection, weights, residual_out, normalized_out, elements, epsilon);
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

void launch_qwen35_fused_q_split_norm_rope(
    const float* qfull,
    const float* q_norm_weight,
    float* query_rope,
    float* gate,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t position,
    float theta,
    float epsilon,
    hipStream_t stream) {
    if (qfull == nullptr || q_norm_weight == nullptr || query_rope == nullptr || gate == nullptr
        || heads == 0 || head_dim == 0 || head_dim > 256 || theta <= 0.0F || epsilon <= 0.0F) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen35_fused_q_split_norm_rope_kernel,
                       dim3(heads), dim3(head_dim), 0, stream,
                       qfull, q_norm_weight, query_rope, gate,
                       heads, head_dim, position, theta, epsilon);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen35_fused_k_norm_rope_kv_store(
    const float* key,
    const float* value,
    const float* k_norm_weight,
    float* key_cache,
    float* value_cache,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t position,
    std::uint32_t cache_capacity,
    float theta,
    float epsilon,
    hipStream_t stream) {
    if (key == nullptr || value == nullptr || k_norm_weight == nullptr
        || key_cache == nullptr || value_cache == nullptr
        || heads == 0 || head_dim == 0 || head_dim > 256 || cache_capacity == 0
        || position >= cache_capacity || theta <= 0.0F || epsilon <= 0.0F) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen35_fused_k_norm_rope_kv_store_kernel,
                       dim3(heads), dim3(head_dim), 0, stream,
                       key, value, k_norm_weight, key_cache, value_cache,
                       heads, head_dim, position, cache_capacity, theta, epsilon);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen35_fused_k_norm_rope_kv_store_f16(
    const float* key,
    const float* value,
    const float* k_norm_weight,
    __half* key_cache,
    __half* value_cache,
    std::uint32_t heads,
    std::uint32_t head_dim,
    std::uint32_t position,
    std::uint32_t cache_capacity,
    float theta,
    float epsilon,
    hipStream_t stream) {
    if (key == nullptr || value == nullptr || k_norm_weight == nullptr
        || key_cache == nullptr || value_cache == nullptr
        || heads == 0 || head_dim == 0 || head_dim > 256 || cache_capacity == 0
        || position >= cache_capacity || theta <= 0.0F || epsilon <= 0.0F) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen35_fused_k_norm_rope_kv_store_f16_kernel,
                       dim3(heads), dim3(head_dim), 0, stream,
                       key, value, k_norm_weight, key_cache, value_cache,
                       heads, head_dim, position, cache_capacity, theta, epsilon);
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

void launch_qwen35_deltanet_state_update_transposed_no_decay_store(
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
    hipLaunchKernelGGL(qwen35_deltanet_state_update_transposed_no_decay_store_kernel,
                       dim3(value_heads), dim3(state_size), 0, stream,
                       query, key, value, beta, decay, state, output,
                       key_heads, value_heads, state_size);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen35_deltanet_state_update_transposed_no_decay_store_lds_inputs(
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
    hipLaunchKernelGGL(qwen35_deltanet_state_update_transposed_no_decay_store_lds_inputs_kernel,
                       dim3(value_heads), dim3(state_size),
                       static_cast<std::size_t>(2) * state_size * sizeof(float), stream,
                       query, key, value, beta, decay, state, output,
                       key_heads, value_heads, state_size);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen35_deltanet_state_update_transposed_fused_beta_alpha(
    const float* query, const float* key, const float* value,
    const float* beta_raw, const float* alpha_raw, const float* dt,
    const float* a, float* state, float* output,
    std::uint32_t key_heads, std::uint32_t value_heads,
    std::uint32_t state_size, hipStream_t stream) {
    if (query == nullptr || key == nullptr || value == nullptr || beta_raw == nullptr
        || alpha_raw == nullptr || dt == nullptr || a == nullptr || state == nullptr
        || output == nullptr || key_heads == 0 || value_heads == 0 || state_size == 0
        || state_size > 1024 || value_heads % key_heads != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen35_deltanet_state_update_transposed_fused_beta_alpha_kernel,
                       dim3(value_heads), dim3(state_size),
                       (static_cast<std::size_t>(2) * state_size + 2) * sizeof(float),
                       stream, query, key, value, beta_raw, alpha_raw, dt, a,
                       state, output, key_heads, value_heads, state_size);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen35_deltanet_state_update_transposed_row_waves(
    const float* query, const float* key, const float* value,
    const float* beta, const float* decay, float* state, float* output,
    std::uint32_t key_heads, std::uint32_t value_heads,
    std::uint32_t state_size, hipStream_t stream) {
    if (query == nullptr || key == nullptr || value == nullptr || beta == nullptr
        || decay == nullptr || state == nullptr || output == nullptr || key_heads == 0
        || value_heads == 0 || state_size != 128 || value_heads % key_heads != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen35_deltanet_state_update_transposed_row_waves_kernel,
                       dim3(value_heads, (state_size + 3) / 4), dim3(256),
                       static_cast<std::size_t>(2) * state_size * sizeof(float), stream,
                       query, key, value, beta, decay, state, output,
                       key_heads, value_heads, state_size);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen35_deltanet_fused_recurrent_core(
    const float* query,
    const float* key,
    const float* value,
    const float* beta,
    const float* decay,
    const float* ssm_norm,
    const float* gate,
    float* state,
    float* gated_output,
    float* recurrent_output,
    std::uint32_t key_heads,
    std::uint32_t value_heads,
    std::uint32_t state_size,
    float epsilon,
    hipStream_t stream,
    Q8_1Block* gated_output_q8) {
    if (query == nullptr || key == nullptr || value == nullptr || beta == nullptr
        || decay == nullptr || ssm_norm == nullptr || gate == nullptr || state == nullptr
        || (gated_output == nullptr && gated_output_q8 == nullptr) || key_heads == 0 || value_heads == 0
        || state_size != 128 || value_heads % key_heads != 0 || epsilon <= 0.0F) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    const std::size_t shared_bytes = static_cast<std::size_t>(2) * state_size * sizeof(float);
    hipLaunchKernelGGL(qwen35_deltanet_fused_recurrent_core_kernel,
                       dim3(value_heads), dim3(state_size), shared_bytes, stream,
                       query, key, value, beta, decay, ssm_norm, gate,
                       state, gated_output, recurrent_output,
                       key_heads, value_heads, state_size, epsilon, gated_output_q8);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen35_deltanet_fused_recurrent_core_batched4(
    const float* query,
    const float* key,
    const float* value,
    const float* beta,
    const float* decay,
    const float* ssm_norm,
    const float* gate,
    float* state,
    float* gated_output,
    std::uint32_t key_heads,
    std::uint32_t value_heads,
    std::uint32_t state_size,
    float epsilon,
    hipStream_t stream,
    Q8_1Block* gated_output_q8) {
    if (query == nullptr || key == nullptr || value == nullptr || beta == nullptr
        || decay == nullptr || ssm_norm == nullptr || gate == nullptr || state == nullptr
        || gated_output == nullptr || key_heads == 0 || value_heads == 0
        || state_size != 128 || value_heads % key_heads != 0 || epsilon <= 0.0F) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    constexpr std::size_t batch = 4;
    const std::size_t shared_bytes = (2 * batch + 1) * state_size * sizeof(float);
    hipLaunchKernelGGL(qwen35_deltanet_fused_recurrent_core_batched4_kernel,
                       dim3(value_heads), dim3(state_size), shared_bytes, stream,
                       query, key, value, beta, decay, ssm_norm, gate,
                       state, gated_output, key_heads, value_heads, state_size, epsilon,
                       gated_output_q8);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen35_deltanet_prefill_stage(
    const float* query,
    const float* key,
    const float* value,
    const float* beta,
    const float* decay,
    const float* gate,
    float* query_batch,
    float* key_batch,
    float* value_batch,
    float* beta_batch,
    float* decay_batch,
    float* gate_batch,
    std::uint32_t token_index,
    std::uint32_t key_heads,
    std::uint32_t value_heads,
    std::uint32_t state_size,
    hipStream_t stream) {
    if (query == nullptr || key == nullptr || value == nullptr || beta == nullptr
        || decay == nullptr || gate == nullptr || query_batch == nullptr || key_batch == nullptr
        || value_batch == nullptr || beta_batch == nullptr || decay_batch == nullptr
        || gate_batch == nullptr || key_heads == 0 || value_heads == 0 || state_size == 0
        || value_heads % key_heads != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    const auto elements = std::max(key_heads * state_size, value_heads * state_size);
    hipLaunchKernelGGL(qwen35_deltanet_prefill_stage_kernel,
                       dim3((elements + 255) / 256), dim3(256), 0, stream,
                       query, key, value, beta, decay, gate,
                       query_batch, key_batch, value_batch, beta_batch, decay_batch, gate_batch,
                       token_index, key_heads, value_heads, state_size);
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

void launch_qwen35_conv_silu_split_batch(
    const float* qkv, const float* conv_weights, float* history,
    float* query, float* key, float* value, std::uint32_t base_position,
    std::uint32_t token_count, std::uint32_t history_capacity,
    std::uint32_t channels, std::uint32_t conv_kernel, hipStream_t stream) {
    if (qkv == nullptr || conv_weights == nullptr || history == nullptr || query == nullptr
        || key == nullptr || value == nullptr || token_count == 0 || token_count > 512
        || history_capacity == 0 || channels == 0 || channels % 5 != 0
        || conv_kernel == 0 || conv_kernel > history_capacity) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen35_conv_silu_split_batch_kernel,
                       dim3((channels + 255) / 256), dim3(256), 0, stream,
                       qkv, conv_weights, history, query, key, value, base_position,
                       token_count, history_capacity, channels, conv_kernel);
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

void launch_qwen35_dual_head_l2_normalize(
    const float* input_a, const float* input_b, float* output_a, float* output_b,
    std::uint32_t heads, std::uint32_t head_dim, hipStream_t stream) {
    if (input_a == nullptr || input_b == nullptr || output_a == nullptr || output_b == nullptr
        || heads == 0 || head_dim == 0 || head_dim > 128) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen35_dual_head_l2_normalize_kernel,
                       dim3(heads), dim3(128), 0, stream,
                       input_a, input_b, output_a, output_b, heads, head_dim);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen35_dual_head_l2_normalize_batch(
    const float* input_a, const float* input_b, float* output_a, float* output_b,
    std::uint32_t token_count, std::uint32_t heads, std::uint32_t head_dim,
    hipStream_t stream) {
    if (input_a == nullptr || input_b == nullptr || output_a == nullptr || output_b == nullptr
        || token_count == 0 || token_count > 512 || heads == 0 || head_dim == 0 || head_dim > 128) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen35_dual_head_l2_normalize_batch_kernel,
                       dim3(token_count * heads), dim3(128), 0, stream,
                       input_a, input_b, output_a, output_b, heads, head_dim);
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

static inline std::uint32_t select_attention_splits(std::uint32_t cache_length) {
    const char* splits_env = std::getenv("MIINFER_ATTENTION_SPLITS");
    if (splits_env != nullptr) {
        return std::min(64u, std::max(1u, static_cast<std::uint32_t>(std::atoi(splits_env))));
    }
    if (cache_length <= 512) {
        return 4;
    } else if (cache_length <= 2048) {
        return 8;
    } else if (cache_length <= 8192) {
        return 16;
    } else if (cache_length <= 16384) {
        return 32;
    } else {
        return 64;
    }
}

void launch_qwen35_tiled_online_attention(
    const float* q, const float* key_cache, const float* value_cache,
    std::uint32_t cache_length, std::uint32_t cache_capacity, float* output,
    const float* gate, float* gated_output,
    std::uint32_t query_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, float scale, hipStream_t stream,
    Q8_1Block* gated_output_q8) {
    if (q == nullptr || key_cache == nullptr || value_cache == nullptr
        || cache_length == 0 || query_heads == 0 || kv_heads == 0
        || head_dim != 256 || (output == nullptr && gated_output == nullptr && gated_output_q8 == nullptr)
        || query_heads % kv_heads != 0 || !std::isfinite(scale)) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    const std::uint32_t splits = select_attention_splits(cache_length);
    hipLaunchKernelGGL(qwen3_wave64_splitk_stage1_f32_kernel,
                       dim3(query_heads, splits), dim3(64), 0, stream,
                       q, key_cache, value_cache, cache_length, cache_capacity,
                       query_heads, kv_heads, head_dim, scale, splits);
    hipLaunchKernelGGL(qwen3_splitk_stage2_kernel,
                       dim3(query_heads), dim3(head_dim), 0, stream,
                       output, gate, gated_output, query_heads, head_dim, splits,
                       gated_output_q8);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen35_tiled_online_attention_f16(
    const float* q, const __half* key_cache, const __half* value_cache,
    std::uint32_t cache_length, std::uint32_t cache_capacity, float* output,
    const float* gate, float* gated_output,
    std::uint32_t query_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, float scale, hipStream_t stream,
    Q8_1Block* gated_output_q8) {
    if (q == nullptr || key_cache == nullptr || value_cache == nullptr
        || cache_length == 0 || query_heads == 0 || kv_heads == 0
        || head_dim != 256 || (output == nullptr && gated_output == nullptr && gated_output_q8 == nullptr)
        || query_heads % kv_heads != 0 || !std::isfinite(scale)) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    const std::uint32_t splits = select_attention_splits(cache_length);
    hipLaunchKernelGGL(qwen3_wave64_splitk_stage1_f16_kernel,
                       dim3(query_heads, splits), dim3(64), 0, stream,
                       q, key_cache, value_cache, cache_length, cache_capacity,
                       query_heads, kv_heads, head_dim, scale, splits);
    hipLaunchKernelGGL(qwen3_splitk_stage2_kernel,
                       dim3(query_heads), dim3(head_dim), 0, stream,
                       output, gate, gated_output, query_heads, head_dim, splits,
                       gated_output_q8);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen35_tiled_online_attention_batch(
    const float* q, const float* key_cache, const float* value_cache,
    const float* gate, float* gated_output, std::uint32_t token_count,
    std::uint32_t base_position, std::uint32_t cache_capacity,
    std::uint32_t query_heads, std::uint32_t kv_heads, std::uint32_t head_dim,
    float scale, hipStream_t stream) {
    if (q == nullptr || key_cache == nullptr || value_cache == nullptr || gate == nullptr
        || gated_output == nullptr || token_count == 0 || base_position > cache_capacity
        || token_count > cache_capacity - base_position || cache_capacity == 0
        || query_heads == 0 || kv_heads == 0 || head_dim != 256
        || query_heads % kv_heads != 0 || !std::isfinite(scale)) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen35_tiled_online_attention_batch_f32_kernel,
                       dim3(token_count * query_heads), dim3(64), 0, stream,
                       q, key_cache, value_cache, gate, gated_output, token_count,
                       base_position, cache_capacity, query_heads, kv_heads, head_dim, scale);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen35_tiled_online_attention_batch_f16(
    const float* q, const __half* key_cache, const __half* value_cache,
    const float* gate, float* gated_output, std::uint32_t token_count,
    std::uint32_t base_position, std::uint32_t cache_capacity,
    std::uint32_t query_heads, std::uint32_t kv_heads, std::uint32_t head_dim,
    float scale, hipStream_t stream) {
    if (q == nullptr || key_cache == nullptr || value_cache == nullptr || gate == nullptr
        || gated_output == nullptr || token_count == 0 || base_position > cache_capacity
        || token_count > cache_capacity - base_position || cache_capacity == 0
        || query_heads == 0 || kv_heads == 0 || head_dim != 256
        || query_heads % kv_heads != 0 || !std::isfinite(scale)) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen35_tiled_online_attention_batch_f16_kernel,
                       dim3(token_count * query_heads), dim3(64), 0, stream,
                       q, key_cache, value_cache, gate, gated_output, token_count,
                       base_position, cache_capacity, query_heads, kv_heads, head_dim, scale);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen35_fused_q_split_norm_rope_batch(
    const float* qfull, const float* q_norm_weight, float* query_rope, float* gate,
    std::uint32_t token_count, std::uint32_t base_position, std::uint32_t heads,
    std::uint32_t kv_heads, std::uint32_t head_dim, float theta, float epsilon,
    hipStream_t stream) {
    if (qfull == nullptr || q_norm_weight == nullptr || query_rope == nullptr || gate == nullptr
        || token_count == 0 || heads == 0 || kv_heads == 0 || heads % kv_heads != 0
        || head_dim != 256 || !std::isfinite(theta)
        || !std::isfinite(epsilon)) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen35_fused_q_split_norm_rope_batch_kernel,
                       dim3(token_count * heads), dim3(256), 0, stream,
                       qfull, q_norm_weight, query_rope, gate, token_count, base_position,
                       heads, kv_heads, head_dim, theta, epsilon);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen35_fused_k_norm_rope_kv_store_batch(
    const float* qfull, const float* value, const float* k_norm_weight,
    float* key_cache, float* value_cache, std::uint32_t token_count,
    std::uint32_t base_position, std::uint32_t cache_capacity,
    std::uint32_t query_heads, std::uint32_t kv_heads, std::uint32_t head_dim,
    float theta, float epsilon, hipStream_t stream) {
    if (qfull == nullptr || value == nullptr || k_norm_weight == nullptr
        || key_cache == nullptr || value_cache == nullptr || token_count == 0
        || cache_capacity == 0 || base_position > cache_capacity
        || token_count > cache_capacity - base_position || query_heads == 0 || kv_heads == 0
        || query_heads % kv_heads != 0
        || head_dim != 256 || !std::isfinite(theta) || !std::isfinite(epsilon)) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen35_fused_k_norm_rope_kv_store_batch_kernel,
                       dim3(token_count * kv_heads), dim3(256), 0, stream,
                       qfull, value, k_norm_weight, key_cache, value_cache, token_count,
                       base_position, cache_capacity, query_heads, kv_heads, head_dim,
                       theta, epsilon);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen35_fused_k_norm_rope_kv_store_batch_f16(
    const float* qfull, const float* value, const float* k_norm_weight,
    __half* key_cache, __half* value_cache, std::uint32_t token_count,
    std::uint32_t base_position, std::uint32_t cache_capacity,
    std::uint32_t query_heads, std::uint32_t kv_heads, std::uint32_t head_dim,
    float theta, float epsilon, hipStream_t stream) {
    if (qfull == nullptr || value == nullptr || k_norm_weight == nullptr
        || key_cache == nullptr || value_cache == nullptr || token_count == 0
        || cache_capacity == 0 || base_position > cache_capacity
        || token_count > cache_capacity - base_position || query_heads == 0 || kv_heads == 0
        || query_heads % kv_heads != 0
        || head_dim != 256 || !std::isfinite(theta) || !std::isfinite(epsilon)) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen35_fused_k_norm_rope_kv_store_batch_f16_kernel,
                       dim3(token_count * kv_heads), dim3(256), 0, stream,
                       qfull, value, k_norm_weight, key_cache, value_cache, token_count,
                       base_position, cache_capacity, query_heads, kv_heads, head_dim,
                       theta, epsilon);
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

void launch_qwen3_kv_cache_store_f16(
    const float* keys, const float* values, __half* key_cache, __half* value_cache,
    std::uint32_t position, std::uint32_t cache_capacity,
    std::uint32_t kv_heads, std::uint32_t head_dim, hipStream_t stream) {
    if (keys == nullptr || values == nullptr || key_cache == nullptr || value_cache == nullptr
        || cache_capacity == 0 || position >= cache_capacity || kv_heads == 0 || head_dim == 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    const auto elements = kv_heads * head_dim;
    hipLaunchKernelGGL(qwen3_kv_cache_store_f16_kernel,
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

void launch_qwen3_q4_k_q8_1_mmvq_lds_input(
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
    const auto shared_bytes = static_cast<std::size_t>(columns / 32) * sizeof(Q8_1Block);
    hipLaunchKernelGGL(qwen3_q4_k_q8_1_mmvq_lds_input_kernel,
                       dim3((rows + 1) / 2), dim3(256), shared_bytes, stream,
                       weights, input, output, rows, columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen3_q4_k_q8_1_mmvq_lds_metadata(
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
    const auto input_bytes = static_cast<std::size_t>(columns / 32) * sizeof(Q8_1Block);
    const auto metadata_bytes = static_cast<std::size_t>(2) * (columns / 256)
                                * sizeof(Q4KMetadata);
    hipLaunchKernelGGL(qwen3_q4_k_q8_1_mmvq_lds_metadata_kernel,
                       dim3((rows + 1) / 2), dim3(256), input_bytes + metadata_bytes, stream,
                       weights, input, output, rows, columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen3_q4_k_q8_1_mmvq_lds_decoded_metadata(
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
    const auto input_bytes = static_cast<std::size_t>(columns / 32) * sizeof(Q8_1Block);
    const auto metadata_bytes = static_cast<std::size_t>(2) * (columns / 256)
                                * sizeof(Q4KDecodedMetadata);
    hipLaunchKernelGGL(qwen3_q4_k_q8_1_mmvq_lds_decoded_metadata_kernel,
                       dim3((rows + 1) / 2), dim3(256), input_bytes + metadata_bytes, stream,
                       weights, input, output, rows, columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen3_q4_k_q8_1_mmvq_expanded(
    const Q4KExpandedDeviceBlock* weights,
    const Q8_1Block* input,
    float* output,
    std::uint32_t rows,
    std::uint32_t columns,
    hipStream_t stream) {
    if (weights == nullptr || input == nullptr || output == nullptr || rows == 0 || columns == 0
        || columns % 256 != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen3_q4_k_q8_1_mmvq_expanded_kernel,
                       dim3((rows + 1) / 2), dim3(256), 0, stream,
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

void launch_qwen35_f32_dual_gemv(
    const float* weights_a, const float* weights_b, const float* input,
    float* output_a, float* output_b, std::uint32_t rows,
    std::uint32_t columns, hipStream_t stream) {
    if (weights_a == nullptr || weights_b == nullptr || input == nullptr
        || output_a == nullptr || output_b == nullptr || rows == 0 || columns == 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen35_f32_dual_gemv_kernel, dim3(rows), dim3(256), 0, stream,
                       weights_a, weights_b, input, output_a, output_b, rows, columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen35_f32_dual_gemm_batch(
    const float* weights_a, const float* weights_b, const float* input,
    float* output_a, float* output_b, std::uint32_t token_count,
    std::uint32_t rows, std::uint32_t columns, hipStream_t stream) {
    if (weights_a == nullptr || weights_b == nullptr || input == nullptr || output_a == nullptr
        || output_b == nullptr || token_count == 0 || token_count > 512 || rows == 0 || columns == 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen35_f32_dual_gemm_batch_kernel,
                       dim3(token_count * rows), dim3(256), 0, stream,
                       weights_a, weights_b, input, output_a, output_b, rows, columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen35_f32_dual_beta_decay(
    const float* beta_weights, const float* alpha_weights, const float* input,
    const float* dt, const float* a, float* beta, float* decay,
    std::uint32_t rows, std::uint32_t columns, hipStream_t stream) {
    if (beta_weights == nullptr || alpha_weights == nullptr || input == nullptr
        || dt == nullptr || a == nullptr || beta == nullptr || decay == nullptr
        || rows == 0 || columns == 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(qwen35_f32_dual_beta_decay_kernel, dim3(rows), dim3(256), 0, stream,
                       beta_weights, alpha_weights, input, dt, a, beta, decay, rows, columns);
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

static bool fast_argmax_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("MIINFER_FAST_ARGMAX");
        return env == nullptr || std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

void launch_qwen3_argmax(
    const float* input, std::uint32_t* output, std::uint32_t elements, hipStream_t stream) {
    if (input == nullptr || output == nullptr || elements == 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    if (elements <= 256 || !fast_argmax_enabled()) {
        hipLaunchKernelGGL(qwen3_argmax_kernel, dim3(1), dim3(256), 0, stream,
                           input, output, elements);
    } else {
        constexpr std::uint32_t kPartials = 256;
        hipLaunchKernelGGL(qwen3_argmax_stage1_kernel, dim3(kPartials), dim3(256), 0, stream,
                           input, elements);
        hipLaunchKernelGGL(qwen3_argmax_stage2_kernel, dim3(1), dim3(256), 0, stream,
                           output, kPartials);
    }
    MIINFER_HIP_CHECK(hipGetLastError());
}

}  // namespace miinfer
