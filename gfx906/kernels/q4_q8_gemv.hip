#include "miinfer/q4_q8_gemv.hpp"

#include "miinfer/hip_check.hpp"

#include <hip/hip_runtime.h>

namespace miinfer {

namespace {

constexpr unsigned int kThreadsPerBlock = 256;

__global__ void q8_1_quantize_kernel(
    const __half* input,
    Q8_1Block* output,
    int elements) {
    const int block = static_cast<int>(blockIdx.x);
    const int index = static_cast<int>(threadIdx.x);
    if (index >= kQ8_1BlockSize || block * kQ8_1BlockSize + index >= elements) {
        return;
    }

    __shared__ float absolute_values[kQ8_1BlockSize];
    __shared__ int quantized[kQ8_1BlockSize];
    __shared__ float scale;
    absolute_values[index] = fabsf(__half2float(input[block * kQ8_1BlockSize + index]));
    __syncthreads();
    for (int stride = kQ8_1BlockSize / 2; stride > 0; stride /= 2) {
        if (index < stride) {
            absolute_values[index] = fmaxf(absolute_values[index], absolute_values[index + stride]);
        }
        __syncthreads();
    }
    if (index == 0) {
        scale = absolute_values[0] / 127.0F;
        output[block].d = __float2half_rn(scale);
    }
    __syncthreads();
    const float inverse_scale = scale != 0.0F ? 1.0F / scale : 0.0F;
    quantized[index] = max(-127, min(127, static_cast<int>(roundf(
        __half2float(input[block * kQ8_1BlockSize + index]) * inverse_scale))));
    __syncthreads();
    if (index == 0) {
        int sum = 0;
        for (int quant_index = 0; quant_index < kQ8_1BlockSize; ++quant_index) {
            sum += quantized[quant_index];
        }
        output[block].s = __float2half_rn(static_cast<float>(sum) * scale);
    }
    output[block].qs[index] = static_cast<std::int8_t>(quantized[index]);
}

__global__ void q8_1_quantize_f32_kernel(
    const float* input,
    Q8_1Block* output,
    int elements) {
    const int block = static_cast<int>(blockIdx.x);
    const int index = static_cast<int>(threadIdx.x);
    if (index >= kQ8_1BlockSize || block * kQ8_1BlockSize + index >= elements) return;

    __shared__ float absolute_values[kQ8_1BlockSize];
    __shared__ int quantized[kQ8_1BlockSize];
    __shared__ float scale;
    absolute_values[index] = fabsf(input[block * kQ8_1BlockSize + index]);
    __syncthreads();
    for (int stride = kQ8_1BlockSize / 2; stride > 0; stride /= 2) {
        if (index < stride) {
            absolute_values[index] = fmaxf(absolute_values[index], absolute_values[index + stride]);
        }
        __syncthreads();
    }
    if (index == 0) {
        scale = absolute_values[0] / 127.0F;
        output[block].d = __float2half_rn(scale);
    }
    __syncthreads();
    const float inverse_scale = scale != 0.0F ? 1.0F / scale : 0.0F;
    quantized[index] = max(-127, min(127, static_cast<int>(roundf(
        input[block * kQ8_1BlockSize + index] * inverse_scale))));
    __syncthreads();
    if (index == 0) {
        int sum = 0;
        for (int quant_index = 0; quant_index < kQ8_1BlockSize; ++quant_index) {
            sum += quantized[quant_index];
        }
        output[block].s = __float2half_rn(static_cast<float>(sum) * scale);
    }
    output[block].qs[index] = static_cast<std::int8_t>(quantized[index]);
}

__global__ void q8_exact_quantize_kernel(
    const __half* input,
    Q8ExactBlock* output,
    int elements) {
    const int block = static_cast<int>(blockIdx.x);
    const int index = static_cast<int>(threadIdx.x);
    if (index >= kQ8_1BlockSize || block * kQ8_1BlockSize + index >= elements) return;

    __shared__ float absolute_values[kQ8_1BlockSize];
    __shared__ int quantized[kQ8_1BlockSize];
    __shared__ float scale;
    absolute_values[index] = fabsf(__half2float(input[block * kQ8_1BlockSize + index]));
    __syncthreads();
    for (int stride = kQ8_1BlockSize / 2; stride > 0; stride /= 2) {
        if (index < stride) {
            absolute_values[index] = fmaxf(absolute_values[index], absolute_values[index + stride]);
        }
        __syncthreads();
    }
    if (index == 0) {
        scale = absolute_values[0] / 127.0F;
        output[block].d = __float2half_rn(scale);
    }
    __syncthreads();
    const float inverse_scale = scale != 0.0F ? 1.0F / scale : 0.0F;
    quantized[index] = max(-127, min(127, static_cast<int>(roundf(
        __half2float(input[block * kQ8_1BlockSize + index]) * inverse_scale))));
    __syncthreads();
    if (index == 0) {
        int sum = 0;
        for (int quant_index = 0; quant_index < kQ8_1BlockSize; ++quant_index) {
            sum += quantized[quant_index];
        }
        output[block].sum = static_cast<std::int16_t>(sum);
    }
    output[block].qs[index] = static_cast<std::int8_t>(quantized[index]);
}

__global__ void q8_exact_quantize_f32_kernel(
    const float* input,
    Q8ExactBlock* output,
    int elements) {
    const int block = static_cast<int>(blockIdx.x);
    const int index = static_cast<int>(threadIdx.x);
    if (index >= kQ8_1BlockSize || block * kQ8_1BlockSize + index >= elements) return;

    __shared__ float absolute_values[kQ8_1BlockSize];
    __shared__ int quantized[kQ8_1BlockSize];
    __shared__ float scale;
    absolute_values[index] = fabsf(input[block * kQ8_1BlockSize + index]);
    __syncthreads();
    for (int stride = kQ8_1BlockSize / 2; stride > 0; stride /= 2) {
        if (index < stride) {
            absolute_values[index] = fmaxf(absolute_values[index], absolute_values[index + stride]);
        }
        __syncthreads();
    }
    if (index == 0) {
        scale = absolute_values[0] / 127.0F;
        output[block].d = __float2half_rn(scale);
    }
    __syncthreads();
    const float inverse_scale = scale != 0.0F ? 1.0F / scale : 0.0F;
    quantized[index] = max(-127, min(127, static_cast<int>(roundf(
        input[block * kQ8_1BlockSize + index] * inverse_scale))));
    __syncthreads();
    if (index == 0) {
        int sum = 0;
        for (int quant_index = 0; quant_index < kQ8_1BlockSize; ++quant_index) {
            sum += quantized[quant_index];
        }
        output[block].sum = static_cast<std::int16_t>(sum);
    }
    output[block].qs[index] = static_cast<std::int8_t>(quantized[index]);
}

// C10c candidate. One workgroup performs the same 256-thread RMS reduction
// as qwen3_rms_normalize_kernel, then processes eight Q8 blocks at a time.
// Keeping the eight blocks in one workgroup avoids recomputing the global RMS
// while retaining the existing per-block Q8 max/reduction and metadata order.
__global__ void qwen3_ffn_norm_to_q8_exact_kernel(
    const float* input,
    const float* weights,
    Q8ExactBlock* output,
    int elements,
    float epsilon,
    __half* materialized_output) {
    constexpr int kBlocksPerRound = 8;
    const int thread = static_cast<int>(threadIdx.x);
    const int lane = thread % kQ8_1BlockSize;
    const int slot = thread / kQ8_1BlockSize;
    const int block_count = elements / kQ8_1BlockSize;

    __shared__ float partials[kThreadsPerBlock];
    __shared__ float absolute_values[kBlocksPerRound * kQ8_1BlockSize];
    __shared__ int quantized[kBlocksPerRound * kQ8_1BlockSize];
    __shared__ float scales[kBlocksPerRound];

    float sum = 0.0F;
    for (int index = thread; index < elements; index += kThreadsPerBlock) {
        sum += input[index] * input[index];
    }
    partials[thread] = sum;
    __syncthreads();
    for (int stride = kThreadsPerBlock / 2; stride > 0; stride /= 2) {
        if (thread < stride) partials[thread] += partials[thread + stride];
        __syncthreads();
    }
    const float inverse_rms = rsqrtf(
        partials[0] / static_cast<float>(elements) + epsilon);

    for (int first_block = 0; first_block < block_count; first_block += kBlocksPerRound) {
        const int block = first_block + slot;
        const bool valid = block < block_count;
        const int element = block * kQ8_1BlockSize + lane;
        __half rounded = __float2half_rn(0.0F);
        if (valid) {
            // Keep these as separate operations: they mirror the separate
            // RMS-normalize and norm-scale kernels before the FP16 boundary.
            volatile float normalized = input[element] * inverse_rms;
            volatile float scaled = normalized * weights[element];
            rounded = __float2half_rn(static_cast<float>(scaled));
            if (materialized_output != nullptr) materialized_output[element] = rounded;
        }
        absolute_values[slot * kQ8_1BlockSize + lane] =
            valid ? fabsf(__half2float(rounded)) : 0.0F;
        __syncthreads();
        for (int stride = kQ8_1BlockSize / 2; stride > 0; stride /= 2) {
            if (lane < stride) {
                const int base = slot * kQ8_1BlockSize;
                absolute_values[base + lane] = fmaxf(
                    absolute_values[base + lane], absolute_values[base + lane + stride]);
            }
            __syncthreads();
        }
        if (lane == 0) {
            scales[slot] = absolute_values[slot * kQ8_1BlockSize] / 127.0F;
            if (valid) output[block].d = __float2half_rn(scales[slot]);
        }
        __syncthreads();
        const float scale = scales[slot];
        const float value = __half2float(rounded);
        const float inverse_scale = scale != 0.0F ? 1.0F / scale : 0.0F;
        quantized[slot * kQ8_1BlockSize + lane] = valid
            ? max(-127, min(127, static_cast<int>(roundf(value * inverse_scale))))
            : 0;
        __syncthreads();
        if (lane == 0 && valid) {
            int block_sum = 0;
            for (int quant_index = 0; quant_index < kQ8_1BlockSize; ++quant_index) {
                block_sum += quantized[slot * kQ8_1BlockSize + quant_index];
            }
            output[block].sum = static_cast<std::int16_t>(block_sum);
        }
        if (valid) {
            output[block].qs[lane] = static_cast<std::int8_t>(
                quantized[slot * kQ8_1BlockSize + lane]);
        }
        __syncthreads();
    }
}

__global__ void silu_mul_q8_exact_kernel(
    const float* gate,
    const float* up,
    Q8ExactBlock* output,
    int elements) {
    const int block = static_cast<int>(blockIdx.x);
    const int index = static_cast<int>(threadIdx.x);
    if (index >= kQ8_1BlockSize || block * kQ8_1BlockSize + index >= elements) return;

    __shared__ __half activated[kQ8_1BlockSize];
    __shared__ float absolute_values[kQ8_1BlockSize];
    __shared__ int quantized[kQ8_1BlockSize];
    __shared__ float scale;

    const int element = block * kQ8_1BlockSize + index;
    const float gate_value = gate[element];
    const float activated_value = (gate_value / (1.0F + expf(-gate_value))) * up[element];
    activated[index] = __float2half_rn(activated_value);
    absolute_values[index] = fabsf(__half2float(activated[index]));
    __syncthreads();
    for (int stride = kQ8_1BlockSize / 2; stride > 0; stride /= 2) {
        if (index < stride) {
            absolute_values[index] = fmaxf(absolute_values[index], absolute_values[index + stride]);
        }
        __syncthreads();
    }
    if (index == 0) {
        scale = absolute_values[0] / 127.0F;
        output[block].d = __float2half_rn(scale);
    }
    __syncthreads();
    const float value = __half2float(activated[index]);
    const float inverse_scale = scale != 0.0F ? 1.0F / scale : 0.0F;
    quantized[index] = max(-127, min(127, static_cast<int>(roundf(value * inverse_scale))));
    __syncthreads();
    if (index == 0) {
        int sum = 0;
        for (int quant_index = 0; quant_index < kQ8_1BlockSize; ++quant_index) {
            sum += quantized[quant_index];
        }
        output[block].sum = static_cast<std::int16_t>(sum);
    }
    output[block].qs[index] = static_cast<std::int8_t>(quantized[index]);
}

__global__ void q4_q8_gemv_kernel(
    const Q4_0Block* weights,
    const Q8_1Block* input,
    __half* output,
    int rows,
    int columns) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) {
        return;
    }
    __shared__ float partials[kThreadsPerBlock];
    float sum = 0.0F;
    const int blocks_per_row = columns / kQ4_0BlockSize;
    const std::size_t row_offset = static_cast<std::size_t>(row) * blocks_per_row;
    for (int index = static_cast<int>(threadIdx.x); index < columns;
         index += static_cast<int>(blockDim.x)) {
        const int block = index / kQ4_0BlockSize;
        const int block_index = index % kQ4_0BlockSize;
        const auto& weight = weights[row_offset + static_cast<std::size_t>(block)];
        const auto& activation = input[static_cast<std::size_t>(block)];
        const std::uint8_t packed = weight.qs[block_index < kQ4_0BlockSize / 2
                                                   ? block_index
                                                   : block_index - kQ4_0BlockSize / 2];
        const int q4 = block_index < kQ4_0BlockSize / 2
                           ? (packed & 0x0F)
                           : ((packed >> 4) & 0x0F);
        sum += static_cast<float>(q4 - 8) * __half2float(weight.d)
               * static_cast<float>(activation.qs[block_index])
               * __half2float(activation.d);
    }
    partials[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int stride = kThreadsPerBlock / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partials[threadIdx.x] += partials[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        output[row] = __float2half_rn(partials[0]);
    }
}

}  // namespace

void launch_q8_1_quantize(
    const __half* input,
    Q8_1Block* output,
    int elements,
    hipStream_t stream) {
    if (elements <= 0 || elements % kQ8_1BlockSize != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(
        q8_1_quantize_kernel,
        dim3(static_cast<unsigned int>(elements / kQ8_1BlockSize)),
        dim3(kQ8_1BlockSize),
        0,
        stream,
        input,
        output,
        elements);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_q8_1_quantize_f32(
    const float* input,
    Q8_1Block* output,
    int elements,
    hipStream_t stream) {
    if (input == nullptr || output == nullptr || elements <= 0
        || elements % kQ8_1BlockSize != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(
        q8_1_quantize_f32_kernel,
        dim3(static_cast<unsigned int>(elements / kQ8_1BlockSize)),
        dim3(kQ8_1BlockSize),
        0,
        stream,
        input,
        output,
        elements);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_q8_exact_quantize(
    const __half* input,
    Q8ExactBlock* output,
    int elements,
    hipStream_t stream) {
    if (input == nullptr || output == nullptr || elements <= 0
        || elements % kQ8_1BlockSize != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(
        q8_exact_quantize_kernel,
        dim3(static_cast<unsigned int>(elements / kQ8_1BlockSize)),
        dim3(kQ8_1BlockSize), 0, stream, input, output, elements);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_q8_exact_quantize_f32(
    const float* input,
    Q8ExactBlock* output,
    int elements,
    hipStream_t stream) {
    if (input == nullptr || output == nullptr || elements <= 0
        || elements % kQ8_1BlockSize != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(
        q8_exact_quantize_f32_kernel,
        dim3(static_cast<unsigned int>(elements / kQ8_1BlockSize)),
        dim3(kQ8_1BlockSize), 0, stream, input, output, elements);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_qwen3_ffn_norm_to_q8_exact(
    const float* input,
    const float* weights,
    Q8ExactBlock* output,
    int elements,
    float epsilon,
    __half* materialized_output,
    hipStream_t stream) {
    if (input == nullptr || weights == nullptr || output == nullptr || elements <= 0
        || elements % kQ8_1BlockSize != 0 || epsilon <= 0.0F) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(
        qwen3_ffn_norm_to_q8_exact_kernel,
        dim3(1), dim3(kThreadsPerBlock), 0, stream,
        input, weights, output, elements, epsilon, materialized_output);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_silu_mul_q8_exact(
    const float* gate,
    const float* up,
    Q8ExactBlock* output,
    int elements,
    hipStream_t stream) {
    if (gate == nullptr || up == nullptr || output == nullptr || elements <= 0
        || elements % kQ8_1BlockSize != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(
        silu_mul_q8_exact_kernel,
        dim3(static_cast<unsigned int>(elements / kQ8_1BlockSize)),
        dim3(kQ8_1BlockSize), 0, stream, gate, up, output, elements);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_q4_q8_gemv(
    const Q4_0Block* weights,
    const Q8_1Block* input,
    __half* output,
    int rows,
    int columns,
    hipStream_t stream) {
    if (rows <= 0 || columns <= 0 || columns % kQ4_0BlockSize != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(
        q4_q8_gemv_kernel,
        dim3(static_cast<unsigned int>(rows)),
        dim3(kThreadsPerBlock),
        kThreadsPerBlock * sizeof(float),
        stream,
        weights,
        input,
        output,
        rows,
        columns);
    MIINFER_HIP_CHECK(hipGetLastError());
}

}  // namespace miinfer
