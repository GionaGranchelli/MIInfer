#include "miinfer/m12_gdn_chunk.hpp"

#include "miinfer/hip_check.hpp"

#include <hip/hip_runtime.h>

namespace {

constexpr unsigned kThreads = 1024;

__global__ void m12_gdn_postprocess_kernel(
    const float* __restrict__ recurrent_output,
    const float* __restrict__ gate,
    const float* __restrict__ ssm_norm,
    float* __restrict__ gated_output,
    std::uint32_t value_heads,
    std::uint32_t state_size,
    float epsilon) {
    const std::uint32_t flat_head = blockIdx.x;
    const std::uint32_t head = flat_head % value_heads;
    const std::uint32_t token = flat_head / value_heads;
    const std::uint32_t row = threadIdx.x;
    if (row >= state_size) return;
    __shared__ float partials[128];
    const std::size_t base = (static_cast<std::size_t>(token) * value_heads + head)
        * state_size;
    const float recurrent = recurrent_output[base + row];
    partials[row] = recurrent * recurrent;
    __syncthreads();
    for (std::uint32_t stride = state_size / 2; stride > 0; stride /= 2) {
        if (row < stride) partials[row] += partials[row + stride];
        __syncthreads();
    }
    const float inverse_rms = rsqrtf(partials[0] / static_cast<float>(state_size) + epsilon);
    const float scaled = recurrent * inverse_rms * ssm_norm[row];
    const float gate_value = gate[base + row];
    gated_output[base + row] = (gate_value / (1.0F + expf(-gate_value))) * scaled;
}

__global__ void m12_gdn_chunk_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ decay,
    float* __restrict__ state,
    float* __restrict__ output,
    miinfer::M12GdnChunkWorkspace workspace,
    std::uint32_t token_count,
    std::uint32_t chunk_start,
    std::uint32_t key_heads,
    std::uint32_t value_heads,
    std::uint32_t state_size,
    std::uint32_t chunk_size) {
    extern __shared__ float shared[];
    float* cumulative = shared;
    float* row_decay = cumulative + chunk_size;
    float* backward_decay = row_decay + chunk_size;
    float* system = backward_decay + chunk_size;
    float* inverse = system + chunk_size * chunk_size;
    float* intra_attention = inverse + chunk_size * chunk_size;
    const std::uint32_t head = blockIdx.x;
    const std::uint32_t tid = threadIdx.x;
    (void)token_count;
    if (head >= value_heads) return;

    const std::uint32_t key_head = head % key_heads;
    const std::size_t key_stride = static_cast<std::size_t>(key_heads) * state_size;
    const std::size_t value_stride = static_cast<std::size_t>(value_heads) * state_size;
    const std::size_t state_stride = static_cast<std::size_t>(state_size) * state_size;
    const std::size_t state_base = static_cast<std::size_t>(head) * state_stride;
    const std::size_t work_values_base = static_cast<std::size_t>(head)
        * chunk_size * state_size;
    const float query_scale = rsqrtf(static_cast<float>(state_size));

    // The current feasibility kernel intentionally launches one workgroup per
    // value head and advances chunks in host order so state dependencies stay
    // explicit. It is a correctness/performance probe, not the final schedule.
    if (tid == 0) {
        float running = 0.0F;
        for (std::uint32_t row = 0; row < chunk_size; ++row) {
            // The runtime supplies multiplicative decay. Accumulate it in log
            // space so the chunk equations still operate on decay products.
            running += logf(decay[static_cast<std::size_t>(chunk_start + row) * value_heads + head]);
            cumulative[row] = running;
        }
    }
    __syncthreads();
    if (tid < chunk_size) {
        row_decay[tid] = expf(cumulative[tid]);
    }
    __syncthreads();
    if (tid == 0) {
        const float final_decay = cumulative[chunk_size - 1];
        for (std::uint32_t row = 0; row < chunk_size; ++row) {
            backward_decay[row] = expf(final_decay - cumulative[row]);
        }
    }
    __syncthreads();

    for (std::size_t flat = tid; flat < static_cast<std::size_t>(chunk_size) * chunk_size;
         flat += blockDim.x) {
        const std::uint32_t row = flat / chunk_size;
        const std::uint32_t column = flat % chunk_size;
        if (column > row) {
            system[flat] = 0.0F;
            intra_attention[flat] = 0.0F;
            continue;
        }
        const float pairwise = expf(cumulative[row] - cumulative[column]);
        float key_beta_dot = 0.0F;
        float query_key_dot = 0.0F;
        const std::size_t current_key_base = static_cast<std::size_t>(chunk_start + row)
            * key_stride + static_cast<std::size_t>(key_head) * state_size;
        const std::size_t previous_key_base = static_cast<std::size_t>(chunk_start + column)
            * key_stride + static_cast<std::size_t>(key_head) * state_size;
        const float beta_value = beta[static_cast<std::size_t>(chunk_start + row)
                                      * value_heads + head];
        for (std::uint32_t dimension = 0; dimension < state_size; ++dimension) {
            const float current_key = key[current_key_base + dimension];
            const float previous_key = key[previous_key_base + dimension];
            key_beta_dot += current_key * beta_value * previous_key;
            query_key_dot += query[current_key_base + dimension] * previous_key;
        }
        system[flat] = row == column
            ? 0.0F : key_beta_dot * pairwise;
        intra_attention[flat] =
            query_key_dot * pairwise * query_scale;
    }
    __syncthreads();

    for (std::size_t flat = tid; flat < static_cast<std::size_t>(chunk_size) * state_size;
         flat += blockDim.x) {
        const std::uint32_t row = flat / state_size;
        const std::uint32_t dimension = flat % state_size;
        const std::size_t source_base = static_cast<std::size_t>(chunk_start + row)
            * key_stride + static_cast<std::size_t>(key_head) * state_size;
        const std::size_t value_base = static_cast<std::size_t>(chunk_start + row)
            * value_stride + static_cast<std::size_t>(head) * state_size;
        const float beta_value = beta[static_cast<std::size_t>(chunk_start + row)
                                      * value_heads + head];
        const float decay_factor = row_decay[row];
        workspace.new_values[work_values_base + flat] = value[value_base + dimension]
            * beta_value;
        workspace.decayed_keys[work_values_base + flat] = key[source_base + dimension]
            * beta_value * decay_factor;
    }
    __syncthreads();

    // Forward substitution for the unit-lower-triangular WY system.
    for (std::uint32_t row = 0; row < chunk_size; ++row) {
        if (tid < chunk_size) {
            float solved = tid == row ? 1.0F : 0.0F;
            if (tid < row) solved = 0.0F;
            for (std::uint32_t column = 0; column < row; ++column) {
                solved -= system[static_cast<std::size_t>(row) * chunk_size + column]
                    * inverse[static_cast<std::size_t>(column) * chunk_size + tid];
            }
            inverse[static_cast<std::size_t>(row) * chunk_size + tid] = solved;
        }
        __syncthreads();
    }

    for (std::size_t flat = tid; flat < static_cast<std::size_t>(chunk_size) * state_size;
         flat += blockDim.x) {
        const std::uint32_t row = flat / state_size;
        const std::uint32_t dimension = flat % state_size;
        float solved_value = 0.0F;
        float solved_key = 0.0F;
        for (std::uint32_t source = 0; source <= row; ++source) {
            const float coefficient = inverse[static_cast<std::size_t>(row) * chunk_size + source];
            solved_value += coefficient * workspace.new_values[work_values_base
                + static_cast<std::size_t>(source) * state_size + dimension];
            solved_key += coefficient * workspace.decayed_keys[work_values_base
                + static_cast<std::size_t>(source) * state_size + dimension];
        }
        workspace.solved_values[work_values_base + flat] = solved_value;
        workspace.solved_keys[work_values_base + flat] = solved_key;
    }
    __syncthreads();

    for (std::size_t flat = tid; flat < static_cast<std::size_t>(chunk_size) * state_size;
         flat += blockDim.x) {
        const std::uint32_t row = flat / state_size;
        const std::uint32_t dimension = flat % state_size;
        float predicted = 0.0F;
        for (std::uint32_t column = 0; column < state_size; ++column) {
            predicted += workspace.solved_keys[work_values_base
                + static_cast<std::size_t>(row) * state_size + column]
                * state[state_base + static_cast<std::size_t>(column) * state_size + dimension];
        }
        workspace.corrected_values[work_values_base + flat] =
            workspace.solved_values[work_values_base + flat] - predicted;
    }
    __syncthreads();

    for (std::size_t flat = tid; flat < static_cast<std::size_t>(chunk_size) * state_size;
         flat += blockDim.x) {
        const std::uint32_t row = flat / state_size;
        const std::uint32_t dimension = flat % state_size;
        const std::size_t query_base = static_cast<std::size_t>(chunk_start + row)
            * key_stride + static_cast<std::size_t>(key_head) * state_size;
        const float query_decay = row_decay[row];
        float inter = 0.0F;
        for (std::uint32_t column = 0; column < state_size; ++column) {
            inter += query[query_base + column] * query_scale * query_decay
                * state[state_base + static_cast<std::size_t>(column) * state_size + dimension];
        }
        float intra = 0.0F;
        for (std::uint32_t source = 0; source <= row; ++source) {
            intra += intra_attention[static_cast<std::size_t>(row) * chunk_size + source]
                * workspace.corrected_values[work_values_base
                + static_cast<std::size_t>(source) * state_size + dimension];
        }
        output[static_cast<std::size_t>(chunk_start + row) * value_stride
            + static_cast<std::size_t>(head) * state_size + dimension] =
            inter + intra;
    }
    __syncthreads();

    const float chunk_decay = row_decay[chunk_size - 1];
    for (std::size_t flat = tid; flat < static_cast<std::size_t>(state_size) * state_size;
         flat += blockDim.x) {
        const std::uint32_t column = flat / state_size;
        const std::uint32_t dimension = flat % state_size;
        float update = 0.0F;
        for (std::uint32_t row = 0; row < chunk_size; ++row) {
            const std::size_t key_base = static_cast<std::size_t>(chunk_start + row)
                * key_stride + static_cast<std::size_t>(key_head) * state_size;
            update += key[key_base + column]
                * backward_decay[row]
                * workspace.corrected_values[work_values_base
                + static_cast<std::size_t>(row) * state_size + dimension];
        }
        const std::size_t state_index = state_base
            + static_cast<std::size_t>(column) * state_size + dimension;
        state[state_index] = state[state_index] * chunk_decay + update;
    }
}

__device__ __forceinline__ float m12_gdn_direct_wave_sum(float value) {
    for (std::uint32_t offset = 32; offset > 0; offset >>= 1) {
        value += __shfl_down(value, offset, 64);
    }
    return value;
}

// Adapted from mxxm-t/mx-llama.cpp @ 2e9d29fe, ggml/src/ggml-cuda/
// gated_delta_net_chunk.cu. The MIT-licensed source is reduced here to the
// fixed Qwen3.8 non-KDA, single-sequence contract. The local state remains
// MIInfer's key x value layout; only the register shard is transposed.
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx906__)
#define MIINFER_MX_GDN_DPP_ADD(name, barrier, dpp_ctrl)                         \
    static __device__ __forceinline__ float name(const float x) {               \
        float r;                                                                \
        asm volatile(                                                           \
            barrier                                                             \
            "v_add_f32_dpp %0, %1, %1 " dpp_ctrl " row_mask:0xf bank_mask:0xf" \
            : "=v"(r) : "v"(x) : "memory");                                   \
        return r;                                                               \
    }

MIINFER_MX_GDN_DPP_ADD(mx_gdn_dpp_add_xor1, "s_nop 4\n", "quad_perm:[1,0,3,2]")
MIINFER_MX_GDN_DPP_ADD(mx_gdn_dpp_add_xor2, "s_nop 1\n", "quad_perm:[2,3,0,1]")
MIINFER_MX_GDN_DPP_ADD(mx_gdn_dpp_add_xor8, "s_nop 1\n", "row_ror:8")

#undef MIINFER_MX_GDN_DPP_ADD

static __device__ __forceinline__ float mx_gdn_dpp_xfer_xor4(const float x) {
    int d;
    asm volatile("v_mov_b32 %0, %1\n"
                 "s_nop 1\n"
                 "v_mov_b32_dpp %0, %1 row_shl:4 row_mask:0xf bank_mask:0x5\n"
                 "v_mov_b32_dpp %0, %1 row_shr:4 row_mask:0xf bank_mask:0xa\n"
                 : "=v"(d) : "v"(__float_as_int(x)) : "memory");
    return __int_as_float(d);
}

static __device__ __forceinline__ float mx_gdn_dpp_xfer_xor16(const float x) {
    int d;
    asm volatile("ds_swizzle_b32 %0, %1 offset:swizzle(SWAP,16)\n"
                 "s_waitcnt lgkmcnt(0)\n"
                 : "=v"(d) : "v"(__float_as_int(x)) : "memory");
    return __int_as_float(d);
}
#endif

template <int width>
static __device__ __forceinline__ float mx_gdn_wave_sum(const float x) {
    static_assert(width >= 1 && (width & (width - 1)) == 0,
                  "wave reduction width must be a power of two");
    float result = x;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx906__)
    if constexpr (width >= 2) result = mx_gdn_dpp_add_xor1(result);
    if constexpr (width >= 4) result = mx_gdn_dpp_add_xor2(result);
    if constexpr (width >= 8) result += mx_gdn_dpp_xfer_xor4(result);
    if constexpr (width >= 16) result = mx_gdn_dpp_add_xor8(result);
    if constexpr (width >= 32) result += mx_gdn_dpp_xfer_xor16(result);
    if constexpr (width >= 64) {
        result += __shfl_xor_sync(0xffffffffffffffffULL, result, 32, 64);
    }
#else
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        result += __shfl_xor(result, offset, width);
    }
#endif
    return result;
}

template <int StateSize, int ChunkSize, int ColumnsPerWave>
__launch_bounds__(128, 2) __global__ void mx_gdn_chunk_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ decay,
    float* __restrict__ state,
    float* __restrict__ output,
    std::uint32_t token_count,
    std::uint32_t key_heads,
    std::uint32_t value_heads) {
    constexpr int kWaveSize = 64;
    constexpr int kRowsPerLane = StateSize / kWaveSize;
    const std::uint32_t value_head = blockIdx.x;
    const std::uint32_t lane = threadIdx.x;
    const std::uint32_t wave = threadIdx.y;
    const std::uint32_t base_column =
        (blockIdx.z * blockDim.y + wave) * ColumnsPerWave;
    if (value_head >= value_heads) return;

    const std::uint32_t key_head = value_head % key_heads;
    const std::size_t state_base = static_cast<std::size_t>(value_head)
        * StateSize * StateSize;
    float shard[kRowsPerLane][ColumnsPerWave] = {};
    for (int column_offset = 0; column_offset < ColumnsPerWave; ++column_offset) {
        const std::uint32_t column = base_column + column_offset;
        for (int row = 0; row < kRowsPerLane; ++row) {
            const std::uint32_t index = row * kWaveSize + lane;
            if (column < StateSize) {
                shard[row][column_offset] = state[state_base
                    + static_cast<std::size_t>(index) * StateSize + column];
            }
        }
    }

    constexpr float kScale = 0.08838834764831843F;
    const std::uint32_t chunk_count = (token_count + ChunkSize - 1) / ChunkSize;
    for (std::uint32_t chunk = 0; chunk < chunk_count; ++chunk) {
        const std::uint32_t chunk_start = chunk * ChunkSize;
        const std::uint32_t local_tokens =
            min(ChunkSize, token_count - chunk_start);
        for (std::uint32_t local = 0; local < local_tokens; ++local) {
            const std::uint32_t token = chunk_start + local;
            const std::size_t qk_base =
                (static_cast<std::size_t>(token) * key_heads + key_head) * StateSize;
            const std::size_t value_base =
                (static_cast<std::size_t>(token) * value_heads + value_head) * StateSize;
            float key_shard[kRowsPerLane];
            float query_shard[kRowsPerLane];
            for (int row = 0; row < kRowsPerLane; ++row) {
                const std::uint32_t index = row * kWaveSize + lane;
                key_shard[row] = key[qk_base + index];
                query_shard[row] = query[qk_base + index];
            }
            const float beta_value = beta[static_cast<std::size_t>(token) * value_heads
                                          + value_head];
            const float decay_value = decay[static_cast<std::size_t>(token) * value_heads
                                            + value_head];

            for (int column_offset = 0; column_offset < ColumnsPerWave; ++column_offset) {
                const std::uint32_t column = base_column + column_offset;
                if (column >= StateSize) break;
                float key_state_dot = 0.0F;
                for (int row = 0; row < kRowsPerLane; ++row) {
                    key_state_dot += shard[row][column_offset] * key_shard[row];
                }
                const float delta = (value[value_base + column]
                    - decay_value * mx_gdn_wave_sum<kWaveSize>(key_state_dot)) * beta_value;
                float query_state_dot = 0.0F;
                for (int row = 0; row < kRowsPerLane; ++row) {
                    shard[row][column_offset] = decay_value * shard[row][column_offset]
                        + key_shard[row] * delta;
                    query_state_dot += shard[row][column_offset] * query_shard[row];
                }
                const float attention = mx_gdn_wave_sum<kWaveSize>(query_state_dot);
                if (lane == 0) {
                    output[(static_cast<std::size_t>(token) * value_heads + value_head)
                        * StateSize + column] = attention * kScale;
                }
            }
        }
    }

    for (int column_offset = 0; column_offset < ColumnsPerWave; ++column_offset) {
        const std::uint32_t column = base_column + column_offset;
        if (column >= StateSize) break;
        for (int row = 0; row < kRowsPerLane; ++row) {
            const std::uint32_t index = row * kWaveSize + lane;
            state[state_base + static_cast<std::size_t>(index) * StateSize + column] =
                shard[row][column_offset];
        }
    }
}

// Matches mx's register-resident scan while retaining MIInfer's canonical
// state orientation: each wave owns two value rows and each lane two key
// columns, so the state shard stays in registers across the complete chunk.
__launch_bounds__(128, 2) __global__ void m12_gdn_direct_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ decay,
    float* __restrict__ state,
    float* __restrict__ output,
    std::uint32_t token_count,
    std::uint32_t key_heads,
    std::uint32_t value_heads,
    std::uint32_t state_size) {
    const std::uint32_t head = blockIdx.x;
    constexpr std::uint32_t kWave = 64;
    constexpr std::uint32_t kRowsPerWave = 2;
    const std::uint32_t lane = threadIdx.x;
    const std::uint32_t warp = threadIdx.y;
    const std::uint32_t key_head = head % key_heads;
    const std::uint32_t row_base = (blockIdx.y * blockDim.y + warp) * kRowsPerWave;
    if (head >= value_heads || state_size != 128 || row_base >= state_size) return;

    const std::size_t state_base = static_cast<std::size_t>(head) * state_size * state_size;
    float shard[kRowsPerWave][2] = {};
    for (std::uint32_t row = 0; row < kRowsPerWave; ++row) {
        const std::uint32_t value_row = row_base + row;
        if (value_row >= state_size) continue;
        shard[row][0] = state[state_base + static_cast<std::size_t>(lane) * state_size + value_row];
        shard[row][1] = state[state_base + static_cast<std::size_t>(lane + kWave) * state_size + value_row];
    }

    extern __shared__ float shared_inputs[];
    float* shared_query = shared_inputs;
    float* shared_key = shared_inputs + state_size;

    for (std::uint32_t token = 0; token < token_count; ++token) {
        const std::size_t qk_base = static_cast<std::size_t>(token) * key_heads * state_size
            + static_cast<std::size_t>(key_head) * state_size;
        shared_query[lane] = query[qk_base + lane];
        shared_query[lane + kWave] = query[qk_base + lane + kWave];
        shared_key[lane] = key[qk_base + lane];
        shared_key[lane + kWave] = key[qk_base + lane + kWave];
        __syncthreads();
        const float beta_value = beta[static_cast<std::size_t>(token) * value_heads + head];
        const float decay_value = decay[static_cast<std::size_t>(token) * value_heads + head];
        const std::size_t value_base = static_cast<std::size_t>(token) * value_heads * state_size
            + static_cast<std::size_t>(head) * state_size;
        for (std::uint32_t row = 0; row < kRowsPerWave; ++row) {
            const std::uint32_t value_row = row_base + row;
            if (value_row >= state_size) continue;
            const float key_partial = shard[row][0] * shared_key[lane]
                + shard[row][1] * shared_key[lane + kWave];
            const float key_dot = __shfl(decay_value * m12_gdn_direct_wave_sum(key_partial), 0, 64);
            const float delta = (value[value_base + value_row] - key_dot) * beta_value;
            shard[row][0] = shard[row][0] * decay_value + delta * shared_key[lane];
            shard[row][1] = shard[row][1] * decay_value + delta * shared_key[lane + kWave];
            const float query_partial = shard[row][0] * shared_query[lane]
                + shard[row][1] * shared_query[lane + kWave];
            const float query_dot = __shfl(m12_gdn_direct_wave_sum(query_partial), 0, 64);
            if (lane == 0) {
                output[(static_cast<std::size_t>(token) * value_heads + head) * state_size + value_row] =
                    query_dot * rsqrtf(static_cast<float>(state_size));
            }
        }
        __syncthreads();
    }

    for (std::uint32_t row = 0; row < kRowsPerWave; ++row) {
        const std::uint32_t value_row = row_base + row;
        if (value_row >= state_size) continue;
        state[state_base + static_cast<std::size_t>(lane) * state_size + value_row] = shard[row][0];
        state[state_base + static_cast<std::size_t>(lane + kWave) * state_size + value_row] = shard[row][1];
    }
}

} // namespace

namespace miinfer {

void launch_m12_gdn_chunk(
    const float* query, const float* key, const float* value,
    const float* beta, const float* decay, float* state, float* output,
    M12GdnChunkWorkspace workspace, std::uint32_t token_count,
    std::uint32_t chunk_start, std::uint32_t key_heads,
    std::uint32_t value_heads, std::uint32_t state_size,
    std::uint32_t chunk_size, hipStream_t stream) {
    if (query == nullptr || key == nullptr || value == nullptr || beta == nullptr
        || decay == nullptr || state == nullptr || output == nullptr
        || workspace.new_values == nullptr || workspace.decayed_keys == nullptr
        || workspace.solved_values == nullptr || workspace.solved_keys == nullptr
        || workspace.corrected_values == nullptr
        || token_count == 0 || chunk_start + chunk_size > token_count
        || key_heads == 0 || value_heads == 0 || value_heads % key_heads != 0
        || state_size != 128 || chunk_size != 64 || chunk_start % chunk_size != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    const std::size_t shared_bytes =
        (3 * chunk_size + 3 * chunk_size * chunk_size) * sizeof(float);
    hipLaunchKernelGGL(m12_gdn_chunk_kernel, dim3(value_heads), dim3(kThreads), shared_bytes, stream,
                       query, key, value, beta, decay, state, output, workspace,
                       token_count, chunk_start, key_heads, value_heads, state_size, chunk_size);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_m12_gdn_direct(
    const float* query, const float* key, const float* value,
    const float* beta, const float* decay, float* state, float* output,
    std::uint32_t token_count, std::uint32_t key_heads,
    std::uint32_t value_heads, std::uint32_t state_size, hipStream_t stream) {
    if (query == nullptr || key == nullptr || value == nullptr || beta == nullptr
        || decay == nullptr || state == nullptr || output == nullptr
        || token_count == 0 || token_count > 512 || key_heads == 0 || value_heads == 0
        || state_size != 128 || value_heads % key_heads != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(m12_gdn_direct_kernel,
                       dim3(value_heads, (state_size + 3) / 4), dim3(64, 2),
                       2 * state_size * sizeof(float), stream,
                       query, key, value, beta, decay, state, output, token_count,
                       key_heads, value_heads, state_size);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_mx_gdn_chunk(
    const float* query, const float* key, const float* value,
    const float* beta, const float* decay, float* state, float* output,
    std::uint32_t token_count, std::uint32_t key_heads,
    std::uint32_t value_heads, std::uint32_t state_size, hipStream_t stream) {
    if (query == nullptr || key == nullptr || value == nullptr || beta == nullptr
        || decay == nullptr || state == nullptr || output == nullptr
        || token_count == 0 || token_count > 512 || token_count % 64 != 0
        || key_heads == 0 || value_heads == 0 || value_heads % key_heads != 0
        || state_size != 128) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL((mx_gdn_chunk_kernel<128, 64, 2>),
                       dim3(value_heads, 1, (state_size + 3) / 4), dim3(64, 2),
                       0, stream, query, key, value, beta, decay, state, output,
                       token_count, key_heads, value_heads);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_m12_gdn_postprocess(
    const float* recurrent_output, const float* gate, const float* ssm_norm,
    float* gated_output, std::uint32_t token_count, std::uint32_t value_heads,
    std::uint32_t state_size, float epsilon, hipStream_t stream) {
    if (recurrent_output == nullptr || gate == nullptr || ssm_norm == nullptr
        || gated_output == nullptr || token_count == 0 || value_heads == 0
        || state_size != 128 || epsilon <= 0.0F) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(m12_gdn_postprocess_kernel,
                       dim3(token_count * value_heads), dim3(state_size), 0, stream,
                       recurrent_output, gate, ssm_norm, gated_output,
                       value_heads, state_size, epsilon);
    MIINFER_HIP_CHECK(hipGetLastError());
}

} // namespace miinfer
