#include "miinfer/m12_gdn_chunk.hpp"

#include "miinfer/hip_check.hpp"

#include <hip/hip_runtime.h>

namespace {

constexpr unsigned kThreads = 256;

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
    const std::uint32_t head = blockIdx.x;
    const std::uint32_t tid = threadIdx.x;
    if (head >= value_heads) return;

    const std::uint32_t key_head = head % key_heads;
    const std::size_t key_stride = static_cast<std::size_t>(token_count) * state_size;
    const std::size_t value_stride = key_stride;
    const std::size_t state_stride = static_cast<std::size_t>(state_size) * state_size;
    const std::size_t key_head_base = static_cast<std::size_t>(key_head) * key_stride;
    const std::size_t value_head_base = static_cast<std::size_t>(head) * value_stride;
    const std::size_t scalar_head_base = static_cast<std::size_t>(head) * token_count;
    const std::size_t state_base = static_cast<std::size_t>(head) * state_stride;
    const std::size_t work_base = static_cast<std::size_t>(head) * chunk_size;
    const std::size_t work_matrix_base = static_cast<std::size_t>(head)
        * chunk_size * chunk_size;
    const std::size_t work_values_base = static_cast<std::size_t>(head)
        * chunk_size * state_size;
    const float query_scale = rsqrtf(static_cast<float>(state_size));

    // The current feasibility kernel intentionally launches one workgroup per
    // value head and advances chunks in host order so state dependencies stay
    // explicit. It is a correctness/performance probe, not the final schedule.
    if (tid == 0) {
        float running = 0.0F;
        for (std::uint32_t row = 0; row < chunk_size; ++row) {
            running += decay[scalar_head_base + chunk_start + row];
            workspace.cumulative[work_base + row] = running;
        }
    }
    __syncthreads();

    for (std::size_t flat = tid; flat < static_cast<std::size_t>(chunk_size) * chunk_size;
         flat += blockDim.x) {
        const std::uint32_t row = flat / chunk_size;
        const std::uint32_t column = flat % chunk_size;
        if (column > row) {
            workspace.pairwise_decay[work_matrix_base + flat] = 0.0F;
            workspace.system[work_matrix_base + flat] = 0.0F;
            workspace.intra_attention[work_matrix_base + flat] = 0.0F;
            continue;
        }
        const float pairwise = expf(workspace.cumulative[work_base + row]
                                    - workspace.cumulative[work_base + column]);
        float key_beta_dot = 0.0F;
        float query_key_dot = 0.0F;
        const std::size_t current_key_base = key_head_base
            + static_cast<std::size_t>(chunk_start + row) * state_size;
        const std::size_t previous_key_base = key_head_base
            + static_cast<std::size_t>(chunk_start + column) * state_size;
        for (std::uint32_t dimension = 0; dimension < state_size; ++dimension) {
            const float current_key = key[current_key_base + dimension];
            const float previous_key = key[previous_key_base + dimension];
            key_beta_dot += current_key * beta[scalar_head_base + chunk_start + row]
                            * previous_key;
            query_key_dot += query[current_key_base + dimension] * previous_key;
        }
        workspace.pairwise_decay[work_matrix_base + flat] = pairwise;
        workspace.system[work_matrix_base + flat] = row == column
            ? 0.0F : key_beta_dot * pairwise;
        workspace.intra_attention[work_matrix_base + flat] =
            query_key_dot * pairwise * query_scale;
    }

    for (std::size_t flat = tid; flat < static_cast<std::size_t>(chunk_size) * state_size;
         flat += blockDim.x) {
        const std::uint32_t row = flat / state_size;
        const std::uint32_t dimension = flat % state_size;
        const std::size_t source_base = key_head_base
            + static_cast<std::size_t>(chunk_start + row) * state_size;
        const std::size_t value_base = value_head_base
            + static_cast<std::size_t>(chunk_start + row) * state_size;
        const float beta_value = beta[scalar_head_base + chunk_start + row];
        const float decay_factor = expf(workspace.cumulative[work_base + row]);
        workspace.new_values[work_values_base + flat] = value[value_base + dimension]
            * beta_value;
        workspace.decayed_keys[work_values_base + flat] = key[source_base + dimension]
            * beta_value * decay_factor;
    }
    __syncthreads();

    for (std::size_t flat = tid; flat < static_cast<std::size_t>(chunk_size) * chunk_size;
         flat += blockDim.x) {
        workspace.inverse[work_matrix_base + flat] = (flat / chunk_size == flat % chunk_size)
            ? 1.0F : 0.0F;
    }
    __syncthreads();
    // Forward substitution for the unit-lower-triangular WY system.
    for (std::uint32_t row = 0; row < chunk_size; ++row) {
        if (tid < chunk_size) {
            float solved = tid == row ? 1.0F : 0.0F;
            if (tid < row) solved = 0.0F;
            for (std::uint32_t column = 0; column < row; ++column) {
                solved -= workspace.system[work_matrix_base
                    + static_cast<std::size_t>(row) * chunk_size + column]
                    * workspace.inverse[work_matrix_base
                    + static_cast<std::size_t>(column) * chunk_size + tid];
            }
            workspace.inverse[work_matrix_base
                + static_cast<std::size_t>(row) * chunk_size + tid] = solved;
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
            const float coefficient = workspace.inverse[work_matrix_base
                + static_cast<std::size_t>(row) * chunk_size + source];
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

    for (std::size_t flat = tid; flat < static_cast<std::size_t>(chunk_size) * state_size;
         flat += blockDim.x) {
        const std::uint32_t row = flat / state_size;
        const std::uint32_t dimension = flat % state_size;
        const std::size_t query_base = key_head_base
            + static_cast<std::size_t>(chunk_start + row) * state_size;
        const float query_decay = expf(workspace.cumulative[work_base + row]);
        float inter = 0.0F;
        for (std::uint32_t column = 0; column < state_size; ++column) {
            inter += query[query_base + column] * query_scale * query_decay
                * state[state_base + static_cast<std::size_t>(column) * state_size + dimension];
        }
        float intra = 0.0F;
        for (std::uint32_t source = 0; source <= row; ++source) {
            intra += workspace.intra_attention[work_matrix_base
                + static_cast<std::size_t>(row) * chunk_size + source]
                * workspace.corrected_values[work_values_base
                + static_cast<std::size_t>(source) * state_size + dimension];
        }
        output[static_cast<std::size_t>(head) * value_stride
            + static_cast<std::size_t>(chunk_start + row) * state_size + dimension] =
            inter + intra;
    }
    __syncthreads();

    const float chunk_decay = expf(workspace.cumulative[work_base + chunk_size - 1]);
    for (std::size_t flat = tid; flat < static_cast<std::size_t>(state_size) * state_size;
         flat += blockDim.x) {
        const std::uint32_t column = flat / state_size;
        const std::uint32_t dimension = flat % state_size;
        float update = 0.0F;
        for (std::uint32_t row = 0; row < chunk_size; ++row) {
            const std::size_t key_base = key_head_base
                + static_cast<std::size_t>(chunk_start + row) * state_size;
            update += key[key_base + column]
                * expf(workspace.cumulative[work_base + chunk_size - 1]
                       - workspace.cumulative[work_base + row])
                * workspace.corrected_values[work_values_base
                + static_cast<std::size_t>(row) * state_size + dimension];
        }
        const std::size_t state_index = state_base
            + static_cast<std::size_t>(column) * state_size + dimension;
        state[state_index] = state[state_index] * chunk_decay + update;
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
        || workspace.cumulative == nullptr || workspace.pairwise_decay == nullptr
        || workspace.system == nullptr || workspace.inverse == nullptr
        || workspace.new_values == nullptr || workspace.decayed_keys == nullptr
        || workspace.solved_values == nullptr || workspace.solved_keys == nullptr
        || workspace.corrected_values == nullptr || workspace.intra_attention == nullptr
        || token_count == 0 || chunk_start + chunk_size > token_count
        || key_heads == 0 || value_heads == 0 || value_heads % key_heads != 0
        || state_size != 128 || chunk_size != 64 || chunk_start % chunk_size != 0) {
        MIINFER_HIP_CHECK(hipErrorInvalidValue);
    }
    hipLaunchKernelGGL(m12_gdn_chunk_kernel, dim3(value_heads), dim3(kThreads), 0, stream,
                       query, key, value, beta, decay, state, output, workspace,
                       token_count, chunk_start, key_heads, value_heads, state_size, chunk_size);
    MIINFER_HIP_CHECK(hipGetLastError());
}

} // namespace miinfer
