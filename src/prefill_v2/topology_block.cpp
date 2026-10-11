#include "miinfer/prefill_v2/topology_block.hpp"
#include "miinfer/hip_check.hpp"

#include <hip/hip_runtime.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace miinfer::prefill_v2 {

PrefillV2TopologyBlock::PrefillV2TopologyBlock(
    const miinfer::Qwen35Model& model, std::size_t block_index, bool mmq_gateup_only)
    : block_index_(block_index),
      gdn0_(model, block_index * kLayersPerBlock + 0, mmq_gateup_only),
      gdn1_(model, block_index * kLayersPerBlock + 1, mmq_gateup_only),
      gdn2_(model, block_index * kLayersPerBlock + 2, mmq_gateup_only),
      gqa3_(model, block_index * kLayersPerBlock + 3, mmq_gateup_only) {}

void PrefillV2TopologyBlock::forward(
    float* d_ping,
    float* d_pong,
    float* d_final_output,
    const RecurrentLayerState& state0_in,
    RecurrentLayerState& state0_out,
    const RecurrentLayerState& state1_in,
    RecurrentLayerState& state1_out,
    const RecurrentLayerState& state2_in,
    RecurrentLayerState& state2_out,
    AttentionKvCacheView kv_cache3,
    PrefillV2Workspace& ws,
    std::uint32_t base_position,
    std::uint32_t token_count,
    hipStream_t stream,
    const DevicePrefillState* prefill_state,
    std::vector<float>* layer_trace,
    std::vector<float>* layer0_row_trace) const {

    const auto capture = [&](std::size_t layer, const float* hidden) {
        if (layer_trace == nullptr) return;
        const auto source = hidden + (token_count - 1) * kHidden;
        auto* target = layer_trace->data() + (layer + 1) * kHidden;
        MIINFER_HIP_CHECK(hipMemcpy(target, source, kHidden * sizeof(float), hipMemcpyDeviceToHost));
    };

    // Layer 0 (GDN): d_ping -> d_pong
    gdn0_.forward(d_ping, d_pong, state0_in, state0_out, ws, token_count, stream, prefill_state);
    capture(block_index_ * kLayersPerBlock, d_pong);
    if (block_index_ == 0 && layer0_row_trace != nullptr) {
        MIINFER_HIP_CHECK(hipMemcpy(layer0_row_trace->data(), d_pong,
                                    token_count * kHidden * sizeof(float), hipMemcpyDeviceToHost));
    }

    // Layer 1 (GDN): d_pong -> d_ping
    gdn1_.forward(d_pong, d_ping, state1_in, state1_out, ws, token_count, stream, prefill_state);
    capture(block_index_ * kLayersPerBlock + 1, d_ping);

    // Layer 2 (GDN): d_ping -> d_pong
    gdn2_.forward(d_ping, d_pong, state2_in, state2_out, ws, token_count, stream, prefill_state);
    capture(block_index_ * kLayersPerBlock + 2, d_pong);

    // Layer 3 (GQA): d_pong -> d_final_output
    gqa3_.forward(d_pong, d_final_output, kv_cache3, ws, base_position, token_count, stream, prefill_state);
    capture(block_index_ * kLayersPerBlock + 3, d_final_output);
}

void PrefillV2TopologyBlock::decode(
    float* d_ping,
    float* d_pong,
    float* d_final_output,
    RecurrentLayerState& state0,
    RecurrentLayerState& state1,
    RecurrentLayerState& state2,
    AttentionKvCacheView kv_cache3,
    PrefillV2Workspace& ws,
    std::uint32_t position,
    const DeviceDecodeState* decode_state,
    hipStream_t stream,
    RecurrentLayerGraphTrace* graph_trace,
    const DeviceDecodeState* attention_decode_state) const {

    // Layer 0 (GDN): d_ping -> d_pong
    gdn0_.decode(d_ping, d_pong, state0, ws, decode_state, stream, graph_trace);

    // Layer 1 (GDN): d_pong -> d_ping
    gdn1_.decode(d_pong, d_ping, state1, ws, decode_state, stream, graph_trace);

    // Layer 2 (GDN): d_ping -> d_pong
    gdn2_.decode(d_ping, d_pong, state2, ws, decode_state, stream, graph_trace);

    // Layer 3 (GQA): d_pong -> d_final_output
    gqa3_.decode(d_pong, d_final_output, kv_cache3, ws, position,
        attention_decode_state != nullptr ? attention_decode_state : decode_state, stream, graph_trace);
    const char* trace_prefix = std::getenv("MIINFER_LC_OP_TRACE_PREFIX");
    const char* trace_decode = std::getenv("MIINFER_LC_DECODE_TRACE");
    static bool block_trace_captured = false;
    static bool block1_trace_captured = false;
    static bool block2_trace_captured = false;
    if (block_index_ <= 2 && graph_trace != nullptr && decode_state != nullptr) {
        graph_trace->copy_async(block_index_ == 0 ? "block0_gqa3_output"
            : block_index_ == 1 ? "block1_gqa3_output" : "block2_gqa3_output", d_final_output,
                                kHidden * sizeof(float), stream);
    } else if (block_index_ <= 2 && trace_prefix != nullptr
        && trace_decode != nullptr && std::string_view(trace_decode) != "0"
        && decode_state == nullptr && !(block_index_ == 0 ? block_trace_captured
            : block_index_ == 1 ? block1_trace_captured : block2_trace_captured)) {
        const auto path = std::filesystem::path(std::string(trace_prefix) + ".decode-pos"
            + std::to_string(position) + (block_index_ == 0 ? ".block0.gqa3_output.f32"
                : block_index_ == 1 ? ".block1.gqa3_output.f32" : ".block2.gqa3_output.f32"));
        if (std::filesystem::exists(path))
            throw std::runtime_error("refusing to overwrite M31 block trace: " + path.string());
        std::vector<float> host(kHidden);
        MIINFER_HIP_CHECK(hipMemcpy(host.data(), d_final_output, kHidden * sizeof(float),
                                    hipMemcpyDeviceToHost));
        std::ofstream output(path, std::ios::binary | std::ios::out);
        if (!output) throw std::runtime_error("cannot create M31 block trace: " + path.string());
        output.write(reinterpret_cast<const char*>(host.data()),
                     static_cast<std::streamsize>(host.size() * sizeof(float)));
        if (!output) throw std::runtime_error("failed writing M31 block trace: " + path.string());
        if (block_index_ == 0) block_trace_captured = true;
        else if (block_index_ == 1) block1_trace_captured = true;
        else block2_trace_captured = true;
    }
}

void PrefillV2TopologyBlock::forward_profiled(
    float* d_ping,
    float* d_pong,
    float* d_final_output,
    const RecurrentLayerState& state0_in,
    RecurrentLayerState& state0_out,
    const RecurrentLayerState& state1_in,
    RecurrentLayerState& state1_out,
    const RecurrentLayerState& state2_in,
    RecurrentLayerState& state2_out,
    AttentionKvCacheView kv_cache3,
    PrefillV2Workspace& ws,
    std::uint32_t base_position,
    std::uint32_t token_count,
    TopologyBlockProfileBreakdown& breakdown,
    hipStream_t stream) const {

    hipEvent_t ev_start, ev_l0, ev_l1, ev_l2, ev_l3;
    MIINFER_HIP_CHECK(hipEventCreate(&ev_start));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_l0));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_l1));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_l2));
    MIINFER_HIP_CHECK(hipEventCreate(&ev_l3));

    MIINFER_HIP_CHECK(hipEventRecord(ev_start, stream));

    // Layer 0 (GDN): d_ping -> d_pong
    gdn0_.forward(d_ping, d_pong, state0_in, state0_out, ws, token_count, stream);
    MIINFER_HIP_CHECK(hipEventRecord(ev_l0, stream));

    // Layer 1 (GDN): d_pong -> d_ping
    gdn1_.forward(d_pong, d_ping, state1_in, state1_out, ws, token_count, stream);
    MIINFER_HIP_CHECK(hipEventRecord(ev_l1, stream));

    // Layer 2 (GDN): d_ping -> d_pong
    gdn2_.forward(d_ping, d_pong, state2_in, state2_out, ws, token_count, stream);
    MIINFER_HIP_CHECK(hipEventRecord(ev_l2, stream));

    // Layer 3 (GQA): d_pong -> d_final_output
    gqa3_.forward(d_pong, d_final_output, kv_cache3, ws, base_position, token_count, stream);
    MIINFER_HIP_CHECK(hipEventRecord(ev_l3, stream));

    MIINFER_HIP_CHECK(hipEventSynchronize(ev_l3));

    const auto elapsed = [](hipEvent_t a, hipEvent_t b) {
        float ms = 0.0F;
        MIINFER_HIP_CHECK(hipEventElapsedTime(&ms, a, b));
        return static_cast<double>(ms);
    };

    breakdown.gdn0_ms = elapsed(ev_start, ev_l0);
    breakdown.gdn1_ms = elapsed(ev_l0, ev_l1);
    breakdown.gdn2_ms = elapsed(ev_l1, ev_l2);
    breakdown.gqa3_ms = elapsed(ev_l2, ev_l3);
    breakdown.total_block_ms = elapsed(ev_start, ev_l3);

    (void)hipEventDestroy(ev_start);
    (void)hipEventDestroy(ev_l0);
    (void)hipEventDestroy(ev_l1);
    (void)hipEventDestroy(ev_l2);
    (void)hipEventDestroy(ev_l3);
}

} // namespace miinfer::prefill_v2
