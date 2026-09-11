#include "miinfer/device_validation.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/m12_dense_stage.hpp"
#include "miinfer/qwen35_model.hpp"

#include "../tools/qwen35_gpu_pipeline.hpp"

#include <hip/hip_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct RawBuffer {
    void* pointer = nullptr;

    explicit RawBuffer(std::size_t bytes) {
        MIINFER_HIP_CHECK(hipMalloc(&pointer, bytes));
    }
    ~RawBuffer() { if (pointer != nullptr) (void)hipFree(pointer); }
    RawBuffer(const RawBuffer&) = delete;
    RawBuffer& operator=(const RawBuffer&) = delete;

    template <typename T>
    T* as() const { return static_cast<T*>(pointer); }
};

struct StageEvents {
    FullAttentionLayer::StageProfile profile;

    StageEvents() {
        for (auto& event : profile.start) MIINFER_HIP_CHECK(hipEventCreate(&event));
        for (auto& event : profile.end) MIINFER_HIP_CHECK(hipEventCreate(&event));
        MIINFER_HIP_CHECK(hipEventCreate(&profile.tail_start));
        MIINFER_HIP_CHECK(hipEventCreate(&profile.tail_end));
        MIINFER_HIP_CHECK(hipEventCreate(&profile.prepare_start));
        MIINFER_HIP_CHECK(hipEventCreate(&profile.prepare_end));
        MIINFER_HIP_CHECK(hipEventCreate(&profile.ordered_start));
        MIINFER_HIP_CHECK(hipEventCreate(&profile.ordered_end));
    }

    ~StageEvents() {
        for (auto& event : profile.start) if (event != nullptr) (void)hipEventDestroy(event);
        for (auto& event : profile.end) if (event != nullptr) (void)hipEventDestroy(event);
        for (auto* event : {profile.tail_start, profile.tail_end, profile.prepare_start,
                            profile.prepare_end, profile.ordered_start, profile.ordered_end}) {
            if (event != nullptr) (void)hipEventDestroy(event);
        }
    }

    void clear() {
        profile.stage_recorded.fill(false);
        profile.tail_recorded = false;
        profile.prepare_recorded = false;
        profile.ordered_recorded = false;
    }
};

struct EventPair {
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;

    EventPair() {
        MIINFER_HIP_CHECK(hipEventCreate(&start));
        MIINFER_HIP_CHECK(hipEventCreate(&stop));
    }
    ~EventPair() {
        if (start != nullptr) (void)hipEventDestroy(start);
        if (stop != nullptr) (void)hipEventDestroy(stop);
    }
    EventPair(const EventPair&) = delete;
    EventPair& operator=(const EventPair&) = delete;
};

struct CompareMetrics {
    float max_abs = 0.0F;
    float rmse = 0.0F;
    bool finite = true;
};

CompareMetrics compare_outputs(const std::vector<float>& actual,
                               const std::vector<float>& expected) {
    if (actual.size() != expected.size()) throw std::runtime_error("attention output size mismatch");
    CompareMetrics result;
    double squared = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) result.finite = false;
        const float error = std::fabs(actual[i] - expected[i]);
        result.max_abs = std::max(result.max_abs, error);
        squared += static_cast<double>(error) * error;
    }
    if (!actual.empty()) result.rmse = static_cast<float>(std::sqrt(squared / actual.size()));
    return result;
}

void set_environment(std::uint32_t batch, const std::string& mode) {
    setenv("MIINFER_PREFILL_LAYER_MAJOR", "1", 1);
    setenv("MIINFER_PREFILL_WIDE_CHUNK", "1", 1);
    setenv("MIINFER_PREFILL_CHUNK", std::to_string(batch).c_str(), 1);
    unsetenv("MIINFER_PREFILL_FULL_LAYER_MAJOR");
    setenv("MIINFER_PREFILL_WIDE_ATTN", "1", 1);
    setenv("MIINFER_PREFILL_WIDE_REPACKED_MMQ", "1", 1);
    setenv("MIINFER_M23_REPACKED_ROW128", "1", 1);
    setenv("MIINFER_PREFILL_REPACKED_RESIDENT_ALL", "1", 1);
    unsetenv("MIINFER_PREFILL_REPACKED_FP16");
    unsetenv("MIINFER_PREFILL_REPACKED_FP16_ATTN_QK");
    unsetenv("MIINFER_PREFILL_REPACKED_FP16_ATTN_V");
    unsetenv("MIINFER_PREFILL_REPACKED_FP16_ATTN_O");
    if (mode == "qk") setenv("MIINFER_PREFILL_REPACKED_FP16_ATTN_QK", "1", 1);
    if (mode == "v") setenv("MIINFER_PREFILL_REPACKED_FP16_ATTN_V", "1", 1);
    if (mode == "o") setenv("MIINFER_PREFILL_REPACKED_FP16_ATTN_O", "1", 1);
    setenv("MIINFER_HIP_GRAPH", "0", 1);
    setenv("MIINFER_PREFILL_WIDE_MMQ_QKV", "1", 1);
    setenv("MIINFER_PREFILL_WIDE_MMQ_SSM_OUT", "1", 1);
    setenv("MIINFER_PREFILL_WIDE_MMQ_FFN", "1", 1);
    setenv("MIINFER_FP16_KV_CACHE", "1", 1);
    setenv("MIINFER_TILED_ONLINE_ATTENTION", "1", 1);
}

void run_layer(const miinfer::Qwen35Model& model, std::uint32_t batch,
               const std::string& mode,
               double& milliseconds, StageEvents& events,
               std::array<double, 15>& stage_ms,
               std::array<double, 3>& phase_ms,
               CompareMetrics& parity, std::size_t& tracked_bytes) {
    set_environment(batch, mode);
    g_cache_capacity = 1024;
    FullAttentionLayer layer(model, 3);
    tracked_bytes = g_device_bytes;
    layer.stage_profile = &events.profile;
    layer.stage_profile_position = batch - 1;
    layer.stage_profile_chunk_base = 0;

    RawBuffer dense_weights(static_cast<std::size_t>(kFfnInner) * kHidden * sizeof(__half));
    RawBuffer dense_input(static_cast<std::size_t>(batch) * kFfnInner * sizeof(__half));
    miinfer::RocblasGemmHandle gemm_handle{};
    std::string gemm_error;
    if (!miinfer::create_rocblas_gemm_handle(gemm_handle, hipStreamPerThread, gemm_error)) {
        throw std::runtime_error(gemm_error);
    }
    layer.set_m12_dense_workspace(dense_weights.as<__half>(), dense_input.as<__half>(), &gemm_handle);

    RawBuffer input(static_cast<std::size_t>(batch) * kHidden * sizeof(float));
    RawBuffer output(static_cast<std::size_t>(batch) * kHidden * sizeof(float));
    std::vector<float> host_input(static_cast<std::size_t>(batch) * kHidden);
    for (std::size_t i = 0; i < host_input.size(); ++i) {
        host_input[i] = std::sin(0.0017F * static_cast<float>(i % kHidden)
                                  + 0.071F * static_cast<float>(i / kHidden));
    }
    MIINFER_HIP_CHECK(hipMemcpy(input.pointer, host_input.data(),
                                host_input.size() * sizeof(float), hipMemcpyHostToDevice));

    const auto execute = [&] {
        if (!layer.prepare_prefill_batch(input.as<float>(), batch, false)) {
            throw std::runtime_error("attention layer rejected wide prefill batch");
        }
        layer.finish_prefill_attention(0, batch);
        layer.finish_prefill_wide(input.as<float>(), output.as<float>(), batch, nullptr, nullptr);
    };
    execute();
    MIINFER_HIP_CHECK(hipDeviceSynchronize());

    events.clear();
    EventPair prepare_events;
    EventPair attention_events;
    EventPair wide_events;
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    MIINFER_HIP_CHECK(hipEventCreate(&start));
    MIINFER_HIP_CHECK(hipEventCreate(&stop));
    MIINFER_HIP_CHECK(hipEventRecord(start, hipStreamPerThread));
    MIINFER_HIP_CHECK(hipEventRecord(prepare_events.start, hipStreamPerThread));
    if (!layer.prepare_prefill_batch(input.as<float>(), batch, false)) {
        throw std::runtime_error("attention layer rejected wide prefill batch");
    }
    MIINFER_HIP_CHECK(hipEventRecord(prepare_events.stop, hipStreamPerThread));
    MIINFER_HIP_CHECK(hipEventRecord(attention_events.start, hipStreamPerThread));
    layer.finish_prefill_attention(0, batch);
    MIINFER_HIP_CHECK(hipEventRecord(attention_events.stop, hipStreamPerThread));
    MIINFER_HIP_CHECK(hipEventRecord(wide_events.start, hipStreamPerThread));
    layer.finish_prefill_wide(input.as<float>(), output.as<float>(), batch, nullptr, nullptr);
    MIINFER_HIP_CHECK(hipEventRecord(wide_events.stop, hipStreamPerThread));
    MIINFER_HIP_CHECK(hipEventRecord(stop, hipStreamPerThread));
    MIINFER_HIP_CHECK(hipEventSynchronize(stop));
    float elapsed = 0.0F;
    MIINFER_HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));
    milliseconds = elapsed;
    phase_ms.fill(0.0);
    for (std::size_t i = 0; i < phase_ms.size(); ++i) {
        const EventPair* pair = i == 0 ? &prepare_events : (i == 1 ? &attention_events : &wide_events);
        float phase_elapsed = 0.0F;
        MIINFER_HIP_CHECK(hipEventElapsedTime(&phase_elapsed, pair->start, pair->stop));
        phase_ms[i] = phase_elapsed;
    }
    MIINFER_HIP_CHECK(hipEventDestroy(start));
    MIINFER_HIP_CHECK(hipEventDestroy(stop));

    stage_ms.fill(0.0);
    for (std::size_t stage = 0; stage < stage_ms.size(); ++stage) {
        if (!events.profile.stage_recorded[stage]) continue;
        float stage_elapsed = 0.0F;
        MIINFER_HIP_CHECK(hipEventElapsedTime(&stage_elapsed, events.profile.start[stage],
                                              events.profile.end[stage]));
        stage_ms[stage] = stage_elapsed;
    }

    std::vector<float> wide_output(static_cast<std::size_t>(batch) * kHidden);
    MIINFER_HIP_CHECK(hipMemcpy(wide_output.data(), output.pointer,
                                wide_output.size() * sizeof(float), hipMemcpyDeviceToHost));
    FullAttentionLayer scalar_oracle(model, 3);
    scalar_oracle.reset();
    RawBuffer scalar_output(static_cast<std::size_t>(batch) * kHidden * sizeof(float));
    for (std::size_t position = 0; position < batch; ++position) {
        scalar_oracle.run(input.as<float>() + position * kHidden,
                          static_cast<std::uint32_t>(position),
                          scalar_output.as<float>() + position * kHidden);
    }
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> scalar_values(wide_output.size());
    MIINFER_HIP_CHECK(hipMemcpy(scalar_values.data(), scalar_output.pointer,
                                scalar_values.size() * sizeof(float), hipMemcpyDeviceToHost));
    parity = compare_outputs(wide_output, scalar_values);
    if (!parity.finite) throw std::runtime_error("non-finite wide/scalar attention parity output");
    if (parity.max_abs > 1.0F) {
        throw std::runtime_error("wide attention candidate exceeds scalar control tolerance");
    }
    miinfer::destroy_rocblas_gemm_handle(gemm_handle);
}

} // namespace

int main(int argc, char** argv) try {
    if (argc != 3 && argc != 4) {
        throw std::runtime_error(
            "usage: miinfer-m24-attention-layer-bakeoff MODEL.gguf BATCH [control|qk|v|o]");
    }
    const auto batch = static_cast<std::uint32_t>(std::stoul(argv[2]));
    if (batch != 128 && batch != 256 && batch != 512) {
        throw std::runtime_error("BATCH must be 128, 256, or 512");
    }
    const std::string mode = argc == 4 ? argv[3] : "control";
    if (mode != "control" && mode != "qk" && mode != "v" && mode != "o") {
        throw std::runtime_error("mode must be control, qk, v, or o");
    }
    miinfer::DeviceInfo device;
    std::string error;
    if (!miinfer::validate_gfx906_device(-1, device, error)) throw std::runtime_error(error);
    const auto model = miinfer::Qwen35Model::load(argv[1]);
    StageEvents events;
    double milliseconds = 0.0;
    std::array<double, 15> stage_ms{};
    std::array<double, 3> phase_ms{};
    CompareMetrics parity;
    std::size_t tracked_bytes = 0;
    run_layer(model, batch, mode, milliseconds, events, stage_ms, phase_ms, parity, tracked_bytes);

    static constexpr std::array<const char*, 15> names{
        "normalization_unprofiled", "qk_projection", "query_norm_rope", "k_norm_rope_kv_store",
        "v_projection", "kv_store_unprofiled", "attention", "attention_output_projection",
        "residual_post_norm", "post_normalization", "ffn_gate_up_projection", "swiglu",
        "ffn_down_projection", "residual", "unused_stage_14"};
    std::cout << std::fixed << std::setprecision(3)
              << "{\"mode\":\"" << mode << "\",\"batch\":" << batch
              << ",\"layer\":3,\"gpu_us\":" << milliseconds * 1000.0
              << ",\"tok_s\":" << (1000.0 * batch / milliseconds)
              << ",\"tracked_layer_bytes\":" << tracked_bytes
              << ",\"phases\":{\"prepare\":" << phase_ms[0]
              << ",\"attention\":" << phase_ms[1]
              << ",\"post_attention_ffn\":" << phase_ms[2]
              << "},\"scalar_control_parity\":{\"finite\":" << (parity.finite ? "true" : "false")
              << ",\"max_abs\":" << parity.max_abs
              << ",\"rmse\":" << parity.rmse
              << "},\"stages\":{";
    for (std::size_t i = 0; i < stage_ms.size(); ++i) {
        if (i != 0) std::cout << ',';
        std::cout << "\"" << names[i] << "\":" << stage_ms[i];
    }
    std::cout << "}}\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
