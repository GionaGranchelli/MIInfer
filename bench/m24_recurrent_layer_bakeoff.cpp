#include "miinfer/device_validation.hpp"
#include "miinfer/fp16_gemv.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/m12_gdn_chunk.hpp"
#include "miinfer/qwen35_model.hpp"

#include "../tools/qwen35_gpu_pipeline.hpp"

#include <hip/hip_runtime.h>

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

void set_common_environment(std::uint32_t batch, bool fp16) {
    setenv("MIINFER_PREFILL_LAYER_MAJOR", "1", 1);
    setenv("MIINFER_PREFILL_WIDE_CHUNK", "1", 1);
    setenv("MIINFER_PREFILL_CHUNK", std::to_string(batch).c_str(), 1);
    setenv("MIINFER_PREFILL_WIDE_REPACKED_MMQ", "1", 1);
    setenv("MIINFER_M23_REPACKED_ROW128", "1", 1);
    setenv("MIINFER_PREFILL_REPACKED_RESIDENT_ALL", "1", 1);
    setenv("MIINFER_PREFILL_REPACKED_RESIDENT_FFN", "1", 1);
    setenv("MIINFER_PREFILL_WIDE_MMQ_QKV", "1", 1);
    setenv("MIINFER_PREFILL_WIDE_MMQ_SSM_OUT", "1", 1);
    setenv("MIINFER_PREFILL_WIDE_MMQ_FFN", "1", 1);
    if (fp16) setenv("MIINFER_PREFILL_REPACKED_FP16", "1", 1);
    else unsetenv("MIINFER_PREFILL_REPACKED_FP16");
}

double run_case(const miinfer::Qwen35Model& model, std::uint32_t batch, bool fp16) {
    set_common_environment(batch, fp16);
    RecurrentLayer layer(model, 0, {});

    const std::size_t gdn_bytes = kVHeads * 64 * kState * sizeof(float);
    RawBuffer gdn_new(gdn_bytes), gdn_decayed(gdn_bytes), gdn_solved_values(gdn_bytes);
    RawBuffer gdn_solved_keys(gdn_bytes), gdn_corrected(gdn_bytes);
    RawBuffer raw_output(static_cast<std::size_t>(batch) * kVHeads * kState * sizeof(float));
    const miinfer::M12GdnChunkWorkspace gdn_workspace{
        gdn_new.as<float>(), gdn_decayed.as<float>(), gdn_solved_values.as<float>(),
        gdn_solved_keys.as<float>(), gdn_corrected.as<float>()};
    layer.set_m12_gdn_workspace(gdn_workspace, raw_output.as<float>());

    RawBuffer dense_weights(kHidden * kFfnInner * sizeof(__half));
    RawBuffer dense_input(static_cast<std::size_t>(batch) * kFfnInner * sizeof(__half));
    miinfer::RocblasGemmHandle gemm_handle{};
    std::string error;
    if (!miinfer::create_rocblas_gemm_handle(gemm_handle, hipStreamPerThread, error)) {
        throw std::runtime_error(error);
    }
    layer.set_m12_dense_workspace(
        dense_weights.as<__half>(), dense_input.as<__half>(), &gemm_handle);

    RawBuffer input(static_cast<std::size_t>(batch) * kHidden * sizeof(float));
    RawBuffer output(static_cast<std::size_t>(batch) * kHidden * sizeof(float));
    std::vector<float> host_input(static_cast<std::size_t>(batch) * kHidden);
    for (std::size_t i = 0; i < host_input.size(); ++i) {
        host_input[i] = std::sin(0.0017F * static_cast<float>(i % kHidden)
                                  + 0.071F * static_cast<float>(i / kHidden));
    }
    MIINFER_HIP_CHECK(hipMemcpy(input.pointer, host_input.data(),
                                host_input.size() * sizeof(float), hipMemcpyHostToDevice));

    layer.prefill_wide(input.as<float>(), output.as<float>(), 0, batch);
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    MIINFER_HIP_CHECK(hipEventCreate(&start));
    MIINFER_HIP_CHECK(hipEventCreate(&stop));
    MIINFER_HIP_CHECK(hipEventRecord(start, hipStreamPerThread));
    layer.prefill_wide(input.as<float>(), output.as<float>(), 0, batch);
    MIINFER_HIP_CHECK(hipEventRecord(stop, hipStreamPerThread));
    MIINFER_HIP_CHECK(hipEventSynchronize(stop));
    float milliseconds = 0.0F;
    MIINFER_HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
    MIINFER_HIP_CHECK(hipEventDestroy(start));
    MIINFER_HIP_CHECK(hipEventDestroy(stop));
    miinfer::destroy_rocblas_gemm_handle(gemm_handle);
    return static_cast<double>(milliseconds) * 1000.0;
}

} // namespace

int main(int argc, char** argv) try {
    if (argc != 4) {
        throw std::runtime_error(
            "usage: miinfer-m24-recurrent-layer-bakeoff MODEL.gguf control|fp16 BATCH");
    }
    const std::string mode = argv[2];
    if (mode != "control" && mode != "fp16") {
        throw std::runtime_error("mode must be control or fp16");
    }
    const auto batch = static_cast<std::uint32_t>(std::stoul(argv[3]));
    if (batch != 128 && batch != 256 && batch != 512) {
        throw std::runtime_error("BATCH must be 128, 256, or 512");
    }
    miinfer::DeviceInfo device;
    std::string error;
    if (!miinfer::validate_gfx906_device(-1, device, error)) throw std::runtime_error(error);
    const auto model = miinfer::Qwen35Model::load(argv[1]);
    const double microseconds = run_case(model, batch, mode == "fp16");
    std::cout << std::fixed << std::setprecision(3)
              << "{\"mode\":\"" << mode << "\",\"batch\":" << batch
              << ",\"layer\":0,\"gpu_us\":" << microseconds
              << ",\"tok_s\":" << (1000000.0 * batch / microseconds)
              << ",\"tracked_layer_bytes\":" << g_device_bytes << "}\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
