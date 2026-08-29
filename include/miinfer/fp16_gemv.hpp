#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <string>
#include <vector>

namespace miinfer {

struct GemvShape {
    const char* id;
    const char* projection;
    int m;
    int k;
};

const std::vector<GemvShape>& qwen3_gemv_shapes();

struct Fp16GemvMetrics {
    double max_abs_error = 0.0;
    double mean_abs_error = 0.0;
    double max_relative_error = 0.0;
    double cosine_similarity = 0.0;
    bool nan_detected = false;
    bool inf_detected = false;
    bool pass = false;
};

constexpr float kFp16GemvAbsoluteTolerance = 5.0e-2F;
constexpr float kFp16GemvRelativeTolerance = 1.0e-2F;

void generate_fp16_gemv_data(
    int m,
    int k,
    std::uint32_t seed,
    std::vector<__half>& weights,
    std::vector<__half>& input);

std::vector<float> fp16_gemv_cpu_reference(
    const std::vector<__half>& weights,
    const std::vector<__half>& input,
    int m,
    int k);

Fp16GemvMetrics evaluate_fp16_gemv(
    const std::vector<__half>& output,
    const std::vector<float>& reference,
    float absolute_tolerance = kFp16GemvAbsoluteTolerance,
    float relative_tolerance = kFp16GemvRelativeTolerance);

void launch_fp16_gemv_baseline(
    const __half* weights,
    const __half* input,
    __half* output,
    int m,
    int k,
    hipStream_t stream = nullptr);

struct GemvKernelResources {
    int registers = 0;
    std::size_t shared_bytes = 0;
    std::size_t local_bytes = 0;
    int max_threads_per_block = 0;
};

GemvKernelResources fp16_gemv_baseline_resources();

struct RocblasGemmHandle {
    void* opaque = nullptr;
};

bool create_rocblas_gemm_handle(
    RocblasGemmHandle& handle,
    hipStream_t stream,
    std::string& error);

void destroy_rocblas_gemm_handle(RocblasGemmHandle& handle);

bool launch_rocblas_gemm_fp16(
    RocblasGemmHandle& handle,
    const __half* weights,
    const __half* input,
    __half* output,
    int m,
    int k,
    std::string& error);

}  // namespace miinfer
