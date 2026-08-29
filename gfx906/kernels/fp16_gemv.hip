#include "miinfer/fp16_gemv.hpp"

#include "miinfer/hip_check.hpp"

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <cstdlib>
#include <cstdio>
#include <string>

namespace miinfer {

namespace {

constexpr unsigned int kThreadsPerBlock = 256;

__global__ void fp16_gemv_baseline_kernel(
    const __half* weights,
    const __half* input,
    __half* output,
    int m,
    int k) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= m) {
        return;
    }

    __shared__ float partials[kThreadsPerBlock];
    float sum = 0.0F;
    const std::size_t row_offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(k);
    for (int column = static_cast<int>(threadIdx.x); column < k;
         column += static_cast<int>(blockDim.x)) {
        sum += __half2float(weights[row_offset + static_cast<std::size_t>(column)])
               * __half2float(input[static_cast<std::size_t>(column)]);
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

// DIAGNOSTIC ONLY. This intentionally stops after the per-thread dot-product
// and exposes partials, allowing EXP-0003 to estimate reduction/synchronization
// cost. It is not a candidate implementation because it changes the output
// contract and adds diagnostic partial stores.
__global__ void fp16_gemv_diagnostic_dot_only_kernel(
    const __half* weights,
    const __half* input,
    float* partials,
    int m,
    int k) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= m) {
        return;
    }
    float sum = 0.0F;
    const std::size_t row_offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(k);
    for (int column = static_cast<int>(threadIdx.x); column < k;
         column += static_cast<int>(blockDim.x)) {
        sum += __half2float(weights[row_offset + static_cast<std::size_t>(column)])
               * __half2float(input[static_cast<std::size_t>(column)]);
    }
    partials[static_cast<std::size_t>(row) * kThreadsPerBlock + threadIdx.x] = sum;
}

bool check_hipblas(hipblasStatus_t status, const char* operation, std::string& error) {
    if (status == HIPBLAS_STATUS_SUCCESS) {
        return true;
    }
    error = std::string(operation) + " failed with hipBLAS status "
            + std::to_string(static_cast<int>(status));
    return false;
}

}  // namespace

void launch_fp16_gemv_baseline(
    const __half* weights,
    const __half* input,
    __half* output,
    int m,
    int k,
    hipStream_t stream) {
    hipLaunchKernelGGL(
        fp16_gemv_baseline_kernel,
        dim3(static_cast<unsigned int>(m)),
        dim3(kThreadsPerBlock),
        kThreadsPerBlock * sizeof(float),
        stream,
        weights,
        input,
        output,
        m,
        k);
    MIINFER_HIP_CHECK(hipGetLastError());
}

void launch_fp16_gemv_diagnostic_dot_only(
    const __half* weights,
    const __half* input,
    float* partials,
    int m,
    int k,
    hipStream_t stream) {
    hipLaunchKernelGGL(
        fp16_gemv_diagnostic_dot_only_kernel,
        dim3(static_cast<unsigned int>(m)),
        dim3(kThreadsPerBlock),
        0,
        stream,
        weights,
        input,
        partials,
        m,
        k);
    MIINFER_HIP_CHECK(hipGetLastError());
}

GemvKernelResources fp16_gemv_baseline_resources() {
    hipFuncAttributes attributes{};
    MIINFER_HIP_CHECK(hipFuncGetAttributes(
        &attributes,
        reinterpret_cast<const void*>(fp16_gemv_baseline_kernel)));
    return {
        attributes.numRegs,
        attributes.sharedSizeBytes,
        attributes.localSizeBytes,
        attributes.maxThreadsPerBlock,
    };
}

bool create_rocblas_gemm_handle(
    RocblasGemmHandle& handle,
    hipStream_t stream,
    std::string& error) {
    auto* hipblas_handle = new hipblasHandle_t{};
    if (!check_hipblas(hipblasCreate(hipblas_handle), "hipblasCreate", error)) {
        delete hipblas_handle;
        return false;
    }
    if (!check_hipblas(hipblasSetStream(*hipblas_handle, stream), "hipblasSetStream", error)) {
        hipblasDestroy(*hipblas_handle);
        delete hipblas_handle;
        return false;
    }
    handle.opaque = hipblas_handle;
    return true;
}

void destroy_rocblas_gemm_handle(RocblasGemmHandle& handle) {
    if (handle.opaque == nullptr) {
        return;
    }
    auto* hipblas_handle = static_cast<hipblasHandle_t*>(handle.opaque);
    std::string error;
    if (!check_hipblas(hipblasDestroy(*hipblas_handle), "hipblasDestroy", error)) {
        std::fprintf(stderr, "MIInfer hipBLAS failure: %s\n", error.c_str());
        std::abort();
    }
    delete hipblas_handle;
    handle.opaque = nullptr;
}

bool launch_rocblas_gemm_fp16(
    RocblasGemmHandle& handle,
    const __half* weights,
    const __half* input,
    __half* output,
    int m,
    int k,
    std::string& error) {
    if (handle.opaque == nullptr) {
        error = "hipBLAS handle is not initialized";
        return false;
    }
    auto* hipblas_handle = static_cast<hipblasHandle_t*>(handle.opaque);
    const float alpha = 1.0F;
    const float beta = 0.0F;

    // MIInfer stores A as row-major M x K. hipBLAS consumes column-major
    // storage, so the same bytes are a K x M matrix and op(A)=transpose
    // produces the intended M x 1 result without a timed transpose/copy.
    return check_hipblas(
        hipblasGemmEx(
            *hipblas_handle,
            HIPBLAS_OP_T,
            HIPBLAS_OP_N,
            m,
            1,
            k,
            &alpha,
            weights,
            HIP_R_16F,
            k,
            input,
            HIP_R_16F,
            k,
            &beta,
            output,
            HIP_R_16F,
            m,
            HIPBLAS_COMPUTE_32F,
            HIPBLAS_GEMM_DEFAULT),
        "hipblasGemmEx",
        error);
}

}  // namespace miinfer
