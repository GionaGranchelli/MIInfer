#include "miinfer/device_validation.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/kquant_wave_layout.hpp"
#include "miinfer/m12_dense_stage.hpp"
#include "miinfer/qwen35_model.hpp"

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

constexpr std::array<std::uint32_t, 4> kBatches{64, 128, 256, 512};

struct Buffer {
    void* pointer = nullptr;

    explicit Buffer(std::size_t bytes) { MIINFER_HIP_CHECK(hipMalloc(&pointer, bytes)); }
    ~Buffer() { if (pointer != nullptr) (void)hipFree(pointer); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    template <typename T>
    T* as() const { return static_cast<T*>(pointer); }
};

struct Event {
    hipEvent_t value = nullptr;
    Event() { MIINFER_HIP_CHECK(hipEventCreate(&value)); }
    ~Event() { if (value != nullptr) (void)hipEventDestroy(value); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

template <typename Fn>
double measure(Fn&& fn, int warmup = 2, int iterations = 7) {
    for (int i = 0; i < warmup; ++i) fn();
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    Event start;
    Event stop;
    std::vector<double> samples;
    samples.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
        MIINFER_HIP_CHECK(hipEventRecord(start.value, hipStreamPerThread));
        fn();
        MIINFER_HIP_CHECK(hipEventRecord(stop.value, hipStreamPerThread));
        MIINFER_HIP_CHECK(hipEventSynchronize(stop.value));
        float milliseconds = 0.0F;
        MIINFER_HIP_CHECK(hipEventElapsedTime(&milliseconds, start.value, stop.value));
        samples.push_back(static_cast<double>(milliseconds) * 1000.0);
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

void check_hipblas(hipblasStatus_t status, const char* operation) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed: "
                                 + std::to_string(static_cast<int>(status)));
    }
}

void gemm(hipblasHandle_t handle, const __half* weights, const __half* input,
          float* output, std::uint32_t rows, std::uint32_t columns,
          std::uint32_t batch) {
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    // Row-major input[B,K], weights[M,K], output[B,M] exposed as column-major.
    check_hipblas(hipblasGemmEx(
        handle, HIPBLAS_OP_N, HIPBLAS_OP_N, batch, rows, columns,
        &alpha, input, HIP_R_16F, batch, weights, HIP_R_16F, columns,
        &beta, output, HIP_R_32F, batch, HIPBLAS_COMPUTE_32F,
        HIPBLAS_GEMM_DEFAULT), "hipblasGemmEx");
}

const miinfer::GgufTensor* find_projection(
    const miinfer::Qwen35Model& model, const std::string& projection,
    miinfer::GgufTensorType type, int layer) {
    const std::string suffix = "." + projection + ".weight";
    const std::string layer_prefix = layer >= 0 ? "blk." + std::to_string(layer) + "." : "";
    for (const auto& tensor : model.tensors()) {
        if (tensor.type == type && tensor.name.ends_with(suffix)
            && (layer_prefix.empty() || tensor.name.starts_with(layer_prefix))) {
            return &tensor;
        }
    }
    return nullptr;
}

template <typename Block>
int mx_affine_quant(const Block& block, std::size_t index) {
    const std::size_t group = index / 32;
    const std::size_t in_group = index % 32;
    const std::size_t pair = group / 2;
    const std::size_t part = group % 2;
    const std::size_t half = in_group / 16;
    const std::size_t in_half = in_group % 16;
    const std::size_t byte = pair * 32 + (in_half / 4) * 4 + (half ? 16 : 0) + in_half % 4;
    const auto* low = [&] {
        if constexpr (std::is_same_v<Block, miinfer::Q5KDeviceBlock>) return block.ql;
        else return block.qs;
    }();
    int value = (low[byte] >> (4 * part)) & 0x0f;
    if constexpr (std::is_same_v<Block, miinfer::Q5KDeviceBlock>) {
        value |= ((block.qh[in_half + (half ? 16 : 0)] >> (2 * pair + part)) & 1) << 4;
    }
    return value;
}

template <typename Block>
void mx_scale_min(const Block& block, std::size_t group, int& scale, int& minimum) {
    if (group < 4) {
        scale = block.scales[group] & 63;
        minimum = block.scales[group + 4] & 63;
    } else {
        scale = (block.scales[group + 4] & 15) | ((block.scales[group - 4] >> 6) << 4);
        minimum = (block.scales[group + 4] >> 4) | ((block.scales[group] >> 6) << 4);
    }
}

int mx_q6_quant(const miinfer::Q6KDeviceBlock& block, std::size_t index) {
    const std::size_t half_block = index / 128;
    const std::size_t lane = index % 128;
    const std::size_t quarter = lane / 32;
    const std::size_t in_quarter = lane % 32;
    const std::size_t low_index = in_quarter + (quarter == 1 || quarter == 3 ? 32 : 0);
    const auto* ql = block.ql + half_block * 64;
    const auto* qh = block.qh + half_block * 32;
    const int low = quarter < 2 ? ql[low_index] & 0x0f : ql[low_index] >> 4;
    const int high = (qh[in_quarter] >> (2 * quarter)) & 3;
    return (low | (high << 4)) - 32;
}

template <typename Block>
float mx_expected_value(const Block* weights, const std::vector<float>& input,
                        std::uint32_t row, std::uint32_t token,
                        std::uint32_t rows, std::uint32_t columns) {
    float result = 0.0F;
    for (std::uint32_t block128 = 0; block128 < columns / 128; ++block128) {
        const auto& activation = input.data() + static_cast<std::size_t>(token) * columns
                                 + block128 * 128;
        for (std::uint32_t group = 0; group < 4; ++group) {
            float max_abs = 0.0F;
            float sum = 0.0F;
            for (std::uint32_t offset = 0; offset < 32; ++offset) {
                max_abs = std::max(max_abs, std::fabs(activation[group * 32 + offset]));
                sum += activation[group * 32 + offset];
            }
            const float d = max_abs == 0.0F ? 0.0F : max_abs / 127.0F;
            const float input_d = std::is_same_v<Block, miinfer::Q6KDeviceBlock>
                ? d : __half2float(__float2half_rn(d));
            const float input_sum = __half2float(__float2half_rn(sum));
            int scale = 0, minimum = 0;
            const std::size_t weight_column = static_cast<std::size_t>(block128) * 128
                                              + group * 32;
            const auto& weight = weights[static_cast<std::size_t>(row) * (columns / 256)
                                         + weight_column / 256];
            const std::size_t element = weight_column % 256;
            int dot_lo = 0, dot_hi = 0;
            for (std::uint32_t offset = 0; offset < 32; ++offset) {
                int q = 0;
                if constexpr (std::is_same_v<Block, miinfer::Q6KDeviceBlock>) {
                    q = mx_q6_quant(weight, element + offset);
                } else {
                    q = mx_affine_quant(weight, element + offset);
                }
                const int quantized = d == 0.0F ? 0 : std::clamp(static_cast<int>(std::round(
                    activation[group * 32 + offset] / d)), -127, 127);
                if (offset < 16) dot_lo += q * quantized;
                else dot_hi += q * quantized;
            }
            if constexpr (std::is_same_v<Block, miinfer::Q6KDeviceBlock>) {
                const int scale0 = weight.scales[element / 16];
                const int scale1 = weight.scales[(element + 16) / 16];
                result += __half2float(weight.d) * input_d
                    * static_cast<float>(scale0 * dot_lo + scale1 * dot_hi);
            } else {
                mx_scale_min(weight, element / 32, scale, minimum);
                result += __half2float(weight.d) * input_d
                    * static_cast<float>(scale * (dot_lo + dot_hi))
                    - __half2float(weight.dmin) * static_cast<float>(minimum) * input_sum;
            }
        }
    }
    (void)rows;
    return result;
}

template <typename Block, typename Tile>
int run_case(
    const miinfer::GgufTensor& tensor,
    const char* type_name,
    std::vector<Tile> (*pack)(const miinfer::GgufTensor&),
    void (*expand_resident)(const Tile*, __half*, std::uint32_t, std::uint32_t, hipStream_t),
    void (*expand_canonical)(const Block*, __half*, std::uint32_t, std::uint32_t, hipStream_t),
    void (*mmq)(const Tile*, const miinfer::M23Q8_1MmqBlock*, float*,
                std::uint32_t, std::uint32_t, std::uint32_t, hipStream_t),
    std::vector<std::uint8_t> (*mx_pack)(const miinfer::GgufTensor&),
    void (*mx_mmq)(const std::uint8_t*, const miinfer::MxQ8_1MmqBlock*, float*,
                   std::uint32_t, std::uint32_t, std::uint32_t, hipStream_t),
    bool mx_affine) {
    if (tensor.dimensions.size() != 2 || tensor.dimensions[0] % 128 != 0
        || tensor.dimensions[1] % 64 != 0) {
        throw std::runtime_error("projection shape is incompatible with M24 MMQ staging");
    }
    const auto columns = static_cast<std::uint32_t>(tensor.dimensions[0]);
    const auto rows = static_cast<std::uint32_t>(tensor.dimensions[1]);
    const std::size_t dense_bytes = static_cast<std::size_t>(rows) * columns * sizeof(__half);
    const std::size_t input_bytes = static_cast<std::size_t>(kBatches.back()) * columns * sizeof(__half);
    const std::size_t output_bytes = static_cast<std::size_t>(kBatches.back()) * rows * sizeof(float);
    const std::size_t mmq_blocks = columns / 128;
    const auto packed = pack(tensor);
    const auto mx_packed = mx_pack(tensor);

    Buffer canonical(tensor.byte_size);
    Buffer resident(packed.size() * sizeof(packed[0]));
    Buffer mx_resident(mx_packed.size());
    Buffer canonical_dense(dense_bytes);
    Buffer resident_dense(dense_bytes);
    Buffer input(input_bytes);
    Buffer input_f32(static_cast<std::size_t>(kBatches.back()) * columns * sizeof(float));
    Buffer mmq_output(output_bytes);
    Buffer mx_output(output_bytes);
    Buffer gemm_output(output_bytes);
    std::array<std::unique_ptr<Buffer>, kBatches.size()> mmq_inputs;
    std::array<std::unique_ptr<Buffer>, kBatches.size()> mx_inputs;

    MIINFER_HIP_CHECK(hipMemcpy(canonical.pointer, tensor.data, tensor.byte_size,
                                hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(resident.pointer, packed.data(),
                                packed.size() * sizeof(packed[0]), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(mx_resident.pointer, mx_packed.data(), mx_packed.size(),
                                hipMemcpyHostToDevice));

    std::vector<__half> input_host(static_cast<std::size_t>(kBatches.back()) * columns);
    std::vector<float> input_f32_host(input_host.size());
    for (std::uint32_t batch = 0; batch < kBatches.back(); ++batch) {
        for (std::uint32_t column = 0; column < columns; ++column) {
            const float value = std::sin(0.0017F * static_cast<float>(column)
                                         + 0.071F * static_cast<float>(batch));
            input_f32_host[static_cast<std::size_t>(batch) * columns + column] = value;
            input_host[static_cast<std::size_t>(batch) * columns + column] = __float2half_rn(value);
        }
    }
    MIINFER_HIP_CHECK(hipMemcpy(input.pointer, input_host.data(), input_bytes,
                                hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(input_f32.pointer, input_f32_host.data(),
                                input_f32_host.size() * sizeof(float), hipMemcpyHostToDevice));
    for (std::size_t index = 0; index < kBatches.size(); ++index) {
        const auto batch = kBatches[index];
        mmq_inputs[index] = std::make_unique<Buffer>(
            static_cast<std::size_t>(batch) * mmq_blocks * sizeof(miinfer::M23Q8_1MmqBlock));
        miinfer::launch_m23_q8_1_mmq_quantize(
            input_f32.as<float>(), mmq_inputs[index]->as<miinfer::M23Q8_1MmqBlock>(),
            batch, columns, hipStreamPerThread);
        mx_inputs[index] = std::make_unique<Buffer>(
            static_cast<std::size_t>(batch) * mmq_blocks * sizeof(miinfer::MxQ8_1MmqBlock));
        miinfer::launch_mx_q8_1_mmq_quantize(
            input_f32.as<float>(), mx_inputs[index]->as<miinfer::MxQ8_1MmqBlock>(),
            batch, columns, mx_affine, hipStreamPerThread);
    }
    MIINFER_HIP_CHECK(hipDeviceSynchronize());

    const double canonical_repack_us = measure([&] {
        expand_canonical(canonical.as<Block>(), canonical_dense.as<__half>(), rows, columns,
                         hipStreamPerThread);
    });
    const double resident_repack_us = measure([&] {
        expand_resident(resident.as<Tile>(), resident_dense.as<__half>(), rows, columns,
                        hipStreamPerThread);
    });

    std::array<std::uint32_t, 3> sample_rows{0, rows / 2, rows - 1};
    double max_resident_error = 0.0;
    std::vector<__half> canonical_row(columns), resident_row(columns);
    for (const auto row : sample_rows) {
        MIINFER_HIP_CHECK(hipMemcpy(
            canonical_row.data(), canonical_dense.as<__half>() + static_cast<std::size_t>(row) * columns,
            static_cast<std::size_t>(columns) * sizeof(__half), hipMemcpyDeviceToHost));
        MIINFER_HIP_CHECK(hipMemcpy(
            resident_row.data(), resident_dense.as<__half>() + static_cast<std::size_t>(row) * columns,
            static_cast<std::size_t>(columns) * sizeof(__half), hipMemcpyDeviceToHost));
        for (std::uint32_t column = 0; column < columns; ++column) {
            max_resident_error = std::max(max_resident_error, static_cast<double>(std::abs(
                __half2float(canonical_row[column]) - __half2float(resident_row[column]))));
        }
    }

    hipblasHandle_t handle = nullptr;
    check_hipblas(hipblasCreate(&handle), "hipblasCreate");
    check_hipblas(hipblasSetStream(handle, hipStreamPerThread), "hipblasSetStream");

    std::cout << std::fixed << std::setprecision(6)
              << "{\"tensor\":\"" << tensor.name << "\",\"type\":\"" << type_name
              << "\",\"rows\":" << rows << ",\"columns\":" << columns
              << ",\"resident_bytes\":" << packed.size() * sizeof(packed[0])
              << ",\"mx_repacked_bytes\":" << mx_packed.size()
              << ",\"dense_bytes\":" << dense_bytes
              << ",\"canonical_repack_us\":" << canonical_repack_us
              << ",\"resident_repack_us\":" << resident_repack_us
              << ",\"max_resident_canonical_fp16_error\":" << max_resident_error
              << ",\"batches\":[";

    bool first = true;
    for (std::size_t index = 0; index < kBatches.size(); ++index) {
        const auto batch = kBatches[index];
        const double mmq_us = measure([&] {
            mmq(resident.as<Tile>(), mmq_inputs[index]->as<miinfer::M23Q8_1MmqBlock>(),
                mmq_output.as<float>(), rows, columns, batch, hipStreamPerThread);
        });
        const double gemm_us = measure([&] {
            gemm(handle, resident_dense.as<__half>(), input.as<__half>(), gemm_output.as<float>(),
                 rows, columns, batch);
        });
        const double mx_quantize_us = measure([&] {
            miinfer::launch_mx_q8_1_mmq_quantize(
                input_f32.as<float>(), mx_inputs[index]->as<miinfer::MxQ8_1MmqBlock>(),
                batch, columns, mx_affine, hipStreamPerThread);
        });
        const double mx_us = measure([&] {
            mx_mmq(mx_resident.as<std::uint8_t>(),
                   mx_inputs[index]->as<miinfer::MxQ8_1MmqBlock>(), mx_output.as<float>(),
                   rows, columns, batch, hipStreamPerThread);
        });
        if (!first) std::cout << ',';
        first = false;
        std::cout << "{\"batch\":" << batch
                  << ",\"resident_mmq_us\":" << mmq_us
                  << ",\"gemm_us\":" << gemm_us
                  << ",\"mx_q8_quantize_us\":" << mx_quantize_us
                  << ",\"mx_repacked_mmq_us\":" << mx_us
                  << ",\"mx_total_us\":" << mx_quantize_us + mx_us
                  << ",\"resident_fp16_total_us\":" << resident_repack_us + gemm_us
                  << ",\"mmq_over_fp16_total\":"
                  << (mmq_us / (resident_repack_us + gemm_us)) << '}';
    }
    std::cout << "]}\n";

    mmq(resident.as<Tile>(), mmq_inputs[0]->as<miinfer::M23Q8_1MmqBlock>(),
        mmq_output.as<float>(), rows, columns, kBatches[0], hipStreamPerThread);
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> output_host(static_cast<std::size_t>(kBatches[0]) * rows);
    MIINFER_HIP_CHECK(hipMemcpy(output_host.data(), mmq_output.pointer,
                                output_host.size() * sizeof(float), hipMemcpyDeviceToHost));
    for (const float value : output_host) {
        if (!std::isfinite(value)) throw std::runtime_error("resident MMQ produced non-finite output");
    }
    mx_mmq(mx_resident.as<std::uint8_t>(), mx_inputs[0]->as<miinfer::MxQ8_1MmqBlock>(),
           mx_output.as<float>(), rows, columns, kBatches[0], hipStreamPerThread);
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    MIINFER_HIP_CHECK(hipMemcpy(output_host.data(), mx_output.pointer,
                                output_host.size() * sizeof(float), hipMemcpyDeviceToHost));
    for (const float value : output_host) {
        if (!std::isfinite(value)) throw std::runtime_error("mx repacked MMQ produced non-finite output");
    }
    double max_mx_error = 0.0;
    const auto* weight_host = reinterpret_cast<const Block*>(tensor.data);
    for (const std::uint32_t row : {0U, rows / 2, rows - 1}) {
        for (const std::uint32_t token : {0U, 1U, 32U, 63U}) {
            const float expected = mx_expected_value(weight_host, input_f32_host, row, token,
                                                     rows, columns);
            const float actual = output_host[static_cast<std::size_t>(token) * rows + row];
            max_mx_error = std::max(max_mx_error, static_cast<double>(std::fabs(expected - actual)));
        }
    }
    if (max_mx_error > 0.05) {
        throw std::runtime_error("mx repacked MMQ contract check failed: max abs error "
                                 + std::to_string(max_mx_error));
    }
    std::cout << "mx_contract_max_abs_error=" << std::fixed << std::setprecision(8)
              << max_mx_error << '\n';
    check_hipblas(hipblasDestroy(handle), "hipblasDestroy");
    return 0;
}

} // namespace

int main(int argc, char** argv) try {
    if (argc < 4 || argc > 5) throw std::runtime_error(
        "usage: miinfer-m24-projection-bakeoff MODEL.gguf gate|up|down|ssm_out q4|q5|q6 [layer]");
    const std::string projection = argv[2];
    const std::string type_name = argv[3];
    const int layer = argc == 5 ? std::stoi(argv[4]) : -1;
    const auto type = type_name == "q4" ? miinfer::GgufTensorType::q4_k
        : type_name == "q5" ? miinfer::GgufTensorType::q5_k
        : type_name == "q6" ? miinfer::GgufTensorType::q6_k
        : throw std::runtime_error("type must be q4, q5, or q6");
    if (projection != "gate" && projection != "up" && projection != "down"
        && projection != "ssm_out") {
        throw std::runtime_error("projection must be gate, up, down, or ssm_out");
    }
    miinfer::DeviceInfo device;
    std::string error;
    if (!miinfer::validate_gfx906_device(-1, device, error)) throw std::runtime_error(error);
    const auto model = miinfer::Qwen35Model::load(argv[1]);
    const std::string tensor_projection = projection == "ssm_out" ? projection : "ffn_" + projection;
    const auto* tensor = find_projection(model, tensor_projection, type, layer);
    if (tensor == nullptr) throw std::runtime_error("requested projection tensor not found");

    if (type == miinfer::GgufTensorType::q4_k) {
        return run_case<miinfer::Q4KDeviceBlock, Q4KMmqTile>(
            *tensor, "q4", pack_q4k_mmq_tensor, launch_m24_q4k_mmq_to_fp16,
            miinfer::launch_m12_q4k_to_fp16, launch_m23_q4k_repacked_mmq,
            pack_mx_q4k_repacked_tensor, launch_mx_q4k_repacked_mmq, true);
    }
    if (type == miinfer::GgufTensorType::q5_k) {
        return run_case<miinfer::Q5KDeviceBlock, Q5KMmqTile>(
            *tensor, "q5", pack_q5k_mmq_tensor, launch_m24_q5k_mmq_to_fp16,
            miinfer::launch_m12_q5k_to_fp16, launch_m23_q5k_repacked_mmq,
            pack_mx_q5k_repacked_tensor, launch_mx_q5k_repacked_mmq, true);
    }
    return run_case<miinfer::Q6KDeviceBlock, Q6KMmqTile>(
        *tensor, "q6", pack_q6k_mmq_tensor, launch_m24_q6k_mmq_to_fp16,
        miinfer::launch_m12_q6k_to_fp16, launch_m23_q6k_repacked_mmq,
        pack_mx_q6k_repacked_tensor, launch_mx_q6k_repacked_mmq, false);
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
