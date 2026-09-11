#pragma once
#include "miinfer/q4k_wave_layout.hpp"
#define MIINFER_M6A3_HELPERS_ONLY
#include "m6a3_qwen35_layer.cpp"

#include "miinfer/hip_check.hpp"
#include "miinfer/fp16_gemv.hpp"
#include "miinfer/m12_dense_stage.hpp"
#include "miinfer/m12_gdn_chunk.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"
#include "miinfer/qwen35_model.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstddef>
#include <ctime>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <time.h>
#include <utility>
#include <vector>

namespace {

// Set once before constructing the runtime. All cache and graph allocations
// in this translation unit use the selected model-load-time capacity.
std::size_t g_cache_capacity = 65536;
// The layer-major prefill schedule stages a causal chunk, then drains it in
// ordered B=4 groups where recurrent state and KV writes require ordering.
constexpr std::size_t kPrefillBatch = 64;
constexpr std::size_t kM12PrefillBatch = 128;
constexpr std::size_t kMaxWidePrefillBatch = 512;
// Projection batches may widen to P512; the causal WY scan stays at its
// independently sized 64-token state chunk until a larger kernel is proven.
constexpr std::size_t kM23CausalChunk = 64;
static_assert(kHidden % 128 == 0 && kInner % 128 == 0 && kFfnInner % 128 == 0,
              "M23 MMQ projections require 128-value K tiles");

struct M23ProfileCounters {
    std::array<std::uint64_t, 7> recurrent_dispatches{};
    std::array<std::uint64_t, 9> attention_dispatches{};
    std::array<std::uint64_t, 7> recurrent_weight_upload_bytes{};
    std::array<std::uint64_t, 9> attention_weight_upload_bytes{};
    std::array<double, 7> recurrent_weight_upload_ms{};
    std::array<double, 9> attention_weight_upload_ms{};
    std::uint64_t weight_upload_bytes = 0;
    double weight_upload_ms = 0.0;
};

std::size_t configured_wide_prefill_batch() {
    const char* env = std::getenv("MIINFER_PREFILL_CHUNK");
    if (env == nullptr) return kM12PrefillBatch;
    const auto requested = std::stoul(env);
    return requested >= kM12PrefillBatch && requested <= kMaxWidePrefillBatch
        && requested % kPrefillBatch == 0 ? requested : kM12PrefillBatch;
}

bool shared_wide_prefill_enabled() {
    const char* wide = std::getenv("MIINFER_PREFILL_WIDE_CHUNK");
    const char* full = std::getenv("MIINFER_PREFILL_FULL_LAYER_MAJOR");
    return wide != nullptr && std::strcmp(wide, "0") != 0
        && full != nullptr && std::strcmp(full, "0") != 0;
}

class DeviceBytes {
public:
    explicit DeviceBytes(std::size_t bytes) : bytes_(bytes) {
        const auto error = hipMalloc(&data_, bytes);
        if (error != hipSuccess) {
            std::fprintf(stderr, "MIInfer device allocation failed: %zu bytes\n", bytes);
            miinfer::hip_check_failed(error, "hipMalloc(&data_, bytes)");
        }
    }
    DeviceBytes(const DeviceBytes&) = delete;
    DeviceBytes& operator=(const DeviceBytes&) = delete;
    ~DeviceBytes() { if (data_ != nullptr) (void)hipFree(data_); }
    void* get() const noexcept { return data_; }

private:
    void* data_ = nullptr;
    std::size_t bytes_ = 0;
};

using Buffer = std::shared_ptr<DeviceBytes>;

struct WidePrefillWorkspace {
    Buffer normalized, qkv, gate, q8_1, mmq_q8;
    Buffer gated, residual, post_normalized, projected;
    Buffer ffn_gate, ffn_up, ffn_activation, ffn_projected;
    Buffer core_query, core_key, core_value, core_beta, core_decay, core_gate;
    Buffer qfull, value, gated_attention, query_rope, attention_gate;
};

std::size_t g_device_allocations = 0;
std::size_t g_device_bytes = 0;
std::size_t g_peak_device_bytes = 0;
std::array<Buffer, 6> g_m23_repacked_scratch;
std::array<std::size_t, 6> g_m23_repacked_scratch_bytes{};

Buffer allocate(std::size_t bytes) {
    auto result = std::make_shared<DeviceBytes>(bytes);
    ++g_device_allocations;
    g_device_bytes += bytes;
    g_peak_device_bytes = std::max(g_peak_device_bytes, g_device_bytes);
    return result;
}

void upload(const void* source, void* destination, std::size_t bytes) {
    MIINFER_HIP_CHECK(hipMemcpy(destination, source, bytes, hipMemcpyHostToDevice));
}

void upload_tensor(const miinfer::GgufTensor& source, const Buffer& destination) {
    upload(source.data, destination->get(), source.byte_size);
}

bool native_down_enabled() {
    const char* value = std::getenv("MIINFER_Q4K_NATIVE_DOWN");
    if (!value || std::strcmp(value, "0") == 0) return false;
    if (std::strcmp(value, "1") == 0) return true;
    throw std::runtime_error("MIINFER_Q4K_NATIVE_DOWN must be 0 or 1");
}

bool native_gate_up_enabled() {
    const char* value = std::getenv("MIINFER_Q4K_NATIVE_GATE_UP");
    if (!value || std::strcmp(value, "0") == 0) return false;
    if (std::strcmp(value, "1") == 0) return true;
    throw std::runtime_error("MIINFER_Q4K_NATIVE_GATE_UP must be 0 or 1");
}

bool native_q_enabled() {
    const char* value = std::getenv("MIINFER_Q4K_NATIVE_Q");
    if (!value || std::strcmp(value, "0") == 0) return false;
    if (std::strcmp(value, "1") == 0) return true;
    throw std::runtime_error("MIINFER_Q4K_NATIVE_Q must be 0 or 1");
}

bool native_attn_gate_enabled() {
    const char* value = std::getenv("MIINFER_Q4K_NATIVE_ATTN_GATE");
    if (!value || std::strcmp(value, "0") == 0) return false;
    if (std::strcmp(value, "1") == 0) return true;
    throw std::runtime_error("MIINFER_Q4K_NATIVE_ATTN_GATE must be 0 or 1");
}

bool native_attn_out_enabled() {
    const char* value = std::getenv("MIINFER_Q4K_NATIVE_ATTN_OUT");
    if (!value || std::strcmp(value, "0") == 0) return false;
    if (std::strcmp(value, "1") == 0) return true;
    throw std::runtime_error("MIINFER_Q4K_NATIVE_ATTN_OUT must be 0 or 1");
}

bool native_k_enabled() {
    const char* value = std::getenv("MIINFER_Q4K_NATIVE_K");
    if (!value || std::strcmp(value, "0") == 0) return false;
    if (std::strcmp(value, "1") == 0) return true;
    throw std::runtime_error("MIINFER_Q4K_NATIVE_K must be 0 or 1");
}

bool native_ssm_out_enabled() {
    const char* value = std::getenv("MIINFER_Q5K_NATIVE_SSM_OUT");
    if (!value || std::strcmp(value, "0") == 0) return false;
    if (std::strcmp(value, "1") == 0) return true;
    throw std::runtime_error("MIINFER_Q5K_NATIVE_SSM_OUT must be 0 or 1");
}

bool native_qkv_enabled() {
    const char* value = std::getenv("MIINFER_KQUANT_NATIVE_QKV");
    if (!value || std::strcmp(value, "0") == 0) return false;
    if (std::strcmp(value, "1") == 0) return true;
    throw std::runtime_error("MIINFER_KQUANT_NATIVE_QKV must be 0 or 1");
}

bool native_v_enabled() {
    const char* value = std::getenv("MIINFER_KQUANT_NATIVE_V");
    if (!value || std::strcmp(value, "0") == 0) return false;
    if (std::strcmp(value, "1") == 0) return true;
    throw std::runtime_error("MIINFER_KQUANT_NATIVE_V must be 0 or 1");
}

bool native_q6k_down_enabled() {
    const char* value = std::getenv("MIINFER_Q6K_NATIVE_DOWN");
    if (!value || std::strcmp(value, "0") == 0) return false;
    if (std::strcmp(value, "1") == 0) return true;
    throw std::runtime_error("MIINFER_Q6K_NATIVE_DOWN must be 0 or 1");
}

Buffer copy_native_tensor(const miinfer::GgufTensor& source) {
    const auto packed = pack_q4k_wave_tensor(source);
    auto result = allocate(packed.size() * sizeof(Q4KWaveTile));
    upload(packed.data(), result->get(), packed.size() * sizeof(Q4KWaveTile));
    return result;
}

Buffer copy_native_q5k_tensor(const miinfer::GgufTensor& source) {
    const auto packed = pack_q5k_wave_tensor(source);
    auto result = allocate(packed.size() * sizeof(Q5KWaveTile));
    upload(packed.data(), result->get(), packed.size() * sizeof(Q5KWaveTile));
    return result;
}

Buffer copy_native_q6k_tensor(const miinfer::GgufTensor& source) {
    const auto packed = pack_q6k_wave_tensor(source);
    auto result = allocate(packed.size() * sizeof(Q6KWaveTile));
    upload(packed.data(), result->get(), packed.size() * sizeof(Q6KWaveTile));
    return result;
}

Buffer copy_native_down(const miinfer::GgufTensor& source) {
    return copy_native_tensor(source);
}

Buffer copy_combined_q4k_tensors(const miinfer::GgufTensor& a, const miinfer::GgufTensor& b) {
    const auto packed_a = pack_q4k_wave_tensor(a);
    const auto packed_b = pack_q4k_wave_tensor(b);
    std::vector<Q4KWaveTile> combined;
    combined.reserve(packed_a.size() + packed_b.size());
    combined.insert(combined.end(), packed_a.begin(), packed_a.end());
    combined.insert(combined.end(), packed_b.begin(), packed_b.end());
    auto result = allocate(combined.size() * sizeof(Q4KWaveTile));
    upload(combined.data(), result->get(), combined.size() * sizeof(Q4KWaveTile));
    return result;
}

Buffer copy_combined_q4k_mmq_tensors(const miinfer::GgufTensor& a, const miinfer::GgufTensor& b) {
    const auto packed_a = pack_q4k_mmq_tensor(a);
    const auto packed_b = pack_q4k_mmq_tensor(b);
    std::vector<Q4KMmqTile> combined;
    combined.reserve(packed_a.size() + packed_b.size());
    combined.insert(combined.end(), packed_a.begin(), packed_a.end());
    combined.insert(combined.end(), packed_b.begin(), packed_b.end());
    auto result = allocate(combined.size() * sizeof(Q4KMmqTile));
    upload(combined.data(), result->get(), combined.size() * sizeof(Q4KMmqTile));
    return result;
}

static bool swiglu_paired_enabled() {
    const char* env = std::getenv("MIINFER_SWIGLU_PAIRED");
    if (!env || env[0] == '\0') return true;
    if (std::strcmp(env, "1") == 0) return true;
    if (std::strcmp(env, "0") == 0) return false;
    throw std::runtime_error("MIINFER_SWIGLU_PAIRED must be 0 or 1");
}

Buffer copy_swiglu_paired_tensors(const miinfer::GgufTensor& gate, const miinfer::GgufTensor& up) {
    const auto packed = pack_q4k_wave_swiglu_fused(gate, up);
    auto result = allocate(packed.size() * sizeof(Q4KWaveSwigluFusedTile));
    upload(packed.data(), result->get(), packed.size() * sizeof(Q4KWaveSwigluFusedTile));
    return result;
}

static bool combined_qkv_gate_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("MIINFER_COMBINED_QKV_GATE");
        return env == nullptr || (std::strcmp(env, "0") != 0 && std::strcmp(env, "OFF") != 0 && std::strcmp(env, "off") != 0);
    }();
    return enabled;
}

static bool combined_attn_qk_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("MIINFER_COMBINED_ATTN_QK");
        return env == nullptr || (std::strcmp(env, "0") != 0 && std::strcmp(env, "OFF") != 0 && std::strcmp(env, "off") != 0);
    }();
    return enabled;
}

void project_native_down(const Buffer& weights, const float* input,
                         miinfer::Q8_1Block* q8, float* output) {
    miinfer::launch_q8_1_quantize_f32(input, q8, 17408);
    launch_q4k_wave_down(static_cast<const Q4KWaveTile*>(weights->get()), q8, output);
}

void project_native_q6k_down(const Buffer& weights, const float* input,
                             miinfer::Q8_1Block* q8, float* output) {
    miinfer::launch_q8_1_quantize_f32(input, q8, 17408);
    launch_q6k_wave_gemv(static_cast<const Q6KWaveTile*>(weights->get()), q8, output, 5120, 17408);
}

Buffer copy_expanded_q4k(const miinfer::GgufTensor& source) {
    if (source.type != miinfer::GgufTensorType::q4_k
        || source.byte_size % sizeof(miinfer::Q4KDeviceBlock) != 0) {
        throw std::runtime_error("invalid Q4_K tensor for expanded copy: " + source.name);
    }
    const auto count = source.byte_size / sizeof(miinfer::Q4KDeviceBlock);
    std::vector<miinfer::Q4KExpandedDeviceBlock> expanded(count);
    const auto* input = reinterpret_cast<const miinfer::Q4KDeviceBlock*>(source.data);
    for (std::size_t block = 0; block < count; ++block) {
        auto& dst = expanded[block];
        dst.d = input[block].d;
        dst.dmin = input[block].dmin;
        for (int group = 0; group < 8; ++group) {
            int scale = 0;
            int minimum = 0;
            const auto& raw = input[block];
            if (group < 4) {
                scale = raw.scales[group] & 63;
                minimum = raw.scales[group + 4] & 63;
            } else {
                scale = (raw.scales[group + 4] & 0x0f)
                        | ((raw.scales[group - 4] >> 6) << 4);
                minimum = (raw.scales[group + 4] >> 4)
                          | ((raw.scales[group] >> 6) << 4);
            }
            dst.scales[group] = static_cast<std::uint8_t>(scale);
            dst.minimums[group] = static_cast<std::uint8_t>(minimum);
        }
        for (int byte = 0; byte < 128; ++byte) {
            dst.qs[2 * byte] = input[block].qs[byte] & 0x0f;
            dst.qs[2 * byte + 1] = input[block].qs[byte] >> 4;
        }
    }
    auto result = allocate(expanded.size() * sizeof(expanded.front()));
    upload(expanded.data(), result->get(), expanded.size() * sizeof(expanded.front()));
    return result;
}

struct DetailedError {
    float max_abs = 0.0F;
    float mean_abs = 0.0F;
    float rms = 0.0F;
    float reference_rms = 0.0F;
    float relative_rms = 0.0F;
};

DetailedError detailed_compare(std::span<const float> actual,
                               std::span<const float> expected) {
    if (actual.size() != expected.size() || actual.empty()) {
        throw std::runtime_error("comparison size mismatch");
    }
    double abs_sum = 0.0;
    double error_sum = 0.0;
    double reference_sum = 0.0;
    DetailedError result;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
            throw std::runtime_error("non-finite comparison value");
        }
        const double delta = static_cast<double>(actual[i]) - expected[i];
        result.max_abs = std::max(result.max_abs, static_cast<float>(std::fabs(delta)));
        abs_sum += std::fabs(delta);
        error_sum += delta * delta;
        reference_sum += static_cast<double>(expected[i]) * expected[i];
    }
    result.mean_abs = static_cast<float>(abs_sum / actual.size());
    result.rms = static_cast<float>(std::sqrt(error_sum / actual.size()));
    result.reference_rms = static_cast<float>(std::sqrt(reference_sum / actual.size()));
    result.relative_rms = result.reference_rms == 0.0F
        ? result.rms : result.rms / result.reference_rms;
    return result;
}

float cosine_similarity(std::span<const float> actual, std::span<const float> expected) {
    double dot = 0.0;
    double actual_norm = 0.0;
    double expected_norm = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        dot += static_cast<double>(actual[i]) * expected[i];
        actual_norm += static_cast<double>(actual[i]) * actual[i];
        expected_norm += static_cast<double>(expected[i]) * expected[i];
    }
    return actual_norm == 0.0 || expected_norm == 0.0
        ? 0.0F : static_cast<float>(dot / std::sqrt(actual_norm * expected_norm));
}

std::size_t first_argmax(std::span<const float> values) {
    std::size_t result = 0;
    for (std::size_t i = 1; i < values.size(); ++i) {
        if (values[i] > values[result]) result = i;
    }
    return result;
}

std::vector<std::size_t> top_indices(std::span<const float> values, std::size_t count) {
    count = std::min(count, values.size());
    std::vector<std::size_t> indices(values.size());
    std::iota(indices.begin(), indices.end(), 0);
    const auto better = [&values](std::size_t a, std::size_t b) {
        return values[a] > values[b] || (values[a] == values[b] && a < b);
    };
    std::partial_sort(indices.begin(), indices.begin() + count, indices.end(), better);
    indices.resize(count);
    return indices;
}

std::size_t rank_of(std::span<const float> values, std::size_t index) {
    std::size_t rank = 1;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (values[i] > values[index] ||
            (values[i] == values[index] && i < index)) ++rank;
    }
    return rank;
}

DetailedError detailed_device_error(const float* device, std::size_t elements,
                                    const std::filesystem::path& expected_path) {
    std::vector<float> actual(elements);
    MIINFER_HIP_CHECK(hipMemcpy(actual.data(), device, elements * sizeof(float),
                                hipMemcpyDeviceToHost));
    return detailed_compare(actual, read_f32(expected_path, elements));
}

struct LocatedError {
    DetailedError metrics;
    std::size_t index = 0;
    float actual = 0.0F;
    float expected = 0.0F;
};

LocatedError located_host_error(std::span<const float> actual,
                                std::span<const float> expected);

LocatedError located_device_error(const float* device, std::size_t elements,
                                  const std::filesystem::path& expected_path) {
    std::vector<float> actual(elements);
    MIINFER_HIP_CHECK(hipMemcpy(actual.data(), device, elements * sizeof(float),
                                hipMemcpyDeviceToHost));
    const auto expected = read_f32(expected_path, elements);
    LocatedError result;
    result.metrics = detailed_compare(actual, expected);
    for (std::size_t i = 0; i < elements; ++i) {
        if (std::fabs(actual[i] - expected[i]) >
            std::fabs(result.actual - result.expected)) {
            result.index = i;
            result.actual = actual[i];
            result.expected = expected[i];
        }
    }
    return result;
}

struct RecurrentTrace {
    const std::filesystem::path& fixture;
    std::size_t layer;
    std::uint32_t position;

    void report_at(std::uint32_t checkpoint_position, const char* label,
                   const float* device, std::size_t elements,
                   const std::string& reference_name) const {
        const auto error = located_device_error(
            device, elements, checkpoint(fixture, checkpoint_position, reference_name));
        std::cout << "trace label=" << label
                  << " max_abs=" << error.metrics.max_abs
                  << " mean_abs=" << error.metrics.mean_abs
                  << " rms=" << error.metrics.rms
                  << " relative_rms=" << error.metrics.relative_rms
                  << " max_index=" << error.index
                  << " reference=" << error.expected
                  << " gpu=" << error.actual << '\n';
    }

    void report(const char* label, const float* device, std::size_t elements,
                const std::string& reference_name) const {
        report_at(position, label, device, elements, reference_name);
    }

    void report_host(const char* label, std::span<const float> actual,
                     const std::string& reference_name) const {
        const auto error = located_host_error(
            actual, read_f32(checkpoint(fixture, position, reference_name), kStateElements));
        std::cout << "trace label=" << label
                  << " max_abs=" << error.metrics.max_abs
                  << " mean_abs=" << error.metrics.mean_abs
                  << " rms=" << error.metrics.rms
                  << " relative_rms=" << error.metrics.relative_rms
                  << " max_index=" << error.index
                  << " reference=" << error.expected
                  << " gpu=" << error.actual << '\n';
    }

private:
    static constexpr std::size_t kStateElements = 48 * 128 * 128;
};

struct UpdateProvenance {
    std::size_t head = 0;
    std::size_t row = 0;
    std::size_t column = 0;
    float previous = 0.0F;
    float decay = 0.0F;
    float beta = 0.0F;
    float value = 0.0F;
    float key = 0.0F;
    float query = 0.0F;
    float key_dot = 0.0F;
    float decayed = 0.0F;
    float delta = 0.0F;
    float candidate = 0.0F;
    float query_dot = 0.0F;
};

struct RecurrentOperands {
    std::vector<float> previous;
    std::vector<float> query;
    std::vector<float> key;
    std::vector<float> value;
    std::vector<float> beta;
    std::vector<float> decay;
};

struct KeyPathCapture {
    std::vector<float> input;
    std::vector<float> normalized;
    std::vector<float> qkv;
    std::vector<float> key;
    std::vector<float> key_norm;
};

struct LayerPathCapture {
    std::vector<float> input;
    std::vector<float> normalized;
    std::vector<float> qkv;
    std::vector<float> recurrent_output;
    std::vector<float> gated;
    std::vector<float> attention_residual;
    std::vector<float> post_normalized;
    std::vector<float> ffn_output;
    std::vector<float> layer_output;
};

struct OutputProjectionPathCapture {
    std::vector<float> gated;
    std::vector<std::byte> q8_input;
    std::vector<float> projected;
    std::vector<float> input;
    std::vector<float> residual;
};

struct GatePathCapture {
    std::vector<float> recurrent_output;
    std::vector<float> head_norm;
    std::vector<float> head_scaled;
    std::vector<float> normalized;
    std::vector<float> gate;
    std::vector<float> gated;
};

LocatedError located_host_error(std::span<const float> actual,
                                 std::span<const float> expected) {
    LocatedError result;
    result.metrics = detailed_compare(actual, expected);
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) >
            std::fabs(result.actual - result.expected)) {
            result.index = i;
            result.actual = actual[i];
            result.expected = expected[i];
        }
    }
    return result;
}

std::vector<float> download(const void* device, std::size_t elements) {
    std::vector<float> host(elements);
    MIINFER_HIP_CHECK(hipMemcpy(host.data(), device, elements * sizeof(float),
                                hipMemcpyDeviceToHost));
    return host;
}

std::vector<std::byte> download_bytes(const void* device, std::size_t bytes) {
    std::vector<std::byte> host(bytes);
    MIINFER_HIP_CHECK(hipMemcpy(host.data(), device, bytes, hipMemcpyDeviceToHost));
    return host;
}

std::uint64_t host_fingerprint(std::span<const std::byte> bytes) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const auto byte : bytes) {
        hash ^= static_cast<std::uint8_t>(byte);
        hash *= 1099511628211ULL;
    }
    return hash;
}

struct ByteMismatch {
    std::size_t count = 0;
    std::size_t first = 0;
};

ByteMismatch compare_bytes(std::span<const std::byte> actual,
                           std::span<const std::byte> expected) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("byte comparison size mismatch");
    }
    ByteMismatch result;
    result.first = actual.size();
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            if (result.count == 0) result.first = i;
            ++result.count;
        }
    }
    return result;
}

std::uint64_t fingerprint(const void* device, std::size_t bytes) {
    std::vector<std::byte> host(bytes);
    MIINFER_HIP_CHECK(hipMemcpy(host.data(), device, bytes, hipMemcpyDeviceToHost));
    std::uint64_t hash = 1469598103934665603ULL;
    for (const auto byte : host) {
        hash ^= static_cast<std::uint8_t>(byte);
        hash *= 1099511628211ULL;
    }
    return hash;
}

void require_type(const miinfer::GgufTensor& tensor_value,
                 std::initializer_list<miinfer::GgufTensorType> types) {
    if (std::find(types.begin(), types.end(), tensor_value.type) == types.end()) {
        throw std::runtime_error("unsupported tensor type for " + tensor_value.name);
    }
}

void project(const miinfer::GgufTensor& weight, const Buffer& device_weight,
             const float* input, miinfer::Q8KDeviceBlock* q8, float* output,
             std::uint32_t rows, std::uint32_t columns,
             bool input_prequantized = false) {
    if (!input_prequantized) {
        miinfer::launch_qwen3_q8_k_quantize(input, q8, columns);
    }
    switch (weight.type) {
    case miinfer::GgufTensorType::q4_k: {
        const char* dot4 = std::getenv("MIINFER_Q4K_DOT4");
        if (dot4 == nullptr || std::strcmp(dot4, "0") != 0) {
            miinfer::launch_qwen3_q4_k_q8_k_gemv_dot4(
                static_cast<const miinfer::Q4KDeviceBlock*>(device_weight->get()), q8,
                output, rows, columns);
        } else {
            miinfer::launch_qwen3_q4_k_q8_k_gemv(
                static_cast<const miinfer::Q4KDeviceBlock*>(device_weight->get()), q8,
                output, rows, columns);
        }
        return;
    }
    case miinfer::GgufTensorType::q5_k:
        miinfer::launch_qwen3_q5_k_q8_k_gemv(
            static_cast<const miinfer::Q5KDeviceBlock*>(device_weight->get()), q8,
            output, rows, columns);
        return;
    case miinfer::GgufTensorType::q6_k:
        miinfer::launch_qwen3_q6_k_q8_k_gemv(
            static_cast<const miinfer::Q6KDeviceBlock*>(device_weight->get()), q8,
            output, rows, columns);
        return;
    default:
        throw std::runtime_error("unsupported quantized projection: " + weight.name);
    }
}

void project_q5_q8_1(const Buffer& device_weight, const float* input,
                     miinfer::Q8_1Block* q8, float* output,
                     std::uint32_t rows, std::uint32_t columns) {
    miinfer::launch_q8_1_quantize_f32(input, q8, columns);
    miinfer::launch_qwen3_q5_k_q8_1_mmvq(
        static_cast<const miinfer::Q5KDeviceBlock*>(device_weight->get()), q8,
        output, rows, columns);
}

void project_q4_q8_1_prequantized(const Buffer& device_weight,
                                  const miinfer::Q8_1Block* q8, float* output,
                                  std::uint32_t rows, std::uint32_t columns,
                                  bool lds_input = false,
                                  bool lds_metadata = false,
                                  bool lds_decoded_metadata = false) {
    if (lds_decoded_metadata) {
        miinfer::launch_qwen3_q4_k_q8_1_mmvq_lds_decoded_metadata(
            static_cast<const miinfer::Q4KDeviceBlock*>(device_weight->get()), q8,
            output, rows, columns);
    } else if (lds_metadata) {
        miinfer::launch_qwen3_q4_k_q8_1_mmvq_lds_metadata(
            static_cast<const miinfer::Q4KDeviceBlock*>(device_weight->get()), q8,
            output, rows, columns);
    } else if (lds_input) {
        miinfer::launch_qwen3_q4_k_q8_1_mmvq_lds_input(
            static_cast<const miinfer::Q4KDeviceBlock*>(device_weight->get()), q8,
            output, rows, columns);
    } else {
        miinfer::launch_qwen3_q4_k_q8_1_mmvq(
            static_cast<const miinfer::Q4KDeviceBlock*>(device_weight->get()), q8,
            output, rows, columns);
    }
}

void project_q4_q8_1_expanded(const Buffer& device_weight, const float* input,
                              miinfer::Q8_1Block* q8, float* output,
                              std::uint32_t rows, std::uint32_t columns) {
    miinfer::launch_q8_1_quantize_f32(input, q8, columns);
    miinfer::launch_qwen3_q4_k_q8_1_mmvq_expanded(
        static_cast<const miinfer::Q4KExpandedDeviceBlock*>(device_weight->get()), q8,
        output, rows, columns);
}

void project_q4_q8_1(const Buffer& device_weight, const float* input,
                     miinfer::Q8_1Block* q8, float* output,
                     std::uint32_t rows, std::uint32_t columns,
                     bool lds_input = false, bool lds_metadata = false,
                     bool lds_decoded_metadata = false) {
    miinfer::launch_q8_1_quantize_f32(input, q8, columns);
    project_q4_q8_1_prequantized(
        device_weight, q8, output, rows, columns, lds_input, lds_metadata,
        lds_decoded_metadata);
}

void project_q6_q8_k_dot4(const Buffer& device_weight, const float* input,
                          miinfer::Q8KDeviceBlock* q8, float* output,
                          std::uint32_t rows, std::uint32_t columns) {
    miinfer::launch_qwen3_q8_k_quantize(input, q8, columns);
    miinfer::launch_qwen3_q6_k_q8_k_gemv_dot4(
        static_cast<const miinfer::Q6KDeviceBlock*>(device_weight->get()), q8,
        output, rows, columns);
}

struct RecurrentLayer {
    std::size_t index;
    const miinfer::Qwen35Model& model;

    const miinfer::GgufTensor& attn_norm;
    const miinfer::GgufTensor& qkv_weight;
    const miinfer::GgufTensor& gate_weight;
    const miinfer::GgufTensor& beta_weight;
    const miinfer::GgufTensor& alpha_weight;
    const miinfer::GgufTensor& dt_weight;
    const miinfer::GgufTensor& a_weight;
    const miinfer::GgufTensor& conv_weight;
    const miinfer::GgufTensor& ssm_norm_weight;
    const miinfer::GgufTensor& ssm_out_weight;
    const miinfer::GgufTensor& post_norm_weight;
    const miinfer::GgufTensor& ffn_gate_weight;
    const miinfer::GgufTensor& ffn_up_weight;
    const miinfer::GgufTensor& ffn_down_weight;

    Buffer d_attn_norm, d_qkv, d_qkv_mmq, d_gate, d_gate_mmq, d_beta, d_alpha, d_conv, d_ssm_norm;
    Buffer d_ssm_out, d_ssm_out_mmq, d_post_norm, d_ffn_gate, d_ffn_gate_mmq, d_ffn_up, d_ffn_up_mmq;
    Buffer d_ffn_down, d_ffn_down_mmq, d_ffn_down_expanded, d_dt, d_a;
    Buffer normalized, qkv, gate, beta_raw, alpha_raw, beta, decay, history;
    Buffer query, key, value, query_norm, key_norm, state, recurrent_output;
    Buffer head_norm, gated, projected, residual, post_normalized;
    Buffer ffn_gate, ffn_up, ffn_activation, layer_output, q8, q8_1;
    RecurrentTrace* trace = nullptr;
    UpdateProvenance* provenance = nullptr;
    std::uint32_t provenance_position = 0;
    std::size_t provenance_index = 0;
    RecurrentOperands* operand_capture = nullptr;
    std::uint32_t operand_capture_position = 0;
    KeyPathCapture* key_path_capture = nullptr;
    std::uint32_t key_path_capture_position = 0;
    LayerPathCapture* layer_path_capture = nullptr;
    std::uint32_t layer_path_capture_position = 0;
    OutputProjectionPathCapture* output_projection_path_capture = nullptr;
    std::uint32_t output_projection_path_capture_position = 0;
    GatePathCapture* gate_path_capture = nullptr;
    std::uint32_t gate_path_capture_position = 0;
    struct StageProfile {
        std::array<hipEvent_t, 14> start{};
        std::array<hipEvent_t, 14> end{};
        hipEvent_t tail_start = nullptr;
        hipEvent_t tail_end = nullptr;
        hipEvent_t prepare_start = nullptr;
        hipEvent_t prepare_end = nullptr;
        hipEvent_t ordered_start = nullptr;
        hipEvent_t ordered_end = nullptr;
        std::array<hipEvent_t, 4> tail_family_start{};
        std::array<hipEvent_t, 4> tail_family_end{};
        mutable bool tail_recorded = false;
        mutable bool prepare_recorded = false;
        mutable bool ordered_recorded = false;
        mutable std::array<bool, 4> tail_family_recorded{};
        mutable std::array<bool, 14> stage_recorded{};
    };
    bool no_decay_store = false;
    StageProfile* stage_profile = nullptr;
    std::uint32_t stage_profile_position = 0;
    std::size_t stage_profile_chunk_base = 0;
    bool reuse_projection_q8 = false;
    bool q5_q8_1_mmvq = false;
    bool q4_q8_1_mmvq = false;
    bool q4_q8_1_gate_up = false;
    bool q4_q8_1_lds_input = true;
    bool q4_q8_1_lds_metadata = true;
    bool q4_q8_1_lds_decoded_metadata = true;
    bool q4_q8_1_attn_gate = false;
    bool direct_layer_output = false;
    bool fused_gate_up_swiglu = false;
    bool fused_core_q8 = false;
    bool fused_norm_q8 = false;
    bool expanded_down = true;
    Buffer d_ffn_down_native;
    Buffer d_ffn_down_dense_source;
    Buffer d_ffn_gate_native, d_ffn_up_native;
    Buffer d_ffn_swiglu_native;
    Buffer d_attn_gate_native;
    Buffer d_ssm_out_native;
    Buffer d_qkv_native;
    Buffer d_qkv_gate_combined;
    bool q6_q8_k_dot4_qkv = false;
    bool transposed_state = true;
    bool transposed_no_decay_store = true;
    bool transposed_lds_inputs = true;
    bool fuse_beta_alpha = false;
    bool dual_beta_alpha_projection = true;
    bool dual_beta_alpha_prepare = false;
    bool dual_head_normalize = false;
    bool row_wave_state = false;
    bool fused_recurrent_core = false;
    bool fused_add_rms_norm = true;
    bool fused_interlayer_norm = true;
    bool prefill_batch_enabled = false;
    bool prefill_recurrent_batch = false;
    bool prefill_recurrent_q8 = false;
    bool dense_projection_prefill = false;
    bool dense_qkv_prefill = false;
    bool prefill_wide_qkv = false;
    bool prefill_gdn_chunkwise = false;
    bool prefill_dense_ffn_down = false;
    std::size_t prefill_capacity = kPrefillBatch;
    bool prefill_wide_repacked = false;
    bool wide_dense_ffn = false;
    bool wide_dense_all = false;
    bool prefill_wide_validate = false;
    std::array<std::uint32_t, 9> m23_dispatch_counts{};
    bool m23_trace_dispatch = false;
    M23ProfileCounters* m23_profile_counters = nullptr;
    bool prefill_core_batch_active = false;
    Buffer prefill_normalized, prefill_qkv, prefill_gate, prefill_q8_1, prefill_mmq_q8;
    Buffer prefill_gated, prefill_residual, prefill_post_normalized;
    Buffer prefill_ffn_gate, prefill_ffn_up, prefill_ffn_activation, prefill_projected;
    Buffer prefill_core_query, prefill_core_key, prefill_core_value;
    Buffer prefill_core_beta, prefill_core_decay, prefill_core_gate;
    miinfer::M12GdnChunkWorkspace m12_gdn_workspace{};
    bool m12_gdn_workspace_ready = false;
    float* m12_gdn_raw_output = nullptr;
    __half* m12_dense_weights = nullptr;
    __half* m12_dense_input = nullptr;
    std::byte* m12_dense_source = nullptr;
    miinfer::RocblasGemmHandle* m12_dense_handle = nullptr;
    bool m12_dense_workspace_ready = false;
    Buffer wide_qkv_fp16, wide_gate_fp16, wide_ssm_out_fp16;
    Buffer wide_ffn_gate_fp16, wide_ffn_up_fp16, wide_ffn_down_fp16;
    Buffer resident_qkv_mmq, resident_gate_mmq, resident_ssm_out_mmq;
    Buffer resident_ffn_gate_mmq, resident_ffn_up_mmq, resident_ffn_down_mmq;
    Buffer decode_mmq_q8;
    bool resident_m23_ffn = false;
    bool resident_m23_all = false;
    bool resident_fp16_prefill = false;
    std::vector<Q4KMmqTile> host_qkv_q4_mmq, host_gate_mmq, host_ffn_gate_mmq, host_ffn_up_mmq;
    std::vector<Q5KMmqTile> host_ssm_out_mmq;
    std::vector<Q4KMmqTile> host_ffn_down_q4_mmq;
    std::vector<Q6KMmqTile> host_qkv_q6_mmq, host_ffn_down_q6_mmq;

    void count_m23_dispatch(std::size_t family) {
        if (m23_trace_dispatch) ++m23_dispatch_counts[family];
        if (m23_profile_counters != nullptr) ++m23_profile_counters->recurrent_dispatches[family];
    }

    RecurrentLayer(const miinfer::Qwen35Model& model_value, std::size_t layer,
                   const std::filesystem::path& fixture)
        : index(layer), model(model_value),
          attn_norm(tensor(*model.file(), name("attn_norm.weight"))),
          qkv_weight(tensor(*model.file(), name("attn_qkv.weight"))),
          gate_weight(tensor(*model.file(), name("attn_gate.weight"))),
          beta_weight(tensor(*model.file(), name("ssm_beta.weight"))),
          alpha_weight(tensor(*model.file(), name("ssm_alpha.weight"))),
          dt_weight(tensor(*model.file(), name("ssm_dt.bias"))),
          a_weight(tensor(*model.file(), name("ssm_a"))),
          conv_weight(tensor(*model.file(), name("ssm_conv1d.weight"))),
          ssm_norm_weight(tensor(*model.file(), name("ssm_norm.weight"))),
          ssm_out_weight(tensor(*model.file(), name("ssm_out.weight"))),
          post_norm_weight(tensor(*model.file(), name("post_attention_norm.weight"))),
          ffn_gate_weight(tensor(*model.file(), name("ffn_gate.weight"))),
          ffn_up_weight(tensor(*model.file(), name("ffn_up.weight"))),
          ffn_down_weight(tensor(*model.file(), name("ffn_down.weight"))) {
        const char* no_decay_store_env = std::getenv("MIINFER_DELTA_NO_DECAY_STORE");
        no_decay_store = no_decay_store_env == nullptr
            || std::strcmp(no_decay_store_env, "0") != 0;
        const char* reuse_projection_q8_env = std::getenv("MIINFER_REUSE_PROJECTION_Q8");
        reuse_projection_q8 = reuse_projection_q8_env == nullptr
            || std::strcmp(reuse_projection_q8_env, "0") != 0;
        const char* q5_q8_1_mmvq_env = std::getenv("MIINFER_Q5K_Q8_1_MMVQ");
        q5_q8_1_mmvq = q5_q8_1_mmvq_env == nullptr
            || std::strcmp(q5_q8_1_mmvq_env, "0") != 0;
        const char* q4_q8_1_mmvq_env = std::getenv("MIINFER_Q4K_Q8_1_MMVQ");
        q4_q8_1_mmvq = q4_q8_1_mmvq_env == nullptr
            || std::strcmp(q4_q8_1_mmvq_env, "0") != 0;
        const char* q4_q8_1_gate_up_env = std::getenv("MIINFER_Q4K_Q8_1_MMVQ_FFN_GATE_UP");
        q4_q8_1_gate_up = q4_q8_1_gate_up_env == nullptr
            || std::strcmp(q4_q8_1_gate_up_env, "0") != 0;
        const char* q4_q8_1_lds_input_env = std::getenv("MIINFER_Q4K_Q8_1_LDS_INPUT");
        q4_q8_1_lds_input = q4_q8_1_lds_input_env == nullptr
            || std::strcmp(q4_q8_1_lds_input_env, "0") != 0;
        const char* q4_q8_1_lds_metadata_env = std::getenv("MIINFER_Q4K_Q8_1_LDS_METADATA");
        q4_q8_1_lds_metadata = q4_q8_1_lds_metadata_env == nullptr
            || std::strcmp(q4_q8_1_lds_metadata_env, "0") != 0;
        const char* q4_q8_1_lds_decoded_metadata_env =
            std::getenv("MIINFER_Q4K_Q8_1_LDS_DECODED_METADATA");
        q4_q8_1_lds_decoded_metadata = q4_q8_1_lds_decoded_metadata_env == nullptr
            || std::strcmp(q4_q8_1_lds_decoded_metadata_env, "0") != 0;
        const char* q4_q8_1_attn_gate_env = std::getenv("MIINFER_Q4K_Q8_1_MMVQ_ATTN_GATE");
        q4_q8_1_attn_gate = q4_q8_1_attn_gate_env == nullptr
            || std::strcmp(q4_q8_1_attn_gate_env, "0") != 0;
        const char* direct_layer_output_env = std::getenv("MIINFER_DIRECT_LAYER_OUTPUT");
        const char* hip_graph_env = std::getenv("MIINFER_HIP_GRAPH");
        const bool use_hip_graph = hip_graph_env != nullptr && std::strcmp(hip_graph_env, "0") != 0;
        direct_layer_output = (direct_layer_output_env != nullptr
            && std::strcmp(direct_layer_output_env, "0") != 0) || use_hip_graph;
        const char* expanded_down_env = std::getenv("MIINFER_Q4K_EXPANDED_DOWN");
        expanded_down = expanded_down_env == nullptr
            || std::strcmp(expanded_down_env, "0") != 0;
        const char* dual_beta_alpha_env = std::getenv("MIINFER_DUAL_BETA_ALPHA_GEMV");
        if (dual_beta_alpha_env != nullptr) {
            dual_beta_alpha_projection = std::strcmp(dual_beta_alpha_env, "0") != 0;
        }
        const char* dual_beta_alpha_prepare_env = std::getenv("MIINFER_DUAL_BETA_ALPHA_PREPARE");
        dual_beta_alpha_prepare = dual_beta_alpha_prepare_env != nullptr
            && std::strcmp(dual_beta_alpha_prepare_env, "0") != 0;
        const char* dual_head_normalize_env = std::getenv("MIINFER_DUAL_HEAD_NORMALIZE");
        dual_head_normalize = dual_head_normalize_env != nullptr
            && std::strcmp(dual_head_normalize_env, "0") != 0;
        const char* fused_gate_up_swiglu_env = std::getenv("MIINFER_FUSED_GATE_UP_SWIGLU");
        fused_gate_up_swiglu = fused_gate_up_swiglu_env == nullptr
            || std::strcmp(fused_gate_up_swiglu_env, "0") != 0;
        const char* fused_core_q8_env = std::getenv("MIINFER_FUSED_CORE_Q8");
        fused_core_q8 = fused_core_q8_env == nullptr
            || std::strcmp(fused_core_q8_env, "0") != 0;
        const char* fused_norm_q8_env = std::getenv("MIINFER_FUSED_NORM_Q8");
        fused_norm_q8 = fused_norm_q8_env != nullptr
            && std::strcmp(fused_norm_q8_env, "0") != 0;
        const char* q6_q8_k_dot4_qkv_env = std::getenv("MIINFER_Q6K_Q8K_DOT4_QKV");
        q6_q8_k_dot4_qkv = q6_q8_k_dot4_qkv_env == nullptr
            || std::strcmp(q6_q8_k_dot4_qkv_env, "0") != 0;
        const char* transposed_state_env = std::getenv("MIINFER_DELTA_TRANSPOSED_STATE");
        if (transposed_state_env != nullptr) {
            transposed_state = std::strcmp(transposed_state_env, "0") != 0;
        }
        const char* transposed_no_decay_store_env =
            std::getenv("MIINFER_DELTA_TRANSPOSED_NO_DECAY_STORE");
        if (transposed_no_decay_store_env != nullptr) {
            transposed_no_decay_store = std::strcmp(transposed_no_decay_store_env, "0") != 0;
        }
        const char* transposed_lds_inputs_env =
            std::getenv("MIINFER_DELTA_TRANSPOSED_LDS_INPUTS");
        if (transposed_lds_inputs_env != nullptr) {
            transposed_lds_inputs = std::strcmp(transposed_lds_inputs_env, "0") != 0;
        }
        const char* fuse_beta_alpha_env = std::getenv("MIINFER_FUSE_BETA_ALPHA");
        fuse_beta_alpha = fuse_beta_alpha_env != nullptr
            && std::strcmp(fuse_beta_alpha_env, "0") != 0;
        const char* row_wave_state_env = std::getenv("MIINFER_DELTA_ROW_WAVES");
        row_wave_state = row_wave_state_env != nullptr
            && std::strcmp(row_wave_state_env, "0") != 0;
        const char* fused_recurrent_core_env = std::getenv("MIINFER_FUSED_RECURRENT_CORE");
        fused_recurrent_core = fused_recurrent_core_env == nullptr
            || std::strcmp(fused_recurrent_core_env, "0") != 0;
        const char* fused_add_rms_norm_env = std::getenv("MIINFER_FUSED_ADD_RMS_NORM");
        fused_add_rms_norm = fused_add_rms_norm_env == nullptr
            || std::strcmp(fused_add_rms_norm_env, "0") != 0;
        const char* fused_interlayer_norm_env = std::getenv("MIINFER_FUSED_INTERLAYER_NORM");
        fused_interlayer_norm = fused_interlayer_norm_env == nullptr
            || std::strcmp(fused_interlayer_norm_env, "0") != 0;
        const char* prefill_batch_env = std::getenv("MIINFER_PREFILL_LAYER_MAJOR");
        prefill_batch_enabled = prefill_batch_env != nullptr
            && std::strcmp(prefill_batch_env, "0") != 0;
        const char* prefill_recurrent_env = std::getenv("MIINFER_PREFILL_RECURRENT_BATCH");
        prefill_recurrent_batch = prefill_batch_enabled
            && (prefill_recurrent_env == nullptr || std::strcmp(prefill_recurrent_env, "0") != 0);
        const char* prefill_recurrent_q8_env = std::getenv("MIINFER_PREFILL_RECURRENT_Q8");
        prefill_recurrent_q8 = prefill_recurrent_batch
            && (prefill_recurrent_q8_env == nullptr || std::strcmp(prefill_recurrent_q8_env, "0") != 0);
        const char* wide_prefill_env = std::getenv("MIINFER_PREFILL_WIDE_CHUNK");
        const bool wide_prefill = prefill_batch_enabled && wide_prefill_env != nullptr
            && std::strcmp(wide_prefill_env, "0") != 0;
        const char* prefill_gdn_env = std::getenv("MIINFER_PREFILL_GDN_CHUNKWISE");
        prefill_gdn_chunkwise = prefill_recurrent_batch
            && (wide_prefill || (prefill_gdn_env != nullptr && std::strcmp(prefill_gdn_env, "0") != 0));
        const char* prefill_dense_env = std::getenv("MIINFER_PREFILL_DENSE_FFN_DOWN");
        prefill_dense_ffn_down = prefill_batch_enabled
            && prefill_dense_env != nullptr && std::strcmp(prefill_dense_env, "0") != 0;
        const char* dense_projection_env = std::getenv("MIINFER_PREFILL_DENSE_PROJECTIONS");
        dense_projection_prefill = prefill_batch_enabled
            && dense_projection_env != nullptr && std::strcmp(dense_projection_env, "0") != 0;
        const char* wide_dense_ffn_env = std::getenv("MIINFER_PREFILL_WIDE_DENSE_FFN");
        const bool wide_dense_ffn_requested = prefill_batch_enabled && wide_prefill
            && wide_dense_ffn_env != nullptr && std::strcmp(wide_dense_ffn_env, "0") != 0;
        wide_dense_ffn = wide_dense_ffn_requested;
        const char* wide_dense_all_env = std::getenv("MIINFER_PREFILL_WIDE_DENSE_ALL");
        wide_dense_all = prefill_batch_enabled && wide_prefill
            && wide_dense_all_env != nullptr && std::strcmp(wide_dense_all_env, "0") != 0;
        wide_dense_ffn = wide_dense_ffn || wide_dense_all;
        const char* dense_qkv_env = std::getenv("MIINFER_PREFILL_DENSE_QKV");
        dense_qkv_prefill = dense_projection_prefill
            && dense_qkv_env != nullptr && std::strcmp(dense_qkv_env, "0") != 0;
        prefill_wide_qkv = wide_prefill;
        const char* wide_repacked_env = std::getenv("MIINFER_PREFILL_WIDE_REPACKED_MMQ");
        prefill_wide_repacked = wide_repacked_env != nullptr
            && std::strcmp(wide_repacked_env, "0") != 0;
        const char* resident_ffn_env = std::getenv("MIINFER_PREFILL_REPACKED_RESIDENT_FFN");
        resident_m23_ffn = prefill_wide_repacked && resident_ffn_env != nullptr
            && std::strcmp(resident_ffn_env, "0") != 0;
        prefill_wide_validate = prefill_wide_repacked && std::getenv("MIINFER_WIDE_VALIDATE") != nullptr
            && index == 0;
        const char* resident_all_env = std::getenv("MIINFER_PREFILL_REPACKED_RESIDENT_ALL");
        resident_m23_all = prefill_wide_repacked && resident_all_env != nullptr
            && std::strcmp(resident_all_env, "0") != 0 && !prefill_wide_validate
            && !wide_dense_ffn && !dense_projection_prefill && !dense_qkv_prefill
            && !prefill_dense_ffn_down;
        if (resident_m23_all) resident_m23_ffn = true;
        const char* resident_fp16_env = std::getenv("MIINFER_PREFILL_REPACKED_FP16");
        resident_fp16_prefill = prefill_wide_repacked && resident_fp16_env != nullptr
            && std::strcmp(resident_fp16_env, "0") != 0;
        const char* trace_dispatch_env = std::getenv("MIINFER_PREFILL_TRACE_DISPATCH");
        m23_trace_dispatch = trace_dispatch_env != nullptr && std::strcmp(trace_dispatch_env, "0") != 0;
        prefill_capacity = (prefill_dense_ffn_down || prefill_gdn_chunkwise || wide_prefill)
            ? (wide_prefill ? configured_wide_prefill_batch() : kM12PrefillBatch)
            : kPrefillBatch;
        require_type(qkv_weight, {miinfer::GgufTensorType::q4_k,
                                   miinfer::GgufTensorType::q6_k});
        require_type(gate_weight, {miinfer::GgufTensorType::q4_k});
        require_type(ssm_out_weight, {miinfer::GgufTensorType::q5_k});
        require_type(ffn_gate_weight, {miinfer::GgufTensorType::q4_k});
        require_type(ffn_up_weight, {miinfer::GgufTensorType::q4_k});
        require_type(ffn_down_weight, {miinfer::GgufTensorType::q4_k,
                                       miinfer::GgufTensorType::q6_k});
        if (prefill_dense_ffn_down && ffn_down_weight.type != miinfer::GgufTensorType::q4_k
            && ffn_down_weight.type != miinfer::GgufTensorType::q6_k) {
            throw std::runtime_error("M12 dense FFN-down requires Q4_K or Q6_K");
        }

        d_attn_norm = allocate(attn_norm.byte_size);
        if (!resident_m23_all && combined_qkv_gate_enabled() &&
            qkv_weight.type == miinfer::GgufTensorType::q4_k &&
            gate_weight.type == miinfer::GgufTensorType::q4_k) {
            d_qkv_gate_combined = copy_combined_q4k_tensors(qkv_weight, gate_weight);
        } else if (!resident_m23_all) {
            if (native_qkv_enabled()) {
                if (qkv_weight.type == miinfer::GgufTensorType::q4_k) {
                    d_qkv_native = copy_native_tensor(qkv_weight);
                } else if (qkv_weight.type == miinfer::GgufTensorType::q6_k) {
                    d_qkv_native = copy_native_q6k_tensor(qkv_weight);
                }
            }
            if (!d_qkv_native) {
                d_qkv = allocate(qkv_weight.byte_size);
            }
            if (dense_qkv_prefill && !d_qkv) {
                d_qkv = allocate(qkv_weight.byte_size);
            }
            if (native_attn_gate_enabled() && gate_weight.type == miinfer::GgufTensorType::q4_k) {
                d_attn_gate_native = copy_native_tensor(gate_weight);
            } else {
                d_gate = allocate(gate_weight.byte_size);
            }
            if (dense_qkv_prefill && !d_gate) {
                d_gate = allocate(gate_weight.byte_size);
            }
        }
        if (prefill_wide_qkv) {
            if (!d_qkv && (!prefill_wide_repacked || prefill_wide_validate)) d_qkv = allocate(qkv_weight.byte_size);
            if (!d_gate && (!prefill_wide_repacked || prefill_wide_validate)) d_gate = allocate(gate_weight.byte_size);
            if (qkv_weight.type == miinfer::GgufTensorType::q6_k) {
                host_qkv_q6_mmq = pack_q6k_mmq_tensor(qkv_weight);
            } else {
                host_qkv_q4_mmq = pack_q4k_mmq_tensor(qkv_weight);
            }
            host_gate_mmq = pack_q4k_mmq_tensor(gate_weight);
        }
        d_beta = allocate(beta_weight.byte_size);
        d_alpha = allocate(alpha_weight.byte_size);
        d_conv = allocate(conv_weight.byte_size);
        d_ssm_norm = allocate(ssm_norm_weight.byte_size);
        if (!resident_m23_all && native_ssm_out_enabled() && ssm_out_weight.type == miinfer::GgufTensorType::q5_k) {
            d_ssm_out_native = copy_native_q5k_tensor(ssm_out_weight);
        } else if (!resident_m23_all) {
            d_ssm_out = allocate(ssm_out_weight.byte_size);
        }
        d_post_norm = allocate(post_norm_weight.byte_size);
        if (!resident_m23_all && swiglu_paired_enabled() && ffn_gate_weight.type == miinfer::GgufTensorType::q4_k
            && ffn_up_weight.type == miinfer::GgufTensorType::q4_k) {
            d_ffn_swiglu_native = copy_swiglu_paired_tensors(ffn_gate_weight, ffn_up_weight);
        } else if (!resident_m23_all && native_gate_up_enabled() && ffn_gate_weight.type == miinfer::GgufTensorType::q4_k
            && ffn_up_weight.type == miinfer::GgufTensorType::q4_k) {
            d_ffn_gate_native = copy_native_tensor(ffn_gate_weight);
            d_ffn_up_native = copy_native_tensor(ffn_up_weight);
        } else if (!resident_m23_all) {
            d_ffn_gate = allocate(ffn_gate_weight.byte_size);
            d_ffn_up = allocate(ffn_up_weight.byte_size);
        }
        if (dense_projection_prefill && !d_ffn_gate) {
            d_ffn_gate = allocate(ffn_gate_weight.byte_size);
            d_ffn_up = allocate(ffn_up_weight.byte_size);
        }
        if (prefill_wide_qkv && !prefill_wide_repacked) {
            if (!d_ffn_gate) d_ffn_gate = allocate(ffn_gate_weight.byte_size);
            if (!d_ffn_up) d_ffn_up = allocate(ffn_up_weight.byte_size);
        }
        if (!resident_m23_all && ((native_down_enabled() && ffn_down_weight.type == miinfer::GgufTensorType::q4_k) ||
            (native_q6k_down_enabled() && ffn_down_weight.type == miinfer::GgufTensorType::q6_k))) {
            if (ffn_down_weight.type == miinfer::GgufTensorType::q4_k) {
                d_ffn_down_native = copy_native_down(ffn_down_weight);
            } else {
                d_ffn_down_native = copy_native_q6k_tensor(ffn_down_weight);
            }
        } else if (!resident_m23_all && expanded_down && ffn_down_weight.type == miinfer::GgufTensorType::q4_k) {
            d_ffn_down_expanded = copy_expanded_q4k(ffn_down_weight);
        }
        if (!resident_m23_all && (!d_ffn_down_native || (wide_dense_ffn && !wide_dense_all))) {
            d_ffn_down = allocate(ffn_down_weight.byte_size);
        }
        if (prefill_wide_qkv && !d_ffn_down && (!prefill_wide_repacked || prefill_wide_validate)) {
            d_ffn_down = allocate(ffn_down_weight.byte_size);
        }
        d_dt = allocate(dt_weight.byte_size);
        d_a = allocate(a_weight.byte_size);
        for (const auto& pair : std::initializer_list<std::pair<const miinfer::GgufTensor*, const Buffer*>>{
                 {&attn_norm, &d_attn_norm},
                 {&beta_weight, &d_beta},
                 {&alpha_weight, &d_alpha}, {&conv_weight, &d_conv},
                 {&ssm_norm_weight, &d_ssm_norm},
                 {&post_norm_weight, &d_post_norm},
                 {&dt_weight, &d_dt}, {&a_weight, &d_a}}) {
            upload_tensor(*pair.first, *pair.second);
        }
        if (d_qkv) upload_tensor(qkv_weight, d_qkv);
        if (d_gate) upload_tensor(gate_weight, d_gate);
        if (prefill_wide_qkv && !prefill_wide_repacked) {
            wide_qkv_fp16 = allocate(kChannels * kHidden * sizeof(__half));
            wide_gate_fp16 = allocate(kInner * kHidden * sizeof(__half));
            if (qkv_weight.type == miinfer::GgufTensorType::q4_k) {
                miinfer::launch_m12_q4k_to_fp16(
                    static_cast<const miinfer::Q4KDeviceBlock*>(d_qkv->get()),
                    static_cast<__half*>(wide_qkv_fp16->get()), kChannels, kHidden,
                    hipStreamPerThread);
            } else {
                miinfer::launch_m12_q6k_to_fp16(
                    static_cast<const miinfer::Q6KDeviceBlock*>(d_qkv->get()),
                    static_cast<__half*>(wide_qkv_fp16->get()), kChannels, kHidden,
                    hipStreamPerThread);
            }
            miinfer::launch_m12_q4k_to_fp16(
                static_cast<const miinfer::Q4KDeviceBlock*>(d_gate->get()),
                static_cast<__half*>(wide_gate_fp16->get()), kInner, kHidden,
                hipStreamPerThread);
        }
        if (d_ssm_out) upload_tensor(ssm_out_weight, d_ssm_out);
        if (prefill_wide_qkv) {
            if (!d_ssm_out && (!prefill_wide_repacked || prefill_wide_validate)) {
                d_ssm_out = allocate(ssm_out_weight.byte_size);
                upload_tensor(ssm_out_weight, d_ssm_out);
            }
            host_ssm_out_mmq = pack_q5k_mmq_tensor(ssm_out_weight);
            if (!prefill_wide_repacked) {
                wide_ssm_out_fp16 = allocate(kHidden * kInner * sizeof(__half));
                miinfer::launch_m12_q5k_to_fp16(
                    static_cast<const miinfer::Q5KDeviceBlock*>(d_ssm_out->get()),
                    static_cast<__half*>(wide_ssm_out_fp16->get()), kHidden, kInner,
                    hipStreamPerThread);
            }
        }
        if (d_ffn_gate) upload_tensor(ffn_gate_weight, d_ffn_gate);
        if (d_ffn_up) upload_tensor(ffn_up_weight, d_ffn_up);
        if (d_ffn_down) upload_tensor(ffn_down_weight, d_ffn_down);
        if (prefill_wide_qkv) {
            host_ffn_gate_mmq = pack_q4k_mmq_tensor(ffn_gate_weight);
            host_ffn_up_mmq = pack_q4k_mmq_tensor(ffn_up_weight);
            if (ffn_down_weight.type == miinfer::GgufTensorType::q4_k) {
                host_ffn_down_q4_mmq = pack_q4k_mmq_tensor(ffn_down_weight);
            } else {
                host_ffn_down_q6_mmq = pack_q6k_mmq_tensor(ffn_down_weight);
            }
            if (resident_m23_ffn) {
                const auto upload_resident = [](const auto& packed) {
                    auto result = allocate(packed.size() * sizeof(packed.front()));
                    upload(packed.data(), result->get(), packed.size() * sizeof(packed.front()));
                    return result;
                };
                resident_ffn_gate_mmq = upload_resident(host_ffn_gate_mmq);
                resident_ffn_up_mmq = upload_resident(host_ffn_up_mmq);
                resident_ffn_down_mmq = ffn_down_weight.type == miinfer::GgufTensorType::q4_k
                    ? upload_resident(host_ffn_down_q4_mmq)
                    : upload_resident(host_ffn_down_q6_mmq);
            }
            if (resident_m23_all) {
                const auto upload_resident = [](const auto& packed) {
                    auto result = allocate(packed.size() * sizeof(packed.front()));
                    upload(packed.data(), result->get(), packed.size() * sizeof(packed.front()));
                    return result;
                };
                resident_qkv_mmq = qkv_weight.type == miinfer::GgufTensorType::q4_k
                    ? upload_resident(host_qkv_q4_mmq)
                    : upload_resident(host_qkv_q6_mmq);
                resident_gate_mmq = upload_resident(host_gate_mmq);
                resident_ssm_out_mmq = upload_resident(host_ssm_out_mmq);
                decode_mmq_q8 = allocate((kFfnInner / 128) * sizeof(miinfer::M23Q8_1MmqBlock));
            }
            if (!prefill_wide_repacked) {
                wide_ffn_gate_fp16 = allocate(kFfnInner * kHidden * sizeof(__half));
                wide_ffn_up_fp16 = allocate(kFfnInner * kHidden * sizeof(__half));
                wide_ffn_down_fp16 = allocate(kHidden * kFfnInner * sizeof(__half));
                miinfer::launch_m12_q4k_to_fp16(
                    static_cast<const miinfer::Q4KDeviceBlock*>(d_ffn_gate->get()),
                    static_cast<__half*>(wide_ffn_gate_fp16->get()), kFfnInner, kHidden,
                    hipStreamPerThread);
                miinfer::launch_m12_q4k_to_fp16(
                    static_cast<const miinfer::Q4KDeviceBlock*>(d_ffn_up->get()),
                    static_cast<__half*>(wide_ffn_up_fp16->get()), kFfnInner, kHidden,
                    hipStreamPerThread);
                if (ffn_down_weight.type == miinfer::GgufTensorType::q4_k) {
                    miinfer::launch_m12_q4k_to_fp16(
                        static_cast<const miinfer::Q4KDeviceBlock*>(d_ffn_down->get()),
                        static_cast<__half*>(wide_ffn_down_fp16->get()), kHidden, kFfnInner,
                        hipStreamPerThread);
                } else {
                    miinfer::launch_m12_q6k_to_fp16(
                        static_cast<const miinfer::Q6KDeviceBlock*>(d_ffn_down->get()),
                        static_cast<__half*>(wide_ffn_down_fp16->get()), kHidden, kFfnInner,
                        hipStreamPerThread);
                }
            }
        }
        if (prefill_dense_ffn_down) {
            d_ffn_down_dense_source = allocate(ffn_down_weight.byte_size);
            upload_tensor(ffn_down_weight, d_ffn_down_dense_source);
        }

        normalized = allocate(kHidden * sizeof(float));
        qkv = allocate((d_qkv_gate_combined ? (kChannels + kInner) : kChannels) * sizeof(float));
        if (!d_qkv_gate_combined) {
            gate = allocate(kInner * sizeof(float));
        }
        beta_raw = allocate(kVHeads * sizeof(float));
        alpha_raw = allocate(kVHeads * sizeof(float));
        beta = allocate(kVHeads * sizeof(float));
        decay = allocate(kVHeads * sizeof(float));
        history = allocate(4 * kChannels * sizeof(float));
        query = allocate(kKHeads * kState * sizeof(float));
        key = allocate(kKHeads * kState * sizeof(float));
        value = allocate(kVHeads * kState * sizeof(float));
        query_norm = allocate(kKHeads * kState * sizeof(float));
        key_norm = allocate(kKHeads * kState * sizeof(float));
        state = allocate(kVHeads * kState * kState * sizeof(float));
        recurrent_output = allocate(kVHeads * kState * sizeof(float));
        head_norm = allocate(kVHeads * kState * sizeof(float));
        gated = allocate(kVHeads * kState * sizeof(float));
        projected = allocate(kHidden * sizeof(float));
        residual = allocate(kHidden * sizeof(float));
        post_normalized = allocate(kHidden * sizeof(float));
        ffn_gate = allocate(kFfnInner * sizeof(float));
        ffn_up = allocate(kFfnInner * sizeof(float));
        ffn_activation = allocate(kFfnInner * sizeof(float));
        layer_output = allocate(kHidden * sizeof(float));
        q8 = allocate((kFfnInner / 256) * sizeof(miinfer::Q8KDeviceBlock));
        q8_1 = allocate((kFfnInner / miinfer::kQ8_1BlockSize) * sizeof(miinfer::Q8_1Block));

        if (prefill_batch_enabled && !shared_wide_prefill_enabled()) {
            prefill_normalized = allocate(prefill_capacity * kHidden * sizeof(float));
            prefill_qkv = allocate(prefill_capacity * (kChannels + kInner) * sizeof(float));
            prefill_gate = allocate(prefill_capacity * kInner * sizeof(float));
            prefill_q8_1 = allocate(prefill_capacity * (kFfnInner / miinfer::kQ8_1BlockSize)
                                     * sizeof(miinfer::Q8_1Block));
            if (prefill_wide_qkv) {
                prefill_mmq_q8 = allocate(prefill_capacity * (kFfnInner / 128)
                                          * sizeof(miinfer::M23Q8_1MmqBlock));
            }
            prefill_gated = allocate(prefill_capacity * kInner * sizeof(float));
            prefill_residual = allocate(prefill_capacity * kHidden * sizeof(float));
            prefill_post_normalized = allocate(prefill_capacity * kHidden * sizeof(float));
            prefill_ffn_gate = allocate(prefill_capacity * kFfnInner * sizeof(float));
            prefill_ffn_up = allocate(prefill_capacity * kFfnInner * sizeof(float));
            prefill_ffn_activation = allocate(prefill_capacity * kFfnInner * sizeof(float));
            prefill_projected = allocate(prefill_capacity * kHidden * sizeof(float));
            if (prefill_recurrent_batch) {
                prefill_core_query = allocate(prefill_capacity * kKHeads * kState * sizeof(float));
                prefill_core_key = allocate(prefill_capacity * kKHeads * kState * sizeof(float));
                prefill_core_value = allocate(prefill_capacity * kVHeads * kState * sizeof(float));
                prefill_core_beta = allocate(prefill_capacity * kVHeads * sizeof(float));
                prefill_core_decay = allocate(prefill_capacity * kVHeads * sizeof(float));
                prefill_core_gate = allocate(prefill_capacity * kVHeads * kState * sizeof(float));
            }
        }

        MIINFER_HIP_CHECK(hipMemset(history->get(), 0, 4 * kChannels * sizeof(float)));
        if (!fixture.empty() && std::filesystem::exists(fixture / "tensors")) {
            const auto initial = read_f32(
                checkpoint(fixture, 0, "state_predelta-" + std::to_string(index)),
                kVHeads * kState * kState);
            upload_state(initial);
        } else {
            MIINFER_HIP_CHECK(hipMemset(state->get(), 0, kVHeads * kState * kState * sizeof(float)));
        }
    }


    std::string name(const char* suffix) const {
        return "blk." + std::to_string(index) + "." + suffix;
    }

    void poison() {
        MIINFER_HIP_CHECK(hipMemset(state->get(), 0xA5,
                                    kVHeads * kState * kState * sizeof(float)));
        MIINFER_HIP_CHECK(hipMemset(history->get(), 0xFF,
                                    4 * kChannels * sizeof(float)));
    }

    void reset(const std::filesystem::path& fixture = {}) {
        MIINFER_HIP_CHECK(hipMemset(history->get(), 0, 4 * kChannels * sizeof(float)));
        if (!fixture.empty() && std::filesystem::exists(fixture / "tensors")) {
            const auto initial = read_f32(
                checkpoint(fixture, 0, "state_predelta-" + std::to_string(index)),
                kVHeads * kState * kState);
            upload_state(initial);
        } else {
            MIINFER_HIP_CHECK(hipMemset(state->get(), 0, kVHeads * kState * kState * sizeof(float)));
        }
    }

    void upload_state(std::span<const float> logical) const {
        if (!transposed_state) {
            upload(logical.data(), state->get(), logical.size() * sizeof(float));
            return;
        }
        const std::size_t matrix = kState * kState;
        std::vector<float> physical(logical.size());
        for (std::size_t head = 0; head < kVHeads; ++head) {
            for (std::size_t row = 0; row < kState; ++row) {
                for (std::size_t column = 0; column < kState; ++column) {
                    physical[head * matrix + column * kState + row] =
                        logical[head * matrix + row * kState + column];
                }
            }
        }
        upload(physical.data(), state->get(), physical.size() * sizeof(float));
    }

    std::vector<float> logical_state() const {
        const std::size_t elements = kVHeads * kState * kState;
        const auto physical = download(state->get(), elements);
        if (!transposed_state) return physical;
        const std::size_t matrix = kState * kState;
        std::vector<float> logical(elements);
        for (std::size_t head = 0; head < kVHeads; ++head) {
            for (std::size_t row = 0; row < kState; ++row) {
                for (std::size_t column = 0; column < kState; ++column) {
                    logical[head * matrix + row * kState + column] =
                        physical[head * matrix + column * kState + row];
                }
            }
        }
        return logical;
    }

    std::uint64_t state_fingerprint() const {
        const auto logical = logical_state();
        return host_fingerprint(std::as_bytes(std::span<const float>(logical)));
    }

    void trace_tensor(std::uint32_t position, const char* label, const float* device,
                      std::size_t elements, const std::string& reference_name) const {
        if (trace != nullptr && trace->layer == index && trace->position == position) {
            trace->report(label, device, elements, reference_name);
        }
    }

    UpdateProvenance sample_update(std::size_t flat_index) const {
        constexpr std::size_t state_size = kState;
        UpdateProvenance result;
        result.head = flat_index / (state_size * state_size);
        result.row = (flat_index / state_size) % state_size;
        result.column = flat_index % state_size;
        const std::size_t state_base =
            (result.head * state_size + result.row) * state_size;
        const std::size_t key_base = (result.head % kKHeads) * state_size;
        std::array<float, state_size> previous{};
        std::array<float, state_size> key_values{};
        std::array<float, state_size> query_values{};
        if (transposed_state) {
            const auto logical = logical_state();
            std::copy_n(logical.data() + state_base, state_size, previous.data());
        } else {
            MIINFER_HIP_CHECK(hipMemcpy(previous.data(),
                                        static_cast<const float*>(state->get()) + state_base,
                                        sizeof(previous), hipMemcpyDeviceToHost));
        }
        MIINFER_HIP_CHECK(hipMemcpy(key_values.data(),
                                    static_cast<const float*>(key_norm->get()) + key_base,
                                    sizeof(key_values), hipMemcpyDeviceToHost));
        MIINFER_HIP_CHECK(hipMemcpy(query_values.data(),
                                    static_cast<const float*>(query_norm->get()) + key_base,
                                    sizeof(query_values), hipMemcpyDeviceToHost));
        MIINFER_HIP_CHECK(hipMemcpy(&result.decay,
                                    static_cast<const float*>(decay->get()) + result.head,
                                    sizeof(float), hipMemcpyDeviceToHost));
        MIINFER_HIP_CHECK(hipMemcpy(&result.beta,
                                    static_cast<const float*>(beta->get()) + result.head,
                                    sizeof(float), hipMemcpyDeviceToHost));
        MIINFER_HIP_CHECK(hipMemcpy(&result.value,
                                    static_cast<const float*>(value->get())
                                        + result.head * state_size + result.row,
                                    sizeof(float), hipMemcpyDeviceToHost));
        result.previous = previous[result.column];
        result.key = key_values[result.column];
        result.query = query_values[result.column];
        for (std::size_t column = 0; column < state_size; ++column) {
            const float decayed = previous[column] * result.decay;
            result.key_dot += decayed * key_values[column];
        }
        result.decayed = result.previous * result.decay;
        result.delta = (result.value - result.key_dot) * result.beta;
        result.candidate = result.decayed + result.delta * result.key;
        for (std::size_t column = 0; column < state_size; ++column) {
            result.query_dot +=
                (previous[column] * result.decay + result.delta * key_values[column])
                * query_values[column];
        }
        return result;
    }

    void capture_operands(RecurrentOperands& captured) const {
        captured.previous = logical_state();
        captured.query = download(query_norm->get(), kKHeads * kState);
        captured.key = download(key_norm->get(), kKHeads * kState);
        captured.value = download(value->get(), kVHeads * kState);
        captured.beta = download(beta->get(), kVHeads);
        captured.decay = download(decay->get(), kVHeads);
    }

    float state_value(std::size_t flat_index) const {
        if (transposed_state) {
            const std::size_t matrix = kState * kState;
            const std::size_t head = flat_index / matrix;
            const std::size_t in_head = flat_index % matrix;
            const std::size_t row = in_head / kState;
            const std::size_t column = in_head % kState;
            float result = 0.0F;
            const std::size_t physical = head * matrix + column * kState + row;
            MIINFER_HIP_CHECK(hipMemcpy(
                &result, static_cast<const float*>(state->get()) + physical,
                sizeof(float), hipMemcpyDeviceToHost));
            return result;
        }
        float result = 0.0F;
        MIINFER_HIP_CHECK(hipMemcpy(
            &result, static_cast<const float*>(state->get()) + flat_index,
            sizeof(float), hipMemcpyDeviceToHost));
        return result;
    }

    void stage_start(std::size_t stage, std::uint32_t position) const {
        if (stage_profile != nullptr && stage_profile_position == position) {
            MIINFER_HIP_CHECK(hipEventRecord(stage_profile->start[stage], hipStreamPerThread));
        }
    }

    void stage_end(std::size_t stage, std::uint32_t position) const {
        if (stage_profile != nullptr && stage_profile_position == position) {
            stage_profile->stage_recorded[stage] = true;
            MIINFER_HIP_CHECK(hipEventRecord(stage_profile->end[stage], hipStreamPerThread));
        }
    }

    void profile_tail_start(std::size_t chunk_base) const {
        if (stage_profile != nullptr && stage_profile_chunk_base == chunk_base) {
            stage_profile->tail_recorded = true;
            MIINFER_HIP_CHECK(hipEventRecord(stage_profile->tail_start, hipStreamPerThread));
        }
    }

    void profile_tail_end(std::size_t chunk_base) const {
        if (stage_profile != nullptr && stage_profile_chunk_base == chunk_base) {
            MIINFER_HIP_CHECK(hipEventRecord(stage_profile->tail_end, hipStreamPerThread));
        }
    }

    void profile_prepare_start(std::size_t chunk_base) const {
        if (stage_profile != nullptr && stage_profile_chunk_base == chunk_base) {
            stage_profile->prepare_recorded = true;
            MIINFER_HIP_CHECK(hipEventRecord(stage_profile->prepare_start, hipStreamPerThread));
        }
    }

    void profile_prepare_end(std::size_t chunk_base) const {
        if (stage_profile != nullptr && stage_profile_chunk_base == chunk_base) {
            MIINFER_HIP_CHECK(hipEventRecord(stage_profile->prepare_end, hipStreamPerThread));
        }
    }

    void profile_ordered_start(std::size_t chunk_base) const {
        if (stage_profile != nullptr && stage_profile_chunk_base == chunk_base) {
            stage_profile->ordered_recorded = true;
            MIINFER_HIP_CHECK(hipEventRecord(stage_profile->ordered_start, hipStreamPerThread));
        }
    }

    void profile_ordered_end(std::size_t chunk_base) const {
        if (stage_profile != nullptr && stage_profile_chunk_base == chunk_base) {
            MIINFER_HIP_CHECK(hipEventRecord(stage_profile->ordered_end, hipStreamPerThread));
        }
    }

    void profile_tail_family_start(std::size_t family, std::size_t offset) const {
        if (stage_profile != nullptr && family < stage_profile->tail_family_start.size()
            && stage_profile_position == stage_profile_chunk_base + offset + 3) {
            MIINFER_HIP_CHECK(hipEventRecord(stage_profile->tail_family_start[family], hipStreamPerThread));
        }
    }

    void profile_tail_family_end(std::size_t family, std::size_t offset) const {
        if (stage_profile != nullptr && family < stage_profile->tail_family_end.size()
            && stage_profile_position == stage_profile_chunk_base + offset + 3) {
            stage_profile->tail_family_recorded[family] = true;
            MIINFER_HIP_CHECK(hipEventRecord(stage_profile->tail_family_end[family], hipStreamPerThread));
        }
    }

    bool prepare_prefill_batch(const float* inputs, std::size_t count, bool normalized_ready = false) {
        if (!prefill_batch_enabled || count == 0 || count > prefill_capacity || count % 4 != 0 || inputs == nullptr
            || (!d_qkv_gate_combined && !d_qkv_native)) {
            return false;
        }
        auto* normalized_out = static_cast<float*>(prefill_normalized->get());
        auto* q8_out = static_cast<miinfer::Q8_1Block*>(prefill_q8_1->get());
        if (dense_qkv_prefill && m12_dense_workspace_ready && d_qkv && d_gate
            && !d_qkv_gate_combined && m12_dense_input != nullptr
            && m12_dense_weights != nullptr && m12_dense_handle != nullptr) {
            for (std::size_t i = 0; i < count; ++i) {
                if (!normalized_ready) {
                    miinfer::launch_qwen3_rms_norm(
                        inputs + i * kHidden, static_cast<const float*>(d_attn_norm->get()),
                        normalized_out + i * kHidden, kHidden, model.config().rms_epsilon);
                }
            }
            miinfer::launch_m12_f32_to_fp16(
                normalized_out, m12_dense_input, count * kHidden, hipStreamPerThread);
            if (qkv_weight.type == miinfer::GgufTensorType::q4_k) {
                miinfer::launch_m12_q4k_to_fp16(
                    static_cast<const miinfer::Q4KDeviceBlock*>(d_qkv->get()),
                    m12_dense_weights, kChannels, kHidden, hipStreamPerThread);
            } else {
                miinfer::launch_m12_q6k_to_fp16(
                    static_cast<const miinfer::Q6KDeviceBlock*>(d_qkv->get()),
                    m12_dense_weights, kChannels, kHidden, hipStreamPerThread);
            }
            std::string error;
            if (!miinfer::launch_rocblas_gemm_fp16_batch(
                    *m12_dense_handle, m12_dense_weights, m12_dense_input,
                    static_cast<float*>(prefill_qkv->get()), kChannels, kHidden,
                    static_cast<int>(count), error)) {
                throw std::runtime_error(error);
            }
            miinfer::launch_m12_q4k_to_fp16(
                static_cast<const miinfer::Q4KDeviceBlock*>(d_gate->get()),
                m12_dense_weights, kInner, kHidden, hipStreamPerThread);
            if (!miinfer::launch_rocblas_gemm_fp16_batch(
                    *m12_dense_handle, m12_dense_weights, m12_dense_input,
                    static_cast<float*>(prefill_gate->get()), kInner, kHidden,
                    static_cast<int>(count), error)) {
                throw std::runtime_error(error);
            }
            return true;
        }
        for (std::size_t i = 0; i < count; ++i) {
            if (!normalized_ready) {
                miinfer::launch_qwen3_rms_norm(
                    inputs + i * kHidden, static_cast<const float*>(d_attn_norm->get()),
                    normalized_out + i * kHidden, kHidden, model.config().rms_epsilon);
            }
            miinfer::launch_q8_1_quantize_f32(
                normalized_out + i * kHidden, q8_out + i * (kHidden / miinfer::kQ8_1BlockSize),
                kHidden, hipStreamPerThread);
        }

        const std::size_t qkv_rows = d_qkv_gate_combined ? kChannels + kInner : kChannels;
        for (std::size_t base = 0; base < count; base += 4) {
            auto* group_q8 = q8_out + base * (kHidden / miinfer::kQ8_1BlockSize);
            auto* group_qkv = static_cast<float*>(prefill_qkv->get()) + base * qkv_rows;
            if (d_qkv_gate_combined) {
                launch_q4k_wave_gemv_batched4(
                    static_cast<const Q4KWaveTile*>(d_qkv_gate_combined->get()), group_q8,
                    group_qkv, qkv_rows, kHidden, hipStreamPerThread);
            } else if (qkv_weight.type == miinfer::GgufTensorType::q4_k) {
                launch_q4k_wave_gemv_batched4(
                    static_cast<const Q4KWaveTile*>(d_qkv_native->get()), group_q8,
                    group_qkv, qkv_rows, kHidden, hipStreamPerThread);
            } else {
                launch_q6k_wave_gemv_batched4(
                    static_cast<const Q6KWaveTile*>(d_qkv_native->get()), group_q8,
                    group_qkv, qkv_rows, kHidden, hipStreamPerThread);
            }
            if (d_attn_gate_native) {
                launch_q4k_wave_gemv_batched4(
                    static_cast<const Q4KWaveTile*>(d_attn_gate_native->get()), group_q8,
                    static_cast<float*>(prefill_gate->get()) + base * kInner, kInner, kHidden,
                    hipStreamPerThread);
            }
        }
        return true;
    }

    const float* prefill_normalized_at(std::size_t index) const {
        return static_cast<const float*>(prefill_normalized->get()) + index * kHidden;
    }

    const float* prefill_qkv_at(std::size_t index) const {
        const std::size_t stride = d_qkv_gate_combined ? kChannels + kInner : kChannels;
        return static_cast<const float*>(prefill_qkv->get()) + index * stride;
    }

    const float* prefill_gate_at(std::size_t index) const {
        return d_qkv_gate_combined ? nullptr
            : static_cast<const float*>(prefill_gate->get()) + index * kInner;
    }

    bool prefill_tail_batch_supported() const {
        return prefill_batch_enabled
            && (d_ffn_down_native || (prefill_dense_ffn_down && m12_dense_workspace_ready))
            && (d_ffn_swiglu_native || (d_ffn_gate_native && d_ffn_up_native));
    }

    bool recurrent_prefill_core_batch_supported() const noexcept {
        return prefill_recurrent_batch && trace == nullptr && provenance == nullptr
            && operand_capture == nullptr && key_path_capture == nullptr
            && layer_path_capture == nullptr && output_projection_path_capture == nullptr
            && gate_path_capture == nullptr;
    }

    void set_m12_gdn_workspace(const miinfer::M12GdnChunkWorkspace& workspace,
                                float* raw_output) {
        m12_gdn_workspace = workspace;
        m12_gdn_workspace_ready = true;
        m12_gdn_raw_output = raw_output;
    }

    void set_m12_dense_workspace(
        __half* weights, __half* input,
        miinfer::RocblasGemmHandle* handle) {
        m12_dense_weights = weights;
        m12_dense_input = input;
        m12_dense_handle = handle;
        m12_dense_workspace_ready = true;
    }

    void set_m12_dense_source(std::byte* source) { m12_dense_source = source; }

    void ensure_m23_repacked() {
        if (!prefill_wide_repacked || (d_qkv_mmq && d_gate_mmq && d_ssm_out_mmq
                                       && d_ffn_gate_mmq && d_ffn_up_mmq && d_ffn_down_mmq)) return;
        const auto upload_packed = [this](const auto& packed, std::size_t slot) -> Buffer {
            const std::size_t bytes = packed.size() * sizeof(packed.front());
            if (!g_m23_repacked_scratch[slot] || g_m23_repacked_scratch_bytes[slot] < bytes) {
                g_m23_repacked_scratch[slot] = allocate(bytes);
                g_m23_repacked_scratch_bytes[slot] = bytes;
            }
            const auto upload_start = std::chrono::steady_clock::now();
            upload(packed.data(), g_m23_repacked_scratch[slot]->get(), bytes);
            const double upload_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - upload_start).count();
            if (m23_profile_counters != nullptr) {
                m23_profile_counters->weight_upload_bytes += bytes;
                m23_profile_counters->recurrent_weight_upload_bytes[slot] += bytes;
                m23_profile_counters->weight_upload_ms += upload_ms;
                m23_profile_counters->recurrent_weight_upload_ms[slot] += upload_ms;
            }
            return g_m23_repacked_scratch[slot];
        };
        if (!d_qkv_mmq) {
            d_qkv_mmq = resident_m23_all ? resident_qkv_mmq
                : (!host_qkv_q6_mmq.empty() ? upload_packed(host_qkv_q6_mmq, 0)
                                             : upload_packed(host_qkv_q4_mmq, 0));
        }
        if (!d_gate_mmq) d_gate_mmq = resident_m23_all
            ? resident_gate_mmq : upload_packed(host_gate_mmq, 1);
        if (!d_ssm_out_mmq) d_ssm_out_mmq = resident_m23_all
            ? resident_ssm_out_mmq : upload_packed(host_ssm_out_mmq, 2);
        if (!d_ffn_gate_mmq) {
            d_ffn_gate_mmq = resident_m23_ffn ? resident_ffn_gate_mmq
                                              : upload_packed(host_ffn_gate_mmq, 3);
        }
        if (!d_ffn_up_mmq) {
            d_ffn_up_mmq = resident_m23_ffn ? resident_ffn_up_mmq
                                            : upload_packed(host_ffn_up_mmq, 4);
        }
        if (!d_ffn_down_mmq) {
            d_ffn_down_mmq = resident_m23_ffn ? resident_ffn_down_mmq
                                              : (!host_ffn_down_q6_mmq.empty()
                                                     ? upload_packed(host_ffn_down_q6_mmq, 5)
                                                     : upload_packed(host_ffn_down_q4_mmq, 5));
        }
    }

    void release_m23_repacked() {
        if (!resident_m23_all) {
            d_qkv_mmq.reset();
            d_gate_mmq.reset();
            d_ssm_out_mmq.reset();
        }
        if (!resident_m23_ffn) {
            d_ffn_gate_mmq.reset();
            d_ffn_up_mmq.reset();
            d_ffn_down_mmq.reset();
        }
    }

    // M23 causal core consumes token-major projections at the caller's batch
    // width; only the mathematically required 64-token GDN scan launches are
    // ordered.
    const float* prefill_wide_causal(const float* projected_qkv,
                                     const float* projected_gate,
                                     const float* projected_beta,
                                     const float* projected_decay,
                                     std::uint32_t base_position,
                                     std::uint32_t token_count) {
        if (projected_qkv == nullptr || projected_gate == nullptr || projected_beta == nullptr
            || projected_decay == nullptr || token_count < kM12PrefillBatch
            || token_count > prefill_capacity || token_count % kPrefillBatch != 0
            || !m12_gdn_workspace_ready || m12_gdn_raw_output == nullptr
            || !prefill_core_query || !prefill_core_key || !prefill_core_value
            || !prefill_gated) {
            throw std::runtime_error("invalid M23 wide recurrent causal inputs");
        }
        const auto profile_position = base_position + token_count - 1;
        stage_start(4, profile_position);
        auto* query_batch = static_cast<float*>(prefill_core_query->get());
        auto* key_batch = static_cast<float*>(prefill_core_key->get());
        auto* value_batch = static_cast<float*>(prefill_core_value->get());
        miinfer::launch_qwen35_conv_silu_split_batch(
            projected_qkv, static_cast<const float*>(d_conv->get()),
            static_cast<float*>(history->get()), query_batch, key_batch, value_batch,
            base_position, token_count, 4, kChannels, 4, hipStreamPerThread);
        miinfer::launch_qwen35_dual_head_l2_normalize_batch(
            query_batch, key_batch, query_batch, key_batch, token_count, kKHeads, kState,
            hipStreamPerThread);
        stage_end(4, profile_position);
        stage_start(5, profile_position);
        const char* direct_env = std::getenv("MIINFER_PREFILL_GDN_DIRECT");
        const bool direct = direct_env != nullptr && std::strcmp(direct_env, "0") != 0;
        if (direct) {
            miinfer::launch_m12_gdn_direct(
                query_batch, key_batch, value_batch, projected_beta, projected_decay,
                static_cast<float*>(state->get()), m12_gdn_raw_output,
                token_count, kKHeads, kVHeads, kState, hipStreamPerThread);
        } else {
            for (std::uint32_t start = 0; start < token_count; start += kM23CausalChunk) {
                miinfer::launch_m12_gdn_chunk(
                    query_batch, key_batch, value_batch, projected_beta, projected_decay,
                    static_cast<float*>(state->get()), m12_gdn_raw_output, m12_gdn_workspace,
                    token_count, start, kKHeads, kVHeads, kState,
                    kM23CausalChunk, hipStreamPerThread);
            }
        }
        stage_end(5, profile_position);
        stage_start(6, profile_position);
        miinfer::launch_m12_gdn_postprocess(
            m12_gdn_raw_output, projected_gate, static_cast<const float*>(d_ssm_norm->get()),
            static_cast<float*>(prefill_gated->get()), token_count, kVHeads, kState,
            model.config().rms_epsilon, hipStreamPerThread);
        stage_end(6, profile_position);
        return static_cast<const float*>(prefill_gated->get());
    }

    const float* prefill_wide_beta_decay(const float* inputs, std::uint32_t base_position,
                                         std::uint32_t token_count) {
        if (inputs == nullptr || token_count < kM12PrefillBatch || token_count > prefill_capacity
            || token_count % kPrefillBatch != 0 || !prefill_normalized
            || !prefill_core_beta || !prefill_core_decay) {
            throw std::runtime_error("invalid M23 wide beta/decay inputs");
        }
        const auto profile_position = base_position + token_count - 1;
        stage_start(0, profile_position);
        auto* normalized_batch = static_cast<float*>(prefill_normalized->get());
        auto* beta_batch = static_cast<float*>(prefill_core_beta->get());
        auto* decay_batch = static_cast<float*>(prefill_core_decay->get());
        miinfer::launch_qwen3_rms_norm_batch(
            inputs, static_cast<const float*>(d_attn_norm->get()), normalized_batch,
            token_count, kHidden, model.config().rms_epsilon, hipStreamPerThread);
        stage_end(0, profile_position);
        stage_start(3, profile_position);
        miinfer::launch_qwen35_f32_dual_gemm_batch(
            static_cast<const float*>(d_beta->get()), static_cast<const float*>(d_alpha->get()),
            normalized_batch, beta_batch, decay_batch, token_count, kVHeads, kHidden,
            hipStreamPerThread);
        count_m23_dispatch(2);
        miinfer::launch_qwen35_prepare_beta_decay(
            beta_batch, decay_batch, static_cast<const float*>(d_dt->get()),
            static_cast<const float*>(d_a->get()), beta_batch, decay_batch,
            token_count * kVHeads, hipStreamPerThread);
        stage_end(3, profile_position);
        return normalized_batch;
    }

    const float* prefill_wide_qkv_gate(const float* inputs, std::uint32_t base_position,
                                       std::uint32_t token_count) {
        if (!prefill_wide_qkv || token_count < kM12PrefillBatch || token_count > prefill_capacity
            || token_count % kPrefillBatch != 0
            || ((!prefill_wide_repacked) && (wide_qkv_fp16 == nullptr || wide_gate_fp16 == nullptr))
            || m12_dense_input == nullptr || m12_dense_handle == nullptr) {
            throw std::runtime_error("invalid M23 wide QKV/Z inputs");
        }
        const auto profile_position = base_position + token_count - 1;
        const float* normalized_batch = prefill_wide_beta_decay(inputs, base_position, token_count);
        stage_start(1, profile_position);
        if (resident_fp16_prefill && d_qkv_mmq && d_gate_mmq) {
            miinfer::launch_m12_f32_to_fp16(
                normalized_batch, m12_dense_input, token_count * kHidden, hipStreamPerThread);
            std::string error;
            if (qkv_weight.type == miinfer::GgufTensorType::q4_k) {
                launch_m24_q4k_mmq_to_fp16(
                    static_cast<const Q4KMmqTile*>(d_qkv_mmq->get()), m12_dense_weights,
                    kChannels, kHidden, hipStreamPerThread);
            } else {
                launch_m24_q6k_mmq_to_fp16(
                    static_cast<const Q6KMmqTile*>(d_qkv_mmq->get()), m12_dense_weights,
                    kChannels, kHidden, hipStreamPerThread);
            }
            if (!miinfer::launch_rocblas_gemm_fp16_batch(
                    *m12_dense_handle, m12_dense_weights, m12_dense_input,
                    static_cast<float*>(prefill_qkv->get()), kChannels, kHidden,
                    token_count, error)) {
                throw std::runtime_error(error);
            }
            stage_end(1, profile_position);
            stage_start(2, profile_position);
            launch_m24_q4k_mmq_to_fp16(
                static_cast<const Q4KMmqTile*>(d_gate_mmq->get()), m12_dense_weights,
                kInner, kHidden, hipStreamPerThread);
            if (!miinfer::launch_rocblas_gemm_fp16_batch(
                    *m12_dense_handle, m12_dense_weights, m12_dense_input,
                    static_cast<float*>(prefill_gate->get()), kInner, kHidden,
                    token_count, error)) {
                throw std::runtime_error(error);
            }
            stage_end(2, profile_position);
            return static_cast<const float*>(prefill_qkv->get());
        }
        if (wide_dense_all) {
            if (m12_dense_source == nullptr) {
                throw std::runtime_error("wide dense source workspace is missing");
            }
            miinfer::launch_m12_f32_to_fp16(
                normalized_batch, m12_dense_input, token_count * kHidden, hipStreamPerThread);
            std::string error;
            upload(qkv_weight.data, m12_dense_source, qkv_weight.byte_size);
            if (qkv_weight.type == miinfer::GgufTensorType::q4_k) {
                miinfer::launch_m12_q4k_to_fp16(
                    reinterpret_cast<const miinfer::Q4KDeviceBlock*>(m12_dense_source), m12_dense_weights,
                    kChannels, kHidden, hipStreamPerThread);
            } else {
                miinfer::launch_m12_q6k_to_fp16(
                    reinterpret_cast<const miinfer::Q6KDeviceBlock*>(m12_dense_source), m12_dense_weights,
                    kChannels, kHidden, hipStreamPerThread);
            }
            if (!miinfer::launch_rocblas_gemm_fp16_batch(
                    *m12_dense_handle, m12_dense_weights,
                    m12_dense_input, static_cast<float*>(prefill_qkv->get()),
                    kChannels, kHidden, token_count, error)) {
                throw std::runtime_error(error);
            }
            stage_end(1, profile_position);
            stage_start(2, profile_position);
            upload(gate_weight.data, m12_dense_source, gate_weight.byte_size);
            miinfer::launch_m12_q4k_to_fp16(
                reinterpret_cast<const miinfer::Q4KDeviceBlock*>(m12_dense_source), m12_dense_weights,
                kInner, kHidden, hipStreamPerThread);
            if (!miinfer::launch_rocblas_gemm_fp16_batch(
                    *m12_dense_handle, m12_dense_weights,
                    m12_dense_input, static_cast<float*>(prefill_gate->get()),
                    kInner, kHidden, token_count, error)) {
                throw std::runtime_error(error);
            }
            stage_end(2, profile_position);
            return static_cast<const float*>(prefill_qkv->get());
        }
        const char* mmq_env = std::getenv("MIINFER_PREFILL_WIDE_MMQ_QKV");
        if (mmq_env != nullptr && std::strcmp(mmq_env, "0") != 0
            && (d_qkv || d_qkv_mmq) && (d_gate || d_gate_mmq) && prefill_mmq_q8) {
            auto* q8 = static_cast<miinfer::M23Q8_1MmqBlock*>(prefill_mmq_q8->get());
            const char* repacked_env = std::getenv("MIINFER_PREFILL_WIDE_REPACKED_MMQ");
            const char* qkv_repacked_env = std::getenv("MIINFER_PREFILL_WIDE_REPACKED_QKV");
            const bool repacked = qkv_repacked_env != nullptr
                ? std::strcmp(qkv_repacked_env, "0") != 0
                : (repacked_env != nullptr && std::strcmp(repacked_env, "0") != 0);
            miinfer::launch_m23_q8_1_mmq_quantize(
                normalized_batch, q8, token_count, kHidden, hipStreamPerThread);
            if (qkv_weight.type == miinfer::GgufTensorType::q4_k) {
                if (repacked && d_qkv_mmq) {
                    launch_m23_q4k_repacked_mmq(
                        static_cast<const Q4KMmqTile*>(d_qkv_mmq->get()), q8,
                        static_cast<float*>(prefill_qkv->get()), kChannels, kHidden, token_count,
                        hipStreamPerThread);
                } else {
                    miinfer::launch_m23_q4_k_q8_1_mmq(
                        static_cast<const miinfer::Q4KDeviceBlock*>(d_qkv->get()), q8,
                        static_cast<float*>(prefill_qkv->get()), kChannels, kHidden, token_count,
                        hipStreamPerThread);
                }
            } else {
                const char* repacked_env = std::getenv("MIINFER_PREFILL_WIDE_REPACKED_QKV");
                const char* all_repacked_env = std::getenv("MIINFER_PREFILL_WIDE_REPACKED_MMQ");
                if (((repacked_env != nullptr && std::strcmp(repacked_env, "0") != 0)
                     || (all_repacked_env != nullptr && std::strcmp(all_repacked_env, "0") != 0))
                    && d_qkv_mmq) {
                    launch_m23_q6k_repacked_mmq(
                        static_cast<const Q6KMmqTile*>(d_qkv_mmq->get()), q8,
                        static_cast<float*>(prefill_qkv->get()), kChannels, kHidden, token_count,
                        hipStreamPerThread);
                } else {
                    miinfer::launch_m23_q6_k_q8_1_mmq(
                        static_cast<const miinfer::Q6KDeviceBlock*>(d_qkv->get()), q8,
                        static_cast<float*>(prefill_qkv->get()), kChannels, kHidden, token_count,
                        hipStreamPerThread);
                }
            }
            count_m23_dispatch(0);
            stage_end(1, profile_position);
            stage_start(2, profile_position);
            if (repacked && d_gate_mmq) {
                launch_m23_q4k_repacked_mmq(
                    static_cast<const Q4KMmqTile*>(d_gate_mmq->get()), q8,
                    static_cast<float*>(prefill_gate->get()), kInner, kHidden, token_count,
                    hipStreamPerThread);
            } else {
                miinfer::launch_m23_q4_k_q8_1_mmq(
                    static_cast<const miinfer::Q4KDeviceBlock*>(d_gate->get()), q8,
                    static_cast<float*>(prefill_gate->get()), kInner, kHidden, token_count,
                    hipStreamPerThread);
            }
            count_m23_dispatch(1);
            stage_end(2, profile_position);
            return static_cast<const float*>(prefill_qkv->get());
        }
        miinfer::launch_m12_f32_to_fp16(
            normalized_batch, m12_dense_input, token_count * kHidden, hipStreamPerThread);
        std::string error;
        if (!miinfer::launch_rocblas_gemm_fp16_batch(
                *m12_dense_handle, static_cast<const __half*>(wide_qkv_fp16->get()), m12_dense_input,
                static_cast<float*>(prefill_qkv->get()), kChannels, kHidden, token_count, error)) {
            throw std::runtime_error(error);
        }
        stage_end(1, profile_position);
        stage_start(2, profile_position);
        if (!miinfer::launch_rocblas_gemm_fp16_batch(
                *m12_dense_handle, static_cast<const __half*>(wide_gate_fp16->get()), m12_dense_input,
                static_cast<float*>(prefill_gate->get()), kInner, kHidden, token_count, error)) {
            throw std::runtime_error(error);
        }
        stage_end(2, profile_position);
        return static_cast<const float*>(prefill_qkv->get());
    }

    const float* prefill_wide_ssm_out(const float* gated_batch, std::uint32_t base_position,
                                      std::uint32_t token_count) {
        if (!prefill_wide_qkv || gated_batch == nullptr || token_count < kM12PrefillBatch
            || token_count > prefill_capacity || token_count % kPrefillBatch != 0
            || ((!prefill_wide_repacked) && wide_ssm_out_fp16 == nullptr)
            || m12_dense_input == nullptr || m12_dense_handle == nullptr) {
            throw std::runtime_error("invalid M23 wide SSM-out inputs");
        }
        const auto profile_position = base_position + token_count - 1;
        stage_start(7, profile_position);
        if (resident_fp16_prefill && d_ssm_out_mmq) {
            miinfer::launch_m12_f32_to_fp16(
                gated_batch, m12_dense_input, token_count * kInner, hipStreamPerThread);
            launch_m24_q5k_mmq_to_fp16(
                static_cast<const Q5KMmqTile*>(d_ssm_out_mmq->get()), m12_dense_weights,
                kHidden, kInner, hipStreamPerThread);
            std::string error;
            if (!miinfer::launch_rocblas_gemm_fp16_batch(
                    *m12_dense_handle, m12_dense_weights, m12_dense_input,
                    static_cast<float*>(prefill_projected->get()), kHidden, kInner,
                    token_count, error)) {
                throw std::runtime_error(error);
            }
            stage_end(7, profile_position);
            return static_cast<const float*>(prefill_projected->get());
        }
        if (wide_dense_all) {
            if (m12_dense_source == nullptr) {
                throw std::runtime_error("wide dense source workspace is missing");
            }
            miinfer::launch_m12_f32_to_fp16(
                gated_batch, m12_dense_input, token_count * kInner, hipStreamPerThread);
            std::string error;
            upload(ssm_out_weight.data, m12_dense_source, ssm_out_weight.byte_size);
            miinfer::launch_m12_q5k_to_fp16(
                reinterpret_cast<const miinfer::Q5KDeviceBlock*>(m12_dense_source), m12_dense_weights,
                kHidden, kInner, hipStreamPerThread);
            if (!miinfer::launch_rocblas_gemm_fp16_batch(
                    *m12_dense_handle, m12_dense_weights,
                    m12_dense_input, static_cast<float*>(prefill_projected->get()),
                    kHidden, kInner, token_count, error)) {
                throw std::runtime_error(error);
            }
            stage_end(7, profile_position);
            return static_cast<const float*>(prefill_projected->get());
        }
        const char* mmq_env = std::getenv("MIINFER_PREFILL_WIDE_MMQ_SSM_OUT");
        if (mmq_env != nullptr && std::strcmp(mmq_env, "0") != 0
            && (d_ssm_out || d_ssm_out_mmq) && prefill_mmq_q8) {
            auto* q8 = static_cast<miinfer::M23Q8_1MmqBlock*>(prefill_mmq_q8->get());
            const char* repacked_env = std::getenv("MIINFER_PREFILL_WIDE_REPACKED_MMQ");
            const bool repacked = repacked_env != nullptr && std::strcmp(repacked_env, "0") != 0;
            miinfer::launch_m23_q8_1_mmq_quantize(
                gated_batch, q8, token_count, kInner, hipStreamPerThread);
            if (repacked && d_ssm_out_mmq) {
                launch_m23_q5k_repacked_mmq(
                    static_cast<const Q5KMmqTile*>(d_ssm_out_mmq->get()), q8,
                    static_cast<float*>(prefill_projected->get()), kHidden, kInner, token_count,
                    hipStreamPerThread);
            } else {
                miinfer::launch_m23_q5_k_q8_1_mmq(
                    static_cast<const miinfer::Q5KDeviceBlock*>(d_ssm_out->get()), q8,
                    static_cast<float*>(prefill_projected->get()), kHidden, kInner, token_count,
                    hipStreamPerThread);
            }
            count_m23_dispatch(3);
            stage_end(7, profile_position);
            return static_cast<const float*>(prefill_projected->get());
        }
        miinfer::launch_m12_f32_to_fp16(
            gated_batch, m12_dense_input, token_count * kInner, hipStreamPerThread);
        std::string error;
        if (!miinfer::launch_rocblas_gemm_fp16_batch(
                *m12_dense_handle, static_cast<const __half*>(wide_ssm_out_fp16->get()), m12_dense_input,
                static_cast<float*>(prefill_projected->get()), kHidden, kInner, token_count, error)) {
            throw std::runtime_error(error);
        }
        stage_end(7, profile_position);
        return static_cast<const float*>(prefill_projected->get());
    }

    void prefill_wide(const float* inputs, float* outputs, std::uint32_t base_position,
                      std::uint32_t token_count) {
        ensure_m23_repacked();
        if (!prefill_wide_qkv || inputs == nullptr || outputs == nullptr || token_count < kM12PrefillBatch
            || token_count > prefill_capacity || token_count % kPrefillBatch != 0
            || ((!prefill_wide_repacked) && (!wide_ffn_gate_fp16 || !wide_ffn_up_fp16 || !wide_ffn_down_fp16))
            || m12_dense_input == nullptr || m12_dense_handle == nullptr) {
            throw std::runtime_error("invalid M23 wide prefill inputs");
        }
        const bool validate = prefill_wide_validate;
        if (m23_trace_dispatch) m23_dispatch_counts.fill(0);
        const auto trace_dispatch = [this] {
            if (!m23_trace_dispatch) return;
            std::cerr << "M23 dispatches: qkv=" << m23_dispatch_counts[0]
                      << " gate=" << m23_dispatch_counts[1]
                      << " beta_alpha=" << m23_dispatch_counts[2]
                      << " ssm_out=" << m23_dispatch_counts[3]
                      << " ffn_gate=" << m23_dispatch_counts[4]
                      << " ffn_up=" << m23_dispatch_counts[5]
                      << " ffn_down=" << m23_dispatch_counts[6] << '\n';
        };
        std::vector<float> initial_state, initial_history;
        if (validate) {
            initial_state = download(state->get(), kVHeads * kState * kState);
            initial_history = download(history->get(), 4 * kChannels);
        }
        const auto profile_position = base_position + token_count - 1;
        const float* qkv_batch = prefill_wide_qkv_gate(inputs, base_position, token_count);
        const float* gated_batch = prefill_wide_causal(
            qkv_batch, static_cast<const float*>(prefill_gate->get()),
            static_cast<const float*>(prefill_core_beta->get()),
            static_cast<const float*>(prefill_core_decay->get()), base_position, token_count);
        const float* ssm_out_batch = prefill_wide_ssm_out(gated_batch, base_position, token_count);
        auto* residual_batch = static_cast<float*>(prefill_residual->get());
        auto* post_normalized_batch = static_cast<float*>(prefill_post_normalized->get());
        stage_start(8, profile_position);
        miinfer::launch_qwen3_fused_add_rms_norm_batch(
            inputs, ssm_out_batch, static_cast<const float*>(d_post_norm->get()), residual_batch,
            post_normalized_batch, token_count, kHidden, model.config().rms_epsilon, hipStreamPerThread);
        stage_end(8, profile_position);
        std::string error;
        const char* ffm_env = std::getenv("MIINFER_PREFILL_WIDE_MMQ_FFN");
        const bool mmq_ffn = ffm_env != nullptr && std::strcmp(ffm_env, "0") != 0
            && (d_ffn_gate_mmq || d_ffn_gate) && (d_ffn_up_mmq || d_ffn_up)
            && (d_ffn_down_mmq || d_ffn_down) && prefill_mmq_q8
            && !wide_dense_ffn
            && ffn_gate_weight.type == miinfer::GgufTensorType::q4_k
            && ffn_up_weight.type == miinfer::GgufTensorType::q4_k;
        const char* repacked_env = std::getenv("MIINFER_PREFILL_WIDE_REPACKED_MMQ");
        const bool repacked_ffn = repacked_env != nullptr && std::strcmp(repacked_env, "0") != 0;
        const bool resident_fp16_ffn = resident_fp16_prefill && d_ffn_gate_mmq && d_ffn_up_mmq
            && d_ffn_down_mmq && m12_dense_weights != nullptr && m12_dense_input != nullptr
            && m12_dense_handle != nullptr;
        if (resident_fp16_ffn) {
            miinfer::launch_m12_f32_to_fp16(
                post_normalized_batch, m12_dense_input, token_count * kHidden, hipStreamPerThread);
            stage_start(10, profile_position);
            launch_m24_q4k_mmq_to_fp16(
                static_cast<const Q4KMmqTile*>(d_ffn_gate_mmq->get()), m12_dense_weights,
                kFfnInner, kHidden, hipStreamPerThread);
            if (!miinfer::launch_rocblas_gemm_fp16_batch(
                    *m12_dense_handle, m12_dense_weights, m12_dense_input,
                    static_cast<float*>(prefill_ffn_gate->get()), kFfnInner, kHidden,
                    token_count, error)) {
                throw std::runtime_error(error);
            }
            launch_m24_q4k_mmq_to_fp16(
                static_cast<const Q4KMmqTile*>(d_ffn_up_mmq->get()), m12_dense_weights,
                kFfnInner, kHidden, hipStreamPerThread);
            if (!miinfer::launch_rocblas_gemm_fp16_batch(
                    *m12_dense_handle, m12_dense_weights, m12_dense_input,
                    static_cast<float*>(prefill_ffn_up->get()), kFfnInner, kHidden,
                    token_count, error)) {
                throw std::runtime_error(error);
            }
            stage_end(10, profile_position);
        } else if (mmq_ffn) {
            auto* q8 = static_cast<miinfer::M23Q8_1MmqBlock*>(prefill_mmq_q8->get());
            miinfer::launch_m23_q8_1_mmq_quantize(post_normalized_batch, q8, token_count, kHidden,
                                                   hipStreamPerThread);
            stage_start(10, profile_position);
            if (repacked_ffn && d_ffn_gate_mmq && d_ffn_up_mmq) {
                launch_m23_q4k_repacked_mmq(
                    static_cast<const Q4KMmqTile*>(d_ffn_gate_mmq->get()), q8,
                    static_cast<float*>(prefill_ffn_gate->get()), kFfnInner, kHidden, token_count,
                    hipStreamPerThread);
                launch_m23_q4k_repacked_mmq(
                    static_cast<const Q4KMmqTile*>(d_ffn_up_mmq->get()), q8,
                    static_cast<float*>(prefill_ffn_up->get()), kFfnInner, kHidden, token_count,
                    hipStreamPerThread);
            } else {
                miinfer::launch_m23_q4_k_q8_1_mmq(
                    static_cast<const miinfer::Q4KDeviceBlock*>(d_ffn_gate->get()), q8,
                    static_cast<float*>(prefill_ffn_gate->get()), kFfnInner, kHidden, token_count,
                    hipStreamPerThread);
                miinfer::launch_m23_q4_k_q8_1_mmq(
                    static_cast<const miinfer::Q4KDeviceBlock*>(d_ffn_up->get()), q8,
                    static_cast<float*>(prefill_ffn_up->get()), kFfnInner, kHidden, token_count,
                    hipStreamPerThread);
            }
            count_m23_dispatch(4);
            count_m23_dispatch(5);
            stage_end(10, profile_position);
        } else {
            miinfer::launch_m12_f32_to_fp16(
                post_normalized_batch, m12_dense_input, token_count * kHidden, hipStreamPerThread);
            stage_start(10, profile_position);
            const auto launch_dense_ffn_projection = [&](const Buffer& staged,
                                                         const miinfer::GgufTensor& source_tensor,
                                                         std::uint32_t rows, std::uint32_t columns,
                                                         float* output) {
                const __half* weights = nullptr;
                if (wide_dense_ffn) {
                    if (m12_dense_source == nullptr) {
                        throw std::runtime_error("wide dense source workspace is missing");
                    }
                    upload(source_tensor.data, m12_dense_source, source_tensor.byte_size);
                    if (source_tensor.type == miinfer::GgufTensorType::q4_k) {
                        miinfer::launch_m12_q4k_to_fp16(
                            reinterpret_cast<const miinfer::Q4KDeviceBlock*>(m12_dense_source),
                            m12_dense_weights, rows, columns, hipStreamPerThread);
                    } else {
                        miinfer::launch_m12_q6k_to_fp16(
                            reinterpret_cast<const miinfer::Q6KDeviceBlock*>(m12_dense_source),
                            m12_dense_weights, rows, columns, hipStreamPerThread);
                    }
                    weights = m12_dense_weights;
                } else {
                    weights = static_cast<const __half*>(staged->get());
                }
                return miinfer::launch_rocblas_gemm_fp16_batch(
                    *m12_dense_handle, weights, m12_dense_input, output,
                    rows, columns, token_count, error);
            };
            if (!launch_dense_ffn_projection(wide_ffn_gate_fp16, ffn_gate_weight,
                                             kFfnInner, kHidden,
                                             static_cast<float*>(prefill_ffn_gate->get()))) {
                throw std::runtime_error(error);
            }
            if (!launch_dense_ffn_projection(wide_ffn_up_fp16, ffn_up_weight,
                                             kFfnInner, kHidden,
                                             static_cast<float*>(prefill_ffn_up->get()))) {
                throw std::runtime_error(error);
            }
            stage_end(10, profile_position);
        }
        auto* activation = static_cast<float*>(prefill_ffn_activation->get());
        stage_start(11, profile_position);
        miinfer::launch_qwen3_silu_mul(
            static_cast<const float*>(prefill_ffn_gate->get()), static_cast<const float*>(prefill_ffn_up->get()),
            activation, token_count * kFfnInner, hipStreamPerThread);
        stage_end(11, profile_position);
        stage_start(12, profile_position);
        if (resident_fp16_ffn) {
            miinfer::launch_m12_f32_to_fp16(
                activation, m12_dense_input, token_count * kFfnInner, hipStreamPerThread);
            if (ffn_down_weight.type == miinfer::GgufTensorType::q4_k) {
                launch_m24_q4k_mmq_to_fp16(
                    static_cast<const Q4KMmqTile*>(d_ffn_down_mmq->get()), m12_dense_weights,
                    kHidden, kFfnInner, hipStreamPerThread);
            } else {
                launch_m24_q6k_mmq_to_fp16(
                    static_cast<const Q6KMmqTile*>(d_ffn_down_mmq->get()), m12_dense_weights,
                    kHidden, kFfnInner, hipStreamPerThread);
            }
            if (!miinfer::launch_rocblas_gemm_fp16_batch(
                    *m12_dense_handle, m12_dense_weights, m12_dense_input,
                    static_cast<float*>(prefill_projected->get()), kHidden, kFfnInner,
                    token_count, error)) {
                throw std::runtime_error(error);
            }
        } else if (mmq_ffn) {
            auto* q8 = static_cast<miinfer::M23Q8_1MmqBlock*>(prefill_mmq_q8->get());
            miinfer::launch_m23_q8_1_mmq_quantize(activation, q8, token_count, kFfnInner, hipStreamPerThread);
            if (ffn_down_weight.type == miinfer::GgufTensorType::q4_k) {
                if (repacked_ffn && d_ffn_down_mmq) {
                    launch_m23_q4k_repacked_mmq(
                        static_cast<const Q4KMmqTile*>(d_ffn_down_mmq->get()), q8,
                        static_cast<float*>(prefill_projected->get()), kHidden, kFfnInner, token_count,
                        hipStreamPerThread);
                } else {
                    miinfer::launch_m23_q4_k_q8_1_mmq(
                        static_cast<const miinfer::Q4KDeviceBlock*>(d_ffn_down->get()), q8,
                        static_cast<float*>(prefill_projected->get()), kHidden, kFfnInner, token_count,
                        hipStreamPerThread);
                }
            } else {
                if (repacked_ffn && d_ffn_down_mmq) {
                    launch_m23_q6k_repacked_mmq(
                        static_cast<const Q6KMmqTile*>(d_ffn_down_mmq->get()), q8,
                        static_cast<float*>(prefill_projected->get()), kHidden, kFfnInner, token_count,
                        hipStreamPerThread);
                } else {
                    miinfer::launch_m23_q6_k_q8_1_mmq(
                        static_cast<const miinfer::Q6KDeviceBlock*>(d_ffn_down->get()), q8,
                        static_cast<float*>(prefill_projected->get()), kHidden, kFfnInner, token_count,
                    hipStreamPerThread);
                }
            }
            count_m23_dispatch(6);
        } else {
            miinfer::launch_m12_f32_to_fp16(
                activation, m12_dense_input, token_count * kFfnInner, hipStreamPerThread);
            const __half* down_weights = nullptr;
            if (wide_dense_ffn) {
                if (m12_dense_source == nullptr) {
                    throw std::runtime_error("wide dense source workspace is missing");
                }
                upload(ffn_down_weight.data, m12_dense_source, ffn_down_weight.byte_size);
                if (ffn_down_weight.type == miinfer::GgufTensorType::q4_k) {
                    miinfer::launch_m12_q4k_to_fp16(
                        reinterpret_cast<const miinfer::Q4KDeviceBlock*>(m12_dense_source),
                        m12_dense_weights, kHidden, kFfnInner, hipStreamPerThread);
                } else {
                    miinfer::launch_m12_q6k_to_fp16(
                        reinterpret_cast<const miinfer::Q6KDeviceBlock*>(m12_dense_source),
                        m12_dense_weights, kHidden, kFfnInner, hipStreamPerThread);
                }
                down_weights = m12_dense_weights;
            } else {
                down_weights = static_cast<const __half*>(wide_ffn_down_fp16->get());
            }
            if (!miinfer::launch_rocblas_gemm_fp16_batch(
                    *m12_dense_handle, down_weights, m12_dense_input,
                    static_cast<float*>(prefill_projected->get()), kHidden, kFfnInner, token_count, error)) {
                throw std::runtime_error(error);
            }
        }
        stage_end(12, profile_position);
        stage_start(13, profile_position);
        miinfer::launch_qwen3_add(residual_batch, static_cast<const float*>(prefill_projected->get()),
                                  outputs, token_count * kHidden, hipStreamPerThread);
        stage_end(13, profile_position);
        if (!validate) {
            trace_dispatch();
            return;
        }

        MIINFER_HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
        const auto wide_output = download(outputs, token_count * kHidden);
        const auto wide_qkv = download(prefill_qkv->get(), token_count * kChannels);
        const auto wide_gate = download(prefill_gate->get(), token_count * kInner);
        const auto wide_beta = download(prefill_core_beta->get(), token_count * kVHeads);
        const auto wide_decay = download(prefill_core_decay->get(), token_count * kVHeads);
        const auto wide_state = download(state->get(), kVHeads * kState * kState);
        const auto wide_history = download(history->get(), 4 * kChannels);
        upload(initial_state.data(), state->get(), initial_state.size() * sizeof(float));
        upload(initial_history.data(), history->get(), initial_history.size() * sizeof(float));
        auto* canonical_output = static_cast<float*>(prefill_residual->get());
        float qkv_error = 0.0F, gate_error = 0.0F, beta_error = 0.0F, decay_error = 0.0F;
        for (std::uint32_t token = 0; token < token_count; ++token) {
            run(inputs + static_cast<std::size_t>(token) * kHidden, base_position + token,
                canonical_output + static_cast<std::size_t>(token) * kHidden);
            const auto canonical_qkv = download(qkv->get(), kChannels);
            const auto canonical_gate = download(gate->get(), kInner);
            const auto canonical_beta = download(beta->get(), kVHeads);
            const auto canonical_decay = download(decay->get(), kVHeads);
            for (std::size_t i = 0; i < kChannels; ++i) {
                qkv_error = std::max(qkv_error, std::fabs(
                    wide_qkv[static_cast<std::size_t>(token) * kChannels + i] - canonical_qkv[i]));
            }
            for (std::size_t i = 0; i < kInner; ++i) {
                gate_error = std::max(gate_error, std::fabs(
                    wide_gate[static_cast<std::size_t>(token) * kInner + i] - canonical_gate[i]));
            }
            for (std::size_t i = 0; i < kVHeads; ++i) {
                beta_error = std::max(beta_error, std::fabs(
                    wide_beta[static_cast<std::size_t>(token) * kVHeads + i] - canonical_beta[i]));
                decay_error = std::max(decay_error, std::fabs(
                    wide_decay[static_cast<std::size_t>(token) * kVHeads + i] - canonical_decay[i]));
            }
        }
        const auto canonical = download(canonical_output, token_count * kHidden);
        const auto canonical_state = download(state->get(), kVHeads * kState * kState);
        const auto canonical_history = download(history->get(), 4 * kChannels);
        upload(wide_state.data(), state->get(), wide_state.size() * sizeof(float));
        upload(wide_history.data(), history->get(), wide_history.size() * sizeof(float));
        float output_error = 0.0F, state_error = 0.0F, history_error = 0.0F;
        for (std::size_t i = 0; i < wide_output.size(); ++i) {
            output_error = std::max(output_error, std::fabs(wide_output[i] - canonical[i]));
        }
        for (std::size_t i = 0; i < wide_state.size(); ++i) {
            state_error = std::max(state_error, std::fabs(wide_state[i] - canonical_state[i]));
        }
        for (std::size_t i = 0; i < wide_history.size(); ++i) {
            history_error = std::max(history_error, std::fabs(wide_history[i] - canonical_history[i]));
        }
        std::cerr << "M23 wide validation: output_max_abs=" << output_error
                  << " qkv_max_abs=" << qkv_error
                  << " gate_max_abs=" << gate_error
                  << " beta_max_abs=" << beta_error
                  << " decay_max_abs=" << decay_error
                  << " state_max_abs=" << state_error
                  << " history_max_abs=" << history_error << '\n';
        trace_dispatch();
    }

    void finish_prefill_batch4(const float* inputs, float* outputs, std::size_t count,
                               std::size_t offset,
                               const float* next_norm_weight = nullptr,
                               float* next_normalized = nullptr,
                               bool defer_dense_down = false,
                               bool defer_dense_gate_up = false) {
        if (!prefill_tail_batch_supported() || inputs == nullptr || outputs == nullptr || count != 4) {
            throw std::runtime_error("invalid recurrent prefill batch tail");
        }
        auto* gated_batch = static_cast<float*>(prefill_gated->get()) + offset * kInner;
        auto* residual_batch = static_cast<float*>(prefill_residual->get()) + offset * kHidden;
        auto* normalized_batch = static_cast<float*>(prefill_post_normalized->get()) + offset * kHidden;
        auto* q8_batch = static_cast<miinfer::Q8_1Block*>(prefill_q8_1->get())
            + offset * (kFfnInner / miinfer::kQ8_1BlockSize);
        auto* projected_batch = static_cast<float*>(prefill_projected->get()) + offset * kHidden;
        auto* ffn_gate_batch = static_cast<float*>(prefill_ffn_gate->get()) + offset * kFfnInner;
        auto* ffn_up_batch = static_cast<float*>(prefill_ffn_up->get()) + offset * kFfnInner;
        auto* ffn_activation_batch = static_cast<float*>(prefill_ffn_activation->get()) + offset * kFfnInner;
        auto* core_query_batch = prefill_core_query
            ? static_cast<float*>(prefill_core_query->get()) + offset * kKHeads * kState : nullptr;
        auto* core_key_batch = prefill_core_key
            ? static_cast<float*>(prefill_core_key->get()) + offset * kKHeads * kState : nullptr;
        auto* core_value_batch = prefill_core_value
            ? static_cast<float*>(prefill_core_value->get()) + offset * kVHeads * kState : nullptr;
        auto* core_beta_batch = prefill_core_beta
            ? static_cast<float*>(prefill_core_beta->get()) + offset * kVHeads : nullptr;
        auto* core_decay_batch = prefill_core_decay
            ? static_cast<float*>(prefill_core_decay->get()) + offset * kVHeads : nullptr;
        auto* core_gate_batch = prefill_core_gate
            ? static_cast<float*>(prefill_core_gate->get()) + offset * kVHeads * kState : nullptr;
        next_normalized = next_normalized != nullptr ? next_normalized + offset * kHidden : nullptr;
        const char* ssm_batch_env = std::getenv("MIINFER_PREFILL_SSM_BATCH");
        const bool batch_ssm_out = d_ssm_out_native
            && (ssm_batch_env == nullptr || std::strcmp(ssm_batch_env, "0") != 0);
        prefill_core_batch_active = recurrent_prefill_core_batch_supported()
            && !prefill_gdn_chunkwise;
        const bool core_q8_ready = prefill_core_batch_active && prefill_recurrent_q8 && batch_ssm_out;
        profile_tail_family_start(0, offset);
        if (prefill_core_batch_active) {
            miinfer::launch_qwen35_deltanet_fused_recurrent_core_batched4(
                core_query_batch,
                core_key_batch,
                core_value_batch,
                core_beta_batch,
                core_decay_batch,
                static_cast<const float*>(d_ssm_norm->get()),
                core_gate_batch,
                static_cast<float*>(state->get()), gated_batch,
                kKHeads, kVHeads, kState, model.config().rms_epsilon,
                hipStreamPerThread, core_q8_ready ? q8_batch : nullptr);
        }
        profile_tail_family_end(0, offset);

        // The generic Q5 batch decoder regressed; this path uses the
        // shape-specific word-reuse kernel, with an opt-out for A/B control.
        profile_tail_family_start(1, offset);
        if (batch_ssm_out) {
            if (!core_q8_ready) {
                for (std::size_t i = 0; i < count; ++i) {
                    miinfer::launch_q8_1_quantize_f32(
                        gated_batch + i * kInner,
                        q8_batch + i * (kInner / miinfer::kQ8_1BlockSize), kInner,
                        hipStreamPerThread);
                }
            }
            launch_q5k_wave_gemv_batched4(
                static_cast<const Q5KWaveTile*>(d_ssm_out_native->get()), q8_batch,
                projected_batch, kHidden, kInner,
                hipStreamPerThread);
            for (std::size_t i = 0; i < count; ++i) {
                miinfer::launch_qwen3_fused_add_rms_norm(
                    inputs + i * kHidden,
                    projected_batch + i * kHidden,
                    static_cast<const float*>(d_post_norm->get()),
                    residual_batch + i * kHidden, normalized_batch + i * kHidden,
                    kHidden, model.config().rms_epsilon, nullptr, nullptr);
            }
        } else {
            for (std::size_t i = 0; i < count; ++i) {
                MIINFER_HIP_CHECK(hipMemcpyAsync(
                    gated->get(), gated_batch + i * kInner, kInner * sizeof(float),
                    hipMemcpyDeviceToDevice, hipStreamPerThread));
                if (d_ssm_out_native) {
                    miinfer::launch_q8_1_quantize_f32(
                        static_cast<const float*>(gated->get()),
                        static_cast<miinfer::Q8_1Block*>(q8_1->get()), kInner,
                        hipStreamPerThread);
                    launch_q5k_wave_gemv(
                        static_cast<const Q5KWaveTile*>(d_ssm_out_native->get()),
                        static_cast<const miinfer::Q8_1Block*>(q8_1->get()),
                        static_cast<float*>(projected->get()), kHidden, kInner,
                        hipStreamPerThread);
                } else if (q5_q8_1_mmvq) {
                    project_q5_q8_1(d_ssm_out, static_cast<const float*>(gated->get()),
                                    static_cast<miinfer::Q8_1Block*>(q8_1->get()),
                                    static_cast<float*>(projected->get()), kHidden, kInner);
                } else {
                    project(ssm_out_weight, d_ssm_out, static_cast<const float*>(gated->get()),
                            static_cast<miinfer::Q8KDeviceBlock*>(q8->get()),
                            static_cast<float*>(projected->get()), kHidden, kInner);
                }
                miinfer::launch_qwen3_fused_add_rms_norm(
                    inputs + i * kHidden, static_cast<const float*>(projected->get()),
                    static_cast<const float*>(d_post_norm->get()),
                    residual_batch + i * kHidden, normalized_batch + i * kHidden,
                    kHidden, model.config().rms_epsilon, nullptr, nullptr);
            }
        }
        profile_tail_family_end(1, offset);

        for (std::size_t i = 0; i < count; ++i) {
            miinfer::launch_q8_1_quantize_f32(
                normalized_batch + i * kHidden,
                q8_batch + i * (kHidden / miinfer::kQ8_1BlockSize), kHidden,
                hipStreamPerThread);
        }
        profile_tail_family_start(2, offset);
        if (defer_dense_gate_up) return;
        if (d_ffn_swiglu_native) {
            launch_q4k_wave_fused_gate_up_swiglu_paired_batched4(
                static_cast<const Q4KWaveSwigluFusedTile*>(d_ffn_swiglu_native->get()),
                q8_batch, ffn_activation_batch,
                kFfnInner, kHidden, hipStreamPerThread);
            for (std::size_t i = 0; i < count; ++i) {
                miinfer::launch_q8_1_quantize_f32(
                    ffn_activation_batch + i * (kFfnInner),
                    q8_batch + i * (kFfnInner / miinfer::kQ8_1BlockSize), kFfnInner,
                    hipStreamPerThread);
            }
        } else {
            launch_q4k_wave_gemv_batched4(
                static_cast<const Q4KWaveTile*>(d_ffn_gate_native->get()), q8_batch,
                ffn_gate_batch, kFfnInner, kHidden,
                hipStreamPerThread);
            launch_q4k_wave_gemv_batched4(
                static_cast<const Q4KWaveTile*>(d_ffn_up_native->get()), q8_batch,
                ffn_up_batch, kFfnInner, kHidden,
                hipStreamPerThread);
            for (std::size_t i = 0; i < count; ++i) {
                miinfer::launch_qwen3_silu_mul(
                    ffn_gate_batch + i * kFfnInner,
                    ffn_up_batch + i * kFfnInner,
                    ffn_activation_batch + i * kFfnInner,
                    kFfnInner, hipStreamPerThread);
                miinfer::launch_q8_1_quantize_f32(
                    ffn_activation_batch + i * kFfnInner,
                    q8_batch + i * (kFfnInner / miinfer::kQ8_1BlockSize), kFfnInner,
                    hipStreamPerThread);
            }
        }
        profile_tail_family_end(2, offset);
        if (defer_dense_down) return;
        profile_tail_family_start(3, offset);
        if (ffn_down_weight.type == miinfer::GgufTensorType::q4_k) {
            launch_q4k_wave_gemv_batched4(
                static_cast<const Q4KWaveTile*>(d_ffn_down_native->get()), q8_batch,
                projected_batch, kHidden, kFfnInner,
                hipStreamPerThread);
        } else {
            launch_q6k_wave_gemv_batched4(
                static_cast<const Q6KWaveTile*>(d_ffn_down_native->get()), q8_batch,
                projected_batch, kHidden, kFfnInner,
                hipStreamPerThread);
        }
        for (std::size_t i = 0; i < count; ++i) {
            if (next_norm_weight != nullptr && next_normalized != nullptr) {
                miinfer::launch_qwen3_fused_add_rms_norm(
                    residual_batch + i * kHidden,
                    projected_batch + i * kHidden,
                    next_norm_weight, outputs + i * kHidden,
                    next_normalized + i * kHidden, kHidden,
                    model.config().rms_epsilon, nullptr, nullptr);
            } else {
                miinfer::launch_qwen3_add(
                    residual_batch + i * kHidden,
                    projected_batch + i * kHidden,
                    outputs + i * kHidden, kHidden, hipStreamPerThread);
            }
        }
        profile_tail_family_end(3, offset);
    }

    void finish_prefill_batch(const float* inputs, float* outputs, std::size_t count,
                              const float* next_norm_weight = nullptr,
                              float* next_normalized = nullptr) {
        if (!prefill_tail_batch_supported() || inputs == nullptr || outputs == nullptr
            || count == 0 || count > prefill_capacity || count % 4 != 0) {
            throw std::runtime_error("invalid recurrent prefill batch tail");
        }
        if (prefill_gdn_chunkwise && m12_gdn_workspace_ready && m12_gdn_raw_output != nullptr
            && count % kPrefillBatch == 0
            && recurrent_prefill_core_batch_supported()) {
            for (std::size_t start = 0; start < count; start += kPrefillBatch) {
                miinfer::launch_m12_gdn_chunk(
                    static_cast<const float*>(prefill_core_query->get()),
                    static_cast<const float*>(prefill_core_key->get()),
                    static_cast<const float*>(prefill_core_value->get()),
                    static_cast<const float*>(prefill_core_beta->get()),
                    static_cast<const float*>(prefill_core_decay->get()),
                    static_cast<float*>(state->get()),
                    m12_gdn_raw_output, m12_gdn_workspace,
                    static_cast<std::uint32_t>(count), static_cast<std::uint32_t>(start),
                    kKHeads, kVHeads, kState, kPrefillBatch, hipStreamPerThread);
            }
            miinfer::launch_m12_gdn_postprocess(
                static_cast<const float*>(m12_gdn_raw_output),
                static_cast<const float*>(prefill_core_gate->get()),
                static_cast<const float*>(d_ssm_norm->get()),
                static_cast<float*>(prefill_gated->get()),
                static_cast<std::uint32_t>(count), kVHeads, kState,
                model.config().rms_epsilon, hipStreamPerThread);
        }
        const bool dense_batch = prefill_dense_ffn_down && m12_dense_workspace_ready
            && (count == kPrefillBatch || count == kM12PrefillBatch);
        const bool dense_gate_up = dense_projection_prefill && m12_dense_workspace_ready
            && d_ffn_gate && d_ffn_up && m12_dense_input != nullptr
            && m12_dense_weights != nullptr && m12_dense_handle != nullptr;
        if (dense_gate_up) {
            for (std::size_t offset = 0; offset < count; offset += 4) {
                finish_prefill_batch4(
                    inputs + offset * kHidden, outputs + offset * kHidden, 4, offset,
                    next_norm_weight, next_normalized, dense_batch, true);
            }
            miinfer::launch_m12_f32_to_fp16(
                static_cast<const float*>(prefill_post_normalized->get()),
                m12_dense_input, count * kHidden, hipStreamPerThread);
            miinfer::launch_m12_q4k_to_fp16(
                static_cast<const miinfer::Q4KDeviceBlock*>(d_ffn_gate->get()),
                m12_dense_weights, kFfnInner, kHidden, hipStreamPerThread);
            std::string error;
            if (!miinfer::launch_rocblas_gemm_fp16_batch(
                    *m12_dense_handle, m12_dense_weights, m12_dense_input,
                    static_cast<float*>(prefill_ffn_gate->get()), kFfnInner, kHidden,
                    static_cast<int>(count), error)) {
                throw std::runtime_error(error);
            }
            miinfer::launch_m12_q4k_to_fp16(
                static_cast<const miinfer::Q4KDeviceBlock*>(d_ffn_up->get()),
                m12_dense_weights, kFfnInner, kHidden, hipStreamPerThread);
            if (!miinfer::launch_rocblas_gemm_fp16_batch(
                    *m12_dense_handle, m12_dense_weights, m12_dense_input,
                    static_cast<float*>(prefill_ffn_up->get()), kFfnInner, kHidden,
                    static_cast<int>(count), error)) {
                throw std::runtime_error(error);
            }
            auto* activation = static_cast<float*>(prefill_ffn_activation->get());
            auto* q8_batch = static_cast<miinfer::Q8_1Block*>(prefill_q8_1->get());
            for (std::size_t i = 0; i < count; ++i) {
                miinfer::launch_qwen3_silu_mul(
                    static_cast<const float*>(prefill_ffn_gate->get()) + i * kFfnInner,
                    static_cast<const float*>(prefill_ffn_up->get()) + i * kFfnInner,
                    activation + i * kFfnInner, kFfnInner, hipStreamPerThread);
                miinfer::launch_q8_1_quantize_f32(
                    activation + i * kFfnInner,
                    q8_batch + i * (kFfnInner / miinfer::kQ8_1BlockSize),
                    kFfnInner, hipStreamPerThread);
            }
            if (dense_batch) {
                miinfer::launch_m12_f32_to_fp16(
                    activation, m12_dense_input, count * kFfnInner, hipStreamPerThread);
                const auto* quantized = static_cast<const std::byte*>(d_ffn_down_dense_source->get());
                if (ffn_down_weight.type == miinfer::GgufTensorType::q4_k) {
                    miinfer::launch_m12_q4k_to_fp16(
                        reinterpret_cast<const miinfer::Q4KDeviceBlock*>(quantized),
                        m12_dense_weights, kHidden, kFfnInner, hipStreamPerThread);
                } else {
                    miinfer::launch_m12_q6k_to_fp16(
                        reinterpret_cast<const miinfer::Q6KDeviceBlock*>(quantized),
                        m12_dense_weights, kHidden, kFfnInner, hipStreamPerThread);
                }
                std::string down_error;
                if (!miinfer::launch_rocblas_gemm_fp16_batch(
                        *m12_dense_handle, m12_dense_weights, m12_dense_input,
                        static_cast<float*>(prefill_projected->get()), kHidden, kFfnInner,
                        static_cast<int>(count), down_error)) {
                    throw std::runtime_error(down_error);
                }
            } else if (ffn_down_weight.type == miinfer::GgufTensorType::q4_k) {
                launch_q4k_wave_gemv_batched4(
                    static_cast<const Q4KWaveTile*>(d_ffn_down_native->get()), q8_batch,
                    static_cast<float*>(prefill_projected->get()), kHidden, kFfnInner,
                    hipStreamPerThread);
            } else {
                launch_q6k_wave_gemv_batched4(
                    static_cast<const Q6KWaveTile*>(d_ffn_down_native->get()), q8_batch,
                    static_cast<float*>(prefill_projected->get()), kHidden, kFfnInner,
                    hipStreamPerThread);
            }
            auto* residual_batch = static_cast<float*>(prefill_residual->get());
            auto* projected_batch = static_cast<float*>(prefill_projected->get());
            auto* output_batch = outputs;
            auto* next_norm = next_normalized;
            for (std::size_t i = 0; i < count; ++i) {
                if (next_norm_weight != nullptr && next_norm != nullptr) {
                    miinfer::launch_qwen3_fused_add_rms_norm(
                        residual_batch + i * kHidden, projected_batch + i * kHidden,
                        next_norm_weight, output_batch + i * kHidden,
                        next_norm + i * kHidden, kHidden, model.config().rms_epsilon,
                        nullptr, nullptr);
                } else {
                    miinfer::launch_qwen3_add(
                        residual_batch + i * kHidden, projected_batch + i * kHidden,
                        output_batch + i * kHidden, kHidden, hipStreamPerThread);
                }
            }
            return;
        }
        if (dense_batch) {
            for (std::size_t offset = 0; offset < count; offset += 4) {
                finish_prefill_batch4(
                    inputs + offset * kHidden, outputs + offset * kHidden, 4, offset,
                    next_norm_weight, next_normalized, true);
            }
            miinfer::launch_m12_f32_to_fp16(
                static_cast<const float*>(prefill_ffn_activation->get()),
                m12_dense_input, count * kFfnInner, hipStreamPerThread);
            const auto* quantized =
                static_cast<const std::byte*>(d_ffn_down_dense_source->get());
            if (ffn_down_weight.type == miinfer::GgufTensorType::q4_k) {
                miinfer::launch_m12_q4k_to_fp16(
                    reinterpret_cast<const miinfer::Q4KDeviceBlock*>(quantized),
                    m12_dense_weights, kHidden, kFfnInner, hipStreamPerThread);
            } else {
                miinfer::launch_m12_q6k_to_fp16(
                    reinterpret_cast<const miinfer::Q6KDeviceBlock*>(quantized),
                    m12_dense_weights, kHidden, kFfnInner, hipStreamPerThread);
            }
            std::string error;
            if (!miinfer::launch_rocblas_gemm_fp16_batch(
                    *m12_dense_handle, m12_dense_weights, m12_dense_input,
                    static_cast<float*>(prefill_projected->get()), kHidden, kFfnInner,
                    static_cast<int>(count), error)) {
                throw std::runtime_error(error);
            }
            auto* residual_batch = static_cast<float*>(prefill_residual->get());
            auto* projected_batch = static_cast<float*>(prefill_projected->get());
            for (std::size_t i = 0; i < count; ++i) {
                if (next_norm_weight != nullptr && next_normalized != nullptr) {
                    miinfer::launch_qwen3_fused_add_rms_norm(
                        residual_batch + i * kHidden, projected_batch + i * kHidden,
                        next_norm_weight, outputs + i * kHidden,
                        next_normalized + i * kHidden, kHidden,
                        model.config().rms_epsilon, nullptr, nullptr);
                } else {
                    miinfer::launch_qwen3_add(
                        residual_batch + i * kHidden, projected_batch + i * kHidden,
                        outputs + i * kHidden, kHidden, hipStreamPerThread);
                }
            }
            return;
        }
        for (std::size_t offset = 0; offset < count; offset += 4) {
            finish_prefill_batch4(inputs + offset * kHidden, outputs + offset * kHidden,
                                  4, offset, next_norm_weight, next_normalized);
        }
    }

    void run(const float* input, std::uint32_t position, float* output,
             const float* next_norm_weight = nullptr,
             float* next_normalized = nullptr,
             bool precomputed_norm = false,
             miinfer::Q8_1Block* next_q8_1 = nullptr,
             bool precomputed_q8 = false,
             const float* prepared_qkv = nullptr,
             const float* prepared_gate = nullptr,
             const float* prepared_normalized = nullptr,
             bool defer_prefill_tail = false,
             std::size_t prefill_index = 0) {
        const float* normalized_input = prepared_normalized != nullptr
            ? prepared_normalized : static_cast<const float*>(normalized->get());
        if (prepared_normalized != nullptr) precomputed_norm = true;
        const bool precomputed_projection = prepared_qkv != nullptr;
        stage_start(0, position);
        if (!precomputed_norm) {
            miinfer::launch_qwen3_rms_norm(
                input, static_cast<const float*>(d_attn_norm->get()),
                static_cast<float*>(normalized->get()), kHidden, model.config().rms_epsilon);
        }
        if (gate_path_capture != nullptr && gate_path_capture_position == position) {
            gate_path_capture->normalized = download(normalized_input, kHidden);
        }
        trace_tensor(position, "attn_norm", normalized_input, kHidden,
                     "attn_norm-" + std::to_string(index));
        stage_end(0, position);
        stage_start(1, position);
        float* qkv_storage = static_cast<float*>(qkv->get());
        const float* qkv_dest = precomputed_projection ? prepared_qkv : qkv_storage;
        float* gate_storage = gate ? static_cast<float*>(gate->get()) : nullptr;
        const float* gate_dest = d_qkv_gate_combined ? qkv_dest + kChannels
            : (precomputed_projection && prepared_gate != nullptr ? prepared_gate : gate_storage);
        if (precomputed_projection) {
            // The batch path already populated qkv/gate with the native B=4 kernel.
        } else if (resident_m23_all) {
            miinfer::launch_m23_q8_1_mmq_quantize(
                normalized_input,
                static_cast<miinfer::M23Q8_1MmqBlock*>(decode_mmq_q8->get()),
                1, kHidden);
            if (qkv_weight.type == miinfer::GgufTensorType::q4_k) {
                launch_m23_q4k_repacked_mmq(
                    static_cast<const Q4KMmqTile*>(resident_qkv_mmq->get()),
                    static_cast<const miinfer::M23Q8_1MmqBlock*>(decode_mmq_q8->get()),
                    qkv_storage, kChannels, kHidden, 1);
            } else {
                launch_m23_q6k_repacked_mmq(
                    static_cast<const Q6KMmqTile*>(resident_qkv_mmq->get()),
                    static_cast<const miinfer::M23Q8_1MmqBlock*>(decode_mmq_q8->get()),
                    qkv_storage, kChannels, kHidden, 1);
            }
        } else if (d_qkv_gate_combined) {
            if (!precomputed_q8 || !fused_norm_q8) {
                miinfer::launch_q8_1_quantize_f32(
                    normalized_input,
                    static_cast<miinfer::Q8_1Block*>(q8_1->get()), kHidden);
            }
            launch_q4k_wave_gemv(
                static_cast<const Q4KWaveTile*>(d_qkv_gate_combined->get()),
                static_cast<const miinfer::Q8_1Block*>(q8_1->get()),
                qkv_storage, kChannels + kInner, kHidden);
        } else if (d_qkv_native) {
            if (!precomputed_q8 || !fused_norm_q8) {
                miinfer::launch_q8_1_quantize_f32(
                    normalized_input,
                    static_cast<miinfer::Q8_1Block*>(q8_1->get()), kHidden);
            }
            if (qkv_weight.type == miinfer::GgufTensorType::q4_k) {
                launch_q4k_wave_gemv(
                    static_cast<const Q4KWaveTile*>(d_qkv_native->get()),
                    static_cast<const miinfer::Q8_1Block*>(q8_1->get()),
                    qkv_storage, kChannels, kHidden);
            } else {
                launch_q6k_wave_gemv(
                    static_cast<const Q6KWaveTile*>(d_qkv_native->get()),
                    static_cast<const miinfer::Q8_1Block*>(q8_1->get()),
                    qkv_storage, kChannels, kHidden);
            }
        } else if (q6_q8_k_dot4_qkv && qkv_weight.type == miinfer::GgufTensorType::q6_k) {
            project_q6_q8_k_dot4(d_qkv, normalized_input,
                                static_cast<miinfer::Q8KDeviceBlock*>(q8->get()),
                                qkv_storage, kChannels, kHidden);
        } else {
            project(qkv_weight, d_qkv, normalized_input,
                    static_cast<miinfer::Q8KDeviceBlock*>(q8->get()),
                    qkv_storage, kChannels, kHidden);
        }
        trace_tensor(position, "qkv", qkv_dest, kChannels,
                     "linear_attn_qkv_mixed-" + std::to_string(index));
        if (layer_path_capture != nullptr && layer_path_capture_position == position) {
            layer_path_capture->input = download(input, kHidden);
            layer_path_capture->normalized = download(normalized_input, kHidden);
            layer_path_capture->qkv = download(qkv_dest, kChannels);
        }
        stage_end(1, position);
        stage_start(2, position);
        if (!d_qkv_gate_combined && !precomputed_projection) {
            if (resident_m23_all) {
                launch_m23_q4k_repacked_mmq(
                    static_cast<const Q4KMmqTile*>(resident_gate_mmq->get()),
                    static_cast<const miinfer::M23Q8_1MmqBlock*>(decode_mmq_q8->get()),
                    gate_storage, kInner, kHidden, 1);
            } else if (d_attn_gate_native) {
                if (!d_qkv_native) {
                    miinfer::launch_q8_1_quantize_f32(
                        normalized_input,
                        static_cast<miinfer::Q8_1Block*>(q8_1->get()), kHidden);
                }
                launch_q4k_wave_gemv(
                    static_cast<const Q4KWaveTile*>(d_attn_gate_native->get()),
                    static_cast<const miinfer::Q8_1Block*>(q8_1->get()),
                    gate_storage, kInner, kHidden);
            } else if (q4_q8_1_attn_gate && gate_weight.type == miinfer::GgufTensorType::q4_k) {
                miinfer::launch_q8_1_quantize_f32(
                    normalized_input,
                    static_cast<miinfer::Q8_1Block*>(q8_1->get()), kHidden);
                project_q4_q8_1_prequantized(
                    d_gate, static_cast<const miinfer::Q8_1Block*>(q8_1->get()),
                    gate_storage, kInner, kHidden,
                    q4_q8_1_lds_input, q4_q8_1_lds_metadata,
                    q4_q8_1_lds_decoded_metadata);
            } else {
                project(gate_weight, d_gate, normalized_input,
                        static_cast<miinfer::Q8KDeviceBlock*>(q8->get()),
                        gate_storage, kInner, kHidden,
                        reuse_projection_q8);
            }
        }
        if (gate_path_capture != nullptr && gate_path_capture_position == position) {
            gate_path_capture->gate = download(gate_dest, kInner);
        }
        stage_end(2, position);
        stage_start(3, position);
        if (dual_beta_alpha_prepare) {
            miinfer::launch_qwen35_f32_dual_beta_decay(
                static_cast<const float*>(d_beta->get()), static_cast<const float*>(d_alpha->get()),
                normalized_input, static_cast<const float*>(d_dt->get()),
                static_cast<const float*>(d_a->get()), static_cast<float*>(beta->get()),
                static_cast<float*>(decay->get()), kVHeads, kHidden);
        } else if (dual_beta_alpha_projection) {
            miinfer::launch_qwen35_f32_dual_gemv(
                static_cast<const float*>(d_beta->get()), static_cast<const float*>(d_alpha->get()),
                normalized_input, static_cast<float*>(beta_raw->get()),
                static_cast<float*>(alpha_raw->get()), kVHeads, kHidden);
        } else {
            miinfer::launch_qwen35_f32_gemv(
                static_cast<const float*>(d_beta->get()), normalized_input,
                static_cast<float*>(beta_raw->get()), kVHeads, kHidden);
            miinfer::launch_qwen35_f32_gemv(
                static_cast<const float*>(d_alpha->get()), normalized_input,
                static_cast<float*>(alpha_raw->get()), kVHeads, kHidden);
        }
        if (!dual_beta_alpha_prepare) {
            miinfer::launch_qwen35_prepare_beta_decay(
                static_cast<const float*>(beta_raw->get()), static_cast<const float*>(alpha_raw->get()),
                static_cast<const float*>(d_dt->get()), static_cast<const float*>(d_a->get()),
                static_cast<float*>(beta->get()), static_cast<float*>(decay->get()), kVHeads);
        }
        stage_end(3, position);
        stage_start(4, position);
        miinfer::launch_qwen35_conv_silu_split(
            qkv_dest, static_cast<const float*>(d_conv->get()),
            static_cast<float*>(history->get()), static_cast<float*>(query->get()),
            static_cast<float*>(key->get()), static_cast<float*>(value->get()),
            position, 4, kChannels, 4);
        if (dual_head_normalize) {
            miinfer::launch_qwen35_dual_head_l2_normalize(
                static_cast<const float*>(query->get()), static_cast<const float*>(key->get()),
                static_cast<float*>(query_norm->get()), static_cast<float*>(key_norm->get()),
                kKHeads, kState);
        } else {
            miinfer::launch_qwen35_head_l2_normalize(
                static_cast<const float*>(query->get()), static_cast<float*>(query_norm->get()),
                kKHeads, kState);
            miinfer::launch_qwen35_head_l2_normalize(
                static_cast<const float*>(key->get()), static_cast<float*>(key_norm->get()),
                kKHeads, kState);
        }
        stage_end(4, position);
        if (provenance != nullptr && provenance_position == position) {
            *provenance = sample_update(provenance_index);
        }
        if (operand_capture != nullptr && operand_capture_position == position) {
            capture_operands(*operand_capture);
        }
        if (key_path_capture != nullptr && key_path_capture_position == position) {
            key_path_capture->input = download(input, kHidden);
            key_path_capture->normalized = download(normalized_input, kHidden);
            key_path_capture->qkv = download(qkv_dest, kChannels);
            key_path_capture->key = download(key->get(), kKHeads * kState);
            key_path_capture->key_norm = download(key_norm->get(), kKHeads * kState);
        }
        stage_start(5, position);
        const bool defer_recurrent_core = defer_prefill_tail
            && recurrent_prefill_core_batch_supported();
        if (defer_recurrent_core) {
            if (prefill_index >= prefill_capacity) throw std::runtime_error("invalid recurrent prefill index");
            miinfer::launch_qwen35_deltanet_prefill_stage(
                static_cast<const float*>(query_norm->get()),
                static_cast<const float*>(key_norm->get()),
                static_cast<const float*>(value->get()),
                static_cast<const float*>(beta->get()),
                static_cast<const float*>(decay->get()), gate_dest,
                static_cast<float*>(prefill_core_query->get()),
                static_cast<float*>(prefill_core_key->get()),
                static_cast<float*>(prefill_core_value->get()),
                static_cast<float*>(prefill_core_beta->get()),
                static_cast<float*>(prefill_core_decay->get()),
                static_cast<float*>(prefill_core_gate->get()), prefill_index,
                kKHeads, kVHeads, kState, hipStreamPerThread);
            stage_end(5, position);
            return;
        }
        if (fused_recurrent_core && transposed_state) {
            float* rec_out_ptr = (layer_path_capture != nullptr || gate_path_capture != nullptr || trace != nullptr)
                ? static_cast<float*>(recurrent_output->get()) : nullptr;
            miinfer::launch_qwen35_deltanet_fused_recurrent_core(
                static_cast<const float*>(query_norm->get()),
                static_cast<const float*>(key_norm->get()),
                static_cast<const float*>(value->get()),
                static_cast<const float*>(beta->get()),
                static_cast<const float*>(decay->get()),
                static_cast<const float*>(d_ssm_norm->get()),
                gate_dest,
                static_cast<float*>(state->get()),
                static_cast<float*>(gated->get()),
                rec_out_ptr,
                kKHeads, kVHeads, kState,
                model.config().rms_epsilon,
                nullptr,
                (fused_core_q8 && d_ssm_out_native) ? static_cast<miinfer::Q8_1Block*>(q8_1->get()) : nullptr);
        } else if (transposed_state) {
            if (transposed_no_decay_store) {
                if (row_wave_state && transposed_lds_inputs) {
                    miinfer::launch_qwen35_deltanet_state_update_transposed_row_waves(
                        static_cast<const float*>(query_norm->get()), static_cast<const float*>(key_norm->get()),
                        static_cast<const float*>(value->get()), static_cast<const float*>(beta->get()),
                        static_cast<const float*>(decay->get()), static_cast<float*>(state->get()),
                        static_cast<float*>(recurrent_output->get()), kKHeads, kVHeads, kState);
                } else if (fuse_beta_alpha && transposed_lds_inputs) {
                    miinfer::launch_qwen35_deltanet_state_update_transposed_fused_beta_alpha(
                        static_cast<const float*>(query_norm->get()), static_cast<const float*>(key_norm->get()),
                        static_cast<const float*>(value->get()), static_cast<const float*>(beta_raw->get()),
                        static_cast<const float*>(alpha_raw->get()), static_cast<const float*>(d_dt->get()),
                        static_cast<const float*>(d_a->get()), static_cast<float*>(state->get()),
                        static_cast<float*>(recurrent_output->get()), kKHeads, kVHeads, kState);
                } else if (transposed_lds_inputs) {
                    miinfer::launch_qwen35_deltanet_state_update_transposed_no_decay_store_lds_inputs(
                        static_cast<const float*>(query_norm->get()), static_cast<const float*>(key_norm->get()),
                        static_cast<const float*>(value->get()), static_cast<const float*>(beta->get()),
                        static_cast<const float*>(decay->get()), static_cast<float*>(state->get()),
                        static_cast<float*>(recurrent_output->get()), kKHeads, kVHeads, kState);
                } else {
                    miinfer::launch_qwen35_deltanet_state_update_transposed_no_decay_store(
                        static_cast<const float*>(query_norm->get()), static_cast<const float*>(key_norm->get()),
                        static_cast<const float*>(value->get()), static_cast<const float*>(beta->get()),
                        static_cast<const float*>(decay->get()), static_cast<float*>(state->get()),
                        static_cast<float*>(recurrent_output->get()), kKHeads, kVHeads, kState);
                }
            } else {
                miinfer::launch_qwen35_deltanet_state_update_transposed(
                    static_cast<const float*>(query_norm->get()), static_cast<const float*>(key_norm->get()),
                    static_cast<const float*>(value->get()), static_cast<const float*>(beta->get()),
                    static_cast<const float*>(decay->get()), static_cast<float*>(state->get()),
                    static_cast<float*>(recurrent_output->get()), kKHeads, kVHeads, kState);
            }
        } else if (no_decay_store) {
            miinfer::launch_qwen35_deltanet_state_update_no_decay_store(
                static_cast<const float*>(query_norm->get()), static_cast<const float*>(key_norm->get()),
                static_cast<const float*>(value->get()), static_cast<const float*>(beta->get()),
                static_cast<const float*>(decay->get()), static_cast<float*>(state->get()),
                static_cast<float*>(recurrent_output->get()), kKHeads, kVHeads, kState);
        } else {
            miinfer::launch_qwen35_deltanet_state_update(
                static_cast<const float*>(query_norm->get()), static_cast<const float*>(key_norm->get()),
                static_cast<const float*>(value->get()), static_cast<const float*>(beta->get()),
                static_cast<const float*>(decay->get()), static_cast<float*>(state->get()),
                static_cast<float*>(recurrent_output->get()), kKHeads, kVHeads, kState);
        }
        if (trace != nullptr && trace->layer == index && position + 1 == trace->position) {
            if (transposed_state) {
                trace->report_host("state_after", logical_state(),
                                   "state_predelta-" + std::to_string(index));
            } else {
                trace->report_at(position + 1, "state_after", static_cast<const float*>(state->get()),
                                 kVHeads * kState * kState,
                                 "state_predelta-" + std::to_string(index));
            }
        }
        stage_end(5, position);
        trace_tensor(position, "recurrent_output",
                     static_cast<const float*>(recurrent_output->get()), kVHeads * kState,
                     "attn_output-" + std::to_string(index));
        if (layer_path_capture != nullptr && layer_path_capture_position == position) {
            layer_path_capture->recurrent_output = download(recurrent_output->get(), kVHeads * kState);
        }
        if (gate_path_capture != nullptr && gate_path_capture_position == position) {
            gate_path_capture->recurrent_output = download(recurrent_output->get(), kVHeads * kState);
        }
        stage_start(6, position);
        if (!fused_recurrent_core || !transposed_state) {
            miinfer::launch_qwen3_head_rms_normalize(
                static_cast<const float*>(recurrent_output->get()), static_cast<float*>(head_norm->get()),
                kVHeads, kState, model.config().rms_epsilon);
            if (gate_path_capture != nullptr && gate_path_capture_position == position) {
                gate_path_capture->head_norm = download(head_norm->get(), kVHeads * kState);
            }
            miinfer::launch_qwen3_head_mul(
                static_cast<const float*>(head_norm->get()), static_cast<const float*>(d_ssm_norm->get()),
                static_cast<float*>(gated->get()), kVHeads, kState);
            if (gate_path_capture != nullptr && gate_path_capture_position == position) {
                gate_path_capture->head_scaled = download(gated->get(), kVHeads * kState);
            }
            miinfer::launch_qwen3_silu_mul(
                gate_dest, static_cast<const float*>(gated->get()),
                static_cast<float*>(gated->get()), kVHeads * kState);
        }
        stage_end(6, position);
        if (defer_prefill_tail) {
            if (prefill_index >= prefill_capacity) throw std::runtime_error("invalid recurrent prefill index");
            MIINFER_HIP_CHECK(hipMemcpyAsync(
                static_cast<float*>(prefill_gated->get()) + prefill_index * kInner,
                gated->get(), kInner * sizeof(float), hipMemcpyDeviceToDevice,
                hipStreamPerThread));
            return;
        }
        trace_tensor(position, "gated", static_cast<const float*>(gated->get()), kVHeads * kState,
                     "final_output-" + std::to_string(index));
        if (layer_path_capture != nullptr && layer_path_capture_position == position) {
            layer_path_capture->gated = download(gated->get(), kVHeads * kState);
        }
        if (output_projection_path_capture != nullptr
            && output_projection_path_capture_position == position) {
            output_projection_path_capture->gated = download(gated->get(), kVHeads * kState);
        }
        if (gate_path_capture != nullptr && gate_path_capture_position == position) {
            gate_path_capture->gated = download(gated->get(), kVHeads * kState);
        }
        stage_start(7, position);
        if (resident_m23_all) {
            miinfer::launch_m23_q8_1_mmq_quantize(
                static_cast<const float*>(gated->get()),
                static_cast<miinfer::M23Q8_1MmqBlock*>(decode_mmq_q8->get()),
                1, kInner);
            launch_m23_q5k_repacked_mmq(
                static_cast<const Q5KMmqTile*>(resident_ssm_out_mmq->get()),
                static_cast<const miinfer::M23Q8_1MmqBlock*>(decode_mmq_q8->get()),
                static_cast<float*>(projected->get()), kHidden, kInner, 1);
        } else if (d_ssm_out_native) {
            if (!fused_core_q8 || !fused_recurrent_core || !transposed_state) {
                miinfer::launch_q8_1_quantize_f32(
                    static_cast<const float*>(gated->get()),
                    static_cast<miinfer::Q8_1Block*>(q8_1->get()), kInner);
            }
            launch_q5k_wave_gemv(
                static_cast<const Q5KWaveTile*>(d_ssm_out_native->get()),
                static_cast<const miinfer::Q8_1Block*>(q8_1->get()),
                static_cast<float*>(projected->get()), kHidden, kInner);
        } else if (q5_q8_1_mmvq) {
            project_q5_q8_1(d_ssm_out, static_cast<const float*>(gated->get()),
                            static_cast<miinfer::Q8_1Block*>(q8_1->get()),
                            static_cast<float*>(projected->get()), kHidden, kInner);
        } else {
            project(ssm_out_weight, d_ssm_out, static_cast<const float*>(gated->get()),
                    static_cast<miinfer::Q8KDeviceBlock*>(q8->get()),
                    static_cast<float*>(projected->get()), kHidden, kInner);
        }
        if (output_projection_path_capture != nullptr
            && output_projection_path_capture_position == position) {
            output_projection_path_capture->q8_input = download_bytes(
                (d_ssm_out_native || q5_q8_1_mmvq) ? q8_1->get() : q8->get(),
                (d_ssm_out_native || q5_q8_1_mmvq)
                    ? (kInner / miinfer::kQ8_1BlockSize) * sizeof(miinfer::Q8_1Block)
                    : (kInner / 256) * sizeof(miinfer::Q8KDeviceBlock));
            output_projection_path_capture->projected = download(projected->get(), kHidden);
        }
        stage_end(7, position);
        stage_start(8, position);
        if (fused_add_rms_norm) {
            miinfer::launch_qwen3_fused_add_rms_norm(
                input, static_cast<const float*>(projected->get()),
                static_cast<const float*>(d_post_norm->get()),
                static_cast<float*>(residual->get()),
                static_cast<float*>(post_normalized->get()),
                kHidden, model.config().rms_epsilon, nullptr,
                (fused_norm_q8 && (d_ffn_swiglu_native || (d_ffn_gate_native && d_ffn_up_native)))
                    ? static_cast<miinfer::Q8_1Block*>(q8_1->get()) : nullptr);
        } else {
            miinfer::launch_qwen3_add(
                input, static_cast<const float*>(projected->get()),
                static_cast<float*>(residual->get()), kHidden);
        }
        if (output_projection_path_capture != nullptr
            && output_projection_path_capture_position == position) {
            output_projection_path_capture->input = download(input, kHidden);
            output_projection_path_capture->residual = download(residual->get(), kHidden);
        }
        if (layer_path_capture != nullptr && layer_path_capture_position == position) {
            layer_path_capture->attention_residual = download(residual->get(), kHidden);
        }
        trace_tensor(position, "attention_residual", static_cast<const float*>(residual->get()),
                     kHidden, "attn_residual-" + std::to_string(index));
        stage_end(8, position);
        stage_start(9, position);
        if (!fused_add_rms_norm) {
            miinfer::launch_qwen3_rms_norm(
                static_cast<const float*>(residual->get()), static_cast<const float*>(d_post_norm->get()),
                static_cast<float*>(post_normalized->get()), kHidden, model.config().rms_epsilon);
        }
        trace_tensor(position, "post_attention_norm",
                     static_cast<const float*>(post_normalized->get()), kHidden,
                     "attn_post_norm-" + std::to_string(index));
        stage_end(9, position);
        stage_start(10, position);
        if (resident_m23_all) {
            miinfer::launch_m23_q8_1_mmq_quantize(
                static_cast<const float*>(post_normalized->get()),
                static_cast<miinfer::M23Q8_1MmqBlock*>(decode_mmq_q8->get()),
                1, kHidden);
            launch_m23_q4k_repacked_mmq(
                static_cast<const Q4KMmqTile*>(resident_ffn_gate_mmq->get()),
                static_cast<const miinfer::M23Q8_1MmqBlock*>(decode_mmq_q8->get()),
                static_cast<float*>(ffn_gate->get()), kFfnInner, kHidden, 1);
            launch_m23_q4k_repacked_mmq(
                static_cast<const Q4KMmqTile*>(resident_ffn_up_mmq->get()),
                static_cast<const miinfer::M23Q8_1MmqBlock*>(decode_mmq_q8->get()),
                static_cast<float*>(ffn_up->get()), kFfnInner, kHidden, 1);
        } else if (d_ffn_swiglu_native) {
            if (!fused_norm_q8) {
                miinfer::launch_q8_1_quantize_f32(
                    static_cast<const float*>(post_normalized->get()),
                    static_cast<miinfer::Q8_1Block*>(q8_1->get()), kHidden);
            }
            launch_q4k_wave_fused_gate_up_swiglu_paired(
                static_cast<const Q4KWaveSwigluFusedTile*>(d_ffn_swiglu_native->get()),
                static_cast<const miinfer::Q8_1Block*>(q8_1->get()),
                static_cast<float*>(ffn_activation->get()), kFfnInner, kHidden);
        } else if (d_ffn_gate_native && d_ffn_up_native) {
            if (!fused_norm_q8) {
                miinfer::launch_q8_1_quantize_f32(
                    static_cast<const float*>(post_normalized->get()),
                    static_cast<miinfer::Q8_1Block*>(q8_1->get()), kHidden);
            }
            if (fused_gate_up_swiglu) {
                launch_q4k_wave_fused_gate_up_swiglu(
                    static_cast<const Q4KWaveTile*>(d_ffn_gate_native->get()),
                    static_cast<const Q4KWaveTile*>(d_ffn_up_native->get()),
                    static_cast<const miinfer::Q8_1Block*>(q8_1->get()),
                    static_cast<float*>(ffn_activation->get()), kFfnInner, kHidden);
            } else {
                launch_q4k_wave_gemv(
                    static_cast<const Q4KWaveTile*>(d_ffn_gate_native->get()),
                    static_cast<const miinfer::Q8_1Block*>(q8_1->get()),
                    static_cast<float*>(ffn_gate->get()), kFfnInner, kHidden);
                launch_q4k_wave_gemv(
                    static_cast<const Q4KWaveTile*>(d_ffn_up_native->get()),
                    static_cast<const miinfer::Q8_1Block*>(q8_1->get()),
                    static_cast<float*>(ffn_up->get()), kFfnInner, kHidden);
            }
        } else if (q4_q8_1_gate_up
            && ffn_gate_weight.type == miinfer::GgufTensorType::q4_k
            && ffn_up_weight.type == miinfer::GgufTensorType::q4_k) {
            miinfer::launch_q8_1_quantize_f32(
                static_cast<const float*>(post_normalized->get()),
                static_cast<miinfer::Q8_1Block*>(q8_1->get()), kHidden);
            project_q4_q8_1_prequantized(
                d_ffn_gate, static_cast<const miinfer::Q8_1Block*>(q8_1->get()),
                static_cast<float*>(ffn_gate->get()), kFfnInner, kHidden,
                q4_q8_1_lds_input, q4_q8_1_lds_metadata,
                q4_q8_1_lds_decoded_metadata);
            project_q4_q8_1_prequantized(
                d_ffn_up, static_cast<const miinfer::Q8_1Block*>(q8_1->get()),
                static_cast<float*>(ffn_up->get()), kFfnInner, kHidden,
                q4_q8_1_lds_input, q4_q8_1_lds_metadata,
                q4_q8_1_lds_decoded_metadata);
        } else {
            project(ffn_gate_weight, d_ffn_gate, static_cast<const float*>(post_normalized->get()),
                    static_cast<miinfer::Q8KDeviceBlock*>(q8->get()),
                    static_cast<float*>(ffn_gate->get()), kFfnInner, kHidden);
            project(ffn_up_weight, d_ffn_up, static_cast<const float*>(post_normalized->get()),
                    static_cast<miinfer::Q8KDeviceBlock*>(q8->get()),
                    static_cast<float*>(ffn_up->get()), kFfnInner, kHidden,
                    reuse_projection_q8);
        }
        stage_end(10, position);
        stage_start(11, position);
        if (resident_m23_all || (!d_ffn_swiglu_native
                                 && (!fused_gate_up_swiglu || !d_ffn_gate_native || !d_ffn_up_native))) {
            miinfer::launch_qwen3_silu_mul(
                static_cast<const float*>(ffn_gate->get()), static_cast<const float*>(ffn_up->get()),
                static_cast<float*>(ffn_activation->get()), kFfnInner);
        }
        stage_end(11, position);
        if (layer_path_capture != nullptr && layer_path_capture_position == position) {
            layer_path_capture->post_normalized = download(post_normalized->get(), kHidden);
        }
        stage_start(12, position);
        if (resident_m23_all) {
            miinfer::launch_m23_q8_1_mmq_quantize(
                static_cast<const float*>(ffn_activation->get()),
                static_cast<miinfer::M23Q8_1MmqBlock*>(decode_mmq_q8->get()),
                1, kFfnInner);
            if (ffn_down_weight.type == miinfer::GgufTensorType::q4_k) {
                launch_m23_q4k_repacked_mmq(
                    static_cast<const Q4KMmqTile*>(resident_ffn_down_mmq->get()),
                    static_cast<const miinfer::M23Q8_1MmqBlock*>(decode_mmq_q8->get()),
                    static_cast<float*>(projected->get()), kHidden, kFfnInner, 1);
            } else {
                launch_m23_q6k_repacked_mmq(
                    static_cast<const Q6KMmqTile*>(resident_ffn_down_mmq->get()),
                    static_cast<const miinfer::M23Q8_1MmqBlock*>(decode_mmq_q8->get()),
                    static_cast<float*>(projected->get()), kHidden, kFfnInner, 1);
            }
        } else if (d_ffn_down_native) {
            if (ffn_down_weight.type == miinfer::GgufTensorType::q4_k) {
                project_native_down(d_ffn_down_native,
                    static_cast<const float*>(ffn_activation->get()),
                    static_cast<miinfer::Q8_1Block*>(q8_1->get()),
                    static_cast<float*>(projected->get()));
            } else {
                project_native_q6k_down(d_ffn_down_native,
                    static_cast<const float*>(ffn_activation->get()),
                    static_cast<miinfer::Q8_1Block*>(q8_1->get()),
                    static_cast<float*>(projected->get()));
            }
        } else if (expanded_down && d_ffn_down_expanded != nullptr) {
            project_q4_q8_1_expanded(d_ffn_down_expanded,
                                     static_cast<const float*>(ffn_activation->get()),
                                     static_cast<miinfer::Q8_1Block*>(q8_1->get()),
                                     static_cast<float*>(projected->get()), kHidden, kFfnInner);
        } else if (q4_q8_1_mmvq && ffn_down_weight.type == miinfer::GgufTensorType::q4_k) {
            project_q4_q8_1(d_ffn_down, static_cast<const float*>(ffn_activation->get()),
                            static_cast<miinfer::Q8_1Block*>(q8_1->get()),
                            static_cast<float*>(projected->get()), kHidden, kFfnInner,
                            q4_q8_1_lds_input, q4_q8_1_lds_metadata,
                            q4_q8_1_lds_decoded_metadata);
        } else {
            project(ffn_down_weight, d_ffn_down, static_cast<const float*>(ffn_activation->get()),
                    static_cast<miinfer::Q8KDeviceBlock*>(q8->get()),
                    static_cast<float*>(projected->get()), kHidden, kFfnInner);
        }
        trace_tensor(position, "ffn_out", static_cast<const float*>(projected->get()), kHidden,
                     "ffn_out-" + std::to_string(index));
        if (layer_path_capture != nullptr && layer_path_capture_position == position) {
            layer_path_capture->ffn_output = download(projected->get(), kHidden);
        }
        stage_end(12, position);
        stage_start(13, position);
        float* completed_output = direct_layer_output
            ? output : static_cast<float*>(layer_output->get());
        if (fused_interlayer_norm && next_norm_weight != nullptr && next_normalized != nullptr) {
            miinfer::launch_qwen3_fused_add_rms_norm(
                static_cast<const float*>(residual->get()),
                static_cast<const float*>(projected->get()),
                next_norm_weight,
                completed_output,
                next_normalized,
                kHidden,
                model.config().rms_epsilon, nullptr,
                (fused_norm_q8 && next_q8_1 != nullptr) ? next_q8_1 : nullptr);
        } else {
            miinfer::launch_qwen3_add(
                static_cast<const float*>(residual->get()), static_cast<const float*>(projected->get()),
                completed_output, kHidden);
        }
        trace_tensor(position, "layer_output", completed_output, kHidden,
                     "l_out-" + std::to_string(index));
        if (layer_path_capture != nullptr && layer_path_capture_position == position) {
            layer_path_capture->layer_output = download(completed_output, kHidden);
        }
        if (!direct_layer_output) {
            MIINFER_HIP_CHECK(hipMemcpyAsync(output, layer_output->get(), kHidden * sizeof(float),
                                            hipMemcpyDeviceToDevice, hipStreamPerThread));
        }
        stage_end(13, position);
    }
};

struct FullAttentionLayer {
    const miinfer::Qwen35Model& model;
    const miinfer::GgufTensor& attn_norm;
    const miinfer::GgufTensor& q_weight;
    const miinfer::GgufTensor& k_weight;
    const miinfer::GgufTensor& v_weight;
    const miinfer::GgufTensor& o_weight;
    const miinfer::GgufTensor& q_norm;
    const miinfer::GgufTensor& k_norm;
    const miinfer::GgufTensor& post_norm;
    const miinfer::GgufTensor& ffn_gate_weight;
    const miinfer::GgufTensor& ffn_up_weight;
    const miinfer::GgufTensor& ffn_down_weight;
    Buffer d_attn_norm, d_q, d_k, d_v, d_o, d_q_norm, d_k_norm, d_post;
    Buffer d_ffn_gate, d_ffn_up, d_ffn_down, d_ffn_down_expanded;
    Buffer normalized, qfull, query, gate, key, key_norm, value, query_norm;
    Buffer query_rope, key_rope, key_cache, value_cache, attention, scores, probabilities;
    Buffer gated_attention, projected, residual, post_normalized, ffn_gate, ffn_up;
    Buffer ffn_activation, layer_output, q8;
    Buffer prefill_normalized, prefill_qfull, prefill_value, prefill_q8_1;
    Buffer prefill_gated_attention, prefill_projected, prefill_residual, prefill_post_normalized;
    Buffer prefill_ffn_gate, prefill_ffn_up, prefill_ffn_activation, prefill_ffn_projected;
    Buffer prefill_query_rope, prefill_gate;
    bool reuse_projection_q8 = false;
    Buffer q8_1;
    bool q4_q8_1_mmvq = false;
    bool q4_q8_1_gate_up = false;
    bool q4_q8_1_lds_input = true;
    bool q4_q8_1_lds_metadata = true;
    bool q4_q8_1_lds_decoded_metadata = true;
    bool direct_layer_output = false;
    bool fused_gate_up_swiglu = false;
    bool fused_core_q8 = false;
    bool fused_norm_q8 = false;
    bool tiled_online_attention = true;
    bool fp16_kv_cache = true;
    bool fused_rope_norm = true;
    bool fused_add_rms_norm = true;
    bool fused_interlayer_norm = true;
    bool expanded_down = true;
    bool prefill_batch_enabled = false;
    std::size_t prefill_capacity = kPrefillBatch;
    Buffer d_ffn_down_native;
    Buffer d_ffn_gate_native, d_ffn_up_native;
    Buffer d_ffn_swiglu_native;
    Buffer d_q_native;
    Buffer d_o_native;
    Buffer d_k_native;
    Buffer d_v_native;
    Buffer d_qk_combined;
    Buffer d_qk_mmq, d_v_mmq, d_o_mmq;
    Buffer resident_qk_mmq, resident_v_mmq, resident_o_mmq;
    Buffer d_ffn_gate_mmq, d_ffn_up_mmq, d_ffn_down_mmq;
    Buffer resident_ffn_gate_mmq, resident_ffn_up_mmq, resident_ffn_down_mmq;
    Buffer decode_mmq_q8;
    bool resident_m23_ffn = false;
    bool resident_m23_all = false;
    std::vector<Q4KMmqTile> host_qk_mmq, host_v_q4_mmq, host_o_q4_mmq;
    std::vector<Q6KMmqTile> host_v_q6_mmq, host_o_q6_mmq;
    std::vector<Q4KMmqTile> host_ffn_gate_mmq, host_ffn_up_mmq, host_ffn_down_q4_mmq;
    std::vector<Q6KMmqTile> host_ffn_down_q6_mmq;
    Buffer prefill_mmq_q8;
    bool wide_attn_prefill = false;
    std::array<std::uint32_t, 9> m23_dispatch_counts{};
    bool m23_trace_dispatch = false;
    M23ProfileCounters* m23_profile_counters = nullptr;
    bool batch_head_rms = false;
    struct StageProfile {
        std::array<hipEvent_t, 15> start{};
        std::array<hipEvent_t, 15> end{};
        hipEvent_t tail_start = nullptr;
        hipEvent_t tail_end = nullptr;
        hipEvent_t prepare_start = nullptr;
        hipEvent_t prepare_end = nullptr;
        hipEvent_t ordered_start = nullptr;
        hipEvent_t ordered_end = nullptr;
        mutable bool tail_recorded = false;
        mutable bool prepare_recorded = false;
        mutable bool ordered_recorded = false;
        mutable std::array<bool, 15> stage_recorded{};
    };
    StageProfile* stage_profile = nullptr;
    std::uint32_t stage_profile_position = 0;
    std::size_t stage_profile_chunk_base = 0;

    void count_m23_dispatch(std::size_t family) {
        if (m23_trace_dispatch) ++m23_dispatch_counts[family];
        if (m23_profile_counters != nullptr) ++m23_profile_counters->attention_dispatches[family];
    }

    FullAttentionLayer(const miinfer::Qwen35Model& model_value, std::size_t layer)
        : model(model_value),
          attn_norm(tensor(*model.file(), prefix(layer, "attn_norm.weight"))),
          q_weight(tensor(*model.file(), prefix(layer, "attn_q.weight"))),
          k_weight(tensor(*model.file(), prefix(layer, "attn_k.weight"))),
          v_weight(tensor(*model.file(), prefix(layer, "attn_v.weight"))),
          o_weight(tensor(*model.file(), prefix(layer, "attn_output.weight"))),
          q_norm(tensor(*model.file(), prefix(layer, "attn_q_norm.weight"))),
          k_norm(tensor(*model.file(), prefix(layer, "attn_k_norm.weight"))),
          post_norm(tensor(*model.file(), prefix(layer, "post_attention_norm.weight"))),
          ffn_gate_weight(tensor(*model.file(), prefix(layer, "ffn_gate.weight"))),
          ffn_up_weight(tensor(*model.file(), prefix(layer, "ffn_up.weight"))),
          ffn_down_weight(tensor(*model.file(), prefix(layer, "ffn_down.weight"))) {
        const char* wide_attn_env = std::getenv("MIINFER_PREFILL_WIDE_ATTN");
        const char* layer_major_env = std::getenv("MIINFER_PREFILL_LAYER_MAJOR");
        wide_attn_prefill = layer_major_env != nullptr && std::strcmp(layer_major_env, "0") != 0
            && wide_attn_env != nullptr && std::strcmp(wide_attn_env, "0") != 0;
        const char* wide_repacked_env = std::getenv("MIINFER_PREFILL_WIDE_REPACKED_MMQ");
        const char* resident_ffn_env = std::getenv("MIINFER_PREFILL_REPACKED_RESIDENT_ATTN_FFN");
        resident_m23_ffn = wide_attn_prefill
            && wide_repacked_env != nullptr && std::strcmp(wide_repacked_env, "0") != 0
            && resident_ffn_env != nullptr && std::strcmp(resident_ffn_env, "0") != 0;
        const char* resident_all_env = std::getenv("MIINFER_PREFILL_REPACKED_RESIDENT_ALL");
        resident_m23_all = wide_attn_prefill
            && wide_repacked_env != nullptr && std::strcmp(wide_repacked_env, "0") != 0
            && resident_all_env != nullptr && std::strcmp(resident_all_env, "0") != 0
            && combined_attn_qk_enabled()
            && q_weight.type == miinfer::GgufTensorType::q4_k
            && k_weight.type == miinfer::GgufTensorType::q4_k;
        if (resident_m23_all) resident_m23_ffn = true;
        const char* trace_dispatch_env = std::getenv("MIINFER_PREFILL_TRACE_DISPATCH");
        m23_trace_dispatch = trace_dispatch_env != nullptr && std::strcmp(trace_dispatch_env, "0") != 0;
        for (const auto* weight : {&q_weight, &k_weight, &v_weight, &o_weight,
                                   &ffn_gate_weight, &ffn_up_weight, &ffn_down_weight}) {
            require_type(*weight, {miinfer::GgufTensorType::q4_k,
                                   miinfer::GgufTensorType::q6_k});
        }
        d_attn_norm = copy_weight(attn_norm);
        if (combined_attn_qk_enabled() &&
            q_weight.type == miinfer::GgufTensorType::q4_k &&
            k_weight.type == miinfer::GgufTensorType::q4_k) {
            if (!resident_m23_all) d_qk_combined = copy_combined_q4k_tensors(q_weight, k_weight);
            if (wide_attn_prefill) {
                const auto q = pack_q4k_mmq_tensor(q_weight);
                const auto k = pack_q4k_mmq_tensor(k_weight);
                host_qk_mmq.reserve(q.size() + k.size());
                host_qk_mmq.insert(host_qk_mmq.end(), q.begin(), q.end());
                host_qk_mmq.insert(host_qk_mmq.end(), k.begin(), k.end());
            }
        } else if (!resident_m23_all) {
            if (native_q_enabled() && q_weight.type == miinfer::GgufTensorType::q4_k) {
                d_q_native = copy_native_tensor(q_weight);
            } else {
                d_q = copy_weight(q_weight);
            }
            if (native_k_enabled() && k_weight.type == miinfer::GgufTensorType::q4_k) {
                d_k_native = copy_native_tensor(k_weight);
            } else {
                d_k = copy_weight(k_weight);
            }
        }
        if (!resident_m23_all && native_v_enabled()) {
            if (v_weight.type == miinfer::GgufTensorType::q4_k) {
                d_v_native = copy_native_tensor(v_weight);
            } else if (v_weight.type == miinfer::GgufTensorType::q6_k) {
                d_v_native = copy_native_q6k_tensor(v_weight);
            }
        }
        if (!resident_m23_all && !d_v_native) {
            d_v = copy_weight(v_weight);
        }
        if (wide_attn_prefill && !host_qk_mmq.empty() && v_weight.type == miinfer::GgufTensorType::q4_k) {
            host_v_q4_mmq = pack_q4k_mmq_tensor(v_weight);
        } else if (wide_attn_prefill && !host_qk_mmq.empty() && v_weight.type == miinfer::GgufTensorType::q6_k) {
            host_v_q6_mmq = pack_q6k_mmq_tensor(v_weight);
        }
        if (!resident_m23_all && native_attn_out_enabled() && o_weight.type == miinfer::GgufTensorType::q4_k) {
            d_o_native = copy_native_tensor(o_weight);
        } else if (!resident_m23_all) {
            d_o = copy_weight(o_weight);
        }
        if (wide_attn_prefill && o_weight.type == miinfer::GgufTensorType::q4_k) {
            host_o_q4_mmq = pack_q4k_mmq_tensor(o_weight);
        } else if (wide_attn_prefill && o_weight.type == miinfer::GgufTensorType::q6_k) {
            host_o_q6_mmq = pack_q6k_mmq_tensor(o_weight);
        }
        d_q_norm = copy_weight(q_norm); d_k_norm = copy_weight(k_norm);
        d_post = copy_weight(post_norm);
        if (!resident_m23_all && swiglu_paired_enabled() && ffn_gate_weight.type == miinfer::GgufTensorType::q4_k
            && ffn_up_weight.type == miinfer::GgufTensorType::q4_k) {
            d_ffn_swiglu_native = copy_swiglu_paired_tensors(ffn_gate_weight, ffn_up_weight);
        } else if (!resident_m23_all && native_gate_up_enabled() && ffn_gate_weight.type == miinfer::GgufTensorType::q4_k
            && ffn_up_weight.type == miinfer::GgufTensorType::q4_k) {
            d_ffn_gate_native = copy_native_tensor(ffn_gate_weight);
            d_ffn_up_native = copy_native_tensor(ffn_up_weight);
        } else if (!resident_m23_all) {
            d_ffn_gate = copy_weight(ffn_gate_weight);
            d_ffn_up = copy_weight(ffn_up_weight);
        }
        if (wide_attn_prefill && ffn_gate_weight.type == miinfer::GgufTensorType::q4_k
            && ffn_up_weight.type == miinfer::GgufTensorType::q4_k) {
            host_ffn_gate_mmq = pack_q4k_mmq_tensor(ffn_gate_weight);
            host_ffn_up_mmq = pack_q4k_mmq_tensor(ffn_up_weight);
        }
        const char* expanded_down_env = std::getenv("MIINFER_Q4K_EXPANDED_DOWN");
        expanded_down = expanded_down_env == nullptr
            || std::strcmp(expanded_down_env, "0") != 0;
        if (!resident_m23_all && ((native_down_enabled() && ffn_down_weight.type == miinfer::GgufTensorType::q4_k) ||
            (native_q6k_down_enabled() && ffn_down_weight.type == miinfer::GgufTensorType::q6_k))) {
            if (ffn_down_weight.type == miinfer::GgufTensorType::q4_k) {
                d_ffn_down_native = copy_native_down(ffn_down_weight);
            } else {
                d_ffn_down_native = copy_native_q6k_tensor(ffn_down_weight);
            }
        } else if (!resident_m23_all && expanded_down && ffn_down_weight.type == miinfer::GgufTensorType::q4_k) {
            d_ffn_down_expanded = copy_expanded_q4k(ffn_down_weight);
        }
        if (!resident_m23_all && !d_ffn_down_native) {
            d_ffn_down = copy_weight(ffn_down_weight);
        }
        if (wide_attn_prefill && ffn_down_weight.type == miinfer::GgufTensorType::q4_k) {
            host_ffn_down_q4_mmq = pack_q4k_mmq_tensor(ffn_down_weight);
        } else if (wide_attn_prefill && ffn_down_weight.type == miinfer::GgufTensorType::q6_k) {
            host_ffn_down_q6_mmq = pack_q6k_mmq_tensor(ffn_down_weight);
        }
        resident_m23_ffn = resident_m23_ffn && !host_ffn_gate_mmq.empty()
            && !host_ffn_up_mmq.empty()
            && (!host_ffn_down_q4_mmq.empty() || !host_ffn_down_q6_mmq.empty());
        if (resident_m23_ffn) {
            const auto upload_resident = [](const auto& packed) {
                auto result = allocate(packed.size() * sizeof(packed.front()));
                upload(packed.data(), result->get(), packed.size() * sizeof(packed.front()));
                return result;
            };
            resident_ffn_gate_mmq = upload_resident(host_ffn_gate_mmq);
            resident_ffn_up_mmq = upload_resident(host_ffn_up_mmq);
            resident_ffn_down_mmq = ffn_down_weight.type == miinfer::GgufTensorType::q4_k
                ? upload_resident(host_ffn_down_q4_mmq)
                : upload_resident(host_ffn_down_q6_mmq);
        }
        if (resident_m23_all) {
            const auto upload_resident = [](const auto& packed) {
                auto result = allocate(packed.size() * sizeof(packed.front()));
                upload(packed.data(), result->get(), packed.size() * sizeof(packed.front()));
                return result;
            };
            resident_qk_mmq = upload_resident(host_qk_mmq);
            resident_v_mmq = v_weight.type == miinfer::GgufTensorType::q4_k
                ? upload_resident(host_v_q4_mmq)
                : upload_resident(host_v_q6_mmq);
            resident_o_mmq = o_weight.type == miinfer::GgufTensorType::q4_k
                ? upload_resident(host_o_q4_mmq)
                : upload_resident(host_o_q6_mmq);
            decode_mmq_q8 = allocate((kFfnInner / 128) * sizeof(miinfer::M23Q8_1MmqBlock));
        }
        const char* reuse_projection_q8_env = std::getenv("MIINFER_REUSE_PROJECTION_Q8");
        reuse_projection_q8 = reuse_projection_q8_env == nullptr
            || std::strcmp(reuse_projection_q8_env, "0") != 0;
        const char* q4_q8_1_mmvq_env = std::getenv("MIINFER_Q4K_Q8_1_MMVQ");
        q4_q8_1_mmvq = q4_q8_1_mmvq_env == nullptr
            || std::strcmp(q4_q8_1_mmvq_env, "0") != 0;
        const char* q4_q8_1_gate_up_env = std::getenv("MIINFER_Q4K_Q8_1_MMVQ_FFN_GATE_UP");
        q4_q8_1_gate_up = q4_q8_1_gate_up_env == nullptr
            || std::strcmp(q4_q8_1_gate_up_env, "0") != 0;
        const char* fused_gate_up_swiglu_env = std::getenv("MIINFER_FUSED_GATE_UP_SWIGLU");
        fused_gate_up_swiglu = fused_gate_up_swiglu_env == nullptr
            || std::strcmp(fused_gate_up_swiglu_env, "0") != 0;
        const char* fused_core_q8_env = std::getenv("MIINFER_FUSED_CORE_Q8");
        fused_core_q8 = fused_core_q8_env == nullptr
            || std::strcmp(fused_core_q8_env, "0") != 0;
        const char* fused_norm_q8_env = std::getenv("MIINFER_FUSED_NORM_Q8");
        fused_norm_q8 = fused_norm_q8_env != nullptr
            && std::strcmp(fused_norm_q8_env, "0") != 0;
        const char* tiled_online_attention_env = std::getenv("MIINFER_TILED_ONLINE_ATTENTION");
        tiled_online_attention = tiled_online_attention_env == nullptr
            || std::strcmp(tiled_online_attention_env, "0") != 0;
        const char* fp16_kv_cache_env = std::getenv("MIINFER_FP16_KV_CACHE");
        fp16_kv_cache = fp16_kv_cache_env == nullptr
            || std::strcmp(fp16_kv_cache_env, "0") != 0;
        const char* direct_layer_output_env = std::getenv("MIINFER_DIRECT_LAYER_OUTPUT");
        const char* hip_graph_env = std::getenv("MIINFER_HIP_GRAPH");
        const bool use_hip_graph = hip_graph_env != nullptr && std::strcmp(hip_graph_env, "0") != 0;
        direct_layer_output = (direct_layer_output_env != nullptr
            && std::strcmp(direct_layer_output_env, "0") != 0) || use_hip_graph;
        const char* q4_q8_1_lds_input_env = std::getenv("MIINFER_Q4K_Q8_1_LDS_INPUT");
        q4_q8_1_lds_input = q4_q8_1_lds_input_env == nullptr
            || std::strcmp(q4_q8_1_lds_input_env, "0") != 0;
        const char* q4_q8_1_lds_metadata_env = std::getenv("MIINFER_Q4K_Q8_1_LDS_METADATA");
        q4_q8_1_lds_metadata = q4_q8_1_lds_metadata_env == nullptr
            || std::strcmp(q4_q8_1_lds_metadata_env, "0") != 0;
        const char* q4_q8_1_lds_decoded_metadata_env =
            std::getenv("MIINFER_Q4K_Q8_1_LDS_DECODED_METADATA");
        q4_q8_1_lds_decoded_metadata = q4_q8_1_lds_decoded_metadata_env == nullptr
            || std::strcmp(q4_q8_1_lds_decoded_metadata_env, "0") != 0;
        const char* batch_head_rms_env = std::getenv("MIINFER_BATCH_HEAD_RMS");
        batch_head_rms = batch_head_rms_env == nullptr
            || std::strcmp(batch_head_rms_env, "0") != 0;
        const char* fused_rope_norm_env = std::getenv("MIINFER_FUSED_ROPE_NORM");
        fused_rope_norm = fused_rope_norm_env == nullptr
            || std::strcmp(fused_rope_norm_env, "0") != 0;
        const char* fused_add_rms_norm_env = std::getenv("MIINFER_FUSED_ADD_RMS_NORM");
        fused_add_rms_norm = fused_add_rms_norm_env == nullptr
            || std::strcmp(fused_add_rms_norm_env, "0") != 0;
        const char* fused_interlayer_norm_env = std::getenv("MIINFER_FUSED_INTERLAYER_NORM");
        fused_interlayer_norm = fused_interlayer_norm_env == nullptr
            || std::strcmp(fused_interlayer_norm_env, "0") != 0;
        const char* prefill_batch_env = std::getenv("MIINFER_PREFILL_LAYER_MAJOR");
        prefill_batch_enabled = prefill_batch_env != nullptr
            && std::strcmp(prefill_batch_env, "0") != 0;
        const char* prefill_gdn_env = std::getenv("MIINFER_PREFILL_GDN_CHUNKWISE");
        const char* prefill_dense_env = std::getenv("MIINFER_PREFILL_DENSE_FFN_DOWN");
        const char* wide_prefill_env = std::getenv("MIINFER_PREFILL_WIDE_CHUNK");
        const bool wide_prefill = prefill_batch_enabled && wide_prefill_env != nullptr
            && std::strcmp(wide_prefill_env, "0") != 0;
        prefill_capacity = prefill_batch_enabled && prefill_dense_env != nullptr
            && std::strcmp(prefill_dense_env, "0") != 0 ? kM12PrefillBatch
            : (wide_prefill ? configured_wide_prefill_batch()
               : (prefill_batch_enabled && prefill_gdn_env != nullptr
                  && std::strcmp(prefill_gdn_env, "0") != 0) ? kM12PrefillBatch : kPrefillBatch);
        normalized = allocate(kHidden * sizeof(float));
        qfull = allocate(((d_qk_combined || resident_m23_all) ? (12288 + 1024) : 12288)
                         * sizeof(float));
        query = allocate(6144 * sizeof(float)); gate = allocate(6144 * sizeof(float));
        if (!d_qk_combined) {
            key = allocate(1024 * sizeof(float));
        }
        key_norm = allocate(1024 * sizeof(float));
        value = allocate(1024 * sizeof(float)); query_norm = allocate(6144 * sizeof(float));
        query_rope = allocate(6144 * sizeof(float)); key_rope = allocate(1024 * sizeof(float));
        const std::size_t kv_element_size = fp16_kv_cache ? sizeof(__half) : sizeof(float);
        key_cache = allocate(4 * g_cache_capacity * 256 * kv_element_size);
        value_cache = allocate(4 * g_cache_capacity * 256 * kv_element_size);
        attention = allocate(6144 * sizeof(float));
        if (!tiled_online_attention) {
            scores = allocate(24 * g_cache_capacity * sizeof(float));
            probabilities = allocate(24 * g_cache_capacity * sizeof(float));
        }
        gated_attention = allocate(6144 * sizeof(float)); projected = allocate(kHidden * sizeof(float));
        residual = allocate(kHidden * sizeof(float)); post_normalized = allocate(kHidden * sizeof(float));
        ffn_gate = allocate(kFfnInner * sizeof(float)); ffn_up = allocate(kFfnInner * sizeof(float));
        ffn_activation = allocate(kFfnInner * sizeof(float)); layer_output = allocate(kHidden * sizeof(float));
        q8 = allocate((kFfnInner / 256) * sizeof(miinfer::Q8KDeviceBlock));
        q8_1 = allocate((kFfnInner / miinfer::kQ8_1BlockSize) * sizeof(miinfer::Q8_1Block));
        if (prefill_batch_enabled && !shared_wide_prefill_enabled()) {
            prefill_normalized = allocate(prefill_capacity * kHidden * sizeof(float));
            prefill_qfull = allocate(prefill_capacity * (12288 + 1024) * sizeof(float));
            prefill_value = allocate(prefill_capacity * 1024 * sizeof(float));
            prefill_q8_1 = allocate(prefill_capacity * (kFfnInner / miinfer::kQ8_1BlockSize)
                                    * sizeof(miinfer::Q8_1Block));
            prefill_gated_attention = allocate(prefill_capacity * kInner * sizeof(float));
            prefill_projected = allocate(prefill_capacity * kHidden * sizeof(float));
            prefill_residual = allocate(prefill_capacity * kHidden * sizeof(float));
            prefill_post_normalized = allocate(prefill_capacity * kHidden * sizeof(float));
            prefill_ffn_gate = allocate(prefill_capacity * kFfnInner * sizeof(float));
            prefill_ffn_up = allocate(prefill_capacity * kFfnInner * sizeof(float));
            prefill_ffn_activation = allocate(prefill_capacity * kFfnInner * sizeof(float));
            prefill_ffn_projected = allocate(prefill_capacity * kHidden * sizeof(float));
            if (wide_attn_prefill && !host_qk_mmq.empty()
                && (!host_v_q4_mmq.empty() || !host_v_q6_mmq.empty())) {
                const std::size_t max_mmq_elements = std::max({kHidden, kInner, kFfnInner});
                prefill_mmq_q8 = allocate(prefill_capacity * (max_mmq_elements / 128)
                                          * sizeof(miinfer::M23Q8_1MmqBlock));
                prefill_query_rope = allocate(prefill_capacity * 6144 * sizeof(float));
                prefill_gate = allocate(prefill_capacity * 6144 * sizeof(float));
            }
        }
        MIINFER_HIP_CHECK(hipMemset(key_cache->get(), 0, 4 * g_cache_capacity * 256 * kv_element_size));
        MIINFER_HIP_CHECK(hipMemset(value_cache->get(), 0, 4 * g_cache_capacity * 256 * kv_element_size));
    }

    void ensure_m23_repacked() {
        if (!wide_attn_prefill || (d_qk_mmq && d_v_mmq && d_o_mmq
                                   && d_ffn_gate_mmq && d_ffn_up_mmq && d_ffn_down_mmq)
            || host_qk_mmq.empty()) return;
        const auto upload_packed = [this](const auto& packed, std::size_t slot) -> Buffer {
            const std::size_t bytes = packed.size() * sizeof(packed.front());
            if (!g_m23_repacked_scratch[slot] || g_m23_repacked_scratch_bytes[slot] < bytes) {
                g_m23_repacked_scratch[slot] = allocate(bytes);
                g_m23_repacked_scratch_bytes[slot] = bytes;
            }
            const auto upload_start = std::chrono::steady_clock::now();
            upload(packed.data(), g_m23_repacked_scratch[slot]->get(), bytes);
            const double upload_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - upload_start).count();
            if (m23_profile_counters != nullptr) {
                m23_profile_counters->weight_upload_bytes += bytes;
                m23_profile_counters->attention_weight_upload_bytes[slot] += bytes;
                m23_profile_counters->weight_upload_ms += upload_ms;
                m23_profile_counters->attention_weight_upload_ms[slot] += upload_ms;
            }
            return g_m23_repacked_scratch[slot];
        };
        if (!d_qk_mmq) d_qk_mmq = resident_m23_all
            ? resident_qk_mmq : upload_packed(host_qk_mmq, 0);
        if (!d_v_mmq) {
            d_v_mmq = resident_m23_all
                ? resident_v_mmq
                : (!host_v_q4_mmq.empty() ? upload_packed(host_v_q4_mmq, 1)
                                           : upload_packed(host_v_q6_mmq, 1));
        }
        if (!d_o_mmq) {
            d_o_mmq = resident_m23_all
                ? resident_o_mmq
                : (!host_o_q4_mmq.empty() ? upload_packed(host_o_q4_mmq, 2)
                                           : upload_packed(host_o_q6_mmq, 2));
        }
        if (!d_ffn_gate_mmq) {
            d_ffn_gate_mmq = resident_m23_ffn ? resident_ffn_gate_mmq
                                              : upload_packed(host_ffn_gate_mmq, 3);
        }
        if (!d_ffn_up_mmq) {
            d_ffn_up_mmq = resident_m23_ffn ? resident_ffn_up_mmq
                                            : upload_packed(host_ffn_up_mmq, 4);
        }
        if (!d_ffn_down_mmq) {
            d_ffn_down_mmq = resident_m23_ffn ? resident_ffn_down_mmq
                                              : (!host_ffn_down_q4_mmq.empty()
                                                     ? upload_packed(host_ffn_down_q4_mmq, 5)
                                                     : upload_packed(host_ffn_down_q6_mmq, 5));
        }
    }

    void release_m23_repacked() {
        if (!resident_m23_all) {
            d_qk_mmq.reset();
            d_v_mmq.reset();
            d_o_mmq.reset();
        }
        if (!resident_m23_ffn) {
            d_ffn_gate_mmq.reset();
            d_ffn_up_mmq.reset();
            d_ffn_down_mmq.reset();
        }
    }

    static std::string prefix(std::size_t layer, const char* suffix) {
        return "blk." + std::to_string(layer) + "." + suffix;
    }
    Buffer copy_weight(const miinfer::GgufTensor& weight) {
        auto result = allocate(weight.byte_size);
        upload_tensor(weight, result);
        return result;
    }

    void poison() {
        const std::size_t kv_element_size = fp16_kv_cache ? sizeof(__half) : sizeof(float);
        MIINFER_HIP_CHECK(hipMemset(key_cache->get(), 0xA5,
                                    4 * g_cache_capacity * 256 * kv_element_size));
        MIINFER_HIP_CHECK(hipMemset(value_cache->get(), 0xA5,
                                    4 * g_cache_capacity * 256 * kv_element_size));
    }

    void reset() {
        const std::size_t kv_element_size = fp16_kv_cache ? sizeof(__half) : sizeof(float);
        MIINFER_HIP_CHECK(hipMemset(key_cache->get(), 0,
                                    4 * g_cache_capacity * 256 * kv_element_size));
        MIINFER_HIP_CHECK(hipMemset(value_cache->get(), 0,
                                    4 * g_cache_capacity * 256 * kv_element_size));
    }

    void stage_start(std::size_t stage, std::uint32_t position) const {
        if (stage_profile != nullptr && stage_profile_position == position) {
            MIINFER_HIP_CHECK(hipEventRecord(stage_profile->start[stage], hipStreamPerThread));
        }
    }

    void stage_end(std::size_t stage, std::uint32_t position) const {
        if (stage_profile != nullptr && stage_profile_position == position) {
            stage_profile->stage_recorded[stage] = true;
            MIINFER_HIP_CHECK(hipEventRecord(stage_profile->end[stage], hipStreamPerThread));
        }
    }

    void profile_tail_start(std::size_t chunk_base) const {
        if (stage_profile != nullptr && stage_profile_chunk_base == chunk_base) {
            stage_profile->tail_recorded = true;
            MIINFER_HIP_CHECK(hipEventRecord(stage_profile->tail_start, hipStreamPerThread));
        }
    }

    void profile_tail_end(std::size_t chunk_base) const {
        if (stage_profile != nullptr && stage_profile_chunk_base == chunk_base) {
            MIINFER_HIP_CHECK(hipEventRecord(stage_profile->tail_end, hipStreamPerThread));
        }
    }

    void profile_prepare_start(std::size_t chunk_base) const {
        if (stage_profile != nullptr && stage_profile_chunk_base == chunk_base) {
            stage_profile->prepare_recorded = true;
            MIINFER_HIP_CHECK(hipEventRecord(stage_profile->prepare_start, hipStreamPerThread));
        }
    }

    void profile_prepare_end(std::size_t chunk_base) const {
        if (stage_profile != nullptr && stage_profile_chunk_base == chunk_base) {
            MIINFER_HIP_CHECK(hipEventRecord(stage_profile->prepare_end, hipStreamPerThread));
        }
    }

    void profile_ordered_start(std::size_t chunk_base) const {
        if (stage_profile != nullptr && stage_profile_chunk_base == chunk_base) {
            stage_profile->ordered_recorded = true;
            MIINFER_HIP_CHECK(hipEventRecord(stage_profile->ordered_start, hipStreamPerThread));
        }
    }

    void profile_ordered_end(std::size_t chunk_base) const {
        if (stage_profile != nullptr && stage_profile_chunk_base == chunk_base) {
            MIINFER_HIP_CHECK(hipEventRecord(stage_profile->ordered_end, hipStreamPerThread));
        }
    }

    bool prepare_prefill_batch(const float* inputs, std::size_t count,
                               bool normalized_ready = false) {
        ensure_m23_repacked();
        if (!prefill_batch_enabled
            || (count != kPrefillBatch && count != prefill_capacity)
            || inputs == nullptr
            || (d_qk_combined == nullptr && d_qk_mmq == nullptr)
            || (d_v_native == nullptr
                && !(wide_attn_prefill && count >= kM12PrefillBatch
                     && count <= prefill_capacity && count % kPrefillBatch == 0
                     && d_v_mmq && prefill_mmq_q8))) {
            return false;
        }
        auto* normalized_out = static_cast<float*>(prefill_normalized->get());
        auto* q8_out = static_cast<miinfer::Q8_1Block*>(prefill_q8_1->get());
        if (wide_attn_prefill && count >= kM12PrefillBatch && count <= prefill_capacity
            && count % kPrefillBatch == 0 && d_qk_mmq && d_v_mmq && prefill_mmq_q8) {
            if (m23_trace_dispatch) m23_dispatch_counts.fill(0);
            if (!normalized_ready) {
                miinfer::launch_qwen3_rms_norm_batch(
                    inputs, static_cast<const float*>(d_attn_norm->get()), normalized_out,
                    static_cast<std::uint32_t>(count), kHidden, model.config().rms_epsilon,
                    hipStreamPerThread);
            }
            auto* mmq_q8 = static_cast<miinfer::M23Q8_1MmqBlock*>(prefill_mmq_q8->get());
            miinfer::launch_m23_q8_1_mmq_quantize(
                normalized_out, mmq_q8, static_cast<std::uint32_t>(count), kHidden,
                hipStreamPerThread);
            stage_start(1, stage_profile_position);
            launch_m23_q4k_repacked_mmq(
                static_cast<const Q4KMmqTile*>(d_qk_mmq->get()), mmq_q8,
                static_cast<float*>(prefill_qfull->get()), 12288 + 1024, kHidden,
                static_cast<std::uint32_t>(count), hipStreamPerThread);
            stage_end(1, stage_profile_position);
            count_m23_dispatch(0);
            stage_start(4, stage_profile_position);
            if (v_weight.type == miinfer::GgufTensorType::q4_k) {
                launch_m23_q4k_repacked_mmq(
                    static_cast<const Q4KMmqTile*>(d_v_mmq->get()), mmq_q8,
                    static_cast<float*>(prefill_value->get()), 1024, kHidden,
                    static_cast<std::uint32_t>(count), hipStreamPerThread);
            } else {
                launch_m23_q6k_repacked_mmq(
                    static_cast<const Q6KMmqTile*>(d_v_mmq->get()), mmq_q8,
                    static_cast<float*>(prefill_value->get()), 1024, kHidden,
                    static_cast<std::uint32_t>(count), hipStreamPerThread);
            }
            stage_end(4, stage_profile_position);
            count_m23_dispatch(1);
            return true;
        }
        for (std::size_t i = 0; i < count; ++i) {
            if (!normalized_ready) {
                miinfer::launch_qwen3_rms_norm(
                    inputs + i * kHidden, static_cast<const float*>(d_attn_norm->get()),
                    normalized_out + i * kHidden, kHidden, model.config().rms_epsilon);
            }
            miinfer::launch_q8_1_quantize_f32(
                normalized_out + i * kHidden,
                q8_out + i * (kHidden / miinfer::kQ8_1BlockSize), kHidden,
                hipStreamPerThread);
        }
        for (std::size_t base = 0; base < count; base += 4) {
            auto* group_q8 = q8_out + base * (kHidden / miinfer::kQ8_1BlockSize);
            auto* group_qfull = static_cast<float*>(prefill_qfull->get()) + base * (12288 + 1024);
            if (d_qk_combined) {
                launch_q4k_wave_gemv_batched4(
                    static_cast<const Q4KWaveTile*>(d_qk_combined->get()), group_q8,
                    group_qfull, 12288 + 1024, kHidden, hipStreamPerThread);
            }
            if (v_weight.type == miinfer::GgufTensorType::q4_k) {
                launch_q4k_wave_gemv_batched4(
                    static_cast<const Q4KWaveTile*>(d_v_native->get()), group_q8,
                    static_cast<float*>(prefill_value->get()) + base * 1024, 1024, kHidden,
                    hipStreamPerThread);
            } else {
                launch_q6k_wave_gemv_batched4(
                    static_cast<const Q6KWaveTile*>(d_v_native->get()), group_q8,
                    static_cast<float*>(prefill_value->get()) + base * 1024, 1024, kHidden,
                    hipStreamPerThread);
            }
        }
        return true;
    }

    const float* prefill_normalized_at(std::size_t index) const {
        return static_cast<const float*>(prefill_normalized->get()) + index * kHidden;
    }

    const float* prefill_qfull_at(std::size_t index) const {
        return static_cast<const float*>(prefill_qfull->get()) + index * (12288 + 1024);
    }

    const float* prefill_value_at(std::size_t index) const {
        return static_cast<const float*>(prefill_value->get()) + index * 1024;
    }

    bool prefill_tail_batch_supported() const {
        const char* env = std::getenv("MIINFER_PREFILL_ATTN_TAIL");
        const bool enabled = env == nullptr || std::strcmp(env, "0") != 0;
        if (enabled && wide_attn_prefill && d_o_mmq && d_ffn_gate_mmq && d_ffn_up_mmq
            && d_ffn_down_mmq && prefill_mmq_q8) {
            return true;
        }
        return enabled && prefill_batch_enabled && d_o_native && d_ffn_down_native
            && (d_ffn_swiglu_native || (d_ffn_gate_native && d_ffn_up_native));
    }

    bool wide_attention_batch_ready() const {
        return wide_attn_prefill && d_qk_mmq && d_v_mmq && prefill_mmq_q8
            && prefill_query_rope && prefill_gate;
    }

    void finish_prefill_attention(std::uint32_t base_position, std::uint32_t count) {
        if (!wide_attention_batch_ready() || count == 0 || count > prefill_capacity
            || base_position > g_cache_capacity || count > g_cache_capacity - base_position) {
            throw std::runtime_error("invalid wide attention prefill batch");
        }
        auto* q = static_cast<float*>(prefill_query_rope->get());
        auto* gate = static_cast<float*>(prefill_gate->get());
        auto* output = static_cast<float*>(prefill_gated_attention->get());
        const auto* qfull = static_cast<const float*>(prefill_qfull->get());
        const auto* value = static_cast<const float*>(prefill_value->get());
        const auto profile_position = base_position + count - 1;
        stage_start(2, profile_position);
        miinfer::launch_qwen35_fused_q_split_norm_rope_batch(
            qfull, static_cast<const float*>(d_q_norm->get()),
            q, gate, count, base_position,
            24, 4, 256, model.config().rope_theta, model.config().rms_epsilon,
            hipStreamPerThread);
        stage_end(2, profile_position);
        count_m23_dispatch(2);
        stage_start(3, profile_position);
        if (fp16_kv_cache) {
            miinfer::launch_qwen35_fused_k_norm_rope_kv_store_batch_f16(
                qfull, value, static_cast<const float*>(d_k_norm->get()),
                static_cast<__half*>(key_cache->get()), static_cast<__half*>(value_cache->get()),
                count, base_position, g_cache_capacity, 24, 4, 256,
                model.config().rope_theta, model.config().rms_epsilon, hipStreamPerThread);
        } else {
            miinfer::launch_qwen35_fused_k_norm_rope_kv_store_batch(
                qfull, value, static_cast<const float*>(d_k_norm->get()),
                static_cast<float*>(key_cache->get()), static_cast<float*>(value_cache->get()),
                count, base_position, g_cache_capacity, 24, 4, 256,
                model.config().rope_theta, model.config().rms_epsilon, hipStreamPerThread);
        }
        stage_end(3, profile_position);
        count_m23_dispatch(3);
        stage_start(6, profile_position);
        if (fp16_kv_cache) {
            miinfer::launch_qwen35_tiled_online_attention_batch_f16(
                q, static_cast<const __half*>(key_cache->get()),
                static_cast<const __half*>(value_cache->get()), gate, output, count,
                base_position, g_cache_capacity, 24, 4, 256,
                1.0F / std::sqrt(256.0F), hipStreamPerThread);
        } else {
            miinfer::launch_qwen35_tiled_online_attention_batch(
                q, static_cast<const float*>(key_cache->get()),
                static_cast<const float*>(value_cache->get()), gate, output, count,
                base_position, g_cache_capacity, 24, 4, 256,
                1.0F / std::sqrt(256.0F), hipStreamPerThread);
        }
        stage_end(6, profile_position);
        count_m23_dispatch(4);
    }

    void finish_prefill_wide(const float* inputs, float* outputs, std::size_t count,
                             const float* next_norm_weight, float* next_normalized) {
        auto* q8 = static_cast<miinfer::M23Q8_1MmqBlock*>(prefill_mmq_q8->get());
        auto* gated = static_cast<const float*>(prefill_gated_attention->get());
        miinfer::launch_m23_q8_1_mmq_quantize(
            gated, q8, static_cast<std::uint32_t>(count), kInner, hipStreamPerThread);
        const auto profile_position = stage_profile_position;
        stage_start(7, profile_position);
        if (o_weight.type == miinfer::GgufTensorType::q4_k) {
            launch_m23_q4k_repacked_mmq(
                static_cast<const Q4KMmqTile*>(d_o_mmq->get()), q8,
                static_cast<float*>(prefill_projected->get()), kHidden, kInner,
                static_cast<std::uint32_t>(count), hipStreamPerThread);
        } else {
            launch_m23_q6k_repacked_mmq(
                static_cast<const Q6KMmqTile*>(d_o_mmq->get()), q8,
                static_cast<float*>(prefill_projected->get()), kHidden, kInner,
                static_cast<std::uint32_t>(count), hipStreamPerThread);
        }
        stage_end(7, profile_position);
        count_m23_dispatch(5);
        auto* residual_batch = static_cast<float*>(prefill_residual->get());
        auto* normalized_batch = static_cast<float*>(prefill_post_normalized->get());
        auto* projected_batch = static_cast<float*>(prefill_projected->get());
        stage_start(8, profile_position);
        miinfer::launch_qwen3_fused_add_rms_norm_batch(
            inputs, projected_batch, static_cast<const float*>(d_post->get()),
            residual_batch, normalized_batch, static_cast<std::uint32_t>(count), kHidden,
            model.config().rms_epsilon, hipStreamPerThread);
        stage_end(8, profile_position);

        stage_start(10, profile_position);
        miinfer::launch_m23_q8_1_mmq_quantize(
            normalized_batch, q8, static_cast<std::uint32_t>(count), kHidden,
            hipStreamPerThread);
        launch_m23_q4k_repacked_mmq(
            static_cast<const Q4KMmqTile*>(d_ffn_gate_mmq->get()), q8,
            static_cast<float*>(prefill_ffn_gate->get()), kFfnInner, kHidden,
            static_cast<std::uint32_t>(count), hipStreamPerThread);
        launch_m23_q4k_repacked_mmq(
            static_cast<const Q4KMmqTile*>(d_ffn_up_mmq->get()), q8,
            static_cast<float*>(prefill_ffn_up->get()), kFfnInner, kHidden,
            static_cast<std::uint32_t>(count), hipStreamPerThread);
        stage_end(10, profile_position);
        count_m23_dispatch(6);
        count_m23_dispatch(7);
        stage_start(11, profile_position);
        miinfer::launch_qwen3_silu_mul(
            static_cast<const float*>(prefill_ffn_gate->get()),
            static_cast<const float*>(prefill_ffn_up->get()),
            static_cast<float*>(prefill_ffn_activation->get()),
            static_cast<std::uint32_t>(count * kFfnInner), hipStreamPerThread);
        stage_end(11, profile_position);
        stage_start(12, profile_position);
        miinfer::launch_m23_q8_1_mmq_quantize(
            static_cast<const float*>(prefill_ffn_activation->get()), q8,
            static_cast<std::uint32_t>(count), kFfnInner, hipStreamPerThread);
        if (ffn_down_weight.type == miinfer::GgufTensorType::q4_k) {
            launch_m23_q4k_repacked_mmq(
                static_cast<const Q4KMmqTile*>(d_ffn_down_mmq->get()), q8,
                static_cast<float*>(prefill_ffn_projected->get()), kHidden, kFfnInner,
                static_cast<std::uint32_t>(count), hipStreamPerThread);
        } else {
            launch_m23_q6k_repacked_mmq(
                static_cast<const Q6KMmqTile*>(d_ffn_down_mmq->get()), q8,
                static_cast<float*>(prefill_ffn_projected->get()), kHidden, kFfnInner,
                static_cast<std::uint32_t>(count), hipStreamPerThread);
        }
        stage_end(12, profile_position);
        count_m23_dispatch(8);
        stage_start(13, profile_position);
        if (next_norm_weight != nullptr && next_normalized != nullptr) {
            miinfer::launch_qwen3_fused_add_rms_norm_batch(
                residual_batch, static_cast<const float*>(prefill_ffn_projected->get()),
                next_norm_weight, outputs, next_normalized,
                static_cast<std::uint32_t>(count), kHidden, model.config().rms_epsilon,
                hipStreamPerThread);
        } else {
            miinfer::launch_qwen3_add(
                residual_batch, static_cast<const float*>(prefill_ffn_projected->get()),
                outputs, static_cast<std::uint32_t>(count * kHidden), hipStreamPerThread);
        }
        stage_end(13, profile_position);
        if (m23_trace_dispatch) {
            std::cerr << "M23 attention dispatches: qk=" << m23_dispatch_counts[0]
                      << " v=" << m23_dispatch_counts[1]
                      << " q_post=" << m23_dispatch_counts[2]
                      << " k_post=" << m23_dispatch_counts[3]
                      << " attention=" << m23_dispatch_counts[4]
                      << " o=" << m23_dispatch_counts[5]
                      << " ffn_gate=" << m23_dispatch_counts[6]
                      << " ffn_up=" << m23_dispatch_counts[7]
                      << " ffn_down=" << m23_dispatch_counts[8] << '\n';
        }
    }

    void finish_prefill_batch4(const float* inputs, float* outputs, std::size_t count,
                               std::size_t offset,
                               const float* next_norm_weight = nullptr,
                               float* next_normalized = nullptr) {
        if (!prefill_tail_batch_supported() || inputs == nullptr || outputs == nullptr
            || count != 4) {
            throw std::runtime_error("invalid attention prefill batch tail");
        }
        auto* q8_batch = static_cast<miinfer::Q8_1Block*>(prefill_q8_1->get())
            + offset * (kFfnInner / miinfer::kQ8_1BlockSize);
        auto* projection_batch = static_cast<float*>(prefill_projected->get()) + offset * kHidden;
        auto* residual_batch = static_cast<float*>(prefill_residual->get()) + offset * kHidden;
        auto* normalized_batch = static_cast<float*>(prefill_post_normalized->get()) + offset * kHidden;
        auto* gated_attention_batch = static_cast<float*>(prefill_gated_attention->get()) + offset * kInner;
        auto* ffn_gate_batch = static_cast<float*>(prefill_ffn_gate->get()) + offset * kFfnInner;
        auto* ffn_up_batch = static_cast<float*>(prefill_ffn_up->get()) + offset * kFfnInner;
        auto* ffn_activation_batch = static_cast<float*>(prefill_ffn_activation->get()) + offset * kFfnInner;
        auto* ffn_projected_batch = static_cast<float*>(prefill_ffn_projected->get()) + offset * kHidden;
        next_normalized = next_normalized != nullptr ? next_normalized + offset * kHidden : nullptr;

        // Q/K/V, causal attention, and KV writes remain position-ordered in
        // run(); only the post-attention O/FFN tail is deferred and batched.
        for (std::size_t i = 0; i < count; ++i) {
            miinfer::launch_q8_1_quantize_f32(
                gated_attention_batch + i * kInner,
                q8_batch + i * (kInner / miinfer::kQ8_1BlockSize), kInner,
                hipStreamPerThread);
        }
        launch_q4k_wave_gemv_batched4(
            static_cast<const Q4KWaveTile*>(d_o_native->get()), q8_batch,
            projection_batch, kHidden, kInner, hipStreamPerThread);
        for (std::size_t i = 0; i < count; ++i) {
            miinfer::launch_qwen3_fused_add_rms_norm(
                inputs + i * kHidden, projection_batch + i * kHidden,
                static_cast<const float*>(d_post->get()),
                residual_batch + i * kHidden, normalized_batch + i * kHidden,
                kHidden, model.config().rms_epsilon, nullptr, nullptr);
            miinfer::launch_q8_1_quantize_f32(
                normalized_batch + i * kHidden,
                q8_batch + i * (kHidden / miinfer::kQ8_1BlockSize), kHidden,
                hipStreamPerThread);
        }
        if (d_ffn_swiglu_native) {
            launch_q4k_wave_fused_gate_up_swiglu_paired_batched4(
                static_cast<const Q4KWaveSwigluFusedTile*>(d_ffn_swiglu_native->get()),
                q8_batch, ffn_activation_batch,
                kFfnInner, kHidden, hipStreamPerThread);
        } else {
            launch_q4k_wave_gemv_batched4(
                static_cast<const Q4KWaveTile*>(d_ffn_gate_native->get()), q8_batch,
                ffn_gate_batch, kFfnInner, kHidden,
                hipStreamPerThread);
            launch_q4k_wave_gemv_batched4(
                static_cast<const Q4KWaveTile*>(d_ffn_up_native->get()), q8_batch,
                ffn_up_batch, kFfnInner, kHidden,
                hipStreamPerThread);
            for (std::size_t i = 0; i < count; ++i) {
                miinfer::launch_qwen3_silu_mul(
                    ffn_gate_batch + i * kFfnInner,
                    ffn_up_batch + i * kFfnInner,
                    ffn_activation_batch + i * kFfnInner,
                    kFfnInner, hipStreamPerThread);
            }
        }
        for (std::size_t i = 0; i < count; ++i) {
            miinfer::launch_q8_1_quantize_f32(
                ffn_activation_batch + i * kFfnInner,
                q8_batch + i * (kFfnInner / miinfer::kQ8_1BlockSize), kFfnInner,
                hipStreamPerThread);
        }
        if (ffn_down_weight.type == miinfer::GgufTensorType::q4_k) {
            launch_q4k_wave_gemv_batched4(
                static_cast<const Q4KWaveTile*>(d_ffn_down_native->get()), q8_batch,
                ffn_projected_batch, kHidden, kFfnInner,
                hipStreamPerThread);
        } else {
            launch_q6k_wave_gemv_batched4(
                static_cast<const Q6KWaveTile*>(d_ffn_down_native->get()), q8_batch,
                ffn_projected_batch, kHidden, kFfnInner,
                hipStreamPerThread);
        }
        for (std::size_t i = 0; i < count; ++i) {
            if (next_norm_weight != nullptr && next_normalized != nullptr) {
                miinfer::launch_qwen3_fused_add_rms_norm(
                    residual_batch + i * kHidden,
                    ffn_projected_batch + i * kHidden,
                    next_norm_weight, outputs + i * kHidden,
                    next_normalized + i * kHidden, kHidden,
                    model.config().rms_epsilon, nullptr, nullptr);
            } else {
                miinfer::launch_qwen3_add(
                    residual_batch + i * kHidden,
                    ffn_projected_batch + i * kHidden,
                    outputs + i * kHidden, kHidden, hipStreamPerThread);
            }
        }
    }

    void finish_prefill_batch(const float* inputs, float* outputs, std::size_t count,
                              const float* next_norm_weight = nullptr,
                              float* next_normalized = nullptr) {
        if (!prefill_tail_batch_supported() || inputs == nullptr || outputs == nullptr
            || count == 0 || count > prefill_capacity || count % 4 != 0) {
            throw std::runtime_error("invalid attention prefill batch tail");
        }
        if (wide_attn_prefill && d_o_mmq && d_ffn_gate_mmq && d_ffn_up_mmq
            && d_ffn_down_mmq && prefill_mmq_q8) {
            finish_prefill_wide(inputs, outputs, count, next_norm_weight, next_normalized);
            return;
        }
        for (std::size_t offset = 0; offset < count; offset += 4) {
            finish_prefill_batch4(inputs + offset * kHidden, outputs + offset * kHidden,
                                  4, offset, next_norm_weight, next_normalized);
        }
    }

    void run(const float* input, std::uint32_t position, float* output,
             const float* next_norm_weight = nullptr,
             float* next_normalized = nullptr,
             bool precomputed_norm = false,
             miinfer::Q8_1Block* next_q8_1 = nullptr,
             bool precomputed_q8 = false,
             const float* prepared_normalized = nullptr,
             const float* prepared_qfull = nullptr,
             const float* prepared_value = nullptr,
             bool defer_prefill_tail = false,
             std::size_t prefill_index = 0) {
        const float* normalized_input = prepared_normalized != nullptr
            ? prepared_normalized : static_cast<const float*>(normalized->get());
        if (prepared_normalized != nullptr) precomputed_norm = true;
        const bool precomputed_projection = prepared_qfull != nullptr;
        stage_start(0, position);
        if (!precomputed_norm) {
            miinfer::launch_qwen3_rms_norm(input, static_cast<const float*>(d_attn_norm->get()),
                static_cast<float*>(normalized->get()), kHidden, model.config().rms_epsilon);
        }
        stage_end(0, position);
        stage_start(1, position);
        float* qfull_storage = static_cast<float*>(qfull->get());
        const float* qfull_dest = precomputed_projection ? prepared_qfull : qfull_storage;
        float* key_storage = key ? static_cast<float*>(key->get()) : nullptr;
        const float* key_dest = (d_qk_combined || resident_m23_all) ? qfull_dest + 12288
            : (precomputed_projection ? qfull_dest + 12288 : key_storage);
        if (precomputed_projection) {
            // The layer-major path already populated Q+gate+K with B=4.
        } else if (resident_m23_all) {
            miinfer::launch_m23_q8_1_mmq_quantize(
                normalized_input,
                static_cast<miinfer::M23Q8_1MmqBlock*>(decode_mmq_q8->get()),
                1, kHidden);
            launch_m23_q4k_repacked_mmq(
                static_cast<const Q4KMmqTile*>(resident_qk_mmq->get()),
                static_cast<const miinfer::M23Q8_1MmqBlock*>(decode_mmq_q8->get()),
                qfull_storage, 12288 + 1024, kHidden, 1);
        } else if (d_qk_combined) {
            if (!precomputed_q8 || !fused_norm_q8) {
                miinfer::launch_q8_1_quantize_f32(
                    normalized_input,
                    static_cast<miinfer::Q8_1Block*>(q8_1->get()), kHidden);
            }
            launch_q4k_wave_gemv(
                static_cast<const Q4KWaveTile*>(d_qk_combined->get()),
                static_cast<const miinfer::Q8_1Block*>(q8_1->get()),
                qfull_storage, 12288 + 1024, kHidden);
        } else if (d_q_native) {
            if (!precomputed_q8 || !fused_norm_q8) {
                miinfer::launch_q8_1_quantize_f32(
                    normalized_input,
                    static_cast<miinfer::Q8_1Block*>(q8_1->get()), kHidden);
            }
            launch_q4k_wave_gemv(
                static_cast<const Q4KWaveTile*>(d_q_native->get()),
                static_cast<const miinfer::Q8_1Block*>(q8_1->get()),
                qfull_storage, 12288, kHidden);
        } else {
            project(q_weight, d_q, normalized_input,
                static_cast<miinfer::Q8KDeviceBlock*>(q8->get()), qfull_storage, 12288, kHidden);
        }
        stage_end(1, position);
        stage_start(2, position);
        if (fused_rope_norm) {
            miinfer::launch_qwen35_fused_q_split_norm_rope(
                qfull_dest,
                static_cast<const float*>(d_q_norm->get()),
                static_cast<float*>(query_rope->get()),
                static_cast<float*>(gate->get()),
                24, 256, position, model.config().rope_theta, model.config().rms_epsilon);
        } else {
            miinfer::launch_qwen35_split_q_gate(qfull_dest,
                static_cast<float*>(query->get()), static_cast<float*>(gate->get()), 24, 256);
            if (batch_head_rms) {
                miinfer::launch_qwen3_head_rms_normalize(
                    static_cast<const float*>(query->get()), static_cast<float*>(query_norm->get()),
                    24, 256, model.config().rms_epsilon);
            } else {
                for (std::uint32_t h = 0; h < 24; ++h) {
                    miinfer::launch_qwen3_rms_normalize(static_cast<const float*>(query->get()) + h * 256,
                        static_cast<float*>(query_norm->get()) + h * 256, 256, model.config().rms_epsilon);
                }
            }
            miinfer::launch_qwen3_head_mul(static_cast<const float*>(query_norm->get()),
                static_cast<const float*>(d_q_norm->get()), static_cast<float*>(query_norm->get()), 24, 256);
        }
        stage_end(2, position);
        stage_start(3, position);
        if (!d_qk_combined && !resident_m23_all) {
            if (d_k_native) {
                if (!d_q_native) {
                    miinfer::launch_q8_1_quantize_f32(
                        normalized_input,
                        static_cast<miinfer::Q8_1Block*>(q8_1->get()), kHidden);
                }
                launch_q4k_wave_gemv(
                    static_cast<const Q4KWaveTile*>(d_k_native->get()),
                    static_cast<const miinfer::Q8_1Block*>(q8_1->get()),
                    key_storage, 1024, kHidden);
            } else {
                project(k_weight, d_k, normalized_input,
                        static_cast<miinfer::Q8KDeviceBlock*>(q8->get()), key_storage,
                        1024, kHidden, d_q_native ? false : reuse_projection_q8);
            }
        }
        stage_end(3, position);
        stage_start(4, position);
        if (!fused_rope_norm) {
            if (batch_head_rms) {
                miinfer::launch_qwen3_head_rms_normalize(
                    key_dest, static_cast<float*>(key_norm->get()),
                    4, 256, model.config().rms_epsilon);
            } else {
                for (std::uint32_t h = 0; h < 4; ++h) {
                    miinfer::launch_qwen3_rms_normalize(key_dest + h * 256,
                        static_cast<float*>(key_norm->get()) + h * 256, 256, model.config().rms_epsilon);
                }
            }
            miinfer::launch_qwen3_head_mul(static_cast<const float*>(key_norm->get()),
                static_cast<const float*>(d_k_norm->get()), static_cast<float*>(key_norm->get()), 4, 256);
        }
        stage_end(4, position);
        stage_start(5, position);
        const float* value_input = prepared_value != nullptr
            ? prepared_value : static_cast<const float*>(value->get());
        if (prepared_value != nullptr) {
            // The layer-major path already populated V with B=4.
        } else if (resident_m23_all) {
            miinfer::launch_m23_q8_1_mmq_quantize(
                normalized_input,
                static_cast<miinfer::M23Q8_1MmqBlock*>(decode_mmq_q8->get()),
                1, kHidden);
            if (v_weight.type == miinfer::GgufTensorType::q4_k) {
                launch_m23_q4k_repacked_mmq(
                    static_cast<const Q4KMmqTile*>(resident_v_mmq->get()),
                    static_cast<const miinfer::M23Q8_1MmqBlock*>(decode_mmq_q8->get()),
                    static_cast<float*>(value->get()), 1024, kHidden, 1);
            } else {
                launch_m23_q6k_repacked_mmq(
                    static_cast<const Q6KMmqTile*>(resident_v_mmq->get()),
                    static_cast<const miinfer::M23Q8_1MmqBlock*>(decode_mmq_q8->get()),
                    static_cast<float*>(value->get()), 1024, kHidden, 1);
            }
        } else if (d_v_native) {
            if (!d_qk_combined && !d_q_native && !d_k_native) {
                miinfer::launch_q8_1_quantize_f32(
                    normalized_input,
                    static_cast<miinfer::Q8_1Block*>(q8_1->get()), kHidden);
            }
            if (v_weight.type == miinfer::GgufTensorType::q4_k) {
                launch_q4k_wave_gemv(
                    static_cast<const Q4KWaveTile*>(d_v_native->get()),
                    static_cast<const miinfer::Q8_1Block*>(q8_1->get()),
                    static_cast<float*>(value->get()), 1024, kHidden);
            } else {
                launch_q6k_wave_gemv(
                    static_cast<const Q6KWaveTile*>(d_v_native->get()),
                    static_cast<const miinfer::Q8_1Block*>(q8_1->get()),
                    static_cast<float*>(value->get()), 1024, kHidden);
            }
        } else {
            project(v_weight, d_v, normalized_input,
                    static_cast<miinfer::Q8KDeviceBlock*>(q8->get()), static_cast<float*>(value->get()),
                    1024, kHidden, (d_qk_combined || d_q_native || d_k_native) ? false : reuse_projection_q8);
        }
        stage_end(5, position);
        stage_start(6, position);
        if (fused_rope_norm) {
            if (fp16_kv_cache) {
                miinfer::launch_qwen35_fused_k_norm_rope_kv_store(
                    key_dest,
                    value_input,
                    static_cast<const float*>(d_k_norm->get()),
                    static_cast<__half*>(key_cache->get()),
                    static_cast<__half*>(value_cache->get()),
                    4, 256, position, g_cache_capacity, model.config().rope_theta, model.config().rms_epsilon);
            } else {
                miinfer::launch_qwen35_fused_k_norm_rope_kv_store(
                    key_dest,
                    value_input,
                    static_cast<const float*>(d_k_norm->get()),
                    static_cast<float*>(key_cache->get()),
                    static_cast<float*>(value_cache->get()),
                    4, 256, position, g_cache_capacity, model.config().rope_theta, model.config().rms_epsilon);
            }
        } else {
            miinfer::launch_qwen35_rope_sections(static_cast<const float*>(query_norm->get()),
                static_cast<float*>(query_rope->get()), 24, 256, position, model.config().rope_theta);
            miinfer::launch_qwen35_rope_sections(static_cast<const float*>(key_norm->get()),
                static_cast<float*>(key_rope->get()), 4, 256, position, model.config().rope_theta);
            if (fp16_kv_cache) {
                miinfer::launch_qwen3_kv_cache_store(static_cast<const float*>(key_rope->get()),
                    value_input, static_cast<__half*>(key_cache->get()),
                    static_cast<__half*>(value_cache->get()), position, g_cache_capacity, 4, 256);
            } else {
                miinfer::launch_qwen3_kv_cache_store(static_cast<const float*>(key_rope->get()),
                    value_input, static_cast<float*>(key_cache->get()),
                    static_cast<float*>(value_cache->get()), position, g_cache_capacity, 4, 256);
            }
        }
        stage_end(6, position);
        stage_start(7, position);
        if (tiled_online_attention) {
            if (fp16_kv_cache) {
                miinfer::launch_qwen35_tiled_online_attention(
                    static_cast<const float*>(query_rope->get()),
                    static_cast<const __half*>(key_cache->get()),
                    static_cast<const __half*>(value_cache->get()),
                    position + 1, g_cache_capacity,
                    static_cast<float*>(attention->get()),
                    static_cast<const float*>(gate->get()),
                    static_cast<float*>(gated_attention->get()),
                    24, 4, 256, 1.0F / std::sqrt(256.0F),
                    nullptr,
                    (fused_core_q8 && d_o_native) ? static_cast<miinfer::Q8_1Block*>(q8_1->get()) : nullptr);
            } else {
                miinfer::launch_qwen35_tiled_online_attention(
                    static_cast<const float*>(query_rope->get()),
                    static_cast<const float*>(key_cache->get()),
                    static_cast<const float*>(value_cache->get()),
                    position + 1, g_cache_capacity,
                    static_cast<float*>(attention->get()),
                    static_cast<const float*>(gate->get()),
                    static_cast<float*>(gated_attention->get()),
                    24, 4, 256, 1.0F / std::sqrt(256.0F),
                    nullptr,
                    (fused_core_q8 && d_o_native) ? static_cast<miinfer::Q8_1Block*>(q8_1->get()) : nullptr);
            }
            stage_end(7, position);
            stage_start(8, position);
        } else {
            if (fp16_kv_cache) {
                throw std::runtime_error("Untiled attention fallback requires FP32 KV cache (MIINFER_FP16_KV_CACHE=0)");
            }
            miinfer::launch_qwen3_cached_attention_parallel(
                static_cast<const float*>(query_rope->get()),
                static_cast<const float*>(key_cache->get()),
                static_cast<const float*>(value_cache->get()),
                position + 1, g_cache_capacity,
                static_cast<float*>(attention->get()),
                static_cast<float*>(scores->get()),
                static_cast<float*>(probabilities->get()),
                24, 4, 256, 1.0F / std::sqrt(256.0F));
            stage_end(7, position);
            stage_start(8, position);
            miinfer::launch_qwen35_sigmoid_mul(
                static_cast<const float*>(attention->get()),
                static_cast<const float*>(gate->get()),
                static_cast<float*>(gated_attention->get()),
                6144);
        }
        if (defer_prefill_tail) {
            if (!prefill_tail_batch_supported() || prefill_index >= prefill_capacity) {
                throw std::runtime_error("invalid deferred attention prefill tail");
            }
            MIINFER_HIP_CHECK(hipMemcpyAsync(
                static_cast<float*>(prefill_gated_attention->get()) + prefill_index * kInner,
                gated_attention->get(), kInner * sizeof(float),
                hipMemcpyDeviceToDevice, hipStreamPerThread));
            return;
        }
        if (resident_m23_all) {
            miinfer::launch_m23_q8_1_mmq_quantize(
                static_cast<const float*>(gated_attention->get()),
                static_cast<miinfer::M23Q8_1MmqBlock*>(decode_mmq_q8->get()),
                1, 6144);
            if (o_weight.type == miinfer::GgufTensorType::q4_k) {
                launch_m23_q4k_repacked_mmq(
                    static_cast<const Q4KMmqTile*>(resident_o_mmq->get()),
                    static_cast<const miinfer::M23Q8_1MmqBlock*>(decode_mmq_q8->get()),
                    static_cast<float*>(projected->get()), kHidden, 6144, 1);
            } else {
                launch_m23_q6k_repacked_mmq(
                    static_cast<const Q6KMmqTile*>(resident_o_mmq->get()),
                    static_cast<const miinfer::M23Q8_1MmqBlock*>(decode_mmq_q8->get()),
                    static_cast<float*>(projected->get()), kHidden, 6144, 1);
            }
        } else if (d_o_native) {
            if (!fused_core_q8 || !tiled_online_attention) {
                miinfer::launch_q8_1_quantize_f32(
                    static_cast<const float*>(gated_attention->get()),
                    static_cast<miinfer::Q8_1Block*>(q8_1->get()), 6144);
            }
            launch_q4k_wave_gemv(
                static_cast<const Q4KWaveTile*>(d_o_native->get()),
                static_cast<const miinfer::Q8_1Block*>(q8_1->get()),
                static_cast<float*>(projected->get()), kHidden, 6144);
        } else {
            project(o_weight, d_o, static_cast<const float*>(gated_attention->get()),
                static_cast<miinfer::Q8KDeviceBlock*>(q8->get()), static_cast<float*>(projected->get()), kHidden, 6144);
        }
        stage_end(8, position);
        stage_start(9, position);
        if (fused_add_rms_norm) {
            miinfer::launch_qwen3_fused_add_rms_norm(
                input, static_cast<const float*>(projected->get()),
                static_cast<const float*>(d_post->get()),
                static_cast<float*>(residual->get()),
                static_cast<float*>(post_normalized->get()),
                kHidden, model.config().rms_epsilon, nullptr,
                (fused_norm_q8 && (d_ffn_swiglu_native || (d_ffn_gate_native && d_ffn_up_native)))
                    ? static_cast<miinfer::Q8_1Block*>(q8_1->get()) : nullptr);
            stage_end(9, position);
            stage_start(10, position);
            stage_end(10, position);
        } else {
            miinfer::launch_qwen3_add(input, static_cast<const float*>(projected->get()),
                static_cast<float*>(residual->get()), kHidden);
            stage_end(9, position);
            stage_start(10, position);
            miinfer::launch_qwen3_rms_norm(static_cast<const float*>(residual->get()),
                static_cast<const float*>(d_post->get()), static_cast<float*>(post_normalized->get()), kHidden,
                model.config().rms_epsilon);
            stage_end(10, position);
        }
        stage_start(11, position);
        if (resident_m23_all) {
            miinfer::launch_m23_q8_1_mmq_quantize(
                static_cast<const float*>(post_normalized->get()),
                static_cast<miinfer::M23Q8_1MmqBlock*>(decode_mmq_q8->get()),
                1, kHidden);
            launch_m23_q4k_repacked_mmq(
                static_cast<const Q4KMmqTile*>(resident_ffn_gate_mmq->get()),
                static_cast<const miinfer::M23Q8_1MmqBlock*>(decode_mmq_q8->get()),
                static_cast<float*>(ffn_gate->get()), kFfnInner, kHidden, 1);
            launch_m23_q4k_repacked_mmq(
                static_cast<const Q4KMmqTile*>(resident_ffn_up_mmq->get()),
                static_cast<const miinfer::M23Q8_1MmqBlock*>(decode_mmq_q8->get()),
                static_cast<float*>(ffn_up->get()), kFfnInner, kHidden, 1);
        } else if (d_ffn_swiglu_native) {
            if (!fused_norm_q8) {
                miinfer::launch_q8_1_quantize_f32(
                    static_cast<const float*>(post_normalized->get()),
                    static_cast<miinfer::Q8_1Block*>(q8_1->get()), kHidden);
            }
            launch_q4k_wave_fused_gate_up_swiglu_paired(
                static_cast<const Q4KWaveSwigluFusedTile*>(d_ffn_swiglu_native->get()),
                static_cast<const miinfer::Q8_1Block*>(q8_1->get()),
                static_cast<float*>(ffn_activation->get()), kFfnInner, kHidden);
        } else if (d_ffn_gate_native && d_ffn_up_native) {
            if (!fused_norm_q8) {
                miinfer::launch_q8_1_quantize_f32(
                    static_cast<const float*>(post_normalized->get()),
                    static_cast<miinfer::Q8_1Block*>(q8_1->get()), kHidden);
            }
            if (fused_gate_up_swiglu) {
                launch_q4k_wave_fused_gate_up_swiglu(
                    static_cast<const Q4KWaveTile*>(d_ffn_gate_native->get()),
                    static_cast<const Q4KWaveTile*>(d_ffn_up_native->get()),
                    static_cast<const miinfer::Q8_1Block*>(q8_1->get()),
                    static_cast<float*>(ffn_activation->get()), kFfnInner, kHidden);
            } else {
                launch_q4k_wave_gemv(
                    static_cast<const Q4KWaveTile*>(d_ffn_gate_native->get()),
                    static_cast<const miinfer::Q8_1Block*>(q8_1->get()),
                    static_cast<float*>(ffn_gate->get()), kFfnInner, kHidden);
                launch_q4k_wave_gemv(
                    static_cast<const Q4KWaveTile*>(d_ffn_up_native->get()),
                    static_cast<const miinfer::Q8_1Block*>(q8_1->get()),
                    static_cast<float*>(ffn_up->get()), kFfnInner, kHidden);
            }
        } else if (q4_q8_1_gate_up
            && ffn_gate_weight.type == miinfer::GgufTensorType::q4_k
            && ffn_up_weight.type == miinfer::GgufTensorType::q4_k) {
            miinfer::launch_q8_1_quantize_f32(
                static_cast<const float*>(post_normalized->get()),
                static_cast<miinfer::Q8_1Block*>(q8_1->get()), kHidden);
            project_q4_q8_1_prequantized(
                d_ffn_gate, static_cast<const miinfer::Q8_1Block*>(q8_1->get()),
                static_cast<float*>(ffn_gate->get()), kFfnInner, kHidden,
                q4_q8_1_lds_input, q4_q8_1_lds_metadata,
                q4_q8_1_lds_decoded_metadata);
            project_q4_q8_1_prequantized(
                d_ffn_up, static_cast<const miinfer::Q8_1Block*>(q8_1->get()),
                static_cast<float*>(ffn_up->get()), kFfnInner, kHidden,
                q4_q8_1_lds_input, q4_q8_1_lds_metadata,
                q4_q8_1_lds_decoded_metadata);
        } else {
            project(ffn_gate_weight, d_ffn_gate, static_cast<const float*>(post_normalized->get()),
                    static_cast<miinfer::Q8KDeviceBlock*>(q8->get()),
                    static_cast<float*>(ffn_gate->get()), kFfnInner, kHidden);
            project(ffn_up_weight, d_ffn_up, static_cast<const float*>(post_normalized->get()),
                    static_cast<miinfer::Q8KDeviceBlock*>(q8->get()),
                    static_cast<float*>(ffn_up->get()), kFfnInner, kHidden,
                    reuse_projection_q8);
        }
        stage_end(11, position);
        stage_start(12, position);
        if (resident_m23_all || (!d_ffn_swiglu_native
                                 && (!fused_gate_up_swiglu || !d_ffn_gate_native || !d_ffn_up_native))) {
            miinfer::launch_qwen3_silu_mul(static_cast<const float*>(ffn_gate->get()),
                static_cast<const float*>(ffn_up->get()), static_cast<float*>(ffn_activation->get()), kFfnInner);
        }
        stage_end(12, position);
        stage_start(13, position);
        if (resident_m23_all) {
            miinfer::launch_m23_q8_1_mmq_quantize(
                static_cast<const float*>(ffn_activation->get()),
                static_cast<miinfer::M23Q8_1MmqBlock*>(decode_mmq_q8->get()),
                1, kFfnInner);
            if (ffn_down_weight.type == miinfer::GgufTensorType::q4_k) {
                launch_m23_q4k_repacked_mmq(
                    static_cast<const Q4KMmqTile*>(resident_ffn_down_mmq->get()),
                    static_cast<const miinfer::M23Q8_1MmqBlock*>(decode_mmq_q8->get()),
                    static_cast<float*>(projected->get()), kHidden, kFfnInner, 1);
            } else {
                launch_m23_q6k_repacked_mmq(
                    static_cast<const Q6KMmqTile*>(resident_ffn_down_mmq->get()),
                    static_cast<const miinfer::M23Q8_1MmqBlock*>(decode_mmq_q8->get()),
                    static_cast<float*>(projected->get()), kHidden, kFfnInner, 1);
            }
        } else if (d_ffn_down_native) {
            if (ffn_down_weight.type == miinfer::GgufTensorType::q4_k) {
                project_native_down(d_ffn_down_native,
                    static_cast<const float*>(ffn_activation->get()),
                    static_cast<miinfer::Q8_1Block*>(q8_1->get()),
                    static_cast<float*>(projected->get()));
            } else {
                project_native_q6k_down(d_ffn_down_native,
                    static_cast<const float*>(ffn_activation->get()),
                    static_cast<miinfer::Q8_1Block*>(q8_1->get()),
                    static_cast<float*>(projected->get()));
            }
        } else if (expanded_down && d_ffn_down_expanded != nullptr) {
            project_q4_q8_1_expanded(d_ffn_down_expanded,
                static_cast<const float*>(ffn_activation->get()),
                static_cast<miinfer::Q8_1Block*>(q8_1->get()),
                static_cast<float*>(projected->get()), kHidden, kFfnInner);
        } else if (q4_q8_1_mmvq && ffn_down_weight.type == miinfer::GgufTensorType::q4_k) {
            project_q4_q8_1(d_ffn_down, static_cast<const float*>(ffn_activation->get()),
                            static_cast<miinfer::Q8_1Block*>(q8_1->get()),
                            static_cast<float*>(projected->get()), kHidden, kFfnInner,
                            q4_q8_1_lds_input, q4_q8_1_lds_metadata,
                            q4_q8_1_lds_decoded_metadata);
        } else {
            project(ffn_down_weight, d_ffn_down, static_cast<const float*>(ffn_activation->get()),
                    static_cast<miinfer::Q8KDeviceBlock*>(q8->get()),
                    static_cast<float*>(projected->get()), kHidden, kFfnInner);
        }
        stage_end(13, position);
        stage_start(14, position);
        float* completed_output = direct_layer_output
            ? output : static_cast<float*>(layer_output->get());
        if (fused_interlayer_norm && next_norm_weight != nullptr && next_normalized != nullptr) {
            miinfer::launch_qwen3_fused_add_rms_norm(
                static_cast<const float*>(residual->get()),
                static_cast<const float*>(projected->get()),
                next_norm_weight,
                completed_output,
                next_normalized,
                kHidden,
                model.config().rms_epsilon, nullptr,
                (fused_norm_q8 && next_q8_1 != nullptr) ? next_q8_1 : nullptr);
        } else {
            miinfer::launch_qwen3_add(static_cast<const float*>(residual->get()),
                static_cast<const float*>(projected->get()), completed_output, kHidden);
        }
        if (!direct_layer_output) {
            MIINFER_HIP_CHECK(hipMemcpyAsync(output, layer_output->get(), kHidden * sizeof(float),
                                            hipMemcpyDeviceToDevice, hipStreamPerThread));
        }
        stage_end(14, position);
    }
};

void run_hybrid_block(RecurrentLayer& recurrent0, RecurrentLayer& recurrent1,
                      RecurrentLayer& recurrent2, FullAttentionLayer& attention,
                      const float* input, std::uint32_t position,
                      float* state1, float* state2, float* state3, float* output) {
    recurrent0.run(input, position, state1);
    recurrent1.run(state1, position, state2);
    recurrent2.run(state2, position, state3);
    attention.run(state3, position, output);
}

struct GpuLayerRef {
    RecurrentLayer* recurrent = nullptr;
    FullAttentionLayer* attention = nullptr;

    const float* attn_norm_weight() const {
        if (recurrent != nullptr) return static_cast<const float*>(recurrent->d_attn_norm->get());
        if (attention != nullptr) return static_cast<const float*>(attention->d_attn_norm->get());
        return nullptr;
    }

    float* normalized_buffer() const {
        if (recurrent != nullptr) return static_cast<float*>(recurrent->normalized->get());
        if (attention != nullptr) return static_cast<float*>(attention->normalized->get());
        return nullptr;
    }

    miinfer::Q8_1Block* q8_1_buffer() const {
        if (recurrent != nullptr && recurrent->q8_1) return static_cast<miinfer::Q8_1Block*>(recurrent->q8_1->get());
        if (attention != nullptr && attention->q8_1) return static_cast<miinfer::Q8_1Block*>(attention->q8_1->get());
        return nullptr;
    }

    bool fused_interlayer_norm() const {
        if (recurrent != nullptr) return recurrent->fused_interlayer_norm;
        if (attention != nullptr) return attention->fused_interlayer_norm;
        return false;
    }

    void profile_ordered_start(std::size_t chunk_base) const {
        if (recurrent != nullptr) recurrent->profile_ordered_start(chunk_base);
        else if (attention != nullptr) attention->profile_ordered_start(chunk_base);
    }

    void profile_ordered_end(std::size_t chunk_base) const {
        if (recurrent != nullptr) recurrent->profile_ordered_end(chunk_base);
        else if (attention != nullptr) attention->profile_ordered_end(chunk_base);
    }

    bool prepare_prefill_batch(const float* inputs, std::size_t count,
                               bool normalized_ready = false,
                               std::size_t profile_chunk_base = std::numeric_limits<std::size_t>::max()) const {
        if (recurrent != nullptr) {
            recurrent->profile_prepare_start(profile_chunk_base);
            const bool result = recurrent->prepare_prefill_batch(inputs, count, normalized_ready);
            recurrent->profile_prepare_end(profile_chunk_base);
            return result;
        }
        if (attention == nullptr) return false;
        attention->profile_prepare_start(profile_chunk_base);
        const bool result = attention->prepare_prefill_batch(inputs, count, normalized_ready);
        attention->profile_prepare_end(profile_chunk_base);
        return result;
    }

    const float* prefill_normalized_at(std::size_t index) const {
        if (recurrent != nullptr) return recurrent->prefill_normalized_at(index);
        return attention == nullptr ? nullptr : attention->prefill_normalized_at(index);
    }

    const float* prefill_qkv_at(std::size_t index) const {
        return recurrent == nullptr ? nullptr : recurrent->prefill_qkv_at(index);
    }

    const float* prefill_gate_at(std::size_t index) const {
        return recurrent == nullptr ? nullptr : recurrent->prefill_gate_at(index);
    }

    const float* prefill_qfull_at(std::size_t index) const {
        return attention == nullptr ? nullptr : attention->prefill_qfull_at(index);
    }

    const float* prefill_value_at(std::size_t index) const {
        return attention == nullptr ? nullptr : attention->prefill_value_at(index);
    }

    bool prefill_tail_batch_supported() const {
        if (recurrent != nullptr) return recurrent->prefill_tail_batch_supported();
        return attention != nullptr && attention->prefill_tail_batch_supported();
    }

    bool wide_attention_batch_ready() const {
        return attention != nullptr && attention->wide_attention_batch_ready();
    }

    void release_m23_repacked() const {
        if (recurrent != nullptr) recurrent->release_m23_repacked();
        else if (attention != nullptr) attention->release_m23_repacked();
    }

    void finish_prefill_attention(std::uint32_t base_position, std::uint32_t count) const {
        if (attention == nullptr) throw std::runtime_error("wide attention requested for recurrent layer");
        attention->finish_prefill_attention(base_position, count);
    }

    void finish_prefill_batch(const float* inputs, float* outputs, std::size_t count,
                              const float* next_norm_weight = nullptr,
                              float* next_normalized = nullptr,
                              std::size_t profile_chunk_base = std::numeric_limits<std::size_t>::max()) const {
        if (recurrent != nullptr) {
            recurrent->profile_tail_start(profile_chunk_base);
            recurrent->finish_prefill_batch(inputs, outputs, count, next_norm_weight, next_normalized);
            recurrent->profile_tail_end(profile_chunk_base);
        } else if (attention != nullptr) {
            attention->profile_tail_start(profile_chunk_base);
            attention->finish_prefill_batch(inputs, outputs, count, next_norm_weight, next_normalized);
            attention->profile_tail_end(profile_chunk_base);
        } else {
            throw std::runtime_error("empty qwen35 GPU layer reference");
        }
    }

    void run(const float* input, std::uint32_t position, float* output,
             const float* next_norm_weight = nullptr,
             float* next_normalized = nullptr,
             bool precomputed_norm = false,
             miinfer::Q8_1Block* next_q8_1 = nullptr,
             bool precomputed_q8 = false,
             const float* prepared_qkv = nullptr,
             const float* prepared_gate = nullptr,
             const float* prepared_normalized = nullptr,
             bool defer_prefill_tail = false,
             std::size_t prefill_index = 0,
             const float* prepared_qfull = nullptr,
             const float* prepared_value = nullptr) const {
        if (recurrent != nullptr) {
            recurrent->run(input, position, output, next_norm_weight, next_normalized,
                           precomputed_norm, next_q8_1, precomputed_q8,
                           prepared_qkv, prepared_gate, prepared_normalized,
                           defer_prefill_tail, prefill_index);
        } else if (attention != nullptr) {
            attention->run(input, position, output, next_norm_weight, next_normalized,
                           precomputed_norm, next_q8_1, precomputed_q8, prepared_normalized,
                           prepared_qfull, prepared_value, defer_prefill_tail, prefill_index);
        } else {
            throw std::runtime_error("empty qwen35 GPU layer reference");
        }
    }
};

void run_prefix(std::span<const GpuLayerRef> layers, std::span<float* const> outputs,
                const float* input, std::uint32_t position,
                const float* final_norm_weight = nullptr,
                float* final_norm_out = nullptr,
                miinfer::Q8_1Block* final_q8_1_out = nullptr) {
    const float* current = input;
    bool precomputed = false;
    bool precomputed_q8 = false;
    for (std::size_t layer = 0; layer < layers.size(); ++layer) {
        const float* next_weight = nullptr;
        float* next_norm = nullptr;
        miinfer::Q8_1Block* next_q8 = nullptr;
        if (layers[layer].fused_interlayer_norm()) {
            if (layer + 1 < layers.size()) {
                next_weight = layers[layer + 1].attn_norm_weight();
                next_norm = layers[layer + 1].normalized_buffer();
                next_q8 = layers[layer + 1].q8_1_buffer();
            } else if (final_norm_weight != nullptr && final_norm_out != nullptr) {
                next_weight = final_norm_weight;
                next_norm = final_norm_out;
                next_q8 = final_q8_1_out;
            }
        }
        layers[layer].run(current, position, outputs[layer], next_weight, next_norm, precomputed, next_q8, precomputed_q8);
        current = outputs[layer];
        precomputed = (next_weight != nullptr && next_norm != nullptr);
        precomputed_q8 = precomputed && (next_q8 != nullptr);
    }
}

RecurrentOperands read_external_operands(const std::filesystem::path& fixture) {
    RecurrentOperands result;
    const auto full_query = read_f32(checkpoint(fixture, 19, "q_in-30"), kVHeads * kState);
    const auto full_key = read_f32(checkpoint(fixture, 19, "k_in-30"), kVHeads * kState);
    result.query.assign(full_query.begin(), full_query.begin() + kKHeads * kState);
    result.key.assign(full_key.begin(), full_key.begin() + kKHeads * kState);
    result.value = read_f32(checkpoint(fixture, 19, "v_in-30"), kVHeads * kState);
    result.beta = read_f32(checkpoint(fixture, 19, "b_in-30"), kVHeads);
    const auto external_gate = read_f32(checkpoint(fixture, 19, "g_in-30"), kVHeads);
    result.decay.resize(kVHeads);
    for (std::size_t i = 0; i < kVHeads; ++i) result.decay[i] = std::exp(external_gate[i]);
    result.previous = read_f32(checkpoint(fixture, 19, "state_predelta-30"),
                               kVHeads * kState * kState);
    return result;
}

std::vector<float> replay_state(const RecurrentOperands& operands) {
    Buffer d_query = allocate(operands.query.size() * sizeof(float));
    Buffer d_key = allocate(operands.key.size() * sizeof(float));
    Buffer d_value = allocate(operands.value.size() * sizeof(float));
    Buffer d_beta = allocate(operands.beta.size() * sizeof(float));
    Buffer d_decay = allocate(operands.decay.size() * sizeof(float));
    Buffer d_state = allocate(operands.previous.size() * sizeof(float));
    Buffer d_output = allocate(kVHeads * kState * sizeof(float));
    upload(operands.query.data(), d_query->get(), operands.query.size() * sizeof(float));
    upload(operands.key.data(), d_key->get(), operands.key.size() * sizeof(float));
    upload(operands.value.data(), d_value->get(), operands.value.size() * sizeof(float));
    upload(operands.beta.data(), d_beta->get(), operands.beta.size() * sizeof(float));
    upload(operands.decay.data(), d_decay->get(), operands.decay.size() * sizeof(float));
    upload(operands.previous.data(), d_state->get(), operands.previous.size() * sizeof(float));
    miinfer::launch_qwen35_deltanet_state_update(
        static_cast<const float*>(d_query->get()), static_cast<const float*>(d_key->get()),
        static_cast<const float*>(d_value->get()), static_cast<const float*>(d_beta->get()),
        static_cast<const float*>(d_decay->get()), static_cast<float*>(d_state->get()),
        static_cast<float*>(d_output->get()), kKHeads, kVHeads, kState);
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    return download(d_state->get(), operands.previous.size());
}

} // namespace
