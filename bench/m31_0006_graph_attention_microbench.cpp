#include "miinfer/hip_check.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

namespace {

constexpr std::uint32_t kQueryHeads = 24;
constexpr std::uint32_t kKvHeads = 4;
constexpr std::uint32_t kHeadDim = 256;
constexpr std::uint32_t kGridSplits = 64;
constexpr std::size_t kSplitScratchBytes =
    kGridSplits * kQueryHeads * kHeadDim * sizeof(float)
    + 2 * kGridSplits * kQueryHeads * sizeof(float);
constexpr std::array<std::uint32_t, 4> kContexts{8192, 32768, 65536, 131072};
constexpr float kScale = 1.0F / 16.0F;
constexpr float kAbsTolerance = 2.0e-3F;

struct DeviceBuffer {
    void* data = nullptr;
    explicit DeviceBuffer(std::size_t bytes) { MIINFER_HIP_CHECK(hipMalloc(&data, bytes)); }
    ~DeviceBuffer() { if (data != nullptr) (void)hipFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

bool parse_positive(const char* text, std::uint32_t& value) {
    char* end = nullptr;
    const auto parsed = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || parsed == 0 || parsed > 30) return false;
    value = static_cast<std::uint32_t>(parsed);
    return true;
}

bool valid_context(std::uint32_t value) {
    return std::find(kContexts.begin(), kContexts.end(), value) != kContexts.end();
}

void launch_attention(const float* q, const __half* key, const __half* value,
                      const miinfer::DeviceDecodeState* state, std::uint32_t capacity,
                      const float* gate, float* output, hipStream_t stream) {
    miinfer::launch_qwen35_tiled_online_attention_quant_dynamic(
        q, key, value, nullptr, nullptr, nullptr, nullptr, state, capacity,
        nullptr, gate, output, kQueryHeads, kKvHeads, kHeadDim, kScale,
        false, false, stream);
}

hipGraphExec_t capture_attention_graph(const float* q, const __half* key,
                                       const __half* value,
                                       const miinfer::DeviceDecodeState* state,
                                       std::uint32_t capacity, const float* gate,
                                       float* output, hipStream_t stream) {
    hipGraph_t graph = nullptr;
    MIINFER_HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed));
    launch_attention(q, key, value, state, capacity, gate, output, stream);
    MIINFER_HIP_CHECK(hipStreamEndCapture(stream, &graph));
    hipGraphExec_t executable = nullptr;
    MIINFER_HIP_CHECK(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    MIINFER_HIP_CHECK(hipGraphDestroy(graph));
    return executable;
}

std::vector<float> cpu_reference(const std::vector<float>& q,
                                 const std::vector<float>& gate,
                                 const std::vector<__half>& key,
                                 const std::vector<__half>& value,
                                 std::uint32_t length, std::uint32_t capacity) {
    std::vector<float> result(q.size());
    std::vector<double> scores(length);
    for (std::uint32_t head = 0; head < kQueryHeads; ++head) {
        const std::uint32_t kv_head = head / (kQueryHeads / kKvHeads);
        const std::size_t q_base = static_cast<std::size_t>(head) * kHeadDim;
        double max_score = -std::numeric_limits<double>::infinity();
        for (std::uint32_t pos = 0; pos < length; ++pos) {
            const std::size_t kv_base =
                (static_cast<std::size_t>(kv_head) * capacity + pos) * kHeadDim;
            double dot = 0.0;
            for (std::uint32_t d = 0; d < kHeadDim; ++d)
                dot += static_cast<double>(q[q_base + d]) * __half2float(key[kv_base + d]);
            scores[pos] = dot * kScale;
            max_score = std::max(max_score, scores[pos]);
        }
        double denominator = 0.0;
        for (auto& score : scores) {
            score = std::exp(score - max_score);
            denominator += score;
        }
        for (std::uint32_t d = 0; d < kHeadDim; ++d) {
            double sum = 0.0;
            for (std::uint32_t pos = 0; pos < length; ++pos) {
                const std::size_t kv_index =
                    (static_cast<std::size_t>(kv_head) * capacity + pos) * kHeadDim + d;
                sum += scores[pos] * __half2float(value[kv_index]);
            }
            const double sigmoid = 1.0 / (1.0 + std::exp(-gate[q_base + d]));
            result[q_base + d] = static_cast<float>((sum / denominator) * sigmoid);
        }
    }
    return result;
}

struct ErrorStats { float max_abs = 0.0F; double rmse = 0.0; bool pass = false; };

ErrorStats compare(const std::vector<float>& expected, const std::vector<float>& actual) {
    ErrorStats stats;
    double square_sum = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const float error = std::fabs(expected[i] - actual[i]);
        stats.max_abs = std::max(stats.max_abs, error);
        square_sum += static_cast<double>(error) * error;
    }
    stats.rmse = std::sqrt(square_sum / expected.size());
    stats.pass = std::isfinite(stats.max_abs) && stats.max_abs <= kAbsTolerance;
    return stats;
}

std::vector<float> copy_output(const float* device, std::size_t count) {
    std::vector<float> host(count);
    MIINFER_HIP_CHECK(hipMemcpy(host.data(), device, count * sizeof(float), hipMemcpyDeviceToHost));
    return host;
}

double timed_direct(const float* q, const __half* key, const __half* value,
                   const miinfer::DeviceDecodeState* state, std::uint32_t capacity,
                   const float* gate, float* output, hipStream_t stream,
                   hipEvent_t start, hipEvent_t stop, double& wall_ms) {
    const auto wall_start = std::chrono::steady_clock::now();
    MIINFER_HIP_CHECK(hipEventRecord(start, stream));
    launch_attention(q, key, value, state, capacity, gate, output, stream);
    MIINFER_HIP_CHECK(hipEventRecord(stop, stream));
    MIINFER_HIP_CHECK(hipEventSynchronize(stop));
    float elapsed = 0.0F;
    MIINFER_HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));
    wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - wall_start).count();
    return elapsed;
}

double timed_graph(hipGraphExec_t graph, hipStream_t stream,
                  hipEvent_t start, hipEvent_t stop, double& wall_ms) {
    const auto wall_start = std::chrono::steady_clock::now();
    MIINFER_HIP_CHECK(hipEventRecord(start, stream));
    MIINFER_HIP_CHECK(hipGraphLaunch(graph, stream));
    MIINFER_HIP_CHECK(hipEventRecord(stop, stream));
    MIINFER_HIP_CHECK(hipEventSynchronize(stop));
    float elapsed = 0.0F;
    MIINFER_HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));
    wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - wall_start).count();
    return elapsed;
}

void usage() {
    std::cout << "usage: miinfer-m31-0006-graph-attention-microbench"
                 " --context 8192|32768|65536|131072"
                 " [--mode baseline|split8|split16|split32] [--repeats 1..30] [--device N]"
                 " (requires M31_GPU_RUN_AUTHORIZED=1)\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::uint32_t context = 0, repeats = 5, device = UINT32_MAX;
    std::string mode = "baseline";
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--help") { usage(); return 0; }
        if (i + 1 >= argc) { usage(); return 2; }
        const char* value = argv[++i];
        if (arg == "--context") {
            char* end = nullptr;
            const auto parsed = std::strtoul(value, &end, 10);
            if (end == value || *end != '\0' || parsed > UINT32_MAX) { usage(); return 2; }
            context = static_cast<std::uint32_t>(parsed);
        } else if (arg == "--mode") {
            mode = value;
        } else if (arg == "--repeats") {
            if (!parse_positive(value, repeats)) { usage(); return 2; }
        } else if (arg == "--device") {
            char* end = nullptr;
            const auto parsed = std::strtoul(value, &end, 10);
            if (end == value || *end != '\0' || parsed > UINT32_MAX) { usage(); return 2; }
            device = static_cast<std::uint32_t>(parsed);
        } else { usage(); return 2; }
    }
    if (!valid_context(context)
        || (mode != "baseline" && mode != "split8" && mode != "split16" && mode != "split32")
        || (mode == "split8" && context != 8192)
        || ((mode == "split16" || mode == "split32")
            && context != 32768 && context != 131072)) {
        usage(); return 2;
    }
    const char* authorized = std::getenv("M31_GPU_RUN_AUTHORIZED");
    if (authorized == nullptr || std::string_view(authorized) != "1") {
        std::cerr << "GPU execution blocked: lead must authorize with M31_GPU_RUN_AUTHORIZED=1\n";
        return 2;
    }

    int device_count = 0;
    MIINFER_HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device != UINT32_MAX && device >= static_cast<std::uint32_t>(device_count)) {
        std::cerr << "selected HIP device index is unavailable\n";
        return 2;
    }
    if (device == UINT32_MAX) {
        std::uint32_t gfx906_count = 0;
        for (int i = 0; i < device_count; ++i) {
            hipDeviceProp_t candidate{};
            MIINFER_HIP_CHECK(hipGetDeviceProperties(&candidate, i));
            if (std::string_view(candidate.gcnArchName).find("gfx906") == 0) {
                device = static_cast<std::uint32_t>(i);
                ++gfx906_count;
            }
        }
        if (gfx906_count != 1) {
            std::cerr << "expected exactly one gfx906; found " << gfx906_count
                      << ", select --device explicitly\n";
            return 2;
        }
    }
    MIINFER_HIP_CHECK(hipSetDevice(static_cast<int>(device)));
    hipDeviceProp_t properties{};
    MIINFER_HIP_CHECK(hipGetDeviceProperties(&properties, static_cast<int>(device)));
    if (std::string_view(properties.gcnArchName).find("gfx906") != 0) {
        std::cerr << "refusing non-gfx906 device: " << properties.gcnArchName << '\n';
        return 2;
    }
    char pci_bus_id[32]{};
    MIINFER_HIP_CHECK(hipDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), static_cast<int>(device)));
    int runtime_version = 0, driver_version = 0;
    MIINFER_HIP_CHECK(hipRuntimeGetVersion(&runtime_version));
    MIINFER_HIP_CHECK(hipDriverGetVersion(&driver_version));
    char hostname[256]{};
    if (gethostname(hostname, sizeof(hostname) - 1) != 0) std::strcpy(hostname, "unknown");
    const std::uint32_t default_splits = context <= 512 ? 4 : context <= 2048 ? 8
        : context <= 8192 ? 16 : context <= 16384 ? 32 : 64;
    const std::uint32_t candidate_splits = mode == "split8" ? 8 : mode == "split16" ? 16 :
        mode == "split32" ? 32 : default_splits;
    const std::string candidate_name = "split" + std::to_string(candidate_splits);
    std::cout << "# host=" << hostname << " gpu_index=" << device
              << " bdf=" << pci_bus_id << " name=\"" << properties.name
              << "\" arch=" << properties.gcnArchName
              << " cus=" << properties.multiProcessorCount
              << " vram_bytes=" << properties.totalGlobalMem
              << " hip_runtime_version=" << runtime_version
              << " hip_driver_version=" << driver_version
              << " context=" << context << " mode=" << mode
              << " default_active_splits=" << default_splits
              << " candidate_active_splits=" << candidate_splits
              << " grid=" << kQueryHeads << 'x' << kGridSplits
              << " block_threads=64 wavefront=64"
              << " scratch_static_bytes=" << kSplitScratchBytes
              << " repeats=" << repeats << " kv=fp16"
              << " q_heads=" << kQueryHeads << " kv_heads=" << kKvHeads
              << " head_dim=" << kHeadDim << " scale=" << kScale
              << " capacity=" << (context + 1)
              << " rng_seed=" << (0x6d310006u + context)
              << " output=fp32 gate=sigmoid graph_nodes=2\n" << std::flush;

    const std::uint32_t capacity = context + 1;
    const std::size_t kv_elements = static_cast<std::size_t>(kKvHeads) * capacity * kHeadDim;
    const std::size_t output_elements = static_cast<std::size_t>(kQueryHeads) * kHeadDim;
    std::vector<float> host_q(output_elements), host_gate(output_elements);
    std::vector<__half> host_key(kv_elements), host_value(kv_elements);
    std::mt19937 rng(0x6d310006u + context);
    std::uniform_real_distribution<float> dist(-0.125F, 0.125F);
    for (auto& x : host_q) x = dist(rng);
    for (auto& x : host_gate) x = dist(rng);
    for (std::uint32_t h = 0; h < kKvHeads; ++h) {
        for (std::uint32_t pos = 0; pos < capacity; ++pos) {
            for (std::uint32_t d = 0; d < kHeadDim; ++d) {
                const std::size_t index =
                    (static_cast<std::size_t>(h) * capacity + pos) * kHeadDim + d;
                const float poison = pos == context ? 10000.0F : dist(rng);
                host_key[index] = __float2half(poison);
                host_value[index] = __float2half(poison);
            }
        }
    }
    const auto reference = cpu_reference(host_q, host_gate, host_key, host_value, context, capacity);

    std::size_t free_before = 0, total_before = 0;
    MIINFER_HIP_CHECK(hipMemGetInfo(&free_before, &total_before));
    DeviceBuffer d_q(host_q.size() * sizeof(float));
    DeviceBuffer d_gate(host_gate.size() * sizeof(float));
    DeviceBuffer d_key(host_key.size() * sizeof(__half));
    DeviceBuffer d_value(host_value.size() * sizeof(__half));
    DeviceBuffer d_state(sizeof(miinfer::DeviceDecodeState));
    DeviceBuffer d_output(output_elements * sizeof(float));
    MIINFER_HIP_CHECK(hipMemcpy(d_q.data, host_q.data(), host_q.size() * sizeof(float), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_gate.data, host_gate.data(), host_gate.size() * sizeof(float), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_key.data, host_key.data(), host_key.size() * sizeof(__half), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(d_value.data, host_value.data(), host_value.size() * sizeof(__half), hipMemcpyHostToDevice));
    const miinfer::DeviceDecodeState state{0, context - 1, 0, 0, 0};
    MIINFER_HIP_CHECK(hipMemcpy(d_state.data, &state, sizeof(state), hipMemcpyHostToDevice));
    std::size_t free_after = 0, total_after = 0;
    MIINFER_HIP_CHECK(hipMemGetInfo(&free_after, &total_after));

    hipStream_t stream = nullptr;
    MIINFER_HIP_CHECK(hipStreamCreate(&stream));
    hipEvent_t start = nullptr, stop = nullptr;
    MIINFER_HIP_CHECK(hipEventCreate(&start));
    MIINFER_HIP_CHECK(hipEventCreate(&stop));
    const auto* q = static_cast<const float*>(d_q.data);
    const auto* gate = static_cast<const float*>(d_gate.data);
    const auto* key = static_cast<const __half*>(d_key.data);
    const auto* value = static_cast<const __half*>(d_value.data);
    const auto* decode_state = static_cast<const miinfer::DeviceDecodeState*>(d_state.data);
    auto* output = static_cast<float*>(d_output.data);

    if (unsetenv("MIINFER_ATTENTION_SPLITS") != 0) return 2;
    double warmup_wall = 0.0;
    (void)timed_direct(q, key, value, decode_state, capacity, gate, output, stream,
                       start, stop, warmup_wall);
    const auto baseline = compare(reference, copy_output(output, output_elements));
    if (!baseline.pass) {
        std::cerr << "default direct output failed CPU oracle: max_abs=" << baseline.max_abs << '\n';
        return 1;
    }
    const auto baseline_output = copy_output(output, output_elements);
    const auto default_capture_start = std::chrono::steady_clock::now();
    hipGraphExec_t default_graph = capture_attention_graph(q, key, value, decode_state,
        capacity, gate, output, stream);
    const double default_capture_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - default_capture_start).count();
    double warmup_graph_wall = 0.0;
    (void)timed_graph(default_graph, stream, start, stop, warmup_graph_wall);
    const auto default_graph_check = compare(reference, copy_output(output, output_elements));
    if (!default_graph_check.pass
        || !compare(baseline_output, copy_output(output, output_elements)).pass) {
        std::cerr << "default HIP-Graph output failed CPU/reference parity\n";
        return 1;
    }

    hipGraphExec_t candidate_graph = nullptr;
    double candidate_capture_ms = 0.0;
    if (mode != "baseline") {
        const auto split_value = std::to_string(candidate_splits);
        if (setenv("MIINFER_ATTENTION_SPLITS", split_value.c_str(), 1) != 0) return 2;
        (void)timed_direct(q, key, value, decode_state, capacity, gate, output, stream,
                           start, stop, warmup_wall);
        const auto candidate_check = compare(reference, copy_output(output, output_elements));
        const auto candidate_output = copy_output(output, output_elements);
        if (!candidate_check.pass || !compare(baseline_output, candidate_output).pass) {
            std::cerr << "candidate-split direct-launch parity failed: max_abs_vs_cpu="
                      << candidate_check.max_abs << '\n';
            return 1;
        }
        const auto candidate_capture_start = std::chrono::steady_clock::now();
        candidate_graph = capture_attention_graph(q, key, value, decode_state,
            capacity, gate, output, stream);
        candidate_capture_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - candidate_capture_start).count();
        double candidate_graph_wall = 0.0;
        (void)timed_graph(candidate_graph, stream, start, stop, candidate_graph_wall);
        const auto candidate_graph_check = compare(reference, copy_output(output, output_elements));
        const auto candidate_graph_output = copy_output(output, output_elements);
        if (!candidate_graph_check.pass || !compare(baseline_output, candidate_graph_output).pass) {
            std::cerr << "candidate-split HIP-Graph parity failed: max_abs_vs_cpu="
                      << candidate_graph_check.max_abs << '\n';
            return 1;
        }
        if (unsetenv("MIINFER_ATTENTION_SPLITS") != 0) return 2;
    }

    std::cout << "# correctness=PASS default_active_splits=" << default_splits
              << " candidate_active_splits=" << candidate_splits
              << " graph_grid=" << kQueryHeads << 'x' << kGridSplits
              << " block_threads=64 wavefront=64"
              << " default_graph_capture_ms=" << default_capture_ms
              << " candidate_graph_capture_ms=" << candidate_capture_ms
              << " static_scratch_bytes=" << kSplitScratchBytes
              << " allocation_delta_bytes=" << (free_before - free_after)
              << " live_allocated_bytes=" << (total_after - free_after)
              << " free_vram_before_bytes=" << free_before
              << " total_vram_before_bytes=" << total_before
              << " free_vram_after_bytes=" << free_after
              << " total_vram_after_bytes=" << total_after
              << " reference_abs_tolerance=" << kAbsTolerance
              << " default_max_abs_vs_reference=" << default_graph_check.max_abs
              << " future_kv_poison=10000.0 ignored\n" << std::flush;

    for (std::uint32_t sample = 1; sample <= repeats; ++sample) {
        for (int candidate = 0; candidate <= (mode == "baseline" ? 0 : 1); ++candidate) {
            const std::string variant_storage = candidate == 0 ? "default" : candidate_name;
            const std::string_view variant = variant_storage;
            if (candidate != 0) {
                const auto split_value = std::to_string(candidate_splits);
                if (setenv("MIINFER_ATTENTION_SPLITS", split_value.c_str(), 1) != 0) return 2;
            }
            if (candidate == 0 && unsetenv("MIINFER_ATTENTION_SPLITS") != 0) return 2;
            double direct_wall = 0.0, graph_wall = 0.0;
            const double direct_device = timed_direct(q, key, value, decode_state,
                capacity, gate, output, stream, start, stop, direct_wall);
            const double graph_device = timed_graph(
                variant == "default" ? default_graph : candidate_graph,
                stream, start, stop, graph_wall);
            const auto actual = copy_output(output, output_elements);
            const auto oracle_check = compare(reference, actual);
            const auto parity_check = compare(baseline_output, actual);
            if (!oracle_check.pass || !parity_check.pass) {
                std::cerr << "sample parity failed variant=" << variant
                          << " max_abs_vs_reference=" << oracle_check.max_abs
                          << " max_abs_vs_default=" << parity_check.max_abs << '\n';
                return 1;
            }
            std::cout << std::fixed << std::setprecision(6)
                      << "sample context=" << context << " variant=" << variant
                      << " active_splits=" << (candidate == 0 ? default_splits : candidate_splits)
                      << " grid=" << kQueryHeads << 'x' << kGridSplits
                      << " sample=" << sample
                      << " direct_kernel_pair_ms=" << direct_device
                      << " direct_submit_to_complete_ms=" << direct_wall
                      << " graph_kernel_pair_ms=" << graph_device
                      << " graph_replay_submit_to_complete_ms=" << graph_wall
                      << " scratch_static_bytes=" << kSplitScratchBytes
                      << " observed_allocation_delta_bytes=" << (free_before - free_after)
                      << " max_abs_vs_reference=" << oracle_check.max_abs
                      << " max_abs_vs_default=" << parity_check.max_abs
                      << "\n" << std::flush;
        }
    }

    if (candidate_graph != nullptr) MIINFER_HIP_CHECK(hipGraphExecDestroy(candidate_graph));
    MIINFER_HIP_CHECK(hipGraphExecDestroy(default_graph));
    MIINFER_HIP_CHECK(hipEventDestroy(stop));
    MIINFER_HIP_CHECK(hipEventDestroy(start));
    MIINFER_HIP_CHECK(hipStreamDestroy(stream));
    return 0;
}
