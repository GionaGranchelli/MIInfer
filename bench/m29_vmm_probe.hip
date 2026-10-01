#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "miinfer/prefill_v2/constants.hpp"
#include "miinfer/qwen35_model.hpp"

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::size_t kKiB = 1024;
constexpr std::size_t kMiB = 1024 * kKiB;
constexpr std::uint32_t kPattern = 0x29a10000u;
constexpr std::size_t kBandwidthBytes = 64 * kMiB;
constexpr int kBandwidthRepeats = 15;
std::ostringstream event_log;
std::streambuf* saved_stdout = nullptr;
std::string result_path;

void emit_status(const char* operation, hipError_t status, double ms = 0.0) {
    std::cout << "{\"event\":\"hip\",\"operation\":\"" << operation
              << "\",\"status\":\"" << hipGetErrorName(status)
              << "\",\"status_code\":" << static_cast<int>(status)
              << ",\"elapsed_ms\":" << std::setprecision(9) << ms << "}\n";
}

void require_status(const char* operation, hipError_t status, double ms = 0.0) {
    emit_status(operation, status, ms);
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorName(status)
                                 + " (" + std::to_string(static_cast<int>(status)) + ")");
    }
}

template<class Fn>
double timed_status(const char* operation, Fn&& fn) {
    const auto start = Clock::now();
    const hipError_t status = fn();
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    require_status(operation, status, ms);
    return ms;
}

std::pair<double, double> mean_stddev(const std::vector<double>& values) {
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double squares = 0.0;
    for (const double value : values) squares += (value - mean) * (value - mean);
    return {mean, std::sqrt(squares / (values.size() - 1))};
}

__global__ void fill_pattern(std::uint32_t* data, std::size_t count, std::uint32_t seed) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) data[i] = seed ^ static_cast<std::uint32_t>(i * 2654435761u);
}

__global__ void check_pattern(const std::uint32_t* data, std::size_t count,
                              std::uint32_t seed, std::uint32_t* mismatch) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count && data[i] != (seed ^ static_cast<std::uint32_t>(i * 2654435761u))) {
        atomicExch(mismatch, 1u);
    }
}

__global__ void copy_transform(const float* src, float* dst, std::size_t count) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = src[i] + 1.0f;
}

__global__ void fill_float(float* data, std::size_t count, float value) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) data[i] = value;
}

__global__ void check_transform(const float* src, const float* dst, std::size_t count,
                                std::uint32_t* mismatch) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count && dst[i] != src[i] + 1.0f) atomicExch(mismatch, 1u);
}

void check_device_pattern(void* address, std::size_t bytes, std::uint32_t seed) {
    auto* mismatch = static_cast<std::uint32_t*>(nullptr);
    require_status("hipMalloc(verify flag)", hipMalloc(&mismatch, sizeof(*mismatch)));
    try {
        const std::size_t count = bytes / sizeof(std::uint32_t);
        require_status("hipMemset(verify flag)", hipMemset(mismatch, 0, sizeof(*mismatch)));
        fill_pattern<<<static_cast<unsigned>((count + 255) / 256), 256>>>(
            static_cast<std::uint32_t*>(address), count, seed);
        require_status("hipGetLastError(fill pattern)", hipGetLastError());
        check_pattern<<<static_cast<unsigned>((count + 255) / 256), 256>>>(
            static_cast<const std::uint32_t*>(address), count, seed, mismatch);
        require_status("hipGetLastError(check pattern)", hipGetLastError());
        require_status("hipDeviceSynchronize(pattern)", hipDeviceSynchronize());
        std::uint32_t host_mismatch = 1;
        require_status("hipMemcpy(verify flag)", hipMemcpy(&host_mismatch, mismatch,
            sizeof(host_mismatch), hipMemcpyDeviceToHost));
        if (host_mismatch != 0) throw std::runtime_error("GPU pattern verification failed");
        std::cout << "{\"event\":\"gpu_pattern\",\"bytes\":" << bytes
                  << ",\"seed\":" << seed << ",\"correct\":true}\n";
    } catch (...) {
        (void)hipFree(mismatch);
        throw;
    }
    require_status("hipFree(verify flag)", hipFree(mismatch));
}

void verify_device_pattern(const void* address, std::size_t bytes, std::uint32_t seed) {
    auto* mismatch = static_cast<std::uint32_t*>(nullptr);
    require_status("hipMalloc(verify-only flag)", hipMalloc(&mismatch, sizeof(*mismatch)));
    require_status("hipMemset(verify-only flag)", hipMemset(mismatch, 0, sizeof(*mismatch)));
    const std::size_t count = bytes / sizeof(std::uint32_t);
    check_pattern<<<static_cast<unsigned>((count + 255) / 256), 256>>>(
        static_cast<const std::uint32_t*>(address), count, seed, mismatch);
    require_status("hipGetLastError(verify-only)", hipGetLastError());
    require_status("hipDeviceSynchronize(verify-only)", hipDeviceSynchronize());
    std::uint32_t host_mismatch = 1;
    require_status("hipMemcpy(verify-only flag)", hipMemcpy(&host_mismatch, mismatch,
        sizeof(host_mismatch), hipMemcpyDeviceToHost));
    require_status("hipFree(verify-only flag)", hipFree(mismatch));
    if (host_mismatch != 0) throw std::runtime_error("GPU verify-only check failed");
    std::cout << "{\"event\":\"gpu_verify_only\",\"bytes\":" << bytes
              << ",\"seed\":" << seed << ",\"correct\":true}\n";
}

struct VmmRange {
    void* address = nullptr;
    std::size_t reserved = 0;
    std::vector<std::pair<void*, std::size_t>> mappings;
    std::vector<hipMemGenericAllocationHandle_t> handles;

    ~VmmRange() {
        for (auto it = mappings.rbegin(); it != mappings.rend(); ++it) {
            const auto status = hipMemUnmap(it->first, it->second);
            emit_status("hipMemUnmap(cleanup)", status);
        }
        for (auto handle : handles) {
            const auto status = hipMemRelease(handle);
            emit_status("hipMemRelease(cleanup)", status);
        }
        if (address != nullptr) {
            const auto status = hipMemAddressFree(address, reserved);
            emit_status("hipMemAddressFree(cleanup)", status);
        }
    }

    void reserve(std::size_t bytes, std::size_t alignment) {
        reserved = bytes;
        timed_status("hipMemAddressReserve", [&] {
            return hipMemAddressReserve(&address, bytes, alignment, nullptr, 0);
        });
        std::cout << "{\"event\":\"reserved_range\",\"base\":"
                  << reinterpret_cast<std::uintptr_t>(address)
                  << ",\"bytes\":" << reserved << ",\"alignment\":" << alignment << "}\n";
    }

    hipMemGenericAllocationHandle_t create(std::size_t bytes,
                                            const hipMemAllocationProp& prop) {
        hipMemGenericAllocationHandle_t handle{};
        timed_status("hipMemCreate", [&] { return hipMemCreate(&handle, bytes, &prop, 0); });
        handles.push_back(handle);
        return handle;
    }

    void map(void* at, std::size_t bytes, hipMemGenericAllocationHandle_t handle) {
        timed_status("hipMemMap", [&] { return hipMemMap(at, bytes, 0, handle, 0); });
        mappings.emplace_back(at, bytes);
        hipMemAccessDesc access{};
        access.location.type = hipMemLocationTypeDevice;
        access.location.id = 0;
        access.flags = hipMemAccessFlagsProtReadWrite;
        timed_status("hipMemSetAccess", [&] { return hipMemSetAccess(at, bytes, &access, 1); });
    }

    void release_handle(hipMemGenericAllocationHandle_t handle) {
        timed_status("hipMemRelease", [&] { return hipMemRelease(handle); });
        std::erase(handles, handle);
    }

    void unmap(void* at, std::size_t bytes) {
        timed_status("hipMemUnmap", [&] { return hipMemUnmap(at, bytes); });
        std::erase(mappings, std::pair<void*, std::size_t>{at, bytes});
    }
};

hipMemAllocationProp allocation_properties(int device) {
    hipMemAllocationProp prop{};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = device;
    prop.requestedHandleTypes = hipMemHandleTypeNone;
    return prop;
}

void check_memory_state(const char* label) {
    std::size_t free_bytes = 0, total_bytes = 0;
    require_status("hipMemGetInfo", hipMemGetInfo(&free_bytes, &total_bytes));
    std::cout << "{\"event\":\"memory_state\",\"label\":\"" << label
              << "\",\"free_bytes\":" << free_bytes
              << ",\"total_bytes\":" << total_bytes << "}\n";
}

std::string json_escape(const std::string& input) {
    std::string result;
    for (const char c : input) {
        if (c == '\\' || c == '"') result.push_back('\\');
        if (c == '\n') result += "\\n";
        else if (c != '\r') result.push_back(c);
    }
    return result;
}

void flush_results(const char* result, const std::string& error = {}) {
    if (saved_stdout != nullptr) {
        std::cout.rdbuf(saved_stdout);
        saved_stdout = nullptr;
    }
    std::ofstream file;
    std::ostream* output = &std::cout;
    if (!result_path.empty()) {
        file.open(result_path);
        if (!file) throw std::runtime_error("cannot open result file: " + result_path);
        output = &file;
    }
    *output << "{\"events\":[";
    const std::string events = event_log.str();
    bool first = true;
    std::size_t start = 0;
    while (start < events.size()) {
        const auto end = events.find('\n', start);
        const auto count = (end == std::string::npos ? events.size() : end) - start;
        if (count != 0) {
            if (!first) *output << ',';
            *output << events.substr(start, count);
            first = false;
        }
        if (end == std::string::npos) break;
        start = end + 1;
    }
    *output << "],\"result\":\"" << result << "\"";
    if (!error.empty()) *output << ",\"error\":\"" << json_escape(error) << "\"";
    *output << "}\n";
}

void graph_tests(std::size_t granularity, const hipMemAllocationProp& prop) {
    const std::size_t region = std::max<std::size_t>(granularity, 4 * kMiB);
    VmmRange range;
    range.reserve(region * 4, granularity);
    auto first = range.create(region, prop);
    range.map(range.address, region, first);
    check_device_pattern(range.address, region, kPattern);

    hipStream_t stream = nullptr;
    hipGraph_t graph = nullptr;
    hipGraphExec_t executable = nullptr;
    require_status("hipStreamCreate(graph)", hipStreamCreate(&stream));
    require_status("hipStreamBeginCapture", hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    fill_pattern<<<1, 256, 0, stream>>>(static_cast<std::uint32_t*>(range.address), 256, kPattern + 1);
    require_status("hipGetLastError(graph capture)", hipGetLastError());
    require_status("hipStreamEndCapture", hipStreamEndCapture(stream, &graph));
    require_status("hipGraphInstantiate", hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (int i = 0; i < 10; ++i) {
        require_status("hipGraphLaunch(stable VMM)", hipGraphLaunch(executable, stream));
        require_status("hipStreamSynchronize(stable VMM)", hipStreamSynchronize(stream));
    }
    verify_device_pattern(range.address, 256 * sizeof(std::uint32_t), kPattern + 1);
    std::cout << "{\"event\":\"graph_stable_mapping\",\"replays\":10,\"result\":\"PASS\"}\n";

    auto second = range.create(region, prop);
    range.map(static_cast<char*>(range.address) + region, region, second);
    require_status("hipGraphLaunch(after adjacent map)", hipGraphLaunch(executable, stream));
    require_status("hipStreamSynchronize(after adjacent map)", hipStreamSynchronize(stream));
    verify_device_pattern(range.address, 256 * sizeof(std::uint32_t), kPattern + 1);
    check_device_pattern(static_cast<char*>(range.address) + region, region, kPattern + 2);
    std::cout << "{\"event\":\"graph_growth_after_capture\",\"base_stable\":true,\"result\":\"PASS\"}\n";

    require_status("hipGraphExecDestroy", hipGraphExecDestroy(executable)); executable = nullptr;
    require_status("hipGraphDestroy", hipGraphDestroy(graph)); graph = nullptr;
    require_status("hipStreamDestroy", hipStreamDestroy(stream)); stream = nullptr;
    range.unmap(static_cast<char*>(range.address) + region, region);
    range.release_handle(second);
    range.unmap(range.address, region);
    range.release_handle(first);

    auto old_backing = range.create(region, prop);
    range.map(range.address, region, old_backing);
    require_status("hipStreamCreate(remap graph)", hipStreamCreate(&stream));
    require_status("hipStreamBeginCapture(remap)", hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    fill_pattern<<<1, 256, 0, stream>>>(static_cast<std::uint32_t*>(range.address), 256, kPattern + 3);
    require_status("hipGetLastError(remap capture)", hipGetLastError());
    require_status("hipStreamEndCapture(remap)", hipStreamEndCapture(stream, &graph));
    require_status("hipGraphInstantiate(remap)", hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    require_status("hipGraphLaunch(before remap)", hipGraphLaunch(executable, stream));
    require_status("hipStreamSynchronize(before remap)", hipStreamSynchronize(stream));
    verify_device_pattern(range.address, 256 * sizeof(std::uint32_t), kPattern + 3);
    range.unmap(range.address, region);
    range.release_handle(old_backing);

    auto new_backing = range.create(region, prop);
    range.map(range.address, region, new_backing);
    require_status("hipGraphLaunch(after same-VA remap)", hipGraphLaunch(executable, stream));
    require_status("hipStreamSynchronize(after same-VA remap)", hipStreamSynchronize(stream));
    verify_device_pattern(range.address, 256 * sizeof(std::uint32_t), kPattern + 3);
    std::cout << "{\"event\":\"graph_same_va_remap\",\"result\":\"SUPPORTED\"}\n";
    require_status("hipGraphExecDestroy(remap)", hipGraphExecDestroy(executable));
    require_status("hipGraphDestroy(remap)", hipGraphDestroy(graph));
    require_status("hipStreamDestroy(remap)", hipStreamDestroy(stream));
}

void bandwidth_case(const char* kind, float* src, float* dst, std::size_t count,
                    std::vector<double>& samples) {
    for (int i = 0; i < 3; ++i) {
        copy_transform<<<static_cast<unsigned>((count + 255) / 256), 256>>>(src, dst, count);
        require_status("hipGetLastError(bandwidth warmup)", hipGetLastError());
    }
    require_status("hipDeviceSynchronize(bandwidth warmup)", hipDeviceSynchronize());
    hipEvent_t start = nullptr, stop = nullptr;
    require_status("hipEventCreate(bandwidth start)", hipEventCreate(&start));
    require_status("hipEventCreate(bandwidth stop)", hipEventCreate(&stop));
    for (int sample = 0; sample < 9; ++sample) {
        require_status("hipEventRecord(start)", hipEventRecord(start));
        for (int i = 0; i < kBandwidthRepeats; ++i) {
            copy_transform<<<static_cast<unsigned>((count + 255) / 256), 256>>>(src, dst, count);
            require_status("hipGetLastError(bandwidth)", hipGetLastError());
        }
        require_status("hipEventRecord(stop)", hipEventRecord(stop));
        require_status("hipEventSynchronize(stop)", hipEventSynchronize(stop));
        float ms = 0.0f;
        require_status("hipEventElapsedTime", hipEventElapsedTime(&ms, start, stop));
        const double gbps = (2.0 * count * sizeof(float) * kBandwidthRepeats)
            / (static_cast<double>(ms) * 1.0e6);
        samples.push_back(gbps);
        std::cout << "{\"event\":\"bandwidth_sample\",\"allocator\":\"" << kind
                  << "\",\"sample\":" << sample << ",\"ms\":" << ms
                  << ",\"effective_GB_s\":" << gbps << "}\n";
    }
    require_status("hipEventDestroy(start)", hipEventDestroy(start));
    require_status("hipEventDestroy(stop)", hipEventDestroy(stop));
}

void compare_allocators(std::size_t granularity, const hipMemAllocationProp& prop) {
    const std::size_t bytes = kBandwidthBytes;
    const std::size_t count = bytes / sizeof(float);
    float *host_src = nullptr, *host_dst = nullptr;
    require_status("hipMalloc(hipMalloc source)", hipMalloc(&host_src, bytes));
    require_status("hipMalloc(hipMalloc destination)", hipMalloc(&host_dst, bytes));
    VmmRange range;
    range.reserve(bytes * 2, granularity);
    auto src_handle = range.create(bytes, prop);
    range.map(range.address, bytes, src_handle);
    auto dst_handle = range.create(bytes, prop);
    auto* vmm_dst = reinterpret_cast<float*>(static_cast<char*>(range.address) + bytes);
    range.map(vmm_dst, bytes, dst_handle);

    fill_float<<<static_cast<unsigned>((count + 255) / 256), 256>>>(host_src, count, 1.25f);
    require_status("hipGetLastError(init hipMalloc)", hipGetLastError());
    require_status("hipMemcpy(hipMalloc input to VMM)", hipMemcpy(range.address, host_src, bytes, hipMemcpyDeviceToDevice));
    std::vector<double> malloc_samples, vmm_samples;
    bandwidth_case("hipMalloc", host_src, host_dst, count, malloc_samples);
    bandwidth_case("VMM", static_cast<float*>(range.address), vmm_dst, count, vmm_samples);

    auto* mismatch = static_cast<std::uint32_t*>(nullptr);
    require_status("hipMalloc(bandwidth verify)", hipMalloc(&mismatch, sizeof(*mismatch)));
    require_status("hipMemset(bandwidth verify malloc)", hipMemset(mismatch, 0, sizeof(*mismatch)));
    check_transform<<<static_cast<unsigned>((count + 255) / 256), 256>>>(host_src, host_dst, count, mismatch);
    require_status("hipGetLastError(verify hipMalloc)", hipGetLastError());
    require_status("hipDeviceSynchronize(verify hipMalloc)", hipDeviceSynchronize());
    std::uint32_t failed = 1;
    require_status("hipMemcpy(verify hipMalloc result)", hipMemcpy(&failed, mismatch, sizeof(failed), hipMemcpyDeviceToHost));
    if (failed) throw std::runtime_error("hipMalloc bandwidth correctness failed");
    require_status("hipMemset(bandwidth verify VMM)", hipMemset(mismatch, 0, sizeof(*mismatch)));
    check_transform<<<static_cast<unsigned>((count + 255) / 256), 256>>>(
        static_cast<float*>(range.address), vmm_dst, count, mismatch);
    require_status("hipGetLastError(verify VMM)", hipGetLastError());
    require_status("hipDeviceSynchronize(verify VMM)", hipDeviceSynchronize());
    require_status("hipMemcpy(verify VMM result)", hipMemcpy(&failed, mismatch, sizeof(failed), hipMemcpyDeviceToHost));
    if (failed) throw std::runtime_error("VMM bandwidth correctness failed");
    require_status("hipFree(bandwidth verify)", hipFree(mismatch));

    const double malloc_median = [&] { auto s = malloc_samples; std::sort(s.begin(), s.end()); return s[s.size()/2]; }();
    const double vmm_median = [&] { auto s = vmm_samples; std::sort(s.begin(), s.end()); return s[s.size()/2]; }();
    const auto [malloc_mean, malloc_sd] = mean_stddev(malloc_samples);
    const auto [vmm_mean, vmm_sd] = mean_stddev(vmm_samples);
    std::cout << "{\"event\":\"bandwidth_summary\",\"bytes_per_buffer\":" << bytes
              << ",\"hipMalloc_median_GB_s\":" << malloc_median
              << ",\"VMM_median_GB_s\":" << vmm_median
              << ",\"hipMalloc_mean_GB_s\":" << malloc_mean
              << ",\"hipMalloc_sd_GB_s\":" << malloc_sd
              << ",\"VMM_mean_GB_s\":" << vmm_mean
              << ",\"VMM_sd_GB_s\":" << vmm_sd
              << ",\"relative_delta_pct\":" << 100.0 * (vmm_median / malloc_median - 1.0)
              << ",\"samples\":9,\"correctness\":\"PASS\"}\n";
    require_status("hipFree(hipMalloc source)", hipFree(host_src));
    require_status("hipFree(hipMalloc destination)", hipFree(host_dst));
}

void incremental_commit(std::size_t granularity, std::size_t bytes_per_token,
                        const hipMemAllocationProp& prop) {
    constexpr std::size_t logical_tokens = 262144;
    const std::size_t reserve_bytes = logical_tokens * bytes_per_token;
    VmmRange range;
    range.reserve(reserve_bytes, granularity);
    std::cout << "{\"event\":\"logical_reservation\",\"logical_tokens\":" << logical_tokens
              << ",\"bytes_per_token\":" << bytes_per_token
              << ",\"requested_bytes\":" << reserve_bytes
              << ",\"base\":" << reinterpret_cast<std::uintptr_t>(range.address)
              << ",\"physical_committed_bytes\":0}\n";

    std::size_t committed = 0;
    std::size_t initial_free = 0, initial_total = 0;
    require_status("hipMemGetInfo(before incremental)", hipMemGetInfo(&initial_free, &initial_total));
    for (const std::size_t tokens : {16384u, 32768u, 65536u, 131072u}) {
        const std::size_t target = ((tokens * bytes_per_token + granularity - 1) / granularity) * granularity;
        const std::size_t growth = target - committed;
        const auto before = Clock::now();
        auto handle = range.create(growth, prop);
        auto* at = static_cast<char*>(range.address) + committed;
        range.map(at, growth, handle);
        range.release_handle(handle);
        const auto after = Clock::now();
        check_device_pattern(at, growth, kPattern + static_cast<std::uint32_t>(tokens));
        if (committed != 0) check_device_pattern(range.address, std::min<std::size_t>(committed, granularity), kPattern + 16384u);
        committed = target;
        std::size_t free_bytes = 0, total_bytes = 0;
        require_status("hipMemGetInfo(incremental)", hipMemGetInfo(&free_bytes, &total_bytes));
        std::cout << "{\"event\":\"incremental_commit\",\"context_tokens\":" << tokens
                  << ",\"physical_committed_bytes\":" << committed
                  << ",\"growth_bytes\":" << growth
                  << ",\"commit_and_map_ms\":"
                  << std::chrono::duration<double, std::milli>(after - before).count()
                  << ",\"free_vram_bytes\":" << free_bytes
                  << ",\"physical_vram_growth_bytes\":" << (initial_free - free_bytes)
                  << ",\"base\":" << reinterpret_cast<std::uintptr_t>(range.address)
                  << ",\"correctness\":\"PASS\"}\n";
    }
}

} // namespace

int main(int argc, char** argv) try {
    std::string model_path = "/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--json-output" && i + 1 < argc) result_path = argv[++i];
        else if (model_path == "/home/fedora-workstation/models/Qwen3.8-27B-Q4_K_M.gguf") model_path = arg;
        else throw std::runtime_error("unexpected argument: " + arg);
    }
    saved_stdout = std::cout.rdbuf(event_log.rdbuf());
    int device = 0;
    require_status("hipGetDevice", hipGetDevice(&device));
    hipDeviceProp_t device_props{};
    require_status("hipGetDeviceProperties", hipGetDeviceProperties(&device_props, device));
    if (std::string(device_props.gcnArchName).find("gfx906") != 0) {
        throw std::runtime_error(std::string("unsupported device: ") + device_props.gcnArchName);
    }
    const auto model = miinfer::Qwen35Model::load(model_path);
    const auto& config = model.config();
    constexpr std::size_t attention_layers_per_block = miinfer::prefill_v2::kAttentionPerBlock;
    constexpr std::size_t layers_per_block = miinfer::prefill_v2::kLayersPerBlock;
    const std::size_t attention_layers = (config.main_layer_count / layers_per_block) * attention_layers_per_block;
    const std::size_t kv_bytes_per_token = attention_layers * 2 * miinfer::prefill_v2::kKvHeads
        * miinfer::prefill_v2::kHeadDim * sizeof(__half);
    if (attention_layers == 0 || config.main_layer_count % layers_per_block != 0) {
        throw std::runtime_error("model layer topology does not match the compiled MIInfer KV layout");
    }

    hipMemAllocationProp prop = allocation_properties(device);
    std::size_t minimum_granularity = 0, recommended_granularity = 0;
    timed_status("hipMemGetAllocationGranularity(minimum)", [&] {
        return hipMemGetAllocationGranularity(&minimum_granularity, &prop, hipMemAllocationGranularityMinimum);
    });
    timed_status("hipMemGetAllocationGranularity(recommended)", [&] {
        return hipMemGetAllocationGranularity(&recommended_granularity, &prop, hipMemAllocationGranularityRecommended);
    });
    int vmm_supported = 0;
    require_status("hipDeviceGetAttribute(VMM)", hipDeviceGetAttribute(
        &vmm_supported, hipDeviceAttributeVirtualMemoryManagementSupported, device));
    std::cout << "{\"event\":\"environment\",\"device\":\"" << device_props.name
              << "\",\"architecture\":\"" << device_props.gcnArchName
              << "\",\"vmm_supported\":" << vmm_supported
              << ",\"minimum_granularity_bytes\":" << minimum_granularity
              << ",\"recommended_granularity_bytes\":" << recommended_granularity
              << ",\"kv_layout\":\"FP16 K+V contiguous per layer, [head, token, dim]\""
              << ",\"attention_layers\":" << attention_layers
              << ",\"kv_bytes_per_token\":" << kv_bytes_per_token << "}\n";
    if (minimum_granularity == 0 || recommended_granularity == 0) {
        throw std::runtime_error("HIP reported zero allocation granularity");
    }
    const std::size_t align = recommended_granularity;

    check_memory_state("before");
    {
        VmmRange lifecycle;
        const std::size_t bytes = std::max<std::size_t>(align, 1 * kMiB);
        lifecycle.reserve(bytes * 2, align);
        auto handle = lifecycle.create(bytes, prop);
        lifecycle.map(lifecycle.address, bytes, handle);
        check_device_pattern(lifecycle.address, bytes, kPattern + 10);
        lifecycle.unmap(lifecycle.address, bytes);
        lifecycle.release_handle(handle);
        auto replacement = lifecycle.create(bytes, prop);
        lifecycle.map(lifecycle.address, bytes, replacement);
        check_device_pattern(lifecycle.address, bytes, kPattern + 11);
        std::cout << "{\"event\":\"basic_lifecycle\",\"result\":\"PASS\"}\n";
    }
    require_status("hipDeviceSynchronize(after lifecycle)", hipDeviceSynchronize());
    check_memory_state("after_basic_lifecycle");

    graph_tests(align, prop);
    require_status("hipDeviceSynchronize(after graph tests)", hipDeviceSynchronize());
    check_memory_state("after_graph_tests");

    compare_allocators(align, prop);
    require_status("hipDeviceSynchronize(after bandwidth)", hipDeviceSynchronize());
    check_memory_state("after_bandwidth");

    incremental_commit(align, kv_bytes_per_token, prop);
    require_status("hipDeviceSynchronize(after incremental commit)", hipDeviceSynchronize());
    check_memory_state("after_incremental_commit");
    std::cout << "{\"event\":\"probe_summary\",\"result\":\"PASS\",\"stable_mapping\":\"PASS\""
              << ",\"incremental_commit\":\"PASS\",\"gpu_access\":\"PASS\""
              << ",\"unmap_recommit\":\"PASS\",\"graph_capture_replay\":\"PASS\""
              << ",\"graph_growth\":\"PASS\",\"graph_same_va_remap\":\"SUPPORTED\"}\n";
    flush_results("PASS");
    return 0;
} catch (const std::exception& error) {
    flush_results("FAIL", error.what());
    return 1;
}
