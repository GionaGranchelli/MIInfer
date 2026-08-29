#include "miinfer/build_config.hpp"
#include "miinfer/build_info.hpp"
#include "miinfer/device_validation.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/memory_stream.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Options {
    std::size_t bytes = 0;
    int warmup = 10;
    int iterations = 1000;
    int device = -1;
    std::string json_output;
};

bool parse_positive(const char* text, int& value) {
    try { value = std::stoi(text); } catch (...) { return false; }
    return value > 0;
}

bool parse_bytes(const char* text, std::size_t& value) {
    try {
        const auto parsed = std::stoull(text, nullptr, 0);
        if (parsed == 0) return false;
        value = static_cast<std::size_t>(parsed);
    } catch (...) { return false; }
    return true;
}

void usage() {
    std::cerr << "usage: miinfer-memory-stream-bench --bytes N [options]\n"
              << "  --bytes N       bytes in one contiguous copy\n"
              << "  --warmup N      warm-up launches (default: 10)\n"
              << "  --iterations N  measured launches (default: 1000)\n"
              << "  --device N      HIP device (default: first gfx906)\n"
              << "  --json-output P write JSONL output\n";
}

bool parse(int argc, char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (i + 1 >= argc) return false;
        if (arg == "--bytes") {
            if (!parse_bytes(argv[++i], options.bytes)) return false;
        } else if (arg == "--warmup") {
            if (!parse_positive(argv[++i], options.warmup)) return false;
        } else if (arg == "--iterations") {
            if (!parse_positive(argv[++i], options.iterations)) return false;
        } else if (arg == "--device") {
            if (!parse_positive(argv[++i], options.device)) return false;
        } else if (arg == "--json-output") {
            options.json_output = argv[++i];
        } else {
            return false;
        }
    }
    return options.bytes != 0;
}

std::string escape(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (const char c : value) {
        if (c == '"' || c == '\\') out << '\\';
        out << c;
    }
    out << '"';
    return out.str();
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2;
    return values.size() % 2 == 0 ? (values[middle - 1] + values[middle]) / 2.0 : values[middle];
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse(argc, argv, options)) {
        usage();
        return 2;
    }
    miinfer::DeviceInfo device;
    std::string error;
    if (!miinfer::validate_gfx906_device(options.device, device, error)) {
        std::cerr << "memory stream benchmark unavailable: " << error << '\n';
        return 1;
    }

    constexpr int buffers = 3;
    std::vector<unsigned char*> sources(buffers, nullptr);
    std::vector<unsigned char*> destinations(buffers, nullptr);
    for (int i = 0; i < buffers; ++i) {
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&sources[i]), options.bytes));
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&destinations[i]), options.bytes));
        MIINFER_HIP_CHECK(hipMemset(sources[i], static_cast<int>(i + 1), options.bytes));
    }
    auto launch = [&](int index) {
        miinfer::launch_diagnostic_stream_copy(sources[index], destinations[index], options.bytes);
    };
    for (int i = 0; i < options.warmup; ++i) launch(i % buffers);
    MIINFER_HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    MIINFER_HIP_CHECK(hipEventCreate(&start));
    MIINFER_HIP_CHECK(hipEventCreate(&stop));
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(options.iterations));
    for (int i = 0; i < options.iterations; ++i) {
        MIINFER_HIP_CHECK(hipEventRecord(start, nullptr));
        launch(i % buffers);
        MIINFER_HIP_CHECK(hipEventRecord(stop, nullptr));
        MIINFER_HIP_CHECK(hipEventSynchronize(stop));
        float milliseconds = 0.0F;
        MIINFER_HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
        samples.push_back(static_cast<double>(milliseconds) * 1000.0);
    }
    const double med = median(samples);
    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    double variance = 0.0;
    for (const double sample : samples) variance += (sample - mean) * (sample - mean);
    variance /= samples.size();
    // A copy moves one read and one write stream; this is an empirical copy
    // throughput number, not a claim about peak HBM traffic.
    const double traffic_bytes = 2.0 * static_cast<double>(options.bytes);
    const double effective_gb_per_second = traffic_bytes / (med * 1000.0);

    std::ostream* output = &std::cout;
    std::ofstream file;
    if (!options.json_output.empty()) {
        file.open(options.json_output);
        if (!file) return 1;
        output = &file;
    }
    *output << std::fixed << std::setprecision(9)
            << "{\"experiment\":\"EXP-0003\",\"workload\":\"DIAGNOSTIC — CONTIGUOUS DEVICE COPY\""
            << ",\"bytes\":" << options.bytes
            << ",\"warmup_iterations\":" << options.warmup
            << ",\"measured_iterations\":" << options.iterations
            << ",\"mean_us\":" << mean << ",\"median_us\":" << med
            << ",\"min_us\":" << *std::min_element(samples.begin(), samples.end())
            << ",\"max_us\":" << *std::max_element(samples.begin(), samples.end())
            << ",\"stddev_us\":" << std::sqrt(variance)
            << ",\"effective_gb_per_second\":" << effective_gb_per_second
            << ",\"traffic_bytes\":" << traffic_bytes
            << ",\"gpu\":" << escape(device.name) << ",\"gfx\":" << escape(device.architecture)
            << ",\"vram_bytes\":" << device.total_vram_bytes
            << ",\"git_commit\":" << escape(MIINFER_GIT_COMMIT)
            << ",\"git_dirty\":" << escape(MIINFER_GIT_DIRTY)
            << ",\"build_type\":" << escape(MIINFER_BUILD_TYPE)
            << ",\"hip_compiler\":" << escape(MIINFER_HIP_COMPILER)
            << ",\"samples_us\":[";
    for (std::size_t i = 0; i < samples.size(); ++i) {
        if (i != 0) *output << ',';
        *output << samples[i];
    }
    *output << "]}\n";

    MIINFER_HIP_CHECK(hipEventDestroy(stop));
    MIINFER_HIP_CHECK(hipEventDestroy(start));
    for (int i = 0; i < buffers; ++i) {
        MIINFER_HIP_CHECK(hipFree(destinations[i]));
        MIINFER_HIP_CHECK(hipFree(sources[i]));
    }
    return 0;
}
