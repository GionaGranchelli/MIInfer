#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "miinfer/device_validation.hpp"
#include "miinfer/hip_check.hpp"
#include "miinfer/prefill_v2/model.hpp"
#include "miinfer/prefill_v2/persistent_session.hpp"
#include "miinfer/qwen35_model.hpp"

#include <hip/hip_runtime.h>

using namespace miinfer;
using namespace miinfer::prefill_v2;

namespace {

constexpr std::uint32_t kPrefix = 1024;
constexpr std::uint32_t kSuffix = 1024;
constexpr std::uint32_t kCapacity = 4096;
constexpr std::uint32_t kControlTile = 512;
constexpr std::uint32_t kCandidateTile = 1024;
constexpr int kSamples = 5;

struct Run {
    double wall_ms = 0.0;
    double gpu_ms = 0.0;
    std::vector<float> hidden;
    std::vector<float> logits;
    std::uint32_t first_token = 0;
    std::uint32_t next_token = 0;
};

struct ErrorStats {
    double max_abs = 0.0;
    double cosine = 0.0;
};

Run run_suffix(
    PrefillV2Model& model,
    const std::uint32_t* d_tokens,
    float* d_prefix_hidden,
    float* d_suffix_hidden,
    std::uint32_t prefix,
    std::uint32_t suffix,
    std::uint32_t tile,
    std::uint32_t vocab_size,
    const std::string& session_path) {

    model.reset_state();
    if (session_path.empty()) {
        for (std::uint32_t pos = 0; pos < prefix; pos += kCandidateTile) {
            model.forward(d_tokens + pos, pos, std::min(kCandidateTile, prefix - pos), d_prefix_hidden);
        }
    } else {
        std::vector<std::uint32_t> restored_tokens;
        if (!model.load_session(session_path, restored_tokens) || restored_tokens.size() != prefix) {
            throw std::runtime_error("failed to restore the expected persistent prefix session");
        }
    }
    MIINFER_HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start{}, end{};
    MIINFER_HIP_CHECK(hipEventCreate(&start));
    MIINFER_HIP_CHECK(hipEventCreate(&end));
    MIINFER_HIP_CHECK(hipEventRecord(start));
    const auto wall_start = std::chrono::steady_clock::now();
    for (std::uint32_t offset = 0; offset < suffix; offset += tile) {
        const std::uint32_t count = std::min(tile, suffix - offset);
        model.forward(
            d_tokens + prefix + offset,
            prefix + offset,
            count,
            d_suffix_hidden + static_cast<std::size_t>(offset) * kHidden);
    }
    MIINFER_HIP_CHECK(hipEventRecord(end));
    MIINFER_HIP_CHECK(hipEventSynchronize(end));
    const auto wall_end = std::chrono::steady_clock::now();
    float gpu_ms = 0.0F;
    MIINFER_HIP_CHECK(hipEventElapsedTime(&gpu_ms, start, end));
    MIINFER_HIP_CHECK(hipEventDestroy(start));
    MIINFER_HIP_CHECK(hipEventDestroy(end));

    Run result;
    result.wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
    result.gpu_ms = gpu_ms;
    result.hidden.resize(static_cast<std::size_t>(suffix) * kHidden);
    MIINFER_HIP_CHECK(hipMemcpy(
        result.hidden.data(), d_suffix_hidden,
        result.hidden.size() * sizeof(float), hipMemcpyDeviceToHost));
    model.compute_logits(d_suffix_hidden + static_cast<std::size_t>(kSuffix - 1) * kHidden);
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    result.logits.resize(vocab_size);
    MIINFER_HIP_CHECK(hipMemcpy(
        result.logits.data(), model.logits_buffer(),
        result.logits.size() * sizeof(float), hipMemcpyDeviceToHost));
    result.first_token = static_cast<std::uint32_t>(
        std::distance(result.logits.begin(), std::max_element(result.logits.begin(), result.logits.end())));
    result.next_token = model.decode_step(42, prefix + suffix);
    return result;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

ErrorStats compare(
    const std::vector<float>& control,
    const std::vector<float>& candidate,
    std::size_t begin,
    std::size_t end) {
    double dot = 0.0, control_norm = 0.0, candidate_norm = 0.0;
    ErrorStats result;
    for (std::size_t i = begin; i < end; ++i) {
        const double a = control[i], b = candidate[i];
        result.max_abs = std::max(result.max_abs, std::abs(a - b));
        dot += a * b;
        control_norm += a * a;
        candidate_norm += b * b;
    }
    result.cosine = dot / (std::sqrt(control_norm * candidate_norm) + 1e-30);
    return result;
}

void record_run(const Run& run, std::vector<double>& wall_ms, std::vector<double>& gpu_ms) {
    if (!std::all_of(run.hidden.begin(), run.hidden.end(), [](float x) { return std::isfinite(x); })) {
        throw std::runtime_error("suffix output contains non-finite hidden values");
    }
    if (!std::all_of(run.logits.begin(), run.logits.end(), [](float x) { return std::isfinite(x); })) {
        throw std::runtime_error("suffix logits contain non-finite values");
    }
    wall_ms.push_back(run.wall_ms);
    gpu_ms.push_back(run.gpu_ms);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::string model_path = argc > 1 ? argv[1] : "/models/Qwen3.8-27B-Q4_K_M.gguf";
        const std::string session_path = argc > 2 ? argv[2] : "";
        std::uint32_t prefix = kPrefix;
        std::uint32_t capacity = kCapacity;
        if (!session_path.empty()) {
            PersistentSessionHeader header{};
            if (!PersistentSession::inspect_file(session_path, header)) {
                throw std::runtime_error("invalid persistent session: " + session_path);
            }
            if (header.kv_quant_mode != static_cast<std::uint8_t>(KvCacheQuantMode::kQ8Q8)) {
                throw std::runtime_error("persistent session is not Q8/Q8 KV");
            }
            prefix = header.prefix_length;
            capacity = std::max(capacity, prefix + kSuffix);
        }
        DeviceInfo device{};
        std::string error;
        if (!validate_gfx906_device(0, device, error)) {
            throw std::runtime_error("device validation failed: " + error);
        }
        char bdf[32]{};
        MIINFER_HIP_CHECK(hipDeviceGetPCIBusId(bdf, sizeof(bdf), 0));

        const auto source = Qwen35Model::load(model_path);
        PrefillV2Model model(source, capacity, true, KvCacheQuantMode::kQ8Q8);
        std::size_t free_bytes = 0, total_bytes = 0;
        MIINFER_HIP_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));

        std::vector<std::uint32_t> tokens(static_cast<std::size_t>(prefix) + kSuffix);
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            tokens[i] = static_cast<std::uint32_t>((i * 37 + 101) % source.config().vocab_size);
        }
        std::uint32_t* d_tokens = nullptr;
        float* d_prefix_hidden = nullptr;
        float* d_suffix_hidden = nullptr;
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_tokens), tokens.size() * sizeof(*d_tokens)));
        MIINFER_HIP_CHECK(hipMalloc(
            reinterpret_cast<void**>(&d_prefix_hidden),
            static_cast<std::size_t>(kCandidateTile) * kHidden * sizeof(*d_prefix_hidden)));
        MIINFER_HIP_CHECK(hipMalloc(
            reinterpret_cast<void**>(&d_suffix_hidden),
            static_cast<std::size_t>(kSuffix) * kHidden * sizeof(*d_suffix_hidden)));
        MIINFER_HIP_CHECK(hipMemcpy(
            d_tokens, tokens.data(), tokens.size() * sizeof(*d_tokens), hipMemcpyHostToDevice));

        std::cout << "device_bdf=" << bdf << " arch=" << device.architecture
                  << " prefix=" << prefix << " suffix=" << kSuffix
                  << " capacity=" << capacity << " kv=Q8/Q8"
                  << (session_path.empty() ? " session=none\n" : " session=restored\n")
                  << "model_vram_gib=" << std::fixed << std::setprecision(3)
                  << static_cast<double>(model.total_vram_bytes()) / (1024.0 * 1024.0 * 1024.0)
                  << " free_after_model_gib="
                  << static_cast<double>(free_bytes) / (1024.0 * 1024.0 * 1024.0)
                  << " total_gib=" << static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0)
                  << '\n';

        std::vector<double> control_wall, candidate_wall, control_gpu, candidate_gpu;
        std::vector<float> control_hidden, candidate_hidden, control_logits, candidate_logits;
        std::uint32_t control_first_token = 0, candidate_first_token = 0;
        std::uint32_t control_next_token = 0, candidate_next_token = 0;
        std::vector<double> warm_wall, warm_gpu;
        record_run(run_suffix(model, d_tokens, d_prefix_hidden, d_suffix_hidden, prefix, kSuffix,
                              kControlTile, source.config().vocab_size, session_path), warm_wall, warm_gpu);
        record_run(run_suffix(model, d_tokens, d_prefix_hidden, d_suffix_hidden, prefix, kSuffix,
                              kCandidateTile, source.config().vocab_size, session_path), warm_wall, warm_gpu);
        const int samples = std::getenv("MIINFER_PREFILL_BLOCK_TRACE_START") != nullptr ? 1 : kSamples;
        for (int i = 0; i < samples; ++i) {
            const bool control_first = (i % 2) == 0;
            if (control_first) {
                Run control = run_suffix(model, d_tokens, d_prefix_hidden, d_suffix_hidden, prefix, kSuffix,
                                         kControlTile, source.config().vocab_size, session_path);
                Run candidate = run_suffix(model, d_tokens, d_prefix_hidden, d_suffix_hidden, prefix, kSuffix,
                                           kCandidateTile, source.config().vocab_size, session_path);
                record_run(control, control_wall, control_gpu);
                record_run(candidate, candidate_wall, candidate_gpu);
                if (i == 0) {
                    control_hidden = std::move(control.hidden);
                    candidate_hidden = std::move(candidate.hidden);
                    control_logits = std::move(control.logits);
                    candidate_logits = std::move(candidate.logits);
                    control_first_token = control.first_token;
                    candidate_first_token = candidate.first_token;
                    control_next_token = control.next_token;
                    candidate_next_token = candidate.next_token;
                }
            } else {
                Run candidate = run_suffix(model, d_tokens, d_prefix_hidden, d_suffix_hidden, prefix, kSuffix,
                                           kCandidateTile, source.config().vocab_size, session_path);
                Run control = run_suffix(model, d_tokens, d_prefix_hidden, d_suffix_hidden, prefix, kSuffix,
                                         kControlTile, source.config().vocab_size, session_path);
                record_run(candidate, candidate_wall, candidate_gpu);
                record_run(control, control_wall, control_gpu);
            }
        }

        const std::size_t half = static_cast<std::size_t>(kSuffix / 2) * kHidden;
        const std::size_t last = control_hidden.size() - kHidden;
        const ErrorStats all = compare(control_hidden, candidate_hidden, 0, control_hidden.size());
        const ErrorStats first_half = compare(control_hidden, candidate_hidden, 0, half);
        const ErrorStats second_half = compare(control_hidden, candidate_hidden, half, control_hidden.size());
        const ErrorStats last_token = compare(control_hidden, candidate_hidden, last, control_hidden.size());
        const ErrorStats logits = compare(control_logits, candidate_logits, 0, control_logits.size());
        const double control_wall_median = median(control_wall);
        const double candidate_wall_median = median(candidate_wall);
        const double control_gpu_median = median(control_gpu);
        const double candidate_gpu_median = median(candidate_gpu);
        std::cout << "control_tiles=2x512 wall_ms=";
        for (double x : control_wall) std::cout << x << ',';
        std::cout << " gpu_ms=";
        for (double x : control_gpu) std::cout << x << ',';
        std::cout << "\ncandidate_tiles=1x1024 wall_ms=";
        for (double x : candidate_wall) std::cout << x << ',';
        std::cout << " gpu_ms=";
        for (double x : candidate_gpu) std::cout << x << ',';
        std::cout << "\nmedian_wall_control_ms=" << control_wall_median
                  << " median_wall_candidate_ms=" << candidate_wall_median
                  << " speedup=" << control_wall_median / candidate_wall_median
                  << " median_gpu_control_ms=" << control_gpu_median
                  << " median_gpu_candidate_ms=" << candidate_gpu_median
                  << std::setprecision(9)
                  << " all_max_abs=" << all.max_abs << " all_cosine=" << all.cosine
                  << " first512_max_abs=" << first_half.max_abs << " first512_cosine=" << first_half.cosine
                  << " last512_max_abs=" << second_half.max_abs << " last512_cosine=" << second_half.cosine
                  << " last_token_max_abs=" << last_token.max_abs
                  << " last_token_cosine=" << last_token.cosine
                  << " logits_max_abs=" << logits.max_abs << " logits_cosine=" << logits.cosine
                  << " first_token_control=" << control_first_token
                  << " first_token_candidate=" << candidate_first_token
                  << " teacher_forced_42_next_control=" << control_next_token
                  << " teacher_forced_42_next_candidate=" << candidate_next_token << '\n';

        MIINFER_HIP_CHECK(hipFree(d_tokens));
        MIINFER_HIP_CHECK(hipFree(d_prefix_hidden));
        MIINFER_HIP_CHECK(hipFree(d_suffix_hidden));
        if (all.max_abs > 0.01 || all.cosine < 0.9999) {
            std::cerr << "tile outputs fail the existing numerical tolerance\n";
            return 2;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
