#include "miinfer/hip_check.hpp"
#include "miinfer/prefill_v2/model.hpp"
#include "miinfer/qwen3_tokenizer.hpp"
#include "miinfer/qwen35_model.hpp"
#include "miinfer/sha256.hpp"

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using miinfer::prefill_v2::PrefillV2Model;

struct DeviceFloats {
    float* data = nullptr;
    explicit DeviceFloats(std::size_t count) {
        MIINFER_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&data), count * sizeof(float)));
    }
    ~DeviceFloats() { if (data) (void)hipFree(data); }
    DeviceFloats(const DeviceFloats&) = delete;
    DeviceFloats& operator=(const DeviceFloats&) = delete;
};

std::vector<std::uint32_t> next_prompt(
    std::size_t count, const miinfer::Qwen3Tokenizer& tokenizer) {
    std::vector<std::uint32_t> result;
    result.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (i == 0 && tokenizer.add_bos()) result.push_back(tokenizer.bos_id());
        else result.push_back(static_cast<std::uint32_t>(std::rand())
                              % static_cast<std::uint32_t>(tokenizer.vocabulary_size()));
    }
    return result;
}

std::vector<std::uint32_t> next_generation(
    std::size_t count, const miinfer::Qwen3Tokenizer& tokenizer) {
    std::vector<std::uint32_t> result;
    result.reserve(count);
    std::uint32_t token = tokenizer.add_bos()
        ? tokenizer.bos_id()
        : static_cast<std::uint32_t>(std::rand())
            % static_cast<std::uint32_t>(tokenizer.vocabulary_size());
    for (std::size_t i = 0; i < count; ++i) {
        result.push_back(token);
        token = static_cast<std::uint32_t>(std::rand())
              % static_cast<std::uint32_t>(tokenizer.vocabulary_size());
    }
    return result;
}

std::string ids_sha256(std::span<const std::uint32_t> ids) {
    return miinfer::sha256_bytes(std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(ids.data()), ids.size_bytes()));
}

double run_prefill(
    PrefillV2Model& model,
    std::span<const std::uint32_t> ids,
    float* hidden,
    float* logits,
    bool measure) {
    model.reset_state();
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    const auto start = Clock::now();
    model.prefill_sequence(ids, hidden);
    model.compute_logits(hidden + (ids.size() - 1) * miinfer::prefill_v2::kHidden, logits);
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    const auto end = Clock::now();
    return measure ? std::chrono::duration<double, std::milli>(end - start).count() : 0.0;
}

void usage() {
    std::cerr << "usage: miinfer-v2-0045-qual-bench MODEL.gguf prefill|decode P [TG] [ITERATIONS]\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4 || argc > 6) {
        usage();
        return 2;
    }
    try {
        const std::string model_path = argv[1];
        const std::string mode = argv[2];
        const auto prompt_tokens = static_cast<std::size_t>(std::stoul(argv[3]));
        const auto generated_tokens = argc >= 5 ? static_cast<std::size_t>(std::stoul(argv[4])) : 0;
        const auto iterations = argc >= 6 ? static_cast<std::size_t>(std::stoul(argv[5])) : 5;
        if ((mode != "prefill" && mode != "decode") || prompt_tokens == 0 || iterations == 0
            || (mode == "prefill" && generated_tokens != 0)
            || (mode == "decode" && generated_tokens == 0)
            || prompt_tokens + generated_tokens + 1 > 16384) {
            throw std::invalid_argument("invalid workload dimensions");
        }

        const auto model_data = miinfer::Qwen35Model::load(model_path);
        const auto tokenizer = miinfer::Qwen3Tokenizer::load(*model_data.file());
        if (tokenizer.vocabulary_size() != model_data.config().vocab_size) {
            throw std::runtime_error("tokenizer/model vocabulary mismatch");
        }
        PrefillV2Model model(model_data, 16384, true);
        DeviceFloats hidden(prompt_tokens * miinfer::prefill_v2::kHidden);
        DeviceFloats logits(model_data.config().vocab_size);
        std::srand(1); // llama-bench uses the C library's default seed (1).

        std::vector<double> samples;
        std::vector<std::string> prompt_hashes;
        std::vector<std::string> generation_hashes;
        samples.reserve(iterations);
        prompt_hashes.reserve(iterations);
        generation_hashes.reserve(iterations);

        if (mode == "prefill") {
            const auto warmup_prompt = next_prompt(prompt_tokens, tokenizer);
            (void)run_prefill(model, warmup_prompt, hidden.data, logits.data, false);
            for (std::size_t i = 0; i < iterations; ++i) {
                const auto prompt = next_prompt(prompt_tokens, tokenizer);
                prompt_hashes.push_back(ids_sha256(prompt));
                samples.push_back(run_prefill(model, prompt, hidden.data, logits.data, true));
                generation_hashes.emplace_back();
            }
        } else {
            // llama-bench warms one decode step before building its depth context.
            model.reset_state();
            const auto warm_token = tokenizer.add_bos()
                ? tokenizer.bos_id()
                : static_cast<std::uint32_t>(std::rand())
                    % static_cast<std::uint32_t>(tokenizer.vocabulary_size());
            (void)model.decode_step(warm_token, 0);
            (void)std::rand();
            const auto prompt = next_prompt(prompt_tokens, tokenizer);
            for (std::size_t i = 0; i < iterations; ++i) {
                prompt_hashes.push_back(ids_sha256(prompt));
                (void)run_prefill(model, prompt, hidden.data, logits.data, false);
                const auto sequence = next_generation(generated_tokens, tokenizer);
                generation_hashes.push_back(ids_sha256(sequence));
                const auto start = Clock::now();
                for (std::size_t step = 0; step < sequence.size(); ++step) {
                    (void)model.decode_step(sequence[step],
                        static_cast<std::uint32_t>(prompt_tokens + step));
                }
                MIINFER_HIP_CHECK(hipDeviceSynchronize());
                const auto end = Clock::now();
                samples.push_back(std::chrono::duration<double, std::milli>(end - start).count());
            }
        }

        std::cout << std::fixed << std::setprecision(6)
                  << "{\"benchmark\":\"v2-0045-equivalent-work\",\"mode\":\"" << mode
                  << "\",\"model_sha256\":\"" << miinfer::sha256_file(model_path)
                  << "\",\"prompt_tokens\":" << prompt_tokens
                  << ",\"decode_tokens\":" << generated_tokens
                  << ",\"iterations\":" << iterations
                  << ",\"prompt_id_sha256\":[";
        for (std::size_t i = 0; i < prompt_hashes.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << '"' << prompt_hashes[i] << '"';
        }
        std::cout << "],\"decode_id_sha256\":[";
        for (std::size_t i = 0; i < generation_hashes.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << '"' << generation_hashes[i] << '"';
        }
        std::cout << "],\"samples_ms\":[";
        for (std::size_t i = 0; i < samples.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << samples[i];
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "benchmark error: " << error.what() << '\n';
        return 1;
    }
}
