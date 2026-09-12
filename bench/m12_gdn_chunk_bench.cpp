#include "miinfer/hip_check.hpp"
#include "miinfer/m12_gdn_chunk.hpp"
#include "miinfer/qwen3_gpu_primitives.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kKeyHeads = 16;
constexpr std::size_t kValueHeads = 48;
constexpr std::size_t kState = 128;
constexpr std::size_t kTokens = 512;
constexpr std::size_t kChunk = 64;

std::size_t at(std::size_t head, std::size_t token, std::size_t dimension,
               std::size_t tokens, std::size_t state) {
    return (head * tokens + token) * state + dimension;
}

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

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

template <typename Fn>
double measure(Fn&& fn, int warmup = 1, int iterations = 5) {
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
    return median(std::move(samples));
}

void normalize_rows(std::vector<float>& values, std::size_t heads) {
    for (std::size_t head = 0; head < heads; ++head) {
        for (std::size_t token = 0; token < kTokens; ++token) {
            double sum = 0.0;
            for (std::size_t dimension = 0; dimension < kState; ++dimension) {
                const float value = values[at(head, token, dimension, kTokens, kState)];
                sum += static_cast<double>(value) * value;
            }
            const float scale = 1.0F / std::sqrt(static_cast<float>(sum) + 1.0e-6F);
            for (std::size_t dimension = 0; dimension < kState; ++dimension) {
                values[at(head, token, dimension, kTokens, kState)] *= scale;
            }
        }
    }
}

void recurrent(const std::vector<float>& query, const std::vector<float>& key,
               const std::vector<float>& value, const std::vector<float>& beta,
               const std::vector<float>& decay, std::vector<float>& state,
               std::vector<float>& output) {
    output.assign(kValueHeads * kTokens * kState, 0.0F);
    for (std::size_t token = 0; token < kTokens; ++token) {
        for (std::size_t head = 0; head < kValueHeads; ++head) {
            const std::size_t key_head = head % kKeyHeads;
            float* state_head = state.data() + head * kState * kState;
            const float* key_token = key.data() + at(key_head, token, 0, kTokens, kState);
            const float* value_token = value.data() + at(head, token, 0, kTokens, kState);
            const float* query_token = query.data() + at(key_head, token, 0, kTokens, kState);
            const float decay_value = decay[head * kTokens + token];
            const float beta_value = beta[head * kTokens + token];
            for (std::size_t row = 0; row < kState; ++row) {
                float key_dot = 0.0F;
                for (std::size_t column = 0; column < kState; ++column) {
                    float& state_value = state_head[column * kState + row];
                    state_value *= decay_value;
                    key_dot += state_value * key_token[column];
                }
                const float delta = (value_token[row] - key_dot) * beta_value;
                for (std::size_t column = 0; column < kState; ++column) {
                    state_head[column * kState + row] += delta * key_token[column];
                }
            }
            float* output_token = output.data() + at(head, token, 0, kTokens, kState);
            for (std::size_t row = 0; row < kState; ++row) {
                for (std::size_t column = 0; column < kState; ++column) {
                    output_token[row] += state_head[column * kState + row] * query_token[column];
                }
                output_token[row] *= 1.0F / std::sqrt(static_cast<float>(kState));
            }
        }
    }
}

double max_error(const std::vector<float>& actual, const std::vector<float>& expected) {
    double result = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        result = std::max(result, static_cast<double>(std::abs(actual[index] - expected[index])));
    }
    return result;
}

} // namespace

int main() try {
    std::mt19937 generator(0x4D3132U);
    std::uniform_real_distribution<float> distribution(-0.5F, 0.5F);
    std::vector<float> query(kKeyHeads * kTokens * kState);
    std::vector<float> key(query.size());
    std::vector<float> value(kValueHeads * kTokens * kState);
    std::vector<float> beta(kValueHeads * kTokens);
    std::vector<float> decay(beta.size());
    std::vector<float> initial_state(kValueHeads * kState * kState);
    for (float& element : query) element = distribution(generator);
    for (float& element : key) element = distribution(generator);
    for (float& element : value) element = distribution(generator);
    for (float& element : beta) element = 0.05F + 0.9F * std::abs(distribution(generator));
    for (float& element : decay) {
        element = std::exp(-0.001F - 0.03F * std::abs(distribution(generator)));
    }
    for (float& element : initial_state) element = 0.01F * distribution(generator);
    normalize_rows(query, kKeyHeads);
    normalize_rows(key, kKeyHeads);

    std::vector<float> expected_state = initial_state;
    std::vector<float> expected_output;
    recurrent(query, key, value, beta, decay, expected_state, expected_output);

    const std::size_t query_token_bytes = kTokens * kKeyHeads * kState * sizeof(float);
    const std::size_t value_token_bytes = kTokens * kValueHeads * kState * sizeof(float);
    std::vector<float> query_tokens(kTokens * kKeyHeads * kState);
    std::vector<float> key_tokens(query_tokens.size());
    std::vector<float> value_tokens(value_token_bytes / sizeof(float));
    std::vector<float> beta_tokens(kTokens * kValueHeads);
    std::vector<float> decay_tokens(beta_tokens.size());
    for (std::size_t token = 0; token < kTokens; ++token) {
        for (std::size_t head = 0; head < kKeyHeads; ++head) {
            for (std::size_t dimension = 0; dimension < kState; ++dimension) {
                query_tokens[(token * kKeyHeads + head) * kState + dimension] =
                    query[at(head, token, dimension, kTokens, kState)];
                key_tokens[(token * kKeyHeads + head) * kState + dimension] =
                    key[at(head, token, dimension, kTokens, kState)];
            }
        }
        for (std::size_t head = 0; head < kValueHeads; ++head) {
            for (std::size_t dimension = 0; dimension < kState; ++dimension) {
                value_tokens[(token * kValueHeads + head) * kState + dimension] =
                    value[at(head, token, dimension, kTokens, kState)];
            }
            beta_tokens[token * kValueHeads + head] = beta[head * kTokens + token];
            decay_tokens[token * kValueHeads + head] = decay[head * kTokens + token];
        }
    }

    Buffer query_device(query.size() * sizeof(float));
    Buffer key_device(key.size() * sizeof(float));
    Buffer value_device(value.size() * sizeof(float));
    Buffer beta_device(beta.size() * sizeof(float));
    Buffer decay_device(decay.size() * sizeof(float));
    Buffer state_device(initial_state.size() * sizeof(float));
    Buffer output_device(expected_output.size() * sizeof(float));
    Buffer gate_device(expected_output.size() * sizeof(float));
    Buffer ssm_norm_device(kState * sizeof(float));
    Buffer postprocessed_device(expected_output.size() * sizeof(float));
    Buffer query_token_device(query_token_bytes);
    Buffer key_token_device(query_token_bytes);
    Buffer value_token_device(value_token_bytes);
    Buffer beta_token_device(beta_tokens.size() * sizeof(float));
    Buffer decay_token_device(decay_tokens.size() * sizeof(float));
    Buffer token_output_device(kValueHeads * kState * sizeof(float));
    Buffer new_values(kValueHeads * kChunk * kState * sizeof(float));
    Buffer decayed_keys(kValueHeads * kChunk * kState * sizeof(float));
    Buffer solved_values(kValueHeads * kChunk * kState * sizeof(float));
    Buffer solved_keys(kValueHeads * kChunk * kState * sizeof(float));
    Buffer corrected_values(kValueHeads * kChunk * kState * sizeof(float));
    MIINFER_HIP_CHECK(hipMemcpy(query_device.pointer, query_tokens.data(), query_token_bytes, hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(key_device.pointer, key_tokens.data(), query_token_bytes, hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(value_device.pointer, value_tokens.data(), value_token_bytes, hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(beta_device.pointer, beta_tokens.data(), beta_tokens.size() * sizeof(float), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(decay_device.pointer, decay_tokens.data(), decay_tokens.size() * sizeof(float), hipMemcpyHostToDevice));
    std::vector<float> ssm_norm_host(kState, 1.0F);
    MIINFER_HIP_CHECK(hipMemset(gate_device.pointer, 0, expected_output.size() * sizeof(float)));
    MIINFER_HIP_CHECK(hipMemcpy(ssm_norm_device.pointer, ssm_norm_host.data(),
                                ssm_norm_host.size() * sizeof(float), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(query_token_device.pointer, query_tokens.data(), query_token_bytes, hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(key_token_device.pointer, key_tokens.data(), query_token_bytes, hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(value_token_device.pointer, value_tokens.data(), value_token_bytes, hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(beta_token_device.pointer, beta_tokens.data(), beta_tokens.size() * sizeof(float), hipMemcpyHostToDevice));
    MIINFER_HIP_CHECK(hipMemcpy(decay_token_device.pointer, decay_tokens.data(), decay_tokens.size() * sizeof(float), hipMemcpyHostToDevice));

    const miinfer::M12GdnChunkWorkspace workspace{
        new_values.as<float>(), decayed_keys.as<float>(), solved_values.as<float>(),
        solved_keys.as<float>(), corrected_values.as<float>()};
    auto reset_state = [&] {
        MIINFER_HIP_CHECK(hipMemcpy(state_device.pointer, initial_state.data(),
                                    initial_state.size() * sizeof(float), hipMemcpyHostToDevice));
    };
    auto run_chunk = [&] {
        reset_state();
        for (std::size_t start = 0; start < kTokens; start += kChunk) {
            miinfer::launch_m12_gdn_chunk(
                query_device.as<float>(), key_device.as<float>(), value_device.as<float>(),
                beta_device.as<float>(), decay_device.as<float>(), state_device.as<float>(),
                output_device.as<float>(), workspace, kTokens, start, kKeyHeads, kValueHeads,
                kState, kChunk, hipStreamPerThread);
        }
    };
    auto run_token = [&] {
        reset_state();
        for (std::size_t token = 0; token < kTokens; ++token) {
            miinfer::launch_qwen35_deltanet_state_update_transposed_no_decay_store(
                query_token_device.as<float>() + token * kKeyHeads * kState,
                key_token_device.as<float>() + token * kKeyHeads * kState,
                value_token_device.as<float>() + token * kValueHeads * kState,
                beta_token_device.as<float>() + token * kValueHeads,
                decay_token_device.as<float>() + token * kValueHeads,
                state_device.as<float>(), token_output_device.as<float>(), kKeyHeads, kValueHeads,
                kState, hipStreamPerThread);
        }
    };
    auto run_mx = [&] {
        reset_state();
        miinfer::launch_mx_gdn_chunk(
            query_token_device.as<float>(), key_token_device.as<float>(),
            value_token_device.as<float>(), beta_token_device.as<float>(),
            decay_token_device.as<float>(), state_device.as<float>(),
            output_device.as<float>(), kTokens, kKeyHeads, kValueHeads, kState,
            hipStreamPerThread);
    };

    const double chunk_us = measure(run_chunk);
    const double token_us = measure(run_token);
    const double mx_us = measure(run_mx);
    run_chunk();
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    miinfer::launch_m12_gdn_postprocess(
        output_device.as<float>(), gate_device.as<float>(), ssm_norm_device.as<float>(),
        postprocessed_device.as<float>(), kTokens, kValueHeads, kState, 1.0e-5F,
        hipStreamPerThread);
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> actual_output(expected_output.size());
    std::vector<float> actual_state(initial_state.size());
    std::vector<float> mx_output(expected_output.size());
    std::vector<float> mx_state(initial_state.size());
    MIINFER_HIP_CHECK(hipMemcpy(actual_output.data(), output_device.pointer,
                                actual_output.size() * sizeof(float), hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(actual_state.data(), state_device.pointer,
                                actual_state.size() * sizeof(float), hipMemcpyDeviceToHost));
    run_mx();
    MIINFER_HIP_CHECK(hipDeviceSynchronize());
    MIINFER_HIP_CHECK(hipMemcpy(mx_output.data(), output_device.pointer,
                                mx_output.size() * sizeof(float), hipMemcpyDeviceToHost));
    MIINFER_HIP_CHECK(hipMemcpy(mx_state.data(), state_device.pointer,
                                mx_state.size() * sizeof(float), hipMemcpyDeviceToHost));
    auto output_error_for = [&](const std::vector<float>& actual) {
        double result = 0.0;
        for (std::size_t token = 0; token < kTokens; ++token) {
            for (std::size_t head = 0; head < kValueHeads; ++head) {
                for (std::size_t dimension = 0; dimension < kState; ++dimension) {
                    const std::size_t device_index =
                        (token * kValueHeads + head) * kState + dimension;
                    const std::size_t expected_index = at(head, token, dimension, kTokens, kState);
                    result = std::max(result, static_cast<double>(std::abs(
                        actual[device_index] - expected_output[expected_index])));
                }
            }
        }
        return result;
    };
    const double output_error = output_error_for(actual_output);
    const double mx_output_error = output_error_for(mx_output);
    const double state_error = max_error(actual_state, expected_state);
    const double mx_state_error = max_error(mx_state, expected_state);
    std::cout << std::fixed << std::setprecision(9)
              << "{\"tokens\":" << kTokens << ",\"chunk\":" << kChunk
              << ",\"key_heads\":" << kKeyHeads << ",\"value_heads\":" << kValueHeads
              << ",\"state_size\":" << kState
              << ",\"chunk_us\":" << chunk_us << ",\"token_us\":" << token_us
              << ",\"mx_us\":" << mx_us
              << ",\"speedup_vs_token\":" << token_us / chunk_us
              << ",\"speedup_mx_vs_token\":" << token_us / mx_us
              << ",\"max_output_error\":" << output_error
              << ",\"max_state_error\":" << state_error
              << ",\"mx_max_output_error\":" << mx_output_error
              << ",\"mx_max_state_error\":" << mx_state_error << "}\n";
    if (output_error > 1.0e-3 || state_error > 1.0e-3
        || mx_output_error > 1.0e-3 || mx_state_error > 1.0e-3) {
        throw std::runtime_error("GDN GPU oracle mismatch");
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
