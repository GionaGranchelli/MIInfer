#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::size_t kKeyHeads = 16;
constexpr std::size_t kValueHeads = 32;
constexpr std::size_t kState = 128;
constexpr std::size_t kTokens = 128;
constexpr std::size_t kChunk = 64;

using Matrix = std::vector<float>;

std::size_t at(std::size_t row, std::size_t column, std::size_t width) {
    return row * width + column;
}

void normalize_rows(Matrix& values, std::size_t rows) {
    for (std::size_t row = 0; row < rows; ++row) {
        double sum = 0.0;
        for (std::size_t column = 0; column < kState; ++column) {
            const float value = values[at(row, column, kState)];
            sum += static_cast<double>(value) * value;
        }
        const float scale = 1.0F / std::sqrt(static_cast<float>(sum) + 1.0e-6F);
        for (std::size_t column = 0; column < kState; ++column) {
            values[at(row, column, kState)] *= scale;
        }
    }
}

void recurrent(const Matrix& query, const Matrix& key, const Matrix& value,
               const Matrix& beta, const Matrix& decay, Matrix& state,
               Matrix& output) {
    output.assign(kValueHeads * kTokens * kState, 0.0F);
    for (std::size_t token = 0; token < kTokens; ++token) {
        for (std::size_t head = 0; head < kValueHeads; ++head) {
            const std::size_t key_head = head % kKeyHeads;
            float* state_head = state.data() + head * kState * kState;
            const float* key_token = key.data() + (key_head * kTokens + token) * kState;
            const float* value_token = value.data() + (head * kTokens + token) * kState;
            const float* query_token = query.data() + (key_head * kTokens + token) * kState;
            const float decay_value = std::exp(decay[head * kTokens + token]);
            const float beta_value = beta[head * kTokens + token];

            for (std::size_t row = 0; row < kState; ++row) {
                float key_dot = 0.0F;
                for (std::size_t column = 0; column < kState; ++column) {
                    float& state_value = state_head[at(column, row, kState)];
                    state_value *= decay_value;
                    key_dot += state_value * key_token[column];
                }
                const float delta = (value_token[row] - key_dot) * beta_value;
                for (std::size_t column = 0; column < kState; ++column) {
                    state_head[at(column, row, kState)] += delta * key_token[column];
                }
            }
            float* output_token = output.data() + (head * kTokens + token) * kState;
            for (std::size_t row = 0; row < kState; ++row) {
                for (std::size_t column = 0; column < kState; ++column) {
                    output_token[row] += state_head[at(column, row, kState)] * query_token[column];
                }
                output_token[row] *= 1.0F / std::sqrt(static_cast<float>(kState));
            }
        }
    }
}

void chunkwise(const Matrix& query, const Matrix& key, const Matrix& value,
               const Matrix& beta, const Matrix& decay, Matrix& state,
               Matrix& output) {
    output.assign(kValueHeads * kTokens * kState, 0.0F);
    const float query_scale = 1.0F / std::sqrt(static_cast<float>(kState));
    for (std::size_t chunk_start = 0; chunk_start < kTokens; chunk_start += kChunk) {
        for (std::size_t head = 0; head < kValueHeads; ++head) {
            const std::size_t key_head = head % kKeyHeads;
            const std::size_t key_base = key_head * kTokens * kState;
            const std::size_t value_base = head * kTokens * kState;
            const std::size_t scalar_base = head * kTokens;
            const std::size_t key_width = kState;
            Matrix cumulative(kChunk, 0.0F);
            Matrix pairwise(kChunk * kChunk, 0.0F);
            Matrix system(kChunk * kChunk, 0.0F);
            Matrix inverse(kChunk * kChunk, 0.0F);
            Matrix new_values(kChunk * kState, 0.0F);
            Matrix decayed_k(kChunk * kState, 0.0F);
            Matrix intra(kChunk * kChunk, 0.0F);

            for (std::size_t row = 0; row < kChunk; ++row) {
                cumulative[row] = decay[scalar_base + chunk_start + row]
                                  + (row == 0 ? 0.0F : cumulative[row - 1]);
                for (std::size_t column = 0; column <= row; ++column) {
                    pairwise[at(row, column, kChunk)] = std::exp(cumulative[row] - cumulative[column]);
                    float key_beta_dot = 0.0F;
                    float query_key_dot = 0.0F;
                    const float beta_value = beta[scalar_base + chunk_start + row];
                    for (std::size_t dimension = 0; dimension < kState; ++dimension) {
                        const float current_key = key[key_base + (chunk_start + row) * key_width + dimension];
                        const float previous_key = key[key_base + (chunk_start + column) * key_width + dimension];
                        const float current_query = query[key_base + (chunk_start + row) * key_width + dimension];
                        key_beta_dot += current_key * beta_value * previous_key;
                        query_key_dot += current_query * previous_key;
                    }
                    system[at(row, column, kChunk)] = key_beta_dot * pairwise[at(row, column, kChunk)];
                    intra[at(row, column, kChunk)] = query_key_dot * pairwise[at(row, column, kChunk)]
                                                     * query_scale;
                }
                for (std::size_t dimension = 0; dimension < kState; ++dimension) {
                    const float current_key = key[key_base + (chunk_start + row) * key_width + dimension];
                    const float current_value = value[value_base + (chunk_start + row) * kState + dimension];
                    const float beta_value = beta[scalar_base + chunk_start + row];
                    const float decay_factor = std::exp(cumulative[row]);
                    decayed_k[at(row, dimension, kState)] = current_key * beta_value * decay_factor;
                    new_values[at(row, dimension, kState)] = current_value * beta_value;
                }
                system[at(row, row, kChunk)] = 0.0F;
                inverse[at(row, row, kChunk)] = 1.0F;
            }

            // Forward substitution for (I + strict_lower(system))^-1.
            for (std::size_t row = 0; row < kChunk; ++row) {
                for (std::size_t column = 0; column < row; ++column) {
                    const float coefficient = system[at(row, column, kChunk)];
                    for (std::size_t rhs = 0; rhs < kChunk; ++rhs) {
                        inverse[at(row, rhs, kChunk)] -= coefficient * inverse[at(column, rhs, kChunk)];
                    }
                }
            }

            Matrix solved_values(kChunk * kState, 0.0F);
            Matrix solved_keys(kChunk * kState, 0.0F);
            for (std::size_t row = 0; row < kChunk; ++row) {
                for (std::size_t source = 0; source <= row; ++source) {
                    const float coefficient = inverse[at(row, source, kChunk)];
                    for (std::size_t dimension = 0; dimension < kState; ++dimension) {
                        solved_values[at(row, dimension, kState)] += coefficient
                            * new_values[at(source, dimension, kState)];
                        solved_keys[at(row, dimension, kState)] += coefficient
                            * decayed_k[at(source, dimension, kState)];
                    }
                }
            }

            const float chunk_decay = std::exp(cumulative.back());
            Matrix corrected_values(kChunk * kState, 0.0F);
            for (std::size_t row = 0; row < kChunk; ++row) {
                for (std::size_t dimension = 0; dimension < kState; ++dimension) {
                    float predicted = 0.0F;
                    for (std::size_t column = 0; column < kState; ++column) {
                        predicted += solved_keys[at(row, column, kState)]
                                     * state[head * kState * kState + at(column, dimension, kState)];
                    }
                    corrected_values[at(row, dimension, kState)] =
                        solved_values[at(row, dimension, kState)] - predicted;
                }
            }

            for (std::size_t row = 0; row < kChunk; ++row) {
                const std::size_t token = chunk_start + row;
                float* output_token = output.data() + (head * kTokens + token) * kState;
                const float* query_token = query.data() + (key_head * kTokens + token) * kState;
                const float query_decay = std::exp(cumulative[row]);
                for (std::size_t dimension = 0; dimension < kState; ++dimension) {
                    float inter = 0.0F;
                    for (std::size_t column = 0; column < kState; ++column) {
                        inter += query_token[column] * query_scale * query_decay
                                 * state[head * kState * kState + at(column, dimension, kState)];
                    }
                    output_token[dimension] = inter;
                    for (std::size_t source = 0; source <= row; ++source) {
                        const float value_correction = corrected_values[at(source, dimension, kState)];
                        output_token[dimension] += intra[at(row, source, kChunk)] * value_correction;
                    }
                }
            }
            for (std::size_t column = 0; column < kState; ++column) {
                for (std::size_t dimension = 0; dimension < kState; ++dimension) {
                    float update = 0.0F;
                    for (std::size_t row = 0; row < kChunk; ++row) {
                        const float key_value = key[key_base + (chunk_start + row) * key_width + column];
                        const float key_decay = std::exp(cumulative.back() - cumulative[row]);
                        update += key_value * key_decay * corrected_values[at(row, dimension, kState)];
                    }
                    state[head * kState * kState + at(column, dimension, kState)] *= chunk_decay;
                    state[head * kState * kState + at(column, dimension, kState)] += update;
                }
            }
        }
    }
}

} // namespace

int main() {
    std::mt19937 generator(0x4D3132U);
    std::uniform_real_distribution<float> distribution(-0.5F, 0.5F);
    Matrix query(kKeyHeads * kTokens * kState);
    Matrix key(query.size());
    Matrix value(kValueHeads * kTokens * kState);
    Matrix beta(kValueHeads * kTokens);
    Matrix decay(beta.size());
    Matrix initial_state(kValueHeads * kState * kState);
    for (float& element : query) element = distribution(generator);
    for (float& element : key) element = distribution(generator);
    for (float& element : value) element = distribution(generator);
    for (float& element : beta) element = 0.05F + 0.9F * std::abs(distribution(generator));
    for (float& element : decay) element = -0.001F - 0.03F * std::abs(distribution(generator));
    for (float& element : initial_state) element = 0.01F * distribution(generator);
    normalize_rows(query, kKeyHeads * kTokens);
    normalize_rows(key, kKeyHeads * kTokens);

    Matrix recurrent_state = initial_state;
    Matrix chunk_state = initial_state;
    Matrix recurrent_output;
    Matrix chunk_output;
    recurrent(query, key, value, beta, decay, recurrent_state, recurrent_output);
    chunkwise(query, key, value, beta, decay, chunk_state, chunk_output);

    double max_output_error = 0.0;
    double max_output_relative_error = 0.0;
    for (std::size_t index = 0; index < recurrent_output.size(); ++index) {
        const double error = std::abs(recurrent_output[index] - chunk_output[index]);
        max_output_error = std::max(max_output_error, error);
        max_output_relative_error = std::max(max_output_relative_error,
            error / std::max(1.0e-3, std::abs(static_cast<double>(recurrent_output[index]))));
    }
    double max_state_error = 0.0;
    for (std::size_t index = 0; index < recurrent_state.size(); ++index) {
        max_state_error = std::max(max_state_error, static_cast<double>(std::abs(
            recurrent_state[index] - chunk_state[index])));
    }
    std::cout << std::fixed << std::setprecision(9)
              << "{\"tokens\":" << kTokens << ",\"chunk\":" << kChunk
              << ",\"key_heads\":" << kKeyHeads << ",\"value_heads\":" << kValueHeads
              << ",\"state_size\":" << kState
              << ",\"max_output_error\":" << max_output_error
              << ",\"max_output_relative_error\":" << max_output_relative_error
              << ",\"max_state_error\":" << max_state_error << "}\n";
    std::cerr << "chunkwise max_output_error=" << max_output_error
              << " max_output_relative_error=" << max_output_relative_error
              << " max_state_error=" << max_state_error << '\n';
    if (max_output_error > 1.0e-2 || max_state_error > 1.0e-2) {
        throw std::runtime_error("chunkwise Gated DeltaNet oracle mismatch");
    }
}
