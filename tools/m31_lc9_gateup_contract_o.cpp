#include "miinfer/gguf.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kColumns = 5120;
constexpr std::size_t kRows = 17408;
constexpr std::size_t kGroups = kColumns / 32;
constexpr std::size_t kQ8BlockBytes = 144;
constexpr std::size_t kQ8Blocks = kColumns / 128;
constexpr std::size_t kF32OperationBound = 7 * kGroups;
constexpr std::size_t kF64OperationBound = 8 * kGroups;

struct Q4KBlock {
    std::uint16_t d;
    std::uint16_t dmin;
    std::uint8_t scales[12];
    std::uint8_t qs[128];
};
static_assert(sizeof(Q4KBlock) == 144);

template<class T>
std::vector<T> read_exact(const std::string& path, std::size_t count) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::vector<T> values(count);
    in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(T)));
    if (!in || in.peek() != std::ifstream::traits_type::eof())
        throw std::runtime_error("wrong byte count in " + path);
    return values;
}

std::uint32_t bits(float value) {
    std::uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

float half_to_float(std::uint16_t value) {
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000U) << 16;
    std::uint32_t exponent = (value >> 10) & 0x1fU;
    std::uint32_t mantissa = value & 0x03ffU;
    std::uint32_t result;
    if (exponent == 0) {
        if (mantissa == 0) result = sign;
        else {
            int e = -14;
            while ((mantissa & 0x0400U) == 0) { mantissa <<= 1; --e; }
            mantissa &= 0x03ffU;
            result = sign | (static_cast<std::uint32_t>(e + 127) << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1fU) {
        result = sign | 0x7f800000U | (mantissa << 13);
    } else {
        result = sign | ((exponent + 112U) << 23) | (mantissa << 13);
    }
    float output;
    std::memcpy(&output, &result, sizeof(output));
    return output;
}

std::uint16_t float_to_half(float value) {
    const std::uint32_t raw = bits(value);
    const std::uint16_t sign = static_cast<std::uint16_t>((raw >> 16) & 0x8000U);
    const int exponent = static_cast<int>((raw >> 23) & 0xffU) - 127 + 15;
    std::uint32_t mantissa = raw & 0x7fffffU;
    if (exponent <= 0) {
        if (exponent < -10) return sign;
        mantissa |= 0x800000U;
        const unsigned shift = static_cast<unsigned>(14 - exponent);
        const std::uint32_t halfway = 1U << (shift - 1);
        const std::uint32_t rounded = mantissa + halfway - 1U + ((mantissa >> shift) & 1U);
        return static_cast<std::uint16_t>(sign | (rounded >> shift));
    }
    if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00U);
    mantissa += 0x0fffU + ((mantissa >> 13) & 1U);
    if (mantissa & 0x800000U) {
        if (exponent + 1 >= 31) return static_cast<std::uint16_t>(sign | 0x7c00U);
        return static_cast<std::uint16_t>(sign | (static_cast<unsigned>(exponent + 1) << 10));
    }
    return static_cast<std::uint16_t>(sign | (static_cast<unsigned>(exponent) << 10)
                                      | (mantissa >> 13));
}

std::uint16_t expected_input_sum(bool quantized_sum, float input_sum, std::uint16_t d,
                                int quantized_codes_sum) {
    const float sum = quantized_sum
        ? half_to_float(d) * static_cast<float>(quantized_codes_sum)
        : input_sum;
    return float_to_half(sum);
}

bool sum_mode_self_check() {
    const auto d = float_to_half(0.5F);
    return expected_input_sum(false, 1.5F, d, -5) == float_to_half(1.5F)
        && expected_input_sum(true, 1.5F, d, -5) == float_to_half(-2.5F);
}

std::pair<unsigned, unsigned> q4k_scale_min(const Q4KBlock& block, std::size_t group) {
    if (group < 4) return {block.scales[group] & 63U, block.scales[group + 4] & 63U};
    const std::size_t i = group - 4;
    return {
        (block.scales[8 + i] & 15U) | ((block.scales[i] >> 6) << 4),
        (block.scales[8 + i] >> 4) | ((block.scales[i + 4] >> 6) << 4)
    };
}

unsigned q4k_code(const Q4KBlock& block, std::size_t group, std::size_t i) {
    const auto* pair = block.qs + (group % 8 / 2) * 32;
    const unsigned shift = static_cast<unsigned>((group % 2) * 4);
    return (pair[i] >> shift) & 15U;
}

struct OperationResult {
    std::vector<double> reference;
    std::vector<double> bounds;
    std::size_t within = 0;
    double max_abs_error = 0.0;
    double max_bound_ratio = 0.0;
    std::size_t worst_row = 0;
    bool all_finite = true;
};

OperationResult check_projection(const Q4KBlock* weights, const std::vector<std::uint8_t>& q8,
                                const std::vector<float>& gpu, std::size_t rows) {
    OperationResult result;
    result.reference.resize(rows);
    result.bounds.resize(rows);
    constexpr double u = 0x1p-24;
    const double gamma_f32 = kF32OperationBound * u / (1.0 - kF32OperationBound * u);
    const double gamma_f64 = kF64OperationBound * std::numeric_limits<double>::epsilon()
        / (1.0 - kF64OperationBound * std::numeric_limits<double>::epsilon());
    const double underflow = kF32OperationBound * std::numeric_limits<float>::denorm_min();

    for (std::size_t row = 0; row < rows; ++row) {
        double reference = 0.0;
        double sum_abs_terms = 0.0;
        const auto* row_weights = weights + row * (kColumns / 256);
        for (std::size_t group = 0; group < kGroups; ++group) {
            const auto& weight = row_weights[group / 8];
            const std::size_t local_q8_block = group / 4;
            const std::size_t local_q8_group = group % 4;
            const auto* block = q8.data() + local_q8_block * kQ8BlockBytes;
            std::uint16_t d_bits, sum_bits;
            std::memcpy(&d_bits, block + local_q8_group * 4, sizeof(d_bits));
            std::memcpy(&sum_bits, block + local_q8_group * 4 + 2, sizeof(sum_bits));
            const double input_d = half_to_float(d_bits);
            const double input_sum = half_to_float(sum_bits);
            const auto [scale, minimum] = q4k_scale_min(weight, group % 8);
            int dot = 0;
            const auto* input_codes = reinterpret_cast<const std::int8_t*>(block + 16)
                + local_q8_group * 32;
            for (std::size_t i = 0; i < 32; ++i)
                dot += static_cast<int>(q4k_code(weight, group, i)) * input_codes[i];

            const double main = static_cast<double>(half_to_float(weight.d)) * input_d
                * static_cast<double>(static_cast<int>(scale) * dot);
            const double correction = static_cast<double>(half_to_float(weight.dmin))
                * static_cast<double>(minimum) * input_sum;
            reference += main - correction;
            sum_abs_terms += std::abs(main) + std::abs(correction);
        }
        result.reference[row] = reference;
        const double bound = gamma_f32 * sum_abs_terms + underflow
            + gamma_f64 * sum_abs_terms;
        result.bounds[row] = bound;
        const double actual = gpu[row];
        if (!std::isfinite(actual) || !std::isfinite(reference)) result.all_finite = false;
        const double error = std::abs(actual - reference);
        const double ratio = bound == 0.0 ? (error == 0.0 ? 0.0 : std::numeric_limits<double>::infinity())
                                          : error / bound;
        if (error <= bound) ++result.within;
        if (error > result.max_abs_error) result.max_abs_error = error;
        if (ratio > result.max_bound_ratio) {
            result.max_bound_ratio = ratio;
            result.worst_row = row;
        }
    }
    return result;
}

} // namespace

int main(int argc, char** argv) {
    const bool qsum = argc == 8 && std::string(argv[7]) == "--qsum";
    if (argc == 2 && std::string(argv[1]) == "--self-check") {
        const bool pass = sum_mode_self_check();
        std::cout << "sum_mode_self_check=" << (pass ? "PASS" : "FAIL") << '\n';
        return pass ? 0 : 1;
    }
    if ((argc != 7 && argc != 8) || (argc == 8 && !qsum)) {
        std::cerr << "usage: m31-lc9-gateup-contract-o MODEL.gguf POST_NORMALIZED.f32 "
                     "GATEUP_MX_Q8.bin GATE.f32 UP.f32 OUTPUT_PREFIX [--qsum]\n";
        return 2;
    }
    try {
        const auto model = miinfer::GgufFile::open(argv[1]);
        const miinfer::GgufTensor *gate_tensor = nullptr, *up_tensor = nullptr;
        for (const auto& tensor : model->tensors()) {
            if (tensor.name == "blk.0.ffn_gate.weight") gate_tensor = &tensor;
            if (tensor.name == "blk.0.ffn_up.weight") up_tensor = &tensor;
        }
        if (!gate_tensor || !up_tensor || gate_tensor->type != miinfer::GgufTensorType::q4_k
            || up_tensor->type != miinfer::GgufTensorType::q4_k
            || gate_tensor->dimensions.size() != 2 || up_tensor->dimensions.size() != 2
            || gate_tensor->dimensions[0] != kColumns || up_tensor->dimensions[0] != kColumns
            || gate_tensor->dimensions[1] != kRows || up_tensor->dimensions[1] != kRows)
            throw std::runtime_error("model Gate/Up tensor shape or Q4_K type mismatch");
        const auto q8 = read_exact<std::uint8_t>(argv[3], kQ8Blocks * kQ8BlockBytes);
        const auto input = read_exact<float>(argv[2], kColumns);
        const auto gate_gpu = read_exact<float>(argv[4], kRows);
        const auto up_gpu = read_exact<float>(argv[5], kRows);

        std::size_t codes_and_scales_match = 0;
        for (std::size_t group = 0; group < kGroups; ++group) {
            const std::size_t q8_block = group / 4, local_group = group % 4;
            const std::size_t begin = group * 32;
            float max_abs = 0.0F;
            std::array<float, 32> partial_sums{};
            for (std::size_t lane = 0; lane < partial_sums.size(); ++lane) {
                partial_sums[lane] = input[begin + lane];
                max_abs = std::max(max_abs, std::abs(partial_sums[lane]));
            }
            for (std::size_t offset : {16U, 8U, 4U, 2U, 1U}) {
                const auto previous = partial_sums;
                for (std::size_t lane = 0; lane < partial_sums.size(); ++lane)
                    partial_sums[lane] = previous[lane] + previous[lane ^ offset];
            }
            const float d = max_abs == 0.0F ? 0.0F : max_abs / 127.0F;
            const std::uint16_t expected_d = float_to_half(d);
            bool group_matches = true;
            int quantized_codes_sum = 0;
            const auto* block = q8.data() + q8_block * kQ8BlockBytes;
            for (std::size_t i = 0; i < 32; ++i) {
                const int quantized = d == 0.0F ? 0 : std::clamp(
                    static_cast<int>(std::round(input[begin + i] / d)), -127, 127);
                quantized_codes_sum += quantized;
                const auto actual = reinterpret_cast<const std::int8_t*>(block + 16)
                    [local_group * 32 + i];
                group_matches &= actual == quantized;
            }
            std::uint16_t actual_d, actual_sum;
            std::memcpy(&actual_d, block + local_group * 4, sizeof(actual_d));
            std::memcpy(&actual_sum, block + local_group * 4 + 2, sizeof(actual_sum));
            const std::uint16_t expected_sum = expected_input_sum(
                qsum, partial_sums[0], expected_d, quantized_codes_sum);
            group_matches &= actual_d == expected_d && actual_sum == expected_sum;
            if (group_matches) ++codes_and_scales_match;
        }

        auto gate = check_projection(reinterpret_cast<const Q4KBlock*>(gate_tensor->data), q8, gate_gpu, kRows);
        auto up = check_projection(reinterpret_cast<const Q4KBlock*>(up_tensor->data), q8, up_gpu, kRows);
        const bool pass = codes_and_scales_match == kGroups && gate.all_finite && up.all_finite
            && gate.within == kRows && up.within == kRows;

        const std::string prefix = argv[6];
        auto write_reference = [&](const std::string& suffix, const std::vector<double>& values) {
            std::ofstream out(prefix + suffix, std::ios::binary);
            if (!out) throw std::runtime_error("cannot write reference output");
            out.write(reinterpret_cast<const char*>(values.data()),
                      static_cast<std::streamsize>(values.size() * sizeof(double)));
        };
        write_reference(".gate.reference.f64", gate.reference);
        write_reference(".up.reference.f64", up.reference);
        std::ofstream rows_out(prefix + ".rows.csv");
        if (!rows_out) throw std::runtime_error("cannot write row diagnostics");
        rows_out << "row,gate_gpu,gate_reference,gate_abs_error,gate_bound,up_gpu,up_reference,up_abs_error,up_bound\n"
                 << std::setprecision(17);
        constexpr double u = 0x1p-24;
        const double gamma_f32 = kF32OperationBound * u / (1.0 - kF32OperationBound * u);
        const double gamma_f64 = kGroups * std::numeric_limits<double>::epsilon()
            / (1.0 - kGroups * std::numeric_limits<double>::epsilon());
        const double underflow = kF32OperationBound * std::numeric_limits<float>::denorm_min();
        for (std::size_t row = 0; row < kRows; ++row) {
            rows_out << row << ',' << gate_gpu[row] << ',' << gate.reference[row] << ','
                     << std::abs(static_cast<double>(gate_gpu[row]) - gate.reference[row]) << ','
                     << gate.bounds[row] << ',' << up_gpu[row] << ',' << up.reference[row] << ','
                     << std::abs(static_cast<double>(up_gpu[row]) - up.reference[row]) << ','
                     << up.bounds[row] << '\n';
        }
        std::ofstream summary(prefix + ".summary.json");
        if (!summary) throw std::runtime_error("cannot write summary");
        summary << std::setprecision(17)
                << "{\n  \"contract\": \"O\",\n  \"operation\": \"Q4_K Gate/Up x MXQ8 affine MMQ\",\n"
                << "  \"rows\": " << kRows << ", \"columns\": " << kColumns << ",\n"
                << "  \"input_sum_mode\": \"" << (qsum ? "qsum" : "ordinary") << "\",\n"
                << "  \"q8_groups\": " << kGroups << ", \"q8_codes_and_scales_matching\": "
                << codes_and_scales_match << ",\n"
                << "  \"fp32_rounding_operations_bound\": " << kF32OperationBound
                << ", \"gamma_n_fp32\": " << gamma_f32
                << ", \"fp64_rounding_operations_bound\": " << kF64OperationBound
                << ", \"gamma_n_fp64\": " << gamma_f64
                << ", \"underflow_allowance\": " << underflow << ",\n"
                << "  \"gate\": {\"finite\": " << (gate.all_finite ? "true" : "false")
                << ", \"rows_within_bound\": " << gate.within
                << ", \"max_abs_error\": " << gate.max_abs_error
                << ", \"max_error_to_bound\": " << gate.max_bound_ratio
                << ", \"worst_row\": " << gate.worst_row << "},\n"
                << "  \"up\": {\"finite\": " << (up.all_finite ? "true" : "false")
                << ", \"rows_within_bound\": " << up.within
                << ", \"max_abs_error\": " << up.max_abs_error
                << ", \"max_error_to_bound\": " << up.max_bound_ratio
                << ", \"worst_row\": " << up.worst_row << "},\n"
                << "  \"status\": \"" << (pass ? "PASS" : "FAIL") << "\"\n}\n";
        std::cout << "contract_o=" << (pass ? "PASS" : "FAIL")
                  << " input_sum_mode=" << (qsum ? "qsum" : "ordinary")
                  << " q8_groups=" << codes_and_scales_match << '/' << kGroups
                  << " gate_rows=" << gate.within << '/' << kRows
                  << " gate_max_ratio=" << gate.max_bound_ratio
                  << " up_rows=" << up.within << '/' << kRows
                  << " up_max_ratio=" << up.max_bound_ratio << '\n';
        return pass ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "contract O scalar reference error: " << error.what() << '\n';
        return 2;
    }
}
