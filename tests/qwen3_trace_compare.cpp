#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Metrics {
    float max_abs = 0.0F;
    float mean_abs = 0.0F;
    float rmse = 0.0F;
    float max_rel = 0.0F;
    std::size_t max_index = 0;
    float first_value = 0.0F;
    float second_value = 0.0F;
};

std::vector<float> read_f32(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open trace: " + path.string());
    file.seekg(0, std::ios::end);
    const auto size = file.tellg();
    if (size < 0 || size % static_cast<std::streamoff>(sizeof(float)) != 0) {
        throw std::runtime_error("trace is not a whole number of F32 values: " + path.string());
    }
    file.seekg(0, std::ios::beg);
    std::vector<float> values(static_cast<std::size_t>(size) / sizeof(float));
    file.read(reinterpret_cast<char*>(values.data()), size);
    if (!file) throw std::runtime_error("short trace: " + path.string());
    return values;
}

Metrics compare(const std::vector<float>& first, const std::vector<float>& second) {
    if (first.size() != second.size()) throw std::runtime_error("trace checkpoint size mismatch");
    Metrics result;
    double sum_abs = 0.0;
    double sum_squared = 0.0;
    for (std::size_t index = 0; index < first.size(); ++index) {
        if (!std::isfinite(first[index]) || !std::isfinite(second[index])) {
            throw std::runtime_error("non-finite value in trace checkpoint");
        }
        const float error = std::fabs(first[index] - second[index]);
        sum_abs += error;
        sum_squared += static_cast<double>(error) * error;
        if (error > result.max_abs) {
            result.max_abs = error;
            result.max_index = index;
            result.first_value = first[index];
            result.second_value = second[index];
        }
        result.max_rel = std::max(
            result.max_rel, error / std::max(1.0F, std::fabs(second[index])));
    }
    if (!first.empty()) {
        result.mean_abs = static_cast<float>(sum_abs / first.size());
        result.rmse = static_cast<float>(std::sqrt(sum_squared / first.size()));
    }
    return result;
}

std::size_t argmax(const std::vector<float>& values) {
    return static_cast<std::size_t>(std::distance(
        values.begin(), std::max_element(values.begin(), values.end())));
}

std::vector<std::size_t> top_indices(const std::vector<float>& values, std::size_t count) {
    std::vector<std::size_t> indices(values.size());
    std::iota(indices.begin(), indices.end(), 0);
    count = std::min(count, indices.size());
    std::partial_sort(indices.begin(), indices.begin() + count, indices.end(),
                      [&](const auto left, const auto right) {
                          return values[left] > values[right];
                      });
    indices.resize(count);
    return indices;
}

void print_metrics(const char* name, const Metrics& result) {
    std::cout << std::left << std::setw(16) << name
              << " max_abs=" << std::setprecision(8) << result.max_abs
              << " mean_abs=" << result.mean_abs
              << " rmse=" << result.rmse
              << " max_rel=" << result.max_rel
              << " max_index=" << result.max_index
              << " first=" << result.first_value
              << " second=" << result.second_value << '\n';
}

int run(const std::filesystem::path& first_path,
        const std::filesystem::path& second_path) {
    std::array<std::string, 39> names{"embedding"};
    for (std::size_t layer = 0; layer < 36; ++layer) {
        names[layer + 1] = "layer-" + std::to_string(layer);
    }
    names[37] = "final-norm";
    names[38] = "logits";

    std::cout << "Qwen3 trace comparison (no acceptance threshold):\n"
              << "  first:  " << first_path << '\n'
              << "  second: " << second_path << '\n';

    for (const auto& name : names) {
        const auto first = read_f32(first_path / (name + ".f32"));
        const auto second = read_f32(second_path / (name + ".f32"));
        const auto result = compare(first, second);
        print_metrics(name.c_str(), result);

        if (name == "logits") {
            const auto first_top = top_indices(first, 5);
            const auto second_top = top_indices(second, 5);
            std::cout << "  first logits argmax=" << argmax(first)
                      << " top5=";
            for (const auto index : first_top) std::cout << ' ' << index;
            std::cout << '\n';
            std::cout << "  second logits argmax=" << argmax(second)
                      << " top5=";
            for (const auto index : second_top) std::cout << ' ' << index;
            std::cout << '\n';
            if (first_top.size() >= 2 && second_top.size() >= 2) {
                std::cout << "  top1 margins: first="
                          << first[first_top[0]] - first[first_top[1]]
                          << " second="
                          << second[second_top[0]] - second[second_top[1]] << '\n';
            }
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: miinfer-qwen3-trace-compare TRACE_A TRACE_B\n";
        return 2;
    }
    try {
        return run(argv[1], argv[2]);
    } catch (const std::exception& error) {
        std::cerr << "qwen3 trace comparison error: " << error.what() << '\n';
        return 1;
    }
}
