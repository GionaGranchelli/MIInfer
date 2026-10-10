#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace m31_0002 {

inline std::vector<std::uint32_t> make_prompt(std::size_t count, std::uint32_t seed) {
    std::vector<std::uint32_t> tokens(count);
    for (auto& token : tokens) {
        seed = seed * 1664525u + 1013904223u;
        token = seed % 151643u + 1;
    }
    return tokens;
}

inline std::uint64_t prompt_fingerprint(const std::vector<std::uint32_t>& tokens) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const auto token : tokens) {
        for (unsigned shift = 0; shift < 32; shift += 8) {
            hash = (hash ^ static_cast<std::uint8_t>(token >> shift)) * 1099511628211ULL;
        }
    }
    return hash;
}

}  // namespace m31_0002
