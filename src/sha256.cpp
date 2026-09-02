#include "miinfer/sha256.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace miinfer {

namespace {

constexpr std::array<std::uint32_t, 64> kRoundConstants = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U,
    0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U,
    0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
    0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
    0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU,
    0x5b9cca4fU, 0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

constexpr std::array<std::uint32_t, 8> kInitialHash = {
    0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
    0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};

constexpr std::uint32_t rotate_right(std::uint32_t value, unsigned amount) {
    return (value >> amount) | (value << (32U - amount));
}

class Sha256 {
public:
    Sha256() : state_(kInitialHash) {}

    void update(const std::byte* data, std::size_t size) {
        total_bytes_ += size;
        while (size != 0) {
            const auto take = std::min(size, block_.size() - block_size_);
            std::memcpy(block_.data() + block_size_, data, take);
            block_size_ += take;
            data += take;
            size -= take;
            if (block_size_ == block_.size()) {
                transform(block_);
                block_size_ = 0;
            }
        }
    }

    [[nodiscard]] std::string finish() {
        const auto bit_length = total_bytes_ * 8U;
        block_[block_size_++] = std::byte{0x80};
        if (block_size_ > 56) {
            while (block_size_ < block_.size()) block_[block_size_++] = std::byte{0};
            transform(block_);
            block_size_ = 0;
        }
        while (block_size_ < 56) block_[block_size_++] = std::byte{0};
        for (unsigned index = 0; index < 8; ++index) {
            block_[56 + index] = std::byte{static_cast<unsigned char>(bit_length >> (56 - 8 * index))};
        }
        transform(block_);

        std::ostringstream output;
        output << std::hex << std::setfill('0');
        for (const auto word : state_) output << std::setw(8) << word;
        return output.str();
    }

private:
    void transform(const std::array<std::byte, 64>& block) {
        std::array<std::uint32_t, 64> schedule{};
        for (unsigned index = 0; index < 16; ++index) {
            const auto offset = index * 4;
            schedule[index] = (static_cast<std::uint32_t>(block[offset]) << 24U)
                            | (static_cast<std::uint32_t>(block[offset + 1]) << 16U)
                            | (static_cast<std::uint32_t>(block[offset + 2]) << 8U)
                            | static_cast<std::uint32_t>(block[offset + 3]);
        }
        for (unsigned index = 16; index < 64; ++index) {
            const auto s0 = rotate_right(schedule[index - 15], 7)
                            ^ rotate_right(schedule[index - 15], 18)
                            ^ (schedule[index - 15] >> 3U);
            const auto s1 = rotate_right(schedule[index - 2], 17)
                            ^ rotate_right(schedule[index - 2], 19)
                            ^ (schedule[index - 2] >> 10U);
            schedule[index] = schedule[index - 16] + s0 + schedule[index - 7] + s1;
        }
        auto working = state_;
        for (unsigned index = 0; index < 64; ++index) {
            const auto s1 = rotate_right(working[4], 6)
                            ^ rotate_right(working[4], 11)
                            ^ rotate_right(working[4], 25);
            const auto choose = (working[4] & working[5]) ^ ((~working[4]) & working[6]);
            const auto temp1 = working[7] + s1 + choose + kRoundConstants[index] + schedule[index];
            const auto s0 = rotate_right(working[0], 2)
                            ^ rotate_right(working[0], 13)
                            ^ rotate_right(working[0], 22);
            const auto majority = (working[0] & working[1])
                                ^ (working[0] & working[2])
                                ^ (working[1] & working[2]);
            const auto temp2 = s0 + majority;
            working[7] = working[6];
            working[6] = working[5];
            working[5] = working[4];
            working[4] = working[3] + temp1;
            working[3] = working[2];
            working[2] = working[1];
            working[1] = working[0];
            working[0] = temp1 + temp2;
        }
        for (unsigned index = 0; index < state_.size(); ++index) state_[index] += working[index];
    }

    std::array<std::uint32_t, 8> state_{};
    std::array<std::byte, 64> block_{};
    std::size_t block_size_ = 0;
    std::uint64_t total_bytes_ = 0;
};

}  // namespace

std::string sha256_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open file for SHA256: " + path);
    std::vector<char> buffer(1U << 20U);
    Sha256 hash;
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = input.gcount();
        if (count > 0) {
            hash.update(reinterpret_cast<const std::byte*>(buffer.data()), static_cast<std::size_t>(count));
        }
    }
    if (!input.eof()) throw std::runtime_error("error while reading file for SHA256: " + path);
    return hash.finish();
}

std::string sha256_bytes(std::span<const std::byte> bytes) {
    Sha256 hash;
    hash.update(bytes.data(), bytes.size());
    return hash.finish();
}

}  // namespace miinfer
