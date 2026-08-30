#include "miinfer/gguf.hpp"
#include "miinfer/qwen3_model.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void append_u32(std::vector<std::uint8_t>& bytes, std::uint32_t value) {
    for (unsigned index = 0; index < 4; ++index) bytes.push_back(static_cast<std::uint8_t>(value >> (8 * index)));
}

void append_u64(std::vector<std::uint8_t>& bytes, std::uint64_t value) {
    for (unsigned index = 0; index < 8; ++index) bytes.push_back(static_cast<std::uint8_t>(value >> (8 * index)));
}

void append_string(std::vector<std::uint8_t>& bytes, const std::string& value) {
    append_u64(bytes, value.size());
    bytes.insert(bytes.end(), value.begin(), value.end());
}

std::filesystem::path write_fixture(
    bool valid_range,
    const std::string& architecture = "qwen3",
    std::uint32_t magic = 0x46554747U,
    std::uint32_t version = 3) {
    std::vector<std::uint8_t> bytes;
    append_u32(bytes, magic);
    append_u32(bytes, version);
    append_u64(bytes, 1);  // tensors
    append_u64(bytes, 3);  // metadata entries

    append_string(bytes, "general.alignment");
    append_u32(bytes, 4);  // UINT32
    append_u32(bytes, 32);
    append_string(bytes, "general.architecture");
    append_u32(bytes, 8);  // STRING
    append_string(bytes, architecture);
    append_string(bytes, "general.name");
    append_u32(bytes, 8);  // STRING
    append_string(bytes, "fixture");

    append_string(bytes, "test.weight");
    append_u32(bytes, 1);  // rank
    append_u64(bytes, 4);
    append_u32(bytes, 0);  // F32
    append_u64(bytes, valid_range ? 0 : 9999);
    while (bytes.size() % 32 != 0) bytes.push_back(0);
    for (unsigned index = 0; index < 16; ++index) bytes.push_back(static_cast<std::uint8_t>(index));

    const auto path = std::filesystem::temp_directory_path() / "miinfer-gguf-fixture.gguf";
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    output.close();
    return path;
}

bool expect_failure(const std::filesystem::path& path) {
    try {
        (void)miinfer::GgufFile::open(path.string());
    } catch (const miinfer::GgufError&) {
        return true;
    }
    return false;
}

}  // namespace

int main() {
    bool passed = true;
    const auto valid = write_fixture(true);
    try {
        const auto file = miinfer::GgufFile::open(valid.string());
        passed = passed && file->version() == 3 && file->data_offset() % 32 == 0;
        passed = passed && file->metadata_string("general.architecture") == "qwen3";
        passed = passed && file->metadata_unsigned("general.alignment") == 32;
        passed = passed && file->tensors().size() == 1;
        passed = passed && file->tensors()[0].byte_size == 16;
        passed = passed && file->tensors()[0].data != nullptr;
        try {
            (void)miinfer::Qwen3Model::load(valid.string());
            passed = false;  // The strict Qwen3 contract is intentionally incomplete.
        } catch (const miinfer::GgufError&) {
        }
    } catch (const std::exception& error) {
        std::cerr << "valid fixture unexpectedly failed: " << error.what() << '\n';
        passed = false;
    }

    const auto bad_range = write_fixture(false);
    passed = expect_failure(bad_range) && passed;
    const auto bad_magic = write_fixture(true, "qwen3", 0x12345678U);
    passed = expect_failure(bad_magic) && passed;
    const auto bad_version = write_fixture(true, "qwen3", 0x46554747U, 2);
    passed = expect_failure(bad_version) && passed;
    const auto unsupported_architecture = write_fixture(true, "llama");
    try {
        (void)miinfer::Qwen3Model::load(unsupported_architecture.string());
        passed = false;
    } catch (const miinfer::GgufError&) {
    }
    {
        std::ofstream truncated(bad_range, std::ios::binary | std::ios::trunc);
        const std::array<std::uint8_t, 8> bytes = {0x47, 0x47, 0x55, 0x46, 3, 0, 0, 0};
        truncated.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
    passed = expect_failure(bad_range) && passed;
    std::filesystem::remove(valid);
    std::filesystem::remove(bad_range);
    std::filesystem::remove(bad_magic);
    std::filesystem::remove(bad_version);
    std::filesystem::remove(unsupported_architecture);
    std::cout << "model loader host tests " << (passed ? "passed" : "failed") << '\n';
    return passed ? 0 : 1;
}
