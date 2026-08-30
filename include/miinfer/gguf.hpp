#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace miinfer {

enum class GgufTensorType : std::uint32_t {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k_s = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
};

using GgufScalar = std::variant<std::uint64_t, std::int64_t, double, bool, std::string>;

struct GgufValue {
    std::variant<GgufScalar, std::vector<GgufScalar>> value;
};

struct GgufTensor {
    std::string name;
    std::vector<std::uint64_t> dimensions;
    GgufTensorType type;
    std::uint64_t offset = 0;
    std::size_t byte_size = 0;
    const std::byte* data = nullptr;
};

class GgufError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class GgufFile {
public:
    static std::shared_ptr<GgufFile> open(const std::string& path);

    GgufFile(const GgufFile&) = delete;
    GgufFile& operator=(const GgufFile&) = delete;
    ~GgufFile();

    [[nodiscard]] std::uint32_t version() const noexcept { return version_; }
    [[nodiscard]] std::size_t file_size() const noexcept { return file_size_; }
    [[nodiscard]] std::size_t data_offset() const noexcept { return data_offset_; }
    [[nodiscard]] const std::vector<GgufTensor>& tensors() const noexcept { return tensors_; }

    [[nodiscard]] const GgufValue* metadata(const std::string& key) const noexcept;
    [[nodiscard]] std::string metadata_string(const std::string& key) const;
    [[nodiscard]] std::uint64_t metadata_unsigned(const std::string& key) const;
    [[nodiscard]] double metadata_float(const std::string& key) const;
    [[nodiscard]] std::size_t metadata_array_size(const std::string& key) const;
    [[nodiscard]] bool metadata_array_is_string(const std::string& key) const;

private:
    GgufFile() = default;

    std::string path_;
    int file_descriptor_ = -1;
    const std::byte* mapping_ = nullptr;
    std::size_t file_size_ = 0;
    std::uint32_t version_ = 0;
    std::size_t data_offset_ = 0;
    std::unordered_map<std::string, GgufValue> metadata_;
    std::vector<GgufTensor> tensors_;
};

[[nodiscard]] std::size_t gguf_tensor_byte_size(
    GgufTensorType type,
    const std::vector<std::uint64_t>& dimensions);

[[nodiscard]] const char* gguf_tensor_type_name(GgufTensorType type) noexcept;

}  // namespace miinfer
