#include "miinfer/gguf.hpp"

#include <algorithm>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <sstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <type_traits>
#include <unistd.h>

namespace miinfer {

namespace {

constexpr std::uint32_t kGgufMagic = 0x46554747U;
constexpr std::uint32_t kGgufVersion = 3U;
constexpr std::uint64_t kMaxStringBytes = 1ULL << 30U;
constexpr std::uint64_t kMaxDimensions = 8U;

[[noreturn]] void fail(const std::string& message) {
    throw GgufError(message);
}

std::size_t checked_size(std::uint64_t value, const char* what) {
    if (value > std::numeric_limits<std::size_t>::max()) {
        fail(std::string("GGUF ") + what + " exceeds host size_t");
    }
    return static_cast<std::size_t>(value);
}

std::size_t checked_add(std::size_t left, std::size_t right, const char* what) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        fail(std::string("GGUF ") + what + " overflows");
    }
    return left + right;
}

std::size_t checked_mul(std::size_t left, std::size_t right, const char* what) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        fail(std::string("GGUF ") + what + " overflows");
    }
    return left * right;
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        fail("GGUF alignment must be a non-zero power of two");
    }
    const std::size_t remainder = value & (alignment - 1);
    return remainder == 0 ? value : checked_add(value, alignment - remainder, "data offset");
}

class Reader {
public:
    Reader(const std::byte* data, std::size_t size) : data_(data), size_(size) {}

    [[nodiscard]] std::size_t position() const noexcept { return position_; }

    template <typename T>
    T read(const char* what) {
        require(sizeof(T), what);
        T value{};
        std::memcpy(&value, data_ + position_, sizeof(T));
        position_ += sizeof(T);
        return value;
    }

    std::string string(const char* what) {
        const auto length = read<std::uint64_t>(what);
        if (length > kMaxStringBytes) {
            fail(std::string("GGUF ") + what + " is unreasonably large");
        }
        const auto bytes = checked_size(length, what);
        require(bytes, what);
        std::string result(reinterpret_cast<const char*>(data_ + position_), bytes);
        position_ += bytes;
        return result;
    }

    void require(std::size_t bytes, const char* what) const {
        if (bytes > size_ - position_) {
            fail(std::string("GGUF truncated while reading ") + what);
        }
    }

    [[nodiscard]] const std::byte* pointer(std::size_t offset) const {
        if (offset > size_) {
            fail("GGUF pointer is outside mapped file");
        }
        return data_ + offset;
    }

private:
    const std::byte* data_;
    std::size_t size_;
    std::size_t position_ = 0;
};

GgufScalar read_scalar(Reader& reader, std::uint32_t type) {
    switch (type) {
    case 0: return static_cast<std::uint64_t>(reader.read<std::uint8_t>("UINT8 metadata"));
    case 1: return static_cast<std::int64_t>(reader.read<std::int8_t>("INT8 metadata"));
    case 2: return static_cast<std::uint64_t>(reader.read<std::uint16_t>("UINT16 metadata"));
    case 3: return static_cast<std::int64_t>(reader.read<std::int16_t>("INT16 metadata"));
    case 4: return static_cast<std::uint64_t>(reader.read<std::uint32_t>("UINT32 metadata"));
    case 5: return static_cast<std::int64_t>(reader.read<std::int32_t>("INT32 metadata"));
    case 6: return static_cast<double>(reader.read<float>("FLOAT32 metadata"));
    case 7: {
        const auto value = reader.read<std::uint8_t>("BOOL metadata");
        if (value > 1) {
            fail("GGUF BOOL metadata is neither zero nor one");
        }
        return value != 0;
    }
    case 8: return reader.string("STRING metadata");
    case 10: return reader.read<std::uint64_t>("UINT64 metadata");
    case 11: return reader.read<std::int64_t>("INT64 metadata");
    case 12: return reader.read<double>("FLOAT64 metadata");
    default: fail("GGUF metadata array contains an unsupported or nested type");
    }
}

GgufValue read_value(Reader& reader, std::uint32_t type) {
    if (type != 9) {
        return {read_scalar(reader, type)};
    }
    const auto element_type = reader.read<std::uint32_t>("metadata array type");
    const auto count = reader.read<std::uint64_t>("metadata array length");
    if (count > 1ULL << 28U) {
        fail("GGUF metadata array is unreasonably large");
    }
    std::vector<GgufScalar> values;
    values.reserve(checked_size(count, "metadata array length"));
    for (std::uint64_t index = 0; index < count; ++index) {
        values.push_back(read_scalar(reader, element_type));
    }
    return {std::move(values)};
}

std::size_t elements(const std::vector<std::uint64_t>& dimensions) {
    std::size_t result = 1;
    for (const auto dimension : dimensions) {
        if (dimension == 0) {
            fail("GGUF tensor has a zero dimension");
        }
        result = checked_mul(result, checked_size(dimension, "tensor dimension"), "tensor elements");
    }
    return result;
}

struct TypeLayout {
    std::size_t block_elements;
    std::size_t block_bytes;
};

TypeLayout layout(GgufTensorType type) {
    switch (type) {
    case GgufTensorType::f32: return {1, 4};
    case GgufTensorType::f16: return {1, 2};
    case GgufTensorType::q4_0: return {32, 18};
    case GgufTensorType::q4_1: return {32, 20};
    case GgufTensorType::q5_0: return {32, 22};
    case GgufTensorType::q5_1: return {32, 24};
    case GgufTensorType::q8_0: return {32, 34};
    case GgufTensorType::q8_1: return {32, 36};
    case GgufTensorType::q2_k: return {256, 84};
    case GgufTensorType::q3_k_s: return {256, 110};
    case GgufTensorType::q4_k: return {256, 144};
    case GgufTensorType::q5_k: return {256, 176};
    case GgufTensorType::q6_k: return {256, 210};
    case GgufTensorType::q8_k: return {256, 292};
    }
    fail("unsupported GGUF tensor type");
}

GgufTensorType tensor_type(std::uint32_t raw) {
    switch (raw) {
    case 0: return GgufTensorType::f32;
    case 1: return GgufTensorType::f16;
    case 2: return GgufTensorType::q4_0;
    case 3: return GgufTensorType::q4_1;
    case 6: return GgufTensorType::q5_0;
    case 7: return GgufTensorType::q5_1;
    case 8: return GgufTensorType::q8_0;
    case 9: return GgufTensorType::q8_1;
    case 10: return GgufTensorType::q2_k;
    case 11: return GgufTensorType::q3_k_s;
    case 12: return GgufTensorType::q4_k;
    case 13: return GgufTensorType::q5_k;
    case 14: return GgufTensorType::q6_k;
    case 15: return GgufTensorType::q8_k;
    default: fail("unsupported GGUF tensor type id " + std::to_string(raw));
    }
}

}  // namespace

std::size_t gguf_tensor_byte_size(
    GgufTensorType type,
    const std::vector<std::uint64_t>& dimensions) {
    const auto tensor_layout = layout(type);
    const auto count = elements(dimensions);
    if (count % tensor_layout.block_elements != 0) {
        fail("GGUF tensor element count is not aligned to its quantization block");
    }
    return checked_mul(count / tensor_layout.block_elements, tensor_layout.block_bytes,
                      "tensor byte size");
}

const char* gguf_tensor_type_name(GgufTensorType type) noexcept {
    switch (type) {
    case GgufTensorType::f32: return "F32";
    case GgufTensorType::f16: return "F16";
    case GgufTensorType::q4_0: return "Q4_0";
    case GgufTensorType::q4_1: return "Q4_1";
    case GgufTensorType::q5_0: return "Q5_0";
    case GgufTensorType::q5_1: return "Q5_1";
    case GgufTensorType::q8_0: return "Q8_0";
    case GgufTensorType::q8_1: return "Q8_1";
    case GgufTensorType::q2_k: return "Q2_K";
    case GgufTensorType::q3_k_s: return "Q3_K_S";
    case GgufTensorType::q4_k: return "Q4_K";
    case GgufTensorType::q5_k: return "Q5_K";
    case GgufTensorType::q6_k: return "Q6_K";
    case GgufTensorType::q8_k: return "Q8_K";
    }
    return "UNKNOWN";
}

std::shared_ptr<GgufFile> GgufFile::open(const std::string& path) {
    auto result = std::shared_ptr<GgufFile>(new GgufFile());
    result->path_ = path;
    result->file_descriptor_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (result->file_descriptor_ < 0) {
        throw GgufError("cannot open GGUF file: " + path);
    }
    struct stat status {};
    if (::fstat(result->file_descriptor_, &status) != 0 || status.st_size <= 0) {
        throw GgufError("cannot stat non-empty GGUF file: " + path);
    }
    if (static_cast<std::uintmax_t>(status.st_size)
        > static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max())) {
        throw GgufError("GGUF file is too large for this host");
    }
    result->file_size_ = static_cast<std::size_t>(status.st_size);
    void* mapped = ::mmap(nullptr, result->file_size_, PROT_READ, MAP_PRIVATE,
                          result->file_descriptor_, 0);
    if (mapped == MAP_FAILED) {
        throw GgufError("cannot mmap GGUF file: " + path);
    }
    result->mapping_ = static_cast<const std::byte*>(mapped);

    try {
        Reader reader(result->mapping_, result->file_size_);
        if (reader.read<std::uint32_t>("magic") != kGgufMagic) {
            fail("invalid GGUF magic");
        }
        result->version_ = reader.read<std::uint32_t>("version");
        if (result->version_ != kGgufVersion) {
            fail("unsupported GGUF version " + std::to_string(result->version_));
        }
        const auto tensor_count = reader.read<std::uint64_t>("tensor count");
        const auto metadata_count = reader.read<std::uint64_t>("metadata count");
        if (tensor_count > 1ULL << 20U || metadata_count > 1ULL << 20U) {
            fail("GGUF header count is unreasonably large");
        }
        for (std::uint64_t index = 0; index < metadata_count; ++index) {
            const auto key = reader.string("metadata key");
            const auto type = reader.read<std::uint32_t>("metadata type");
            if (!result->metadata_.emplace(key, read_value(reader, type)).second) {
                fail("duplicate GGUF metadata key: " + key);
            }
        }
        result->tensors_.reserve(checked_size(tensor_count, "tensor count"));
        for (std::uint64_t index = 0; index < tensor_count; ++index) {
            GgufTensor tensor;
            tensor.name = reader.string("tensor name");
            const auto rank = reader.read<std::uint32_t>("tensor rank");
            if (rank == 0 || rank > kMaxDimensions) {
                fail("GGUF tensor rank is outside the supported range");
            }
            tensor.dimensions.reserve(rank);
            for (std::uint32_t dimension = 0; dimension < rank; ++dimension) {
                tensor.dimensions.push_back(reader.read<std::uint64_t>("tensor dimension"));
            }
            tensor.type = tensor_type(reader.read<std::uint32_t>("tensor type"));
            tensor.offset = reader.read<std::uint64_t>("tensor offset");
            tensor.byte_size = gguf_tensor_byte_size(tensor.type, tensor.dimensions);
            if (std::find_if(result->tensors_.begin(), result->tensors_.end(),
                             [&](const GgufTensor& existing) { return existing.name == tensor.name; })
                != result->tensors_.end()) {
                fail("duplicate GGUF tensor name: " + tensor.name);
            }
            result->tensors_.push_back(std::move(tensor));
        }
        const auto alignment_value = result->metadata_.find("general.alignment");
        const std::uint64_t alignment = alignment_value == result->metadata_.end()
                                             ? 32
                                             : std::visit([](const auto& value) -> std::uint64_t {
                                                   using T = std::decay_t<decltype(value)>;
                                                   if constexpr (std::is_same_v<T, std::uint64_t>) {
                                                       return value;
                                                   } else if constexpr (std::is_same_v<T, std::int64_t>) {
                                                       if (value < 0) fail("negative GGUF alignment");
                                                       return static_cast<std::uint64_t>(value);
                                                   } else {
                                                       fail("GGUF general.alignment is not an integer");
                                                   }
                                               }, std::get<GgufScalar>(alignment_value->second.value));
        result->data_offset_ = align_up(reader.position(), checked_size(alignment, "alignment"));
        if (result->data_offset_ > result->file_size_) {
            fail("GGUF tensor data offset is outside the file");
        }
        for (auto& tensor : result->tensors_) {
            const auto relative = checked_size(tensor.offset, "tensor offset");
            if (relative % checked_size(alignment, "alignment") != 0) {
                fail("GGUF tensor offset is not aligned: " + tensor.name);
            }
            const auto absolute = checked_add(result->data_offset_, relative, "tensor range");
            const auto end = checked_add(absolute, tensor.byte_size, "tensor range");
            if (end > result->file_size_) {
                fail("GGUF tensor range is outside the file: " + tensor.name);
            }
            tensor.data = result->mapping_ + absolute;
        }
    } catch (...) {
        ::munmap(const_cast<std::byte*>(result->mapping_), result->file_size_);
        result->mapping_ = nullptr;
        throw;
    }
    return result;
}

GgufFile::~GgufFile() {
    if (mapping_ != nullptr) {
        ::munmap(const_cast<std::byte*>(mapping_), file_size_);
    }
    if (file_descriptor_ >= 0) {
        ::close(file_descriptor_);
    }
}

const GgufValue* GgufFile::metadata(const std::string& key) const noexcept {
    const auto found = metadata_.find(key);
    return found == metadata_.end() ? nullptr : &found->second;
}

std::string GgufFile::metadata_string(const std::string& key) const {
    const auto* value = metadata(key);
    if (value == nullptr || !std::holds_alternative<GgufScalar>(value->value)) {
        throw GgufError("missing or non-scalar GGUF metadata: " + key);
    }
    const auto& scalar = std::get<GgufScalar>(value->value);
    if (!std::holds_alternative<std::string>(scalar)) {
        throw GgufError("GGUF metadata is not a string: " + key);
    }
    return std::get<std::string>(scalar);
}

std::uint64_t GgufFile::metadata_unsigned(const std::string& key) const {
    const auto* value = metadata(key);
    if (value == nullptr || !std::holds_alternative<GgufScalar>(value->value)) {
        throw GgufError("missing or non-scalar GGUF metadata: " + key);
    }
    const auto& scalar = std::get<GgufScalar>(value->value);
    return std::visit([](const auto& item) -> std::uint64_t {
        using T = std::decay_t<decltype(item)>;
        if constexpr (std::is_same_v<T, std::uint64_t>) return item;
        if constexpr (std::is_same_v<T, std::uint32_t>) return item;
        if constexpr (std::is_same_v<T, std::int64_t>) {
            if (item < 0) fail("negative unsigned GGUF metadata");
            return static_cast<std::uint64_t>(item);
        }
        fail("GGUF metadata is not an unsigned integer");
    }, scalar);
}

double GgufFile::metadata_float(const std::string& key) const {
    const auto* value = metadata(key);
    if (value == nullptr || !std::holds_alternative<GgufScalar>(value->value)) {
        throw GgufError("missing or non-scalar GGUF metadata: " + key);
    }
    const auto& scalar = std::get<GgufScalar>(value->value);
    return std::visit([](const auto& item) -> double {
        using T = std::decay_t<decltype(item)>;
        if constexpr (std::is_same_v<T, double>) return item;
        if constexpr (std::is_same_v<T, std::uint64_t>) return static_cast<double>(item);
        if constexpr (std::is_same_v<T, std::int64_t>) return static_cast<double>(item);
        fail("GGUF metadata is not numeric");
    }, scalar);
}

std::size_t GgufFile::metadata_array_size(const std::string& key) const {
    const auto* value = metadata(key);
    if (value == nullptr || !std::holds_alternative<std::vector<GgufScalar>>(value->value)) {
        throw GgufError("missing or non-array GGUF metadata: " + key);
    }
    return std::get<std::vector<GgufScalar>>(value->value).size();
}

bool GgufFile::metadata_array_is_string(const std::string& key) const {
    const auto* value = metadata(key);
    if (value == nullptr || !std::holds_alternative<std::vector<GgufScalar>>(value->value)) {
        throw GgufError("missing or non-array GGUF metadata: " + key);
    }
    const auto& values = std::get<std::vector<GgufScalar>>(value->value);
    return std::all_of(values.begin(), values.end(), [](const GgufScalar& scalar) {
        return std::holds_alternative<std::string>(scalar);
    });
}

}  // namespace miinfer
