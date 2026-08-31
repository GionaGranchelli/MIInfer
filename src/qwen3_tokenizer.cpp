#include "miinfer/qwen3_tokenizer.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace miinfer {

namespace {

struct Codepoint {
    std::uint32_t value;
    std::size_t begin;
    std::size_t end;
};

std::vector<Codepoint> codepoints(std::string_view text) {
    std::vector<Codepoint> result;
    for (std::size_t offset = 0; offset < text.size();) {
        const auto begin = offset;
        const auto first = static_cast<unsigned char>(text[offset++]);
        std::uint32_t value = first;
        std::size_t continuation = 0;
        if ((first & 0x80U) == 0) {
            continuation = 0;
        } else if ((first & 0xE0U) == 0xC0U) {
            value = first & 0x1FU;
            continuation = 1;
        } else if ((first & 0xF0U) == 0xE0U) {
            value = first & 0x0FU;
            continuation = 2;
        } else if ((first & 0xF8U) == 0xF0U) {
            value = first & 0x07U;
            continuation = 3;
        } else {
            value = 0xFFFD;
        }
        if (offset + continuation > text.size()) {
            throw std::invalid_argument("input text is not valid UTF-8");
        }
        for (std::size_t i = 0; i < continuation; ++i) {
            const auto byte = static_cast<unsigned char>(text[offset]);
            if ((byte & 0xC0U) != 0x80U) {
                throw std::invalid_argument("input text is not valid UTF-8");
            }
            value = (value << 6U) | (byte & 0x3FU);
            ++offset;
        }
        result.push_back({value, begin, offset});
    }
    return result;
}

std::string utf8(std::uint32_t value) {
    std::string result;
    if (value <= 0x7FU) {
        result.push_back(static_cast<char>(value));
    } else if (value <= 0x7FFU) {
        result.push_back(static_cast<char>(0xC0U | (value >> 6U)));
        result.push_back(static_cast<char>(0x80U | (value & 0x3FU)));
    } else if (value <= 0xFFFFU) {
        result.push_back(static_cast<char>(0xE0U | (value >> 12U)));
        result.push_back(static_cast<char>(0x80U | ((value >> 6U) & 0x3FU)));
        result.push_back(static_cast<char>(0x80U | (value & 0x3FU)));
    } else {
        result.push_back(static_cast<char>(0xF0U | (value >> 18U)));
        result.push_back(static_cast<char>(0x80U | ((value >> 12U) & 0x3FU)));
        result.push_back(static_cast<char>(0x80U | ((value >> 6U) & 0x3FU)));
        result.push_back(static_cast<char>(0x80U | (value & 0x3FU)));
    }
    return result;
}

bool letter(std::uint32_t value) {
    return (value >= 'A' && value <= 'Z') || (value >= 'a' && value <= 'z')
        || value >= 0x80U;
}

bool number(std::uint32_t value) {
    return value >= '0' && value <= '9';
}

bool whitespace(std::uint32_t value) {
    return value == ' ' || value == '\t' || value == '\n' || value == '\r'
        || value == '\v' || value == '\f';
}

bool punctuation(std::uint32_t value) {
    return !whitespace(value) && !letter(value) && !number(value);
}

bool value_is_newline(std::uint32_t value) {
    return value == '\r' || value == '\n';
}

std::string merge_key(std::string_view left, std::string_view right) {
    std::string key;
    key.reserve(left.size() + right.size() + 1);
    key.append(left);
    key.push_back('\x01');
    key.append(right);
    return key;
}

std::uint32_t byte_to_codepoint(std::uint8_t byte) {
    if ((byte >= 0x21U && byte <= 0x7EU)
        || (byte >= 0xA1U && byte <= 0xACU)
        || byte >= 0xAEU) {
        return byte;
    }
    std::uint32_t missing = 0;
    for (std::uint16_t candidate = 0; candidate < byte; ++candidate) {
        const auto prior = static_cast<std::uint8_t>(candidate);
        if (!((prior >= 0x21U && prior <= 0x7EU)
              || (prior >= 0xA1U && prior <= 0xACU)
              || prior >= 0xAEU)) {
            ++missing;
        }
    }
    return 256U + missing;
}

bool codepoint_to_byte(std::uint32_t value, std::uint8_t& byte) {
    for (std::uint16_t candidate = 0; candidate < 256; ++candidate) {
        if (byte_to_codepoint(static_cast<std::uint8_t>(candidate)) == value) {
            byte = static_cast<std::uint8_t>(candidate);
            return true;
        }
    }
    return false;
}

std::vector<std::string> split_qwen2(std::string_view text) {
    const auto points = codepoints(text);
    std::vector<std::string> result;
    std::size_t pos = 0;
    while (pos < points.size()) {
        const auto value = points[pos].value;
        if (value == '\'' && pos + 1 < points.size()) {
            const auto next = points[pos + 1].value;
            if (next == 's' || next == 'S' || next == 't' || next == 'T'
                || next == 'm' || next == 'M' || next == 'd' || next == 'D') {
                result.emplace_back(text.substr(points[pos].begin,
                                                points[pos + 1].end - points[pos].begin));
                pos += 2;
                continue;
            }
            if (pos + 2 < points.size()) {
                const auto third = points[pos + 2].value;
                if ((next == 'r' || next == 'R') && (third == 'e' || third == 'E')) {
                    result.emplace_back(text.substr(points[pos].begin,
                                                    points[pos + 2].end - points[pos].begin));
                    pos += 3;
                    continue;
                }
                if ((next == 'v' || next == 'V') && (third == 'e' || third == 'E')) {
                    result.emplace_back(text.substr(points[pos].begin,
                                                    points[pos + 2].end - points[pos].begin));
                    pos += 3;
                    continue;
                }
                if ((next == 'l' || next == 'L') && (third == 'l' || third == 'L')) {
                    result.emplace_back(text.substr(points[pos].begin,
                                                    points[pos + 2].end - points[pos].begin));
                    pos += 3;
                    continue;
                }
            }
        }

        const bool next_is_letter = pos + 1 < points.size() && letter(points[pos + 1].value);
        if (!value_is_newline(value) && !number(value) && (letter(value) || next_is_letter)) {
            const auto begin = points[pos].begin;
            ++pos;
            while (pos < points.size() && letter(points[pos].value)) ++pos;
            result.emplace_back(text.substr(begin, points[pos - 1].end - begin));
            continue;
        }

        if (number(value)) {
            result.emplace_back(text.substr(points[pos].begin,
                                            points[pos].end - points[pos].begin));
            ++pos;
            continue;
        }

        const bool starts_with_space = value == ' ';
        const std::size_t punctuation_start = pos;
        if (starts_with_space && pos + 1 < points.size() && punctuation(points[pos + 1].value)) ++pos;
        if (pos < points.size() && punctuation(points[pos].value)) {
            ++pos;
            while (pos < points.size() && punctuation(points[pos].value)) ++pos;
            while (pos < points.size() && (points[pos].value == '\r' || points[pos].value == '\n')) ++pos;
            result.emplace_back(text.substr(points[punctuation_start].begin,
                                            points[pos - 1].end - points[punctuation_start].begin));
            continue;
        }

        const auto whitespace_start = pos;
        while (pos < points.size() && whitespace(points[pos].value)) ++pos;
        if (pos > whitespace_start) {
            result.emplace_back(text.substr(points[whitespace_start].begin,
                                            points[pos - 1].end - points[whitespace_start].begin));
            continue;
        }

        throw std::runtime_error("Qwen2 pre-tokenizer could not classify input");
    }
    return result;
}

std::string byte_encode(std::string_view piece) {
    std::string result;
    for (const auto byte : piece) result += utf8(byte_to_codepoint(static_cast<std::uint8_t>(byte)));
    return result;
}

std::string decode_piece(std::string_view piece) {
    std::string result;
    for (const auto point : codepoints(piece)) {
        std::uint8_t byte = 0;
        if (codepoint_to_byte(point.value, byte)) result.push_back(static_cast<char>(byte));
        else result += utf8(point.value);
    }
    return result;
}

const std::vector<GgufScalar>& metadata_array(const Qwen3Model& model, const char* key) {
    const auto* value = model.file()->metadata(key);
    if (value == nullptr || !std::holds_alternative<std::vector<GgufScalar>>(value->value)) {
        throw std::invalid_argument(std::string("missing or non-array GGUF metadata: ") + key);
    }
    return std::get<std::vector<GgufScalar>>(value->value);
}

std::uint32_t metadata_uint(const Qwen3Model& model, const char* key) {
    const auto* value = model.file()->metadata(key);
    if (value == nullptr || !std::holds_alternative<GgufScalar>(value->value)) {
        throw std::invalid_argument(std::string("missing GGUF metadata: ") + key);
    }
    const auto& scalar = std::get<GgufScalar>(value->value);
    if (!std::holds_alternative<std::uint64_t>(scalar)) {
        throw std::invalid_argument(std::string("GGUF metadata is not unsigned: ") + key);
    }
    return static_cast<std::uint32_t>(std::get<std::uint64_t>(scalar));
}

bool metadata_bool(const Qwen3Model& model, const char* key, bool fallback) {
    const auto* value = model.file()->metadata(key);
    if (value == nullptr) return fallback;
    if (!std::holds_alternative<GgufScalar>(value->value)) {
        throw std::invalid_argument(std::string("GGUF metadata is not scalar: ") + key);
    }
    const auto& scalar = std::get<GgufScalar>(value->value);
    if (!std::holds_alternative<bool>(scalar)) {
        throw std::invalid_argument(std::string("GGUF metadata is not boolean: ") + key);
    }
    return std::get<bool>(scalar);
}

}  // namespace

Qwen3Tokenizer Qwen3Tokenizer::load(const Qwen3Model& model) {
    const auto* model_value = model.file()->metadata("tokenizer.ggml.model");
    const auto* pre_value = model.file()->metadata("tokenizer.ggml.pre");
    if (model_value == nullptr || pre_value == nullptr
        || !std::holds_alternative<GgufScalar>(model_value->value)
        || !std::holds_alternative<GgufScalar>(pre_value->value)
        || !std::holds_alternative<std::string>(std::get<GgufScalar>(model_value->value))
        || !std::holds_alternative<std::string>(std::get<GgufScalar>(pre_value->value))
        || std::get<std::string>(std::get<GgufScalar>(model_value->value)) != "gpt2"
        || std::get<std::string>(std::get<GgufScalar>(pre_value->value)) != "qwen2") {
        throw std::invalid_argument("unsupported tokenizer; expected GGUF gpt2/qwen2");
    }

    Qwen3Tokenizer tokenizer;
    const auto& token_values = metadata_array(model, "tokenizer.ggml.tokens");
    tokenizer.tokens_.reserve(token_values.size());
    for (std::size_t id = 0; id < token_values.size(); ++id) {
        if (!std::holds_alternative<std::string>(token_values[id])) {
            throw std::invalid_argument("tokenizer.ggml.tokens contains a non-string value");
        }
        tokenizer.tokens_.push_back(std::get<std::string>(token_values[id]));
        tokenizer.token_to_id_.emplace(tokenizer.tokens_.back(), static_cast<std::uint32_t>(id));
    }
    const auto& merge_values = metadata_array(model, "tokenizer.ggml.merges");
    for (std::size_t rank = 0; rank < merge_values.size(); ++rank) {
        if (!std::holds_alternative<std::string>(merge_values[rank])) {
            throw std::invalid_argument("tokenizer.ggml.merges contains a non-string value");
        }
        const auto& merge = std::get<std::string>(merge_values[rank]);
        const auto separator = merge.find(' ');
        if (separator == std::string::npos) throw std::invalid_argument("invalid Qwen2 BPE merge");
        const auto left = merge.substr(0, separator);
        const auto right = merge.substr(separator + 1);
        tokenizer.merge_ranks_.emplace(merge_key(left, right), static_cast<std::uint32_t>(rank));
    }
    tokenizer.bos_id_ = metadata_uint(model, "tokenizer.ggml.bos_token_id");
    tokenizer.eos_id_ = metadata_uint(model, "tokenizer.ggml.eos_token_id");
    tokenizer.add_bos_ = metadata_bool(model, "tokenizer.ggml.add_bos_token", false);
    tokenizer.add_eos_ = metadata_bool(model, "tokenizer.ggml.add_eos_token", false);
    return tokenizer;
}

std::vector<std::uint32_t> Qwen3Tokenizer::encode(std::string_view text) const {
    std::vector<std::uint32_t> result;
    if (add_bos_) result.push_back(bos_id_);
    for (const auto& raw_piece : split_qwen2(text)) {
        std::vector<std::string> symbols;
        const auto encoded = byte_encode(raw_piece);
        for (const auto point : codepoints(encoded)) {
            symbols.emplace_back(encoded.substr(point.begin, point.end - point.begin));
        }
        while (symbols.size() > 1) {
            std::size_t best = symbols.size();
            std::uint32_t best_rank = 0;
            for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
                const auto it = merge_ranks_.find(merge_key(symbols[i], symbols[i + 1]));
                if (it != merge_ranks_.end() && (best == symbols.size() || it->second < best_rank)) {
                    best = i;
                    best_rank = it->second;
                }
            }
            if (best == symbols.size()) break;
            symbols[best] += symbols[best + 1];
            symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best + 1));
        }
        for (const auto& symbol : symbols) {
            const auto it = token_to_id_.find(symbol);
            if (it == token_to_id_.end()) throw std::runtime_error("Qwen2 BPE produced an unknown token");
            result.push_back(it->second);
        }
    }
    if (add_eos_) result.push_back(eos_id_);
    return result;
}

std::string Qwen3Tokenizer::decode(std::span<const std::uint32_t> tokens) const {
    std::string result;
    for (const auto id : tokens) {
        if (id >= tokens_.size()) throw std::out_of_range("token ID is outside tokenizer vocabulary");
        result += decode_piece(tokens_[id]);
    }
    return result;
}

}  // namespace miinfer
