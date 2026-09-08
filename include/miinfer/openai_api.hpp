#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace miinfer {

struct OpenAiChatRequest {
    std::string chatml_prompt;
    bool stream = false;
    std::size_t max_tokens = 256;
};

struct OpenAiParseResult {
    std::optional<OpenAiChatRequest> request;
    std::string error;
};

OpenAiParseResult parse_openai_chat_request(std::string_view body);

} // namespace miinfer
