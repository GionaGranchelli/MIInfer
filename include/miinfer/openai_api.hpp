#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace miinfer {

struct ChatMessage {
    std::string role;
    std::string content;
};

struct OpenAiChatRequest {
    std::optional<std::string> model;
    std::vector<ChatMessage> messages;
    bool stream = false;
    std::size_t max_tokens = 256;
};

struct OpenAiParseResult {
    std::optional<OpenAiChatRequest> request;
    std::string error;
};

constexpr std::size_t kMaxOutputTokens = 4096;

OpenAiParseResult parse_openai_chat_request(std::string_view body);
std::string build_chatml(const OpenAiChatRequest& request);

} // namespace miinfer
