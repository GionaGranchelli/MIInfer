#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace miinfer {

struct ChatToolCall {
    std::string id;
    std::string name;
    std::string arguments;
};

struct ChatTool {
    std::string name;
    std::string description;
    std::string parameters_json;
};

struct ChatMessage {
    std::string role;
    std::string content;
    std::optional<std::string> tool_call_id;
    std::vector<ChatToolCall> tool_calls;
};

struct OpenAiChatRequest {
    std::optional<std::string> model;
    std::vector<ChatMessage> messages;
    std::vector<ChatTool> tools;
    std::string tool_choice = "auto";
    bool stream = false;
    std::size_t max_tokens = 256;
};

struct OpenAiParseResult {
    std::optional<OpenAiChatRequest> request;
    std::string error;
};

constexpr std::size_t kMaxOutputTokens = 4096;

struct OpenAiGeneratedToolCalls {
    std::vector<ChatToolCall> calls;
};

OpenAiParseResult parse_openai_chat_request(std::string_view body);
std::string build_chatml(const OpenAiChatRequest& request);
OpenAiGeneratedToolCalls parse_generated_tool_calls(std::string_view text);

} // namespace miinfer
