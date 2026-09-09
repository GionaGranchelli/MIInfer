#include "miinfer/openai_api.hpp"

#include <algorithm>
#include <nlohmann/json.hpp>

namespace miinfer {

OpenAiParseResult parse_openai_chat_request(std::string_view body) {
    try {
        const auto json = nlohmann::json::parse(body);
        if (!json.is_object() || !json.contains("messages") || !json["messages"].is_array()
            || json["messages"].empty()) return {{}, "messages must be a non-empty array"};
        OpenAiChatRequest request;
        if (json.contains("model")) {
            if (!json["model"].is_string()) return {{}, "model must be a string"};
            request.model = json["model"].get<std::string>();
        }
        if (json.contains("stream")) {
            if (!json["stream"].is_boolean()) return {{}, "stream must be a boolean"};
            request.stream = json["stream"].get<bool>();
        }
        if (json.contains("max_tokens")) {
            if (!json["max_tokens"].is_number_unsigned()) return {{}, "max_tokens must be an unsigned integer"};
            request.max_tokens = json["max_tokens"].get<std::size_t>();
            if (request.max_tokens > kMaxOutputTokens) {
                return {{}, "max_tokens must be at most 4096"};
            }
        }
        for (const auto& message : json["messages"]) {
            if (!message.is_object() || !message.contains("role") || !message["role"].is_string()
                || !message.contains("content") || !message["content"].is_string()) return {{}, "each message needs string role and content"};
            const auto role = message["role"].get<std::string>();
            if (role != "system" && role != "user" && role != "assistant") return {{}, "unsupported message role"};
            request.messages.push_back({role, message["content"].get<std::string>()});
        }
        return {std::move(request), {}};
    } catch (const nlohmann::json::exception&) {
        return {{}, "invalid JSON"};
    }
}

std::string build_chatml(const OpenAiChatRequest& request) {
    std::string prompt;
    for (const auto& message : request.messages) {
        prompt += "<|im_start|>" + message.role + "\n" + message.content + "<|im_end|>\n";
    }
    return prompt + "<|im_start|>assistant\n";
}

} // namespace miinfer
