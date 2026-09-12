#include "miinfer/openai_api.hpp"

#include <algorithm>
#include <cctype>
#include <nlohmann/json.hpp>

namespace miinfer {

namespace {

using json = nlohmann::json;

std::string trim_ascii(std::string_view value) {
    std::size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) ++begin;
    std::size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) --end;
    return std::string(value.substr(begin, end - begin));
}

bool read_content(const json& value, std::string& content) {
    if (value.is_string()) {
        content = value.get<std::string>();
        return true;
    }
    if (value.is_null()) {
        content.clear();
        return true;
    }
    if (!value.is_array()) return false;
    content.clear();
    for (const auto& part : value) {
        if (!part.is_object() || !part.contains("type") || !part["type"].is_string()
            || part["type"] != "text" || !part.contains("text") || !part["text"].is_string()) {
            return false;
        }
        content += part["text"].get<std::string>();
    }
    return true;
}

std::string value_for_qwen_parameter(const json& value) {
    return value.is_string() ? value.get<std::string>() : value.dump();
}

std::string tool_json(const ChatTool& tool) {
    json function = {
        {"name", tool.name},
        {"description", tool.description},
        {"parameters", json::parse(tool.parameters_json)},
    };
    return json{{"type", "function"}, {"function", std::move(function)}}.dump();
}

std::string tool_instructions(const OpenAiChatRequest& request) {
    if (request.tools.empty() || request.tool_choice == "none") return {};
    std::string result = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
    for (const auto& tool : request.tools) result += "\n" + tool_json(tool);
    result += "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
              "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n"
              "value_1\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
              "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
              "- Required parameters MUST be specified\n"
              "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
              "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
              "</IMPORTANT>";
    return result;
}

void append_assistant_tool_calls(std::string& prompt, const ChatMessage& message) {
    for (std::size_t i = 0; i < message.tool_calls.size(); ++i) {
        const auto& call = message.tool_calls[i];
        json arguments = json::parse(call.arguments);
        if (i == 0) prompt += message.content.empty() ? "<tool_call>\n" : "\n\n<tool_call>\n";
        else prompt += "\n<tool_call>\n";
        prompt += "<function=" + call.name + ">\n";
        if (arguments.is_object()) {
            for (const auto& [name, value] : arguments.items()) {
                prompt += "<parameter=" + name + ">\n" + value_for_qwen_parameter(value)
                    + "\n</parameter>\n";
            }
        }
        prompt += "</function>\n</tool_call>";
    }
}

void append_tool_result_messages(std::string& prompt, const std::vector<ChatMessage>& messages,
                                 std::size_t& index) {
    prompt += "<|im_start|>user";
    while (index < messages.size() && messages[index].role == "tool") {
        prompt += "\n<tool_response>\n" + messages[index].content + "\n</tool_response>";
        ++index;
    }
    prompt += "<|im_end|>\n";
}

void append_generated_xml_calls(std::string_view text, OpenAiGeneratedToolCalls& result) {
    std::size_t position = 0;
    while ((position = text.find("<function=", position)) != std::string_view::npos) {
        const std::size_t name_end = text.find('>', position + 10);
        if (name_end == std::string_view::npos) return;
        const std::string name(text.substr(position + 10, name_end - position - 10));
        const std::size_t function_end = text.find("</function>", name_end + 1);
        if (name.empty() || function_end == std::string_view::npos) return;

        json arguments = json::object();
        std::size_t argument_position = name_end + 1;
        while ((argument_position = text.find("<parameter=", argument_position)) != std::string_view::npos
               && argument_position < function_end) {
            const std::size_t parameter_end = text.find('>', argument_position + 11);
            if (parameter_end == std::string_view::npos || parameter_end >= function_end) return;
            const std::string parameter(text.substr(argument_position + 11,
                                                     parameter_end - argument_position - 11));
            const std::size_t value_end = text.find("</parameter>", parameter_end + 1);
            if (parameter.empty() || value_end == std::string_view::npos || value_end > function_end) return;
            const std::string value = trim_ascii(text.substr(parameter_end + 1,
                                                              value_end - parameter_end - 1));
            try {
                arguments[parameter] = json::parse(value);
            } catch (const json::exception&) {
                arguments[parameter] = value;
            }
            argument_position = value_end + 12;
        }
        result.calls.push_back({"call_" + std::to_string(result.calls.size()), name, arguments.dump()});
        position = function_end + 11;
    }
}

void append_generated_json_calls(std::string_view text, OpenAiGeneratedToolCalls& result) {
    const std::size_t open = text.find("<tool_call>");
    if (open == std::string_view::npos) return;
    const std::size_t begin = open + 11;
    const std::size_t close = text.find("</tool_call>", begin);
    if (close == std::string_view::npos) return;
    try {
        const json value = json::parse(text.substr(begin, close - begin));
        const auto add = [&](const json& call) {
            if (!call.is_object() || !call.contains("name") || !call["name"].is_string()) return;
            const json arguments = call.contains("arguments") ? call["arguments"] : json::object();
            result.calls.push_back({"call_" + std::to_string(result.calls.size()),
                                    call["name"].get<std::string>(),
                                    arguments.is_string() ? arguments.get<std::string>() : arguments.dump()});
        };
        if (value.is_array()) for (const auto& call : value) add(call);
        else add(value);
    } catch (const json::exception&) {
    }
}

} // namespace

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
        if (json.contains("tool_choice")) {
            const auto& choice = json["tool_choice"];
            if (choice.is_string()) {
                request.tool_choice = choice.get<std::string>();
                if (request.tool_choice != "none" && request.tool_choice != "auto"
                    && request.tool_choice != "required") return {{}, "unsupported tool_choice"};
            } else if (choice.is_object() && choice.contains("type") && choice["type"] == "function"
                       && choice.contains("function") && choice["function"].is_object()
                       && choice["function"].contains("name") && choice["function"]["name"].is_string()) {
                request.tool_choice = "required";
            } else {
                return {{}, "tool_choice must be auto, none, required, or a function choice"};
            }
        }
        if (json.contains("tools")) {
            if (!json["tools"].is_array()) return {{}, "tools must be an array"};
            for (const auto& tool : json["tools"]) {
                if (!tool.is_object() || tool.value("type", "") != "function"
                    || !tool.contains("function") || !tool["function"].is_object()) {
                    return {{}, "only function tools are supported"};
                }
                const auto& function = tool["function"];
                if (!function.contains("name") || !function["name"].is_string()) {
                    return {{}, "each function tool needs a string name"};
                }
                const std::string description = function.contains("description")
                    ? function.value("description", "") : "";
                const auto parameters = function.contains("parameters")
                    ? function["parameters"] : nlohmann::json::object();
                if (!parameters.is_object()) return {{}, "tool parameters must be an object"};
                request.tools.push_back({function["name"].get<std::string>(), description, parameters.dump()});
            }
        }
        for (const auto& message : json["messages"]) {
            if (!message.is_object() || !message.contains("role") || !message["role"].is_string()) {
                return {{}, "each message needs a string role"};
            }
            const auto role = message["role"].get<std::string>();
            if (role != "system" && role != "developer" && role != "user"
                && role != "assistant" && role != "tool") return {{}, "unsupported message role"};
            ChatMessage parsed_message;
            parsed_message.role = role;
            if (message.contains("content")) {
                if (!read_content(message["content"], parsed_message.content)) {
                    return {{}, "message content must be a string, null, or text parts"};
                }
            } else if (role != "assistant" || !message.contains("tool_calls")) {
                return {{}, "each message needs content unless it has tool_calls"};
            }
            if (role == "tool") {
                if (!message.contains("tool_call_id") || !message["tool_call_id"].is_string()) {
                    return {{}, "tool messages need a string tool_call_id"};
                }
                parsed_message.tool_call_id = message["tool_call_id"].get<std::string>();
            }
            if (message.contains("tool_calls")) {
                if (role != "assistant" || !message["tool_calls"].is_array()) {
                    return {{}, "tool_calls are only supported on assistant messages"};
                }
                for (const auto& call : message["tool_calls"]) {
                    if (!call.is_object() || !call.contains("id") || !call["id"].is_string()
                        || call.value("type", "") != "function" || !call.contains("function")
                        || !call["function"].is_object()) return {{}, "invalid assistant tool_call"};
                    const auto& function = call["function"];
                    if (!function.contains("name") || !function["name"].is_string()
                        || !function.contains("arguments") || !function["arguments"].is_string()) {
                        return {{}, "tool_call function needs name and string arguments"};
                    }
                    try {
                        const auto arguments = nlohmann::json::parse(function["arguments"].get<std::string>());
                        (void)arguments;
                    }
                    catch (const nlohmann::json::exception&) { return {{}, "tool_call arguments must be valid JSON"}; }
                    parsed_message.tool_calls.push_back({call["id"].get<std::string>(),
                        function["name"].get<std::string>(), function["arguments"].get<std::string>()});
                }
            }
            request.messages.push_back(std::move(parsed_message));
        }
        return {std::move(request), {}};
    } catch (const nlohmann::json::exception&) {
        return {{}, "invalid JSON"};
    }
}

std::string build_chatml(const OpenAiChatRequest& request) {
    std::string prompt;
    const std::string tools = tool_instructions(request);
    std::size_t index = 0;
    if (!tools.empty()) {
        prompt = "<|im_start|>system\n" + tools;
        if (!request.messages.empty()
            && (request.messages.front().role == "system" || request.messages.front().role == "developer")) {
            if (!request.messages.front().content.empty()) prompt += "\n\n" + request.messages.front().content;
            index = 1;
        }
        prompt += "<|im_end|>\n";
    }
    while (index < request.messages.size()) {
        const auto& message = request.messages[index];
        if (message.role == "tool") {
            append_tool_result_messages(prompt, request.messages, index);
            continue;
        }
        const std::string role = message.role == "developer" ? "system" : message.role;
        prompt += "<|im_start|>" + role + "\n" + message.content;
        if (role == "assistant" && !message.tool_calls.empty()) append_assistant_tool_calls(prompt, message);
        prompt += "<|im_end|>\n";
        ++index;
    }
    return prompt + "<|im_start|>assistant\n";
}

OpenAiGeneratedToolCalls parse_generated_tool_calls(std::string_view text) {
    OpenAiGeneratedToolCalls result;
    append_generated_xml_calls(text, result);
    if (result.calls.empty()) append_generated_json_calls(text, result);
    return result;
}

} // namespace miinfer
