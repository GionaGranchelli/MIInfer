#include "miinfer/openai_api.hpp"

#include <iostream>

#define CHECK(condition) do { if (!(condition)) { std::cerr << "check failed: " #condition "\n"; return 1; } } while (false)

int main() {
    const auto default_budget = miinfer::parse_openai_chat_request(
        R"({"messages":[{"role":"user","content":"hi"}]})");
    CHECK(default_budget.request);
    CHECK(!default_budget.request->max_tokens);

    const auto explicit_budget = miinfer::parse_openai_chat_request(
        R"({"messages":[{"role":"user","content":"hi"}],"max_tokens":32})");
    CHECK(explicit_budget.request && explicit_budget.request->max_tokens == 32);
    const auto completion_budget = miinfer::parse_openai_chat_request(
        R"({"messages":[{"role":"user","content":"hi"}],"max_completion_tokens":256})");
    CHECK(completion_budget.request && completion_budget.request->max_tokens == 256);

    CHECK(miinfer::resolve_output_token_limit(std::nullopt, 55, 8192)
          == miinfer::kDefaultMaxOutputTokens);
    CHECK(miinfer::resolve_output_token_limit(std::nullopt, 8191, 8192) == 1);
    CHECK(miinfer::resolve_output_token_limit(32, 8000, 8192) == 32);
    CHECK(!miinfer::resolve_output_token_limit(256, 8000, 8192));
    CHECK(!miinfer::resolve_output_token_limit(std::nullopt, 8192, 8192));

    CHECK(miinfer::openai_finish_reason(miinfer::GenerationStopReason::kStopToken) == "stop");
    CHECK(miinfer::openai_finish_reason(miinfer::GenerationStopReason::kOutputLimit) == "length");
    CHECK(miinfer::openai_finish_reason(miinfer::GenerationStopReason::kCancelled) == "cancelled");
    CHECK(miinfer::openai_finish_reason(miinfer::GenerationStopReason::kOutputLimit, true) == "tool_calls");

    const auto parsed = miinfer::parse_openai_chat_request(R"({"model":"test","messages":[{"role":"system","content":"Be concise."},{"role":"user","content":"He said \"hi\"\\ok\n"},{"role":"assistant","content":"Hi"},{"role":"user","content":"\u03bb"}],"stream":true,"max_tokens":4096,"ignored":1})");
    CHECK(parsed.request);
    CHECK(parsed.request->model && *parsed.request->model == "test");
    CHECK(parsed.request->stream);
    CHECK(parsed.request->max_tokens == 4096);
    CHECK(parsed.request->messages.size() == 4);
    CHECK(miinfer::build_chatml(*parsed.request) == "<|im_start|>system\nBe concise.<|im_end|>\n<|im_start|>user\nHe said \"hi\"\\ok\n<|im_end|>\n<|im_start|>assistant\nHi<|im_end|>\n<|im_start|>user\nλ<|im_end|>\n<|im_start|>assistant\n");
    const auto tool_request = miinfer::parse_openai_chat_request(R"({"messages":[{"role":"system","content":"Use tools."},{"role":"user","content":[{"type":"text","text":"Read a file."}]},{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"a.txt\"}"}}]},{"role":"tool","tool_call_id":"call_1","content":"contents"}],"tools":[{"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}],"stream":true})");
    CHECK(tool_request.request);
    CHECK(tool_request.request->tools.size() == 1);
    CHECK(tool_request.request->messages[1].content == "Read a file.");
    CHECK(tool_request.request->messages[2].tool_calls.size() == 1);
    CHECK(miinfer::build_chatml(*tool_request.request).find("<tools>") != std::string::npos);
    CHECK(miinfer::build_chatml(*tool_request.request).find("<tool_response>\ncontents") != std::string::npos);

    const auto stop_array_request = miinfer::parse_openai_chat_request(R"({"messages":[{"role":"user","content":"hi"}],"stop":["<|im_end|>","<|endoftext|>"]})");
    CHECK(stop_array_request.request);
    CHECK(stop_array_request.request->stop.size() == 2);
    CHECK(stop_array_request.request->stop[0] == "<|im_end|>");
    CHECK(stop_array_request.request->stop[1] == "<|endoftext|>");

    const auto stop_str_request = miinfer::parse_openai_chat_request(R"({"messages":[{"role":"user","content":"hi"}],"stop":"<|im_end|>"})");
    CHECK(stop_str_request.request);
    CHECK(stop_str_request.request->stop.size() == 1);
    CHECK(stop_str_request.request->stop[0] == "<|im_end|>");

    const auto generated = miinfer::parse_generated_tool_calls(
        "<think>I'll inspect it.</think>\n<tool_call>\n<function=read_file>\n"
        "<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>");
    CHECK(generated.calls.size() == 1);
    CHECK(generated.calls[0].name == "read_file");
    CHECK(generated.calls[0].arguments == R"({"path":"a.txt"})");

    for (const char* invalid : {"{", "{}", R"({"messages":[]})", R"({"messages":{}})", R"({"messages":[{"role":"tool","content":"x"}]})", R"({"messages":[{"role":"user","content":1}]})", R"({"messages":[{"role":"user","content":"x"}],"stream":1})", R"({"messages":[{"role":"user","content":"x"}],"max_tokens":-1})", R"({"messages":[{"role":"user","content":"x"}],"max_tokens":0})", R"({"messages":[{"role":"user","content":"x"}],"max_tokens":65537})", R"({"messages":[{"role":"user","content":"x"}],"max_completion_tokens":0})"}) {
        CHECK(!miinfer::parse_openai_chat_request(invalid).request);
    }
    std::cout << "openai API host test passed\n";
}
