#include "miinfer/openai_api.hpp"

#include <cassert>
#include <iostream>

int main() {
    const auto parsed = miinfer::parse_openai_chat_request(R"({"model":"test","messages":[{"role":"system","content":"Be concise."},{"role":"user","content":"He said \"hi\"\\ok\n"},{"role":"assistant","content":"Hi"},{"role":"user","content":"\u03bb"}],"stream":true,"max_tokens":9999,"ignored":1})");
    assert(parsed.request);
    assert(parsed.request->model && *parsed.request->model == "test");
    assert(parsed.request->stream);
    assert(parsed.request->max_tokens == 4096);
    assert(parsed.request->messages.size() == 4);
    assert(miinfer::build_chatml(*parsed.request) == "<|im_start|>system\nBe concise.<|im_end|>\n<|im_start|>user\nHe said \"hi\"\\ok\n<|im_end|>\n<|im_start|>assistant\nHi<|im_end|>\n<|im_start|>user\nλ<|im_end|>\n<|im_start|>assistant\n");
    for (const char* invalid : {"{", "{}", R"({"messages":[]})", R"({"messages":{}})", R"({"messages":[{"role":"tool","content":"x"}]})", R"({"messages":[{"role":"user","content":1}]})", R"({"messages":[{"role":"user","content":"x"}],"stream":1})", R"({"messages":[{"role":"user","content":"x"}],"max_tokens":-1})"}) {
        assert(!miinfer::parse_openai_chat_request(invalid).request);
    }
    std::cout << "openai API host test passed\n";
}
