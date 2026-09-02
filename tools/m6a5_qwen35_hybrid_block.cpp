#define MIINFER_M6A4_HELPERS_ONLY
#include "m6a4_qwen35_attention_layer.cpp"

#include <array>
#include <iostream>
#include <stdexcept>

namespace {

constexpr std::size_t kRecurrentLayers = 3;
constexpr std::size_t kFullLayer = 3;

std::vector<float> recurrent_block(const GgufFile& model, const std::filesystem::path& fixture,
                                   std::size_t layer, std::size_t position,
                                   std::span<const float> input, std::span<const float> state,
                                   std::vector<std::array<float, kChannels>>& history,
                                   std::vector<float>& state_out) {
    const auto norm_weights = f32_values(tensor(model, "blk." + std::to_string(layer) + ".attn_norm.weight"), kHidden);
    std::vector<float> normalized(kHidden);
    rms_rows(input, norm_weights, normalized, kHidden);
    const auto prefix = "blk." + std::to_string(layer) + ".";
    auto qkv = gemv(tensor(model, prefix + "attn_qkv.weight"), normalized, kChannels, kHidden);
    const auto expected_qkv = read_f32(checkpoint(fixture, position, "linear_attn_qkv_mixed-" + std::to_string(layer)), kChannels);
    require_match("hybrid layer " + std::to_string(layer) + " QKV", compare(qkv, expected_qkv), 1.0e-1F);
    std::array<float, kChannels> saved{};
    std::copy(qkv.begin(), qkv.end(), saved.begin());
    auto raw = conv_output(qkv, history, tensor(model, prefix + "ssm_conv1d.weight"));
    history.push_back(saved);
    const auto expected_raw = read_f32(checkpoint(fixture, position, "conv_output_raw-" + std::to_string(layer)), kChannels);
    require_match("hybrid layer " + std::to_string(layer) + " convolution", compare(raw, expected_raw), 1.0e-2F);
    silu(raw);

    const std::size_t q_size = kKHeads * kState;
    std::vector<float> q(raw.begin(), raw.begin() + q_size);
    std::vector<float> k(raw.begin() + q_size, raw.begin() + 2 * q_size);
    std::vector<float> v(raw.begin() + 2 * q_size, raw.end());
    l2_norm(q, kState);
    l2_norm(k, kState);
    const auto z = gemv(tensor(model, prefix + "attn_gate.weight"), normalized, kVHeads * kState, kHidden);
    const auto beta_raw = f32_gemv(tensor(model, prefix + "ssm_beta.weight"), normalized, kVHeads, kHidden);
    const auto alpha_raw = f32_gemv(tensor(model, prefix + "ssm_alpha.weight"), normalized, kVHeads, kHidden);
    const auto dt = f32_values(tensor(model, prefix + "ssm_dt.bias"), kVHeads);
    const auto a = f32_values(tensor(model, prefix + "ssm_a"), kVHeads);
    std::vector<float> beta(kVHeads), gate(kVHeads);
    for (std::size_t h = 0; h < kVHeads; ++h) {
        beta[h] = 1.0F / (1.0F + std::exp(-beta_raw[h]));
        const float x = alpha_raw[h] + dt[h];
        gate[h] = (x <= 20.0F ? std::log1p(std::exp(x)) : x) * a[h];
    }
    auto recurrent_output = recurrent(q, k, v, gate, beta, state, state_out);
    const auto expected_attention = read_f32(checkpoint(fixture, position, "attn_output-" + std::to_string(layer)), kInner);
    require_match("hybrid layer " + std::to_string(layer) + " recurrent output",
                  compare(recurrent_output, expected_attention), 2.0e-3F);
    const auto ssm_norm = f32_values(tensor(model, prefix + "ssm_norm.weight"), kState);
    std::vector<float> gated(recurrent_output.size());
    rms_rows(recurrent_output, ssm_norm, gated, kState);
    for (std::size_t i = 0; i < gated.size(); ++i) gated[i] *= z[i] / (1.0F + std::exp(-z[i]));
    auto projected = gemv(tensor(model, prefix + "ssm_out.weight"), gated, kHidden, kInner);
    std::vector<float> residual(kHidden);
    for (std::size_t i = 0; i < kHidden; ++i) residual[i] = input[i] + projected[i];
    const auto expected_residual = read_f32(checkpoint(fixture, position, "attn_residual-" + std::to_string(layer)), kHidden);
    require_match("hybrid layer " + std::to_string(layer) + " residual", compare(residual, expected_residual), 1.0e-1F);
    const auto post_weights = f32_values(tensor(model, prefix + "post_attention_norm.weight"), kHidden);
    std::vector<float> post_norm(kHidden);
    rms_rows(residual, post_weights, post_norm, kHidden);
    auto ffn_gate = gemv(tensor(model, prefix + "ffn_gate.weight"), post_norm, kFfnInner, kHidden);
    auto ffn_up = gemv(tensor(model, prefix + "ffn_up.weight"), post_norm, kFfnInner, kHidden);
    for (std::size_t i = 0; i < kFfnInner; ++i) ffn_gate[i] = ffn_gate[i] / (1.0F + std::exp(-ffn_gate[i])) * ffn_up[i];
    auto ffn = gemv(tensor(model, prefix + "ffn_down.weight"), ffn_gate, kHidden, kFfnInner);
    for (std::size_t i = 0; i < kHidden; ++i) ffn[i] += residual[i];
    const auto expected_output = read_f32(checkpoint(fixture, position, "l_out-" + std::to_string(layer)), kHidden);
    require_match("hybrid layer " + std::to_string(layer) + " output", compare(ffn, expected_output), 1.0e-1F);
    return ffn;
}

std::vector<float> full_block(const GgufFile& model, const std::filesystem::path& fixture,
                              std::size_t position, std::span<const float> input,
                              std::vector<std::vector<float>>& keys,
                              std::vector<std::vector<float>>& values) {
    const std::string prefix = "blk." + std::to_string(kFullLayer) + ".";
    const auto norm_weights = f32_values(tensor(model, prefix + "attn_norm.weight"), kHidden);
    std::vector<float> normalized(kHidden);
    rms_rows(input, norm_weights, normalized, kHidden);
    auto qfull = gemv(tensor(model, prefix + "attn_q.weight"), normalized, kQFull, kHidden);
    std::vector<float> query(kAttentionWidth), gate(kAttentionWidth);
    for (std::size_t h = 0; h < kHeads; ++h) {
        std::copy_n(qfull.data() + h * 2 * kHeadDim, kHeadDim, query.data() + h * kHeadDim);
        std::copy_n(qfull.data() + h * 2 * kHeadDim + kHeadDim, kHeadDim, gate.data() + h * kHeadDim);
    }
    rms_rows(query, f32_values(tensor(model, prefix + "attn_q_norm.weight"), kHeadDim), query, kHeadDim);
    auto key = gemv(tensor(model, prefix + "attn_k.weight"), normalized, kKvWidth, kHidden);
    auto value = gemv(tensor(model, prefix + "attn_v.weight"), normalized, kKvWidth, kHidden);
    rms_rows(key, f32_values(tensor(model, prefix + "attn_k_norm.weight"), kHeadDim), key, kHeadDim);
    rope_mrope(query, kHeads, position);
    rope_mrope(key, kKvHeads, position);
    keys.push_back(key);
    values.push_back(value);
    auto attention = cached_attention(query, keys, values, position);
    for (std::size_t i = 0; i < attention.size(); ++i) attention[i] *= 1.0F / (1.0F + std::exp(-gate[i]));
    auto projected = gemv(tensor(model, prefix + "attn_output.weight"), attention, kHidden, kAttentionWidth);
    std::vector<float> residual(kHidden);
    for (std::size_t i = 0; i < kHidden; ++i) residual[i] = input[i] + projected[i];
    const auto expected_residual = read_f32(checkpoint(fixture, position, "attn_residual-3"), kHidden);
    require_match("hybrid full layer residual", compare(residual, expected_residual), 1.0e-1F);
    std::vector<float> post_norm(kHidden);
    rms_rows(residual, f32_values(tensor(model, prefix + "post_attention_norm.weight"), kHidden), post_norm, kHidden);
    auto ffn_gate = gemv(tensor(model, prefix + "ffn_gate.weight"), post_norm, kFfnInner, kHidden);
    auto ffn_up = gemv(tensor(model, prefix + "ffn_up.weight"), post_norm, kFfnInner, kHidden);
    for (std::size_t i = 0; i < kFfnInner; ++i) ffn_gate[i] = ffn_gate[i] / (1.0F + std::exp(-ffn_gate[i])) * ffn_up[i];
    auto output = gemv(tensor(model, prefix + "ffn_down.weight"), ffn_gate, kHidden, kFfnInner);
    for (std::size_t i = 0; i < kHidden; ++i) output[i] += residual[i];
    require_match("hybrid full layer output",
                  compare(output, read_f32(checkpoint(fixture, position, "l_out-3"), kHidden)), 1.0e-1F);
    return output;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: miinfer-m6a5-qwen35-hybrid-block MODEL.gguf FIXTURE_DIR\n";
        return 2;
    }
    try {
        const auto model = GgufFile::open(argv[1]);
        const std::filesystem::path fixture = argv[2];
        std::array<std::vector<std::array<float, kChannels>>, kRecurrentLayers> histories;
        std::array<std::vector<float>, kRecurrentLayers> states;
        std::vector<std::vector<float>> keys;
        std::vector<std::vector<float>> values;
        for (std::size_t position = 0; position <= 8; ++position) {
            auto current = read_f32(checkpoint(fixture, position, "model_input_embed"), kHidden);
            for (std::size_t layer = 0; layer < kRecurrentLayers; ++layer) {
                const auto expected_state = read_f32(checkpoint(fixture, position, "state_predelta-" + std::to_string(layer)), kVHeads * kState * kState);
                if (position == 0) states[layer] = expected_state;
                else require_match("hybrid layer " + std::to_string(layer) + " state input",
                                    compare(states[layer], expected_state), 2.0e-2F);
                const auto state_input = states[layer];
                current = recurrent_block(*model, fixture, layer, position, current, state_input, histories[layer], states[layer]);
            }
            current = full_block(*model, fixture, position, current, keys, values);
            std::cout << "position=" << position << " hybrid_block_output_checked\n";
        }
        std::cout << "M6-A5 four-layer hybrid block reference harness complete\n";
    } catch (const std::exception& error) {
        std::cerr << "M6-A5 failed: " << error.what() << '\n';
        return 1;
    }
}
