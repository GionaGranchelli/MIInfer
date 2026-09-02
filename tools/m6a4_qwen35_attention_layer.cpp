#define MIINFER_M6A3_HELPERS_ONLY
#include "m6a3_qwen35_layer.cpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::size_t kHeads = 24;
constexpr std::size_t kKvHeads = 4;
constexpr std::size_t kHeadDim = 256;
constexpr std::size_t kQFull = kHeads * 2 * kHeadDim;
constexpr std::size_t kAttentionWidth = kHeads * kHeadDim;
constexpr std::size_t kKvWidth = kKvHeads * kHeadDim;

void rope_mrope(std::span<float> values, std::size_t heads, std::size_t position) {
    if (values.size() != heads * kHeadDim) throw std::runtime_error("invalid RoPE shape");
    constexpr int sections[] = {11, 11, 10, 0};
    constexpr int section_count = sections[0] + sections[1] + sections[2] + sections[3];
    constexpr float base = 10000000.0F;
    const float theta_scale = std::pow(base, -2.0F / 64.0F);
    for (std::size_t head = 0; head < heads; ++head) {
        float theta_t = static_cast<float>(position);
        float theta_h = static_cast<float>(position);
        float theta_w = static_cast<float>(position);
        float theta_e = 0.0F;
        auto row = values.subspan(head * kHeadDim, kHeadDim);
        for (std::size_t pair = 0; pair < 32; ++pair) {
            float theta = theta_t;
            const int sector = static_cast<int>(pair % section_count);
            if (sector % 3 == 1 && sector < 3 * sections[1]) theta = theta_h;
            else if (sector % 3 == 2 && sector < 3 * sections[2]) theta = theta_w;
            else if (sector % 3 == 0 && sector < 3 * sections[0]) theta = theta_t;
            else theta = theta_e;
            const float angle = theta;
            const float c = std::cos(angle);
            const float s = std::sin(angle);
            const float x0 = row[pair];
            const float x1 = row[pair + 32];
            row[pair] = x0 * c - x1 * s;
            row[pair + 32] = x0 * s + x1 * c;
            theta_t *= theta_scale;
            theta_h *= theta_scale;
            theta_w *= theta_scale;
            theta_e *= theta_scale;
        }
    }
}

std::vector<float> cached_attention(std::span<const float> query,
                                    const std::vector<std::vector<float>>& keys,
                                    const std::vector<std::vector<float>>& values,
                                    std::size_t position) {
    if (query.size() != kAttentionWidth || keys.size() != position + 1 || values.size() != position + 1) {
        throw std::runtime_error("invalid cached attention history");
    }
    std::vector<float> output(kAttentionWidth);
    std::vector<float> scores(position + 1);
    for (std::size_t head = 0; head < kHeads; ++head) {
        const std::size_t kv_head = head / (kHeads / kKvHeads);
        const auto q = query.subspan(head * kHeadDim, kHeadDim);
        float maximum = -INFINITY;
        for (std::size_t p = 0; p <= position; ++p) {
            const auto key = std::span<const float>(keys[p]).subspan(kv_head * kHeadDim, kHeadDim);
            float dot = 0.0F;
            for (std::size_t i = 0; i < kHeadDim; ++i) dot += q[i] * key[i];
            scores[p] = dot / std::sqrt(static_cast<float>(kHeadDim));
            maximum = std::max(maximum, scores[p]);
        }
        float denominator = 0.0F;
        for (std::size_t p = 0; p <= position; ++p) {
            scores[p] = std::exp(scores[p] - maximum);
            denominator += scores[p];
        }
        for (std::size_t i = 0; i < kHeadDim; ++i) {
            float sum = 0.0F;
            for (std::size_t p = 0; p <= position; ++p) {
                sum += scores[p] * values[p][kv_head * kHeadDim + i];
            }
            output[head * kHeadDim + i] = sum / denominator;
        }
    }
    return output;
}

void report(std::string_view label, const Metrics& error, float tolerance) {
    require_match(label, error, tolerance);
    std::cout << label << " max_abs=" << error.max_abs << " rmse=" << error.rmse
              << " index=" << error.index << '\n';
}

void validate_position(const GgufFile& model, const std::filesystem::path& fixture,
                       std::size_t position, const std::vector<std::vector<float>>& key_history,
                       const std::vector<std::vector<float>>& value_history) {
    const auto input = read_f32(checkpoint(fixture, position, "l_out-2"), kHidden);
    const auto norm_weights = f32_values(tensor(model, "blk.3.attn_norm.weight"), kHidden);
    std::vector<float> normalized(kHidden);
    rms_rows(input, norm_weights, normalized, kHidden);
    report("position=" + std::to_string(position) + " attention_norm",
           compare(normalized, read_f32(checkpoint(fixture, position, "attn_norm-3"), kHidden)), 1.0e-6F);

    const auto qfull = gemv(tensor(model, "blk.3.attn_q.weight"), normalized, kQFull, kHidden);
    report("position=" + std::to_string(position) + " Qcur_full",
           compare(qfull, read_f32(checkpoint(fixture, position, "Qcur_full-3"), kQFull)), 1.0e-3F);

    std::vector<float> query(kAttentionWidth);
    std::vector<float> gate(kAttentionWidth);
    for (std::size_t head = 0; head < kHeads; ++head) {
        std::copy_n(qfull.data() + head * 2 * kHeadDim, kHeadDim, query.data() + head * kHeadDim);
        std::copy_n(qfull.data() + head * 2 * kHeadDim + kHeadDim, kHeadDim, gate.data() + head * kHeadDim);
    }
    const auto q_norm_weights = f32_values(tensor(model, "blk.3.attn_q_norm.weight"), kHeadDim);
    rms_rows(query, q_norm_weights, query, kHeadDim);
    report("position=" + std::to_string(position) + " Qcur_normed",
           compare(query, read_f32(checkpoint(fixture, position, "Qcur_normed-3"), kAttentionWidth)), 1.0e-3F);

    auto key = gemv(tensor(model, "blk.3.attn_k.weight"), normalized, kKvWidth, kHidden);
    auto value = gemv(tensor(model, "blk.3.attn_v.weight"), normalized, kKvWidth, kHidden);
    report("position=" + std::to_string(position) + " Vcur_projection",
           compare(value, read_f32(checkpoint(fixture, position, "Vcur-3"), kKvWidth)), 1.0e-3F);
    const auto k_norm_weights = f32_values(tensor(model, "blk.3.attn_k_norm.weight"), kHeadDim);
    rms_rows(key, k_norm_weights, key, kHeadDim);
    report("position=" + std::to_string(position) + " Kcur_projection_normed",
           compare(key, read_f32(checkpoint(fixture, position, "Kcur_normed-3"), kKvWidth)), 1.0e-3F);

    rope_mrope(query, kHeads, position);
    rope_mrope(key, kKvHeads, position);
    report("position=" + std::to_string(position) + " Qcur_rope",
           compare(query, read_f32(checkpoint(fixture, position, "Qcur-3"), kAttentionWidth)), 1.0e-3F);
    report("position=" + std::to_string(position) + " Kcur_rope",
           compare(key, read_f32(checkpoint(fixture, position, "Kcur-3"), kKvWidth)), 1.0e-3F);

    std::vector<float> gate_sigmoid(gate.size());
    for (std::size_t i = 0; i < gate.size(); ++i) gate_sigmoid[i] = 1.0F / (1.0F + std::exp(-gate[i]));
    const auto expected_gate = read_f32(checkpoint(fixture, position, "gate_sigmoid-3"), kAttentionWidth);
    report("position=" + std::to_string(position) + " gate_sigmoid",
           compare(gate_sigmoid, expected_gate), 1.0e-3F);

    auto all_keys = key_history;
    auto all_values = value_history;
    all_keys.push_back(key);
    all_values.push_back(value);
    auto attention = cached_attention(query, all_keys, all_values, position);
    report("position=" + std::to_string(position) + " attention",
           compare(attention, read_f32(checkpoint(fixture, position, "attn_pregate-3"), kAttentionWidth)), 4.0e-3F);
    for (std::size_t i = 0; i < attention.size(); ++i) attention[i] *= gate_sigmoid[i];
    report("position=" + std::to_string(position) + " gated_attention",
           compare(attention, read_f32(checkpoint(fixture, position, "attn_gated-3"), kAttentionWidth)), 2.0e-3F);

    auto projected = gemv(tensor(model, "blk.3.attn_output.weight"), attention, kHidden, kAttentionWidth);
    report("position=" + std::to_string(position) + " attention_output",
           compare(projected, read_f32(checkpoint(fixture, position, "attn_output-3"), kHidden)), 1.0e-2F);
    std::vector<float> residual(kHidden);
    for (std::size_t i = 0; i < kHidden; ++i) residual[i] = input[i] + projected[i];
    report("position=" + std::to_string(position) + " attention_residual",
           compare(residual, read_f32(checkpoint(fixture, position, "attn_residual-3"), kHidden)), 1.0e-2F);
    const auto post_weights = f32_values(tensor(model, "blk.3.post_attention_norm.weight"), kHidden);
    std::vector<float> post_norm(kHidden);
    rms_rows(residual, post_weights, post_norm, kHidden);
    report("position=" + std::to_string(position) + " post_attention_norm",
           compare(post_norm, read_f32(checkpoint(fixture, position, "attn_post_norm-3"), kHidden)), 1.0e-2F);

    auto ffn_gate = gemv(tensor(model, "blk.3.ffn_gate.weight"), post_norm, kFfnInner, kHidden);
    auto ffn_up = gemv(tensor(model, "blk.3.ffn_up.weight"), post_norm, kFfnInner, kHidden);
    for (std::size_t i = 0; i < kFfnInner; ++i) ffn_gate[i] = ffn_gate[i] / (1.0F + std::exp(-ffn_gate[i])) * ffn_up[i];
    auto ffn = gemv(tensor(model, "blk.3.ffn_down.weight"), ffn_gate, kHidden, kFfnInner);
    report("position=" + std::to_string(position) + " FFN",
           compare(ffn, read_f32(checkpoint(fixture, position, "ffn_out-3"), kHidden)), 1.0e-2F);
    for (std::size_t i = 0; i < kHidden; ++i) ffn[i] += residual[i];
    report("position=" + std::to_string(position) + " layer_output",
           compare(ffn, read_f32(checkpoint(fixture, position, "l_out-3"), kHidden)), 1.0e-2F);
}

}  // namespace

#ifndef MIINFER_M6A4_HELPERS_ONLY
int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: miinfer-m6a4-qwen35-attention-layer MODEL.gguf FIXTURE_DIR\n";
        return 2;
    }
    try {
        const auto model = GgufFile::open(argv[1]);
        const std::filesystem::path fixture = argv[2];
        std::vector<std::vector<float>> keys;
        std::vector<std::vector<float>> values;
        for (std::size_t position = 0; position <= 8; ++position) {
            validate_position(*model, fixture, position, keys, values);
            keys.push_back(read_f32(checkpoint(fixture, position, "Kcur-3"), kKvWidth));
            values.push_back(read_f32(checkpoint(fixture, position, "Vcur-3"), kKvWidth));
        }
        std::cout << "M6-A4 full-attention layer reference harness complete\n";
    } catch (const std::exception& error) {
        std::cerr << "M6-A4 failed: " << error.what() << '\n';
        return 1;
    }
}
#endif
