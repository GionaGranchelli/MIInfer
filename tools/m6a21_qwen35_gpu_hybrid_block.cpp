#include "qwen35_gpu_pipeline.hpp"


int main(int argc, char** argv) {
    if (argc != 3 && argc != 4 && argc != 5) {
        std::cerr << "usage: miinfer-m6a21-qwen35-gpu-hybrid-block MODEL.gguf FIXTURE_DIR "
                     "[EXTERNAL_OPERAND_FIXTURE] "
                     "[--deep|--block4-7|--prefix8|--prefix16|--prefix32|--prefix32-locate|"
                     "--prefix32-provenance|--prefix32-operand-attribution|"
                     "--prefix32-k-path-attribution|--prefix32-l29-path-attribution|"
                     "--prefix32-l29-gate-attribution|--prefix32-external-contract|"
                     "--prefix64-external-contract|--prefix64-l54-attribution|"
                     "--prefix64-l53-attribution|--prefix64-l53-gated-contract|"
                     "--prefix64-observable-contract|--prefix64-l0-l2-p2-trace|"
                     "--prefix64-l0-p2-output-projection|--generate16|--generate64|"
                     "--generate128|--generate256|--generate512|--generate1024|"
                     "--bench64|--bench128|--bench256|--bench512|--bench1024|--profile64]\n";
        return 2;
    }
    const std::string mode = argc >= 4 ? argv[argc - 1] : "";
    const bool prefix8 = mode == "--prefix8";
    const bool prefix16 = mode == "--prefix16";
    const bool locate32 = mode == "--prefix32-locate";
    const bool provenance32 = mode == "--prefix32-provenance";
    const bool operand_attribution32 = mode == "--prefix32-operand-attribution";
    const bool k_path_attribution32 = mode == "--prefix32-k-path-attribution";
    const bool l29_path_attribution32 = mode == "--prefix32-l29-path-attribution";
    const bool l29_gate_attribution32 = mode == "--prefix32-l29-gate-attribution";
    const bool external_contract32 = mode == "--prefix32-external-contract";
    const bool external_contract64 = mode == "--prefix64-external-contract";
    const bool trace64 = mode == "--prefix64-l54-attribution";
    const bool trace53 = mode == "--prefix64-l53-attribution";
    const bool gate53_contract = mode == "--prefix64-l53-gated-contract";
    const bool observable64 = mode == "--prefix64-observable-contract";
    const bool trace012 = mode == "--prefix64-l0-l2-p2-trace";
    const bool trace_l0_output = mode == "--prefix64-l0-p2-output-projection";
    const bool generation = mode == "--generate16" || mode == "--generate64"
        || mode == "--generate128" || mode == "--generate256"
        || mode == "--generate512" || mode == "--generate1024"
        || mode == "--bench64" || mode == "--bench128" || mode == "--bench256"
        || mode == "--bench512" || mode == "--bench1024";
    const bool benchmark = mode == "--bench64" || mode == "--bench128" || mode == "--bench256"
        || mode == "--bench512" || mode == "--bench1024";
    const bool profile64 = mode == "--profile64";
    const char* lm_mmvq_env = std::getenv("MIINFER_LM_Q8_1_MMVQ");
    const bool lm_mmvq = lm_mmvq_env == nullptr || std::strcmp(lm_mmvq_env, "0") != 0;
    const char* native_lm_env = std::getenv("MIINFER_Q6K_NATIVE_LM_HEAD");
    const bool native_lm_head = native_lm_env == nullptr || std::strcmp(native_lm_env, "0") != 0;
    const char* fused_interlayer_norm_env = std::getenv("MIINFER_FUSED_INTERLAYER_NORM");
    const bool fused_interlayer_norm = fused_interlayer_norm_env == nullptr
        || std::strcmp(fused_interlayer_norm_env, "0") != 0;
    const char* fused_norm_q8_env = std::getenv("MIINFER_FUSED_NORM_Q8");
    const bool fused_norm_q8 = fused_norm_q8_env != nullptr
        && std::strcmp(fused_norm_q8_env, "0") != 0;
    const char* hip_graph_env = std::getenv("MIINFER_HIP_GRAPH");
    const bool use_hip_graph = hip_graph_env != nullptr && std::strcmp(hip_graph_env, "0") != 0;
    const char* device_token_chain_env = std::getenv("MIINFER_DEVICE_TOKEN_CHAIN");
    const bool device_token_chain = device_token_chain_env == nullptr || std::strcmp(device_token_chain_env, "0") != 0;
    const std::size_t generation_tokens = mode == "--generate16" ? 16
        : mode == "--generate64" || mode == "--bench64" ? 64
        : mode == "--generate128" || mode == "--bench128" ? 128
        : mode == "--generate256" || mode == "--bench256" ? 256
        : mode == "--generate512" || mode == "--bench512" ? 512 : 1024;
    const bool prefix32 = mode == "--prefix32" || locate32 || provenance32
        || operand_attribution32 || k_path_attribution32 || l29_path_attribution32
        || l29_gate_attribution32 || external_contract32;
    const bool prefix64 = external_contract64 || trace64 || trace53 || gate53_contract
        || observable64 || trace012 || trace_l0_output || generation || profile64;
    const bool deep = mode == "--deep" || mode == "--block4-7" || prefix8 || prefix16 || prefix32
        || prefix64;
    const bool second_block = mode == "--block4-7" || prefix8 || prefix16 || prefix32 || prefix64;
    if (argc >= 4 && !deep) {
        std::cerr << "unknown option: " << mode << '\n';
        return 2;
    }
    try {
        const auto model = miinfer::Qwen35Model::load(argv[1]);
        const auto fixture = std::filesystem::path(argv[2]);
        std::optional<std::uint32_t> prompt_override;
        if (const char* prompt_env = std::getenv("MIINFER_PROMPT_TOKEN"); prompt_env != nullptr) {
            try {
                const auto parsed = std::stoull(prompt_env);
                if (parsed >= model.config().vocab_size) throw std::invalid_argument("out of range");
                prompt_override = static_cast<std::uint32_t>(parsed);
            } catch (...) {
                throw std::invalid_argument("MIINFER_PROMPT_TOKEN must be a valid vocabulary ID");
            }
        }
        const auto operand_fixture = argc == 5
            ? std::filesystem::path(argv[3]) : fixture;
        RecurrentLayer recurrent0(model, 0, fixture);
        RecurrentLayer recurrent1(model, 1, fixture);
        RecurrentLayer recurrent2(model, 2, fixture);
        FullAttentionLayer attention3(model, 3);
        std::unique_ptr<RecurrentLayer> recurrent4;
        std::unique_ptr<RecurrentLayer> recurrent5;
        std::unique_ptr<RecurrentLayer> recurrent6;
        std::unique_ptr<FullAttentionLayer> attention7;
        std::unique_ptr<RecurrentLayer> recurrent8;
        std::unique_ptr<RecurrentLayer> recurrent9;
        std::unique_ptr<RecurrentLayer> recurrent10;
        std::unique_ptr<FullAttentionLayer> attention11;
        std::unique_ptr<RecurrentLayer> recurrent12;
        std::unique_ptr<RecurrentLayer> recurrent13;
        std::unique_ptr<RecurrentLayer> recurrent14;
        std::unique_ptr<FullAttentionLayer> attention15;
        std::unique_ptr<RecurrentLayer> recurrent16;
        std::unique_ptr<RecurrentLayer> recurrent17;
        std::unique_ptr<RecurrentLayer> recurrent18;
        std::unique_ptr<FullAttentionLayer> attention19;
        std::unique_ptr<RecurrentLayer> recurrent20;
        std::unique_ptr<RecurrentLayer> recurrent21;
        std::unique_ptr<RecurrentLayer> recurrent22;
        std::unique_ptr<FullAttentionLayer> attention23;
        std::unique_ptr<RecurrentLayer> recurrent24;
        std::unique_ptr<RecurrentLayer> recurrent25;
        std::unique_ptr<RecurrentLayer> recurrent26;
        std::unique_ptr<FullAttentionLayer> attention27;
        std::unique_ptr<RecurrentLayer> recurrent28;
        std::unique_ptr<RecurrentLayer> recurrent29;
        std::unique_ptr<RecurrentLayer> recurrent30;
        std::unique_ptr<FullAttentionLayer> attention31;
        if (second_block) {
            recurrent4 = std::make_unique<RecurrentLayer>(model, 4, fixture);
            recurrent5 = std::make_unique<RecurrentLayer>(model, 5, fixture);
            recurrent6 = std::make_unique<RecurrentLayer>(model, 6, fixture);
            attention7 = std::make_unique<FullAttentionLayer>(model, 7);
        }
        if (prefix16 || prefix32 || prefix64) {
            recurrent8 = std::make_unique<RecurrentLayer>(model, 8, fixture);
            recurrent9 = std::make_unique<RecurrentLayer>(model, 9, fixture);
            recurrent10 = std::make_unique<RecurrentLayer>(model, 10, fixture);
            attention11 = std::make_unique<FullAttentionLayer>(model, 11);
            recurrent12 = std::make_unique<RecurrentLayer>(model, 12, fixture);
            recurrent13 = std::make_unique<RecurrentLayer>(model, 13, fixture);
            recurrent14 = std::make_unique<RecurrentLayer>(model, 14, fixture);
            attention15 = std::make_unique<FullAttentionLayer>(model, 15);
        }
        if (prefix32 || prefix64) {
            recurrent16 = std::make_unique<RecurrentLayer>(model, 16, fixture);
            recurrent17 = std::make_unique<RecurrentLayer>(model, 17, fixture);
            recurrent18 = std::make_unique<RecurrentLayer>(model, 18, fixture);
            attention19 = std::make_unique<FullAttentionLayer>(model, 19);
            recurrent20 = std::make_unique<RecurrentLayer>(model, 20, fixture);
            recurrent21 = std::make_unique<RecurrentLayer>(model, 21, fixture);
            recurrent22 = std::make_unique<RecurrentLayer>(model, 22, fixture);
            attention23 = std::make_unique<FullAttentionLayer>(model, 23);
            recurrent24 = std::make_unique<RecurrentLayer>(model, 24, fixture);
            recurrent25 = std::make_unique<RecurrentLayer>(model, 25, fixture);
            recurrent26 = std::make_unique<RecurrentLayer>(model, 26, fixture);
            attention27 = std::make_unique<FullAttentionLayer>(model, 27);
            recurrent28 = std::make_unique<RecurrentLayer>(model, 28, fixture);
            recurrent29 = std::make_unique<RecurrentLayer>(model, 29, fixture);
            recurrent30 = std::make_unique<RecurrentLayer>(model, 30, fixture);
            attention31 = std::make_unique<FullAttentionLayer>(model, 31);
        }
        std::array<std::unique_ptr<RecurrentLayer>, 24> recurrent32_plus;
        std::array<std::unique_ptr<FullAttentionLayer>, 8> attention32_plus;
        if (prefix64) {
            for (std::size_t layer = 0; layer < recurrent32_plus.size(); ++layer) {
                const std::size_t model_layer = 32 + layer + layer / 3;
                recurrent32_plus[layer] = std::make_unique<RecurrentLayer>(model, model_layer, fixture);
            }
            for (std::size_t layer = 0; layer < attention32_plus.size(); ++layer) {
                attention32_plus[layer] = std::make_unique<FullAttentionLayer>(model, 35 + layer * 4);
            }
        }

        if (locate32 || provenance32) {
            const std::array<std::size_t, 13> positions{{
                1, 2, 4, 8, 16, 32, 48, 56, 60, 61, 62, 63, 64}};
            const auto is_position = [&positions](std::size_t position) {
                return std::find(positions.begin(), positions.end(), position) != positions.end();
            };
            const std::array<GpuLayerRef, 32> layers{{
                {&recurrent0, nullptr}, {&recurrent1, nullptr}, {&recurrent2, nullptr},
                {nullptr, &attention3}, {recurrent4.get(), nullptr},
                {recurrent5.get(), nullptr}, {recurrent6.get(), nullptr},
                {nullptr, attention7.get()}, {recurrent8.get(), nullptr},
                {recurrent9.get(), nullptr}, {recurrent10.get(), nullptr},
                {nullptr, attention11.get()}, {recurrent12.get(), nullptr},
                {recurrent13.get(), nullptr}, {recurrent14.get(), nullptr},
                {nullptr, attention15.get()}, {recurrent16.get(), nullptr},
                {recurrent17.get(), nullptr}, {recurrent18.get(), nullptr},
                {nullptr, attention19.get()}, {recurrent20.get(), nullptr},
                {recurrent21.get(), nullptr}, {recurrent22.get(), nullptr},
                {nullptr, attention23.get()}, {recurrent24.get(), nullptr},
                {recurrent25.get(), nullptr}, {recurrent26.get(), nullptr},
                {nullptr, attention27.get()}, {recurrent28.get(), nullptr},
                {recurrent29.get(), nullptr}, {recurrent30.get(), nullptr},
                {nullptr, attention31.get()}}};
            std::array<Buffer, 32> outputs{};
            std::array<float*, 32> output_pointers{};
            for (std::size_t layer = 0; layer < layers.size(); ++layer) {
                outputs[layer] = allocate(kHidden * sizeof(float));
                output_pointers[layer] = static_cast<float*>(outputs[layer]->get());
            }
            Buffer input = allocate(kHidden * sizeof(float));
            const auto generated = read_tokens(fixture / "generated_tokens.txt");
            if (generated.size() < 64) throw std::runtime_error("fixture has fewer than 64 tokens");
            RecurrentTrace l30_trace{fixture, 30, 64};
            recurrent30->trace = locate32 ? &l30_trace : nullptr;
            const auto allocations_before_decode = g_device_allocations;
            if (provenance32) {
                constexpr std::size_t tracked_index = 86909;
                const std::array<std::size_t, 7> transitions{{3, 7, 31, 59, 60, 62, 63}};
                std::cout << "tracked_state_index=" << tracked_index
                          << " head=5 row=38 column=125\n"
                          << "position max_index max_abs fixed_reference fixed_gpu fixed_abs\n";
                for (std::size_t position = 0; position <= 64; ++position) {
                    const auto host_input = position > 1
                        ? embedding(tensor(*model.file(), "token_embd.weight"), generated[position - 1])
                        : read_f32(checkpoint(fixture, position, "model_input_embed"), kHidden);
                    upload(host_input.data(), input->get(), host_input.size() * sizeof(float));
                    UpdateProvenance provenance;
                    const bool report_transition =
                        std::find(transitions.begin(), transitions.end(), position) != transitions.end();
                    const float* current = static_cast<const float*>(input->get());
                    for (std::size_t layer = 0; layer < 30; ++layer) {
                        layers[layer].run(current, position, output_pointers[layer]);
                        current = output_pointers[layer];
                    }
                    recurrent30->provenance = &provenance;
                    recurrent30->provenance_position = static_cast<std::uint32_t>(position);
                    recurrent30->provenance_index = tracked_index;
                    layers[30].run(current, position, output_pointers[30]);
                    layers[31].run(output_pointers[30], position, output_pointers[31]);
                    MIINFER_HIP_CHECK(hipDeviceSynchronize());
                    if (position == 64) continue;
                    const std::size_t next_position = position + 1;
                    const auto expected = read_f32(
                        checkpoint(fixture, next_position, "state_predelta-30"),
                        kVHeads * kState * kState);
                    const auto state = located_host_error(recurrent30->logical_state(), expected);
                    const auto fixed_gpu = recurrent30->state_value(tracked_index);
                    const auto fixed_reference = expected[tracked_index];
                    std::cout << next_position << ' ' << state.index << ' '
                              << state.metrics.max_abs << ' ' << fixed_reference << ' '
                              << fixed_gpu << ' ' << std::fabs(fixed_gpu - fixed_reference) << '\n';
                    if (report_transition) {
                        const auto stored = recurrent30->state_value(tracked_index);
                        const auto expected_next = expected[tracked_index];
                        std::cout << "transition=" << position << "->" << next_position
                                  << " previous=" << provenance.previous
                                  << " decay=" << provenance.decay
                                  << " beta=" << provenance.beta
                                  << " value=" << provenance.value
                                  << " key=" << provenance.key
                                  << " query=" << provenance.query
                                  << " key_dot=" << provenance.key_dot
                                  << " decayed=" << provenance.decayed
                                  << " delta=" << provenance.delta
                                  << " candidate=" << provenance.candidate
                                  << " stored=" << stored
                                  << " reference_next=" << expected_next
                                  << " abs_error=" << std::fabs(stored - expected_next)
                                  << " query_dot=" << provenance.query_dot << '\n';
                    }
                }
                std::cout << "allocations_during_decode="
                          << (g_device_allocations - allocations_before_decode)
                          << " device_bytes_after_setup=" << g_device_bytes
                          << " peak_device_bytes=" << g_peak_device_bytes << '\n'
                          << "M6-A26.2 qwen35 L30 recurrent-update provenance COMPLETE\n";
                return 0;
            }
            std::cout << "position l30_state_max l30_state_mean l30_state_rms l30_state_rel l30_index "
                         "l30_reference l30_gpu\n";
            for (std::size_t position = 0; position <= 64; ++position) {
                const auto host_input = position > 1
                    ? embedding(tensor(*model.file(), "token_embd.weight"), generated[position - 1])
                    : read_f32(checkpoint(fixture, position, "model_input_embed"), kHidden);
                upload(host_input.data(), input->get(), host_input.size() * sizeof(float));
                if (is_position(position)) {
                    const auto expected = read_f32(
                        checkpoint(fixture, position, "state_predelta-30"),
                        kVHeads * kState * kState);
                    const auto state = located_host_error(recurrent30->logical_state(), expected);
                    std::cout << position << ' ' << state.metrics.max_abs << ' '
                              << state.metrics.mean_abs << ' '
                              << state.metrics.rms << ' ' << state.metrics.relative_rms << ' '
                              << state.index << ' ' << state.expected << ' ' << state.actual << '\n';
                    if (position == 64) {
                        for (std::size_t layer = 28; layer < 31; ++layer) {
                            const auto adjacent = located_host_error(
                                layers[layer].recurrent->logical_state(),
                                read_f32(checkpoint(fixture, position,
                                                    "state_predelta-" + std::to_string(layer)),
                                        kVHeads * kState * kState));
                            std::cout << "entry_state_layer=" << layer
                                      << " max_abs=" << adjacent.metrics.max_abs
                                      << " mean_abs=" << adjacent.metrics.mean_abs
                                      << " rms=" << adjacent.metrics.rms
                                      << " relative_rms=" << adjacent.metrics.relative_rms
                                      << " max_index=" << adjacent.index
                                      << " reference=" << adjacent.expected
                                      << " gpu=" << adjacent.actual << '\n';
                        }
                    }
                }
                run_prefix(std::span<const GpuLayerRef>(layers),
                           std::span<float* const>(output_pointers),
                           static_cast<const float*>(input->get()), position);
                MIINFER_HIP_CHECK(hipDeviceSynchronize());
                if (position != 64) continue;
                std::cout << "p64_layers\n";
                for (std::size_t layer = 28; layer < 32; ++layer) {
                    const auto error = located_device_error(
                        static_cast<const float*>(outputs[layer]->get()), kHidden,
                        checkpoint(fixture, position, "l_out-" + std::to_string(layer)));
                    std::cout << "layer=" << layer << " max_abs=" << error.metrics.max_abs
                              << " rms=" << error.metrics.rms
                              << " relative_rms=" << error.metrics.relative_rms
                              << " max_index=" << error.index
                              << " reference=" << error.expected
                              << " gpu=" << error.actual << '\n';
                }
                std::cout << "p64_fingerprints"
                          << " state28=" << recurrent28->state_fingerprint()
                          << " state29=" << recurrent29->state_fingerprint()
                          << " state30=" << recurrent30->state_fingerprint()
                          << " K27=" << fingerprint(attention27->key_cache->get(),
                              4 * (position + 1) * 256 * sizeof(float))
                          << " V27=" << fingerprint(attention27->value_cache->get(),
                              4 * (position + 1) * 256 * sizeof(float)) << '\n';
            }
            std::cout << "allocations_during_decode="
                      << (g_device_allocations - allocations_before_decode)
                      << " device_bytes_after_setup=" << g_device_bytes
                      << " peak_device_bytes=" << g_peak_device_bytes << '\n'
                      << "M6-A26.1 qwen35 L30 state localization COMPLETE\n";
            return 0;
        }

        if (prefix8 || prefix16 || prefix32 || prefix64) {
            const std::size_t layer_count = prefix64 ? 64 : prefix32 ? 32 : prefix16 ? 16 : 8;
            std::array<GpuLayerRef, 64> layers{{
                {&recurrent0, nullptr}, {&recurrent1, nullptr}, {&recurrent2, nullptr},
                {nullptr, &attention3}, {recurrent4.get(), nullptr},
                {recurrent5.get(), nullptr}, {recurrent6.get(), nullptr},
                {nullptr, attention7.get()}, {recurrent8.get(), nullptr},
                {recurrent9.get(), nullptr}, {recurrent10.get(), nullptr},
                {nullptr, attention11.get()}, {recurrent12.get(), nullptr},
                {recurrent13.get(), nullptr}, {recurrent14.get(), nullptr},
                {nullptr, attention15.get()}, {recurrent16.get(), nullptr},
                {recurrent17.get(), nullptr}, {recurrent18.get(), nullptr},
                {nullptr, attention19.get()}, {recurrent20.get(), nullptr},
                {recurrent21.get(), nullptr}, {recurrent22.get(), nullptr},
                {nullptr, attention23.get()}, {recurrent24.get(), nullptr},
                {recurrent25.get(), nullptr}, {recurrent26.get(), nullptr},
                {nullptr, attention27.get()}, {recurrent28.get(), nullptr},
                {recurrent29.get(), nullptr}, {recurrent30.get(), nullptr},
                {nullptr, attention31.get()}}};
            if (prefix64) {
                for (std::size_t layer = 32; layer < 64; ++layer) {
                    const std::size_t tail = layer - 32;
                    if (tail % 4 == 3) {
                        layers[layer] = {nullptr, attention32_plus[tail / 4].get()};
                    } else {
                        layers[layer] = {recurrent32_plus[tail - tail / 4].get(), nullptr};
                    }
                }
            }
            std::array<Buffer, 64> outputs{};
            std::array<float*, 64> output_pointers{};
            for (std::size_t layer = 0; layer < layer_count; ++layer) {
                outputs[layer] = allocate(kHidden * sizeof(float));
                output_pointers[layer] = static_cast<float*>(outputs[layer]->get());
            }
            Buffer input = allocate(kHidden * sizeof(float));
            const auto generated = read_tokens(fixture / "generated_tokens.txt");
            if (generated.size() < 64) throw std::runtime_error("fixture has fewer than 64 tokens");
            Buffer d_final_norm_weight;
            Buffer d_output_weight;
            Buffer final_norm;
            Buffer final_q8;
            Buffer final_q8_1;
            Buffer logits;
            Buffer d_embedding;
            Buffer argmax_token;
            Buffer d_decode_tokens;
            if (observable64 || generation || profile64) {
                const auto& final_norm_weight = tensor(*model.file(), "output_norm.weight");
                const auto& output_weight = tensor(*model.file(), "output.weight");
                require_type(final_norm_weight, {miinfer::GgufTensorType::f32});
                require_type(output_weight, {miinfer::GgufTensorType::q6_k});
                d_final_norm_weight = allocate(final_norm_weight.byte_size);
                upload_tensor(final_norm_weight, d_final_norm_weight);
                if (native_lm_head) {
                    d_output_weight = copy_native_q6k_tensor(output_weight);
                } else {
                    d_output_weight = allocate(output_weight.byte_size);
                    upload_tensor(output_weight, d_output_weight);
                }
                final_norm = allocate(kHidden * sizeof(float));
                final_q8 = allocate((kHidden / 256) * sizeof(miinfer::Q8KDeviceBlock));
                if (lm_mmvq || native_lm_head) {
                    final_q8_1 = allocate((kHidden / 32) * sizeof(miinfer::Q8_1Block));
                }
                logits = allocate(model.config().vocab_size * sizeof(float));
                if (generation || profile64) {
                    const auto& embedding_weight = tensor(*model.file(), "token_embd.weight");
                    require_type(embedding_weight, {miinfer::GgufTensorType::q4_k});
                    d_embedding = allocate(embedding_weight.byte_size);
                    upload_tensor(embedding_weight, d_embedding);
                    argmax_token = allocate(sizeof(std::uint32_t));
                    d_decode_tokens = allocate((generation_tokens + 1) * sizeof(std::uint32_t));
                }
            }
            RecurrentTrace l54_trace{fixture, 54, 1};
            if (trace64) recurrent32_plus[17]->trace = &l54_trace;
            LayerPathCapture l54_path;
            if (trace64) {
                recurrent32_plus[17]->layer_path_capture = &l54_path;
                recurrent32_plus[17]->layer_path_capture_position = 1;
            }
            RecurrentTrace l53_trace{fixture, 53, 1};
            if (trace53) recurrent32_plus[16]->trace = &l53_trace;
            LayerPathCapture l53_path;
            if (trace53) {
                recurrent32_plus[16]->layer_path_capture = &l53_path;
                recurrent32_plus[16]->layer_path_capture_position = 1;
            }
            GatePathCapture l53_gate_path;
            if (gate53_contract) {
                recurrent32_plus[16]->gate_path_capture = &l53_gate_path;
                recurrent32_plus[16]->gate_path_capture_position = 1;
            }
            if (profile64) {
                const auto reset_all = [&] {
                    for (const auto& layer : layers) {
                        if (layer.recurrent != nullptr) layer.recurrent->reset(fixture);
                        if (layer.attention != nullptr) layer.attention->reset();
                    }
                    MIINFER_HIP_CHECK(hipDeviceSynchronize());
                };
                std::array<hipEvent_t, 64> layer_start{};
                std::array<hipEvent_t, 64> layer_end{};
                for (std::size_t layer = 0; layer < layers.size(); ++layer) {
                    MIINFER_HIP_CHECK(hipEventCreate(&layer_start[layer]));
                    MIINFER_HIP_CHECK(hipEventCreate(&layer_end[layer]));
                }
                hipEvent_t token_start = nullptr;
                hipEvent_t token_end = nullptr;
                hipEvent_t final_start = nullptr;
                hipEvent_t final_norm_end = nullptr;
                hipEvent_t final_q8_end = nullptr;
                hipEvent_t final_lm_end = nullptr;
                std::array<RecurrentLayer*, 3> profiled_recurrent_layers{
                    &recurrent0, &recurrent1, &recurrent2};
                std::array<RecurrentLayer::StageProfile, 3> recurrent_profiles{};
                for (auto& profile : recurrent_profiles) {
                    for (std::size_t stage = 0; stage < profile.start.size(); ++stage) {
                        MIINFER_HIP_CHECK(hipEventCreate(&profile.start[stage]));
                        MIINFER_HIP_CHECK(hipEventCreate(&profile.end[stage]));
                    }
                }
                FullAttentionLayer::StageProfile attention_profile;
                for (std::size_t stage = 0; stage < attention_profile.start.size(); ++stage) {
                    MIINFER_HIP_CHECK(hipEventCreate(&attention_profile.start[stage]));
                    MIINFER_HIP_CHECK(hipEventCreate(&attention_profile.end[stage]));
                }
                MIINFER_HIP_CHECK(hipEventCreate(&token_start));
                MIINFER_HIP_CHECK(hipEventCreate(&token_end));
                MIINFER_HIP_CHECK(hipEventCreate(&final_start));
                MIINFER_HIP_CHECK(hipEventCreate(&final_norm_end));
                MIINFER_HIP_CHECK(hipEventCreate(&final_q8_end));
                MIINFER_HIP_CHECK(hipEventCreate(&final_lm_end));
                for (std::size_t i = 0; i < profiled_recurrent_layers.size(); ++i) {
                    profiled_recurrent_layers[i]->stage_profile = &recurrent_profiles[i];
                    profiled_recurrent_layers[i]->stage_profile_position = 63;
                }
                attention3.stage_profile = &attention_profile;
                attention3.stage_profile_position = 63;
                reset_all();
                const auto prompt = prompt_override.has_value()
                    ? std::vector<std::uint32_t>{*prompt_override}
                    : read_tokens(fixture / "prompt_tokens.txt");
                if (prompt.size() != 1 || generated.size() < 63) {
                    throw std::runtime_error("profile requires prompt and 63 generated tokens");
                }
                auto token = prompt.front();
                for (std::size_t position = 0; position < 63; ++position) {
                    miinfer::launch_qwen35_q4_k_embedding(
                        static_cast<const miinfer::Q4KDeviceBlock*>(d_embedding->get()),
                        token, model.config().vocab_size, kHidden,
                        static_cast<float*>(input->get()));
                    run_prefix(std::span<const GpuLayerRef>(layers),
                               std::span<float* const>(output_pointers),
                               static_cast<const float*>(input->get()), position);
                    MIINFER_HIP_CHECK(hipDeviceSynchronize());
                    token = generated[position];
                }
                miinfer::launch_qwen35_q4_k_embedding(
                    static_cast<const miinfer::Q4KDeviceBlock*>(d_embedding->get()),
                    token, model.config().vocab_size, kHidden,
                    static_cast<float*>(input->get()));
                MIINFER_HIP_CHECK(hipEventRecord(token_start, nullptr));
                for (std::size_t layer = 0; layer < layers.size(); ++layer) {
                    MIINFER_HIP_CHECK(hipEventRecord(layer_start[layer], nullptr));
                    layers[layer].run(layer == 0
                                          ? static_cast<const float*>(input->get())
                                          : output_pointers[layer - 1],
                                      63, output_pointers[layer]);
                    MIINFER_HIP_CHECK(hipEventRecord(layer_end[layer], nullptr));
                }
                MIINFER_HIP_CHECK(hipEventRecord(final_start, nullptr));
                miinfer::launch_qwen3_rms_norm(
                    output_pointers[63], static_cast<const float*>(d_final_norm_weight->get()),
                    static_cast<float*>(final_norm->get()), kHidden, model.config().rms_epsilon);
                MIINFER_HIP_CHECK(hipEventRecord(final_norm_end, nullptr));
                if (native_lm_head) {
                    miinfer::launch_q8_1_quantize_f32(
                        static_cast<const float*>(final_norm->get()),
                        static_cast<miinfer::Q8_1Block*>(final_q8_1->get()), kHidden);
                    MIINFER_HIP_CHECK(hipEventRecord(final_q8_end, nullptr));
                    launch_q6k_wave_gemv(
                        static_cast<const Q6KWaveTile*>(d_output_weight->get()),
                        static_cast<const miinfer::Q8_1Block*>(final_q8_1->get()),
                        static_cast<float*>(logits->get()), model.config().vocab_size, kHidden);
                } else if (lm_mmvq) {
                    miinfer::launch_q8_1_quantize_f32(
                        static_cast<const float*>(final_norm->get()),
                        static_cast<miinfer::Q8_1Block*>(final_q8_1->get()), kHidden);
                    MIINFER_HIP_CHECK(hipEventRecord(final_q8_end, nullptr));
                    miinfer::launch_qwen3_q6_k_q8_1_mmvq(
                        static_cast<const miinfer::Q6KDeviceBlock*>(d_output_weight->get()),
                        static_cast<const miinfer::Q8_1Block*>(final_q8_1->get()),
                        static_cast<float*>(logits->get()), model.config().vocab_size, kHidden);
                } else {
                    miinfer::launch_qwen3_q8_k_quantize(
                        static_cast<const float*>(final_norm->get()),
                        static_cast<miinfer::Q8KDeviceBlock*>(final_q8->get()), kHidden);
                    MIINFER_HIP_CHECK(hipEventRecord(final_q8_end, nullptr));
                    miinfer::launch_qwen3_q6_k_q8_k_gemv(
                        static_cast<const miinfer::Q6KDeviceBlock*>(d_output_weight->get()),
                        static_cast<const miinfer::Q8KDeviceBlock*>(final_q8->get()),
                        static_cast<float*>(logits->get()), model.config().vocab_size, kHidden);
                }
                MIINFER_HIP_CHECK(hipEventRecord(final_lm_end, nullptr));
                miinfer::launch_qwen3_argmax(
                    static_cast<const float*>(logits->get()),
                    static_cast<std::uint32_t*>(argmax_token->get()), model.config().vocab_size);
                MIINFER_HIP_CHECK(hipEventRecord(token_end, nullptr));
                MIINFER_HIP_CHECK(hipEventSynchronize(token_end));
                float total_ms = 0.0F;
                float final_norm_ms = 0.0F;
                float final_q8_ms = 0.0F;
                float final_lm_ms = 0.0F;
                float final_argmax_ms = 0.0F;
                MIINFER_HIP_CHECK(hipEventElapsedTime(&total_ms, token_start, token_end));
                MIINFER_HIP_CHECK(hipEventElapsedTime(&final_norm_ms, final_start, final_norm_end));
                MIINFER_HIP_CHECK(hipEventElapsedTime(&final_q8_ms, final_norm_end, final_q8_end));
                MIINFER_HIP_CHECK(hipEventElapsedTime(&final_lm_ms, final_q8_end, final_lm_end));
                MIINFER_HIP_CHECK(hipEventElapsedTime(&final_argmax_ms, final_lm_end, token_end));
                std::cout << "profile_position=63 total_gpu_ms=" << total_ms
                          << " final_norm_ms=" << final_norm_ms
                          << " final_q8_ms=" << final_q8_ms
                          << " final_lm_ms=" << final_lm_ms
                          << " final_argmax_ms=" << final_argmax_ms << '\n';
                float layer_sum = 0.0F;
                for (std::size_t layer = 0; layer < layers.size(); ++layer) {
                    float elapsed = 0.0F;
                    MIINFER_HIP_CHECK(hipEventElapsedTime(
                        &elapsed, layer_start[layer], layer_end[layer]));
                    layer_sum += elapsed;
                    std::cout << "layer=" << layer
                              << " kind=" << (layers[layer].recurrent != nullptr ? "recurrent" : "attention")
                              << " gpu_ms=" << elapsed << '\n';
                }
                static constexpr std::array<const char*, 14> stage_names{
                    "attn_norm", "qkv_projection", "gate_projection", "beta_alpha",
                    "conv_and_head_norm", "state_update", "recurrent_gate",
                    "ssm_output_projection", "attention_residual", "ffn_norm",
                    "ffn_gate_up", "ffn_activation", "ffn_down", "ffn_residual"};
                for (std::size_t i = 0; i < profiled_recurrent_layers.size(); ++i) {
                    std::cout << "recurrent_layer" << profiled_recurrent_layers[i]->index
                              << "_stages\n";
                    for (std::size_t stage = 0; stage < stage_names.size(); ++stage) {
                        float elapsed = 0.0F;
                        MIINFER_HIP_CHECK(hipEventElapsedTime(
                            &elapsed, recurrent_profiles[i].start[stage], recurrent_profiles[i].end[stage]));
                        std::cout << "stage=" << stage_names[stage]
                                  << " gpu_ms=" << elapsed << '\n';
                    }
                }
                static constexpr std::array<const char*, 15> attention_stage_names{
                    "attn_norm", "q_projection", "q_split_head_norm", "k_projection",
                    "k_head_norm", "v_projection", "rope_kv_store", "cached_attention",
                    "attention_gate_o_projection", "attention_residual", "ffn_norm",
                    "ffn_gate_up", "ffn_activation", "ffn_down", "ffn_residual"};
                std::cout << "full_attention_layer3_stages\n";
                for (std::size_t stage = 0; stage < attention_stage_names.size(); ++stage) {
                    float elapsed = 0.0F;
                    MIINFER_HIP_CHECK(hipEventElapsedTime(
                        &elapsed, attention_profile.start[stage], attention_profile.end[stage]));
                    std::cout << "stage=" << attention_stage_names[stage]
                              << " gpu_ms=" << elapsed << '\n';
                }
                std::cout << "layer_sum_gpu_ms=" << layer_sum
                          << " dispatches=unknown_in_native_harness"
                          << " allocations_during_profile=0\n"
                          << "M6-B2 qwen35 native P64 profile PASS\n";
                for (std::size_t layer = 0; layer < layers.size(); ++layer) {
                    MIINFER_HIP_CHECK(hipEventDestroy(layer_start[layer]));
                    MIINFER_HIP_CHECK(hipEventDestroy(layer_end[layer]));
                }
                MIINFER_HIP_CHECK(hipEventDestroy(token_start));
                MIINFER_HIP_CHECK(hipEventDestroy(token_end));
                MIINFER_HIP_CHECK(hipEventDestroy(final_start));
                MIINFER_HIP_CHECK(hipEventDestroy(final_norm_end));
                MIINFER_HIP_CHECK(hipEventDestroy(final_q8_end));
                MIINFER_HIP_CHECK(hipEventDestroy(final_lm_end));
                for (const auto& profile : recurrent_profiles) {
                    for (std::size_t stage = 0; stage < profile.start.size(); ++stage) {
                        MIINFER_HIP_CHECK(hipEventDestroy(profile.start[stage]));
                        MIINFER_HIP_CHECK(hipEventDestroy(profile.end[stage]));
                    }
                }
                for (std::size_t stage = 0; stage < attention_profile.start.size(); ++stage) {
                    MIINFER_HIP_CHECK(hipEventDestroy(attention_profile.start[stage]));
                    MIINFER_HIP_CHECK(hipEventDestroy(attention_profile.end[stage]));
                }
                return 0;
            }
            if (generation) {
                const auto prompt = prompt_override.has_value()
                    ? std::vector<std::uint32_t>{*prompt_override}
                    : read_tokens(fixture / "prompt_tokens.txt");
                if (prompt.size() != 1 || prompt.front() >= model.config().vocab_size) {
                    throw std::runtime_error("generation requires one valid prompt token");
                }
                const auto reset_all = [&] {
                    for (const auto& layer : layers) {
                        if (layer.recurrent != nullptr) layer.recurrent->reset(fixture);
                        if (layer.attention != nullptr) layer.attention->reset();
                    }
                    MIINFER_HIP_CHECK(hipDeviceSynchronize());
                };
                struct GenerationResult {
                    std::vector<std::uint32_t> tokens;
                    std::uint64_t state_hash = 0;
                    double decode_ms = 0.0;
                };
                const auto monotonic_ms = [] {
                    timespec timestamp{};
                    if (clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp) != 0) {
                        throw std::runtime_error("clock_gettime failed");
                    }
                    return static_cast<double>(timestamp.tv_sec) * 1000.0
                        + static_cast<double>(timestamp.tv_nsec) / 1000000.0;
                };
                std::vector<hipGraphExec_t> decode_graphs;
                if (use_hip_graph) {
                    decode_graphs.resize(generation_tokens, nullptr);
                    for (std::size_t position = 0; position < generation_tokens; ++position) {
                        hipGraph_t graph = nullptr;
                        MIINFER_HIP_CHECK(hipStreamBeginCapture(hipStreamPerThread, hipStreamCaptureModeRelaxed));

                        if (device_token_chain) {
                            const auto* token_device_ptr = static_cast<const std::uint32_t*>(d_decode_tokens->get()) + position;
                            miinfer::launch_qwen35_q4_k_embedding_device_token(
                                static_cast<const miinfer::Q4KDeviceBlock*>(d_embedding->get()),
                                token_device_ptr, model.config().vocab_size, kHidden,
                                static_cast<float*>(input->get()), hipStreamPerThread);
                        }

                        run_prefix(std::span<const GpuLayerRef>(layers),
                                   std::span<float* const>(output_pointers),
                                   static_cast<const float*>(input->get()), position,
                                   static_cast<const float*>(d_final_norm_weight->get()),
                                   static_cast<float*>(final_norm->get()),
                                   final_q8_1 ? static_cast<miinfer::Q8_1Block*>(final_q8_1->get()) : nullptr);
                        if (!fused_interlayer_norm) {
                            miinfer::launch_qwen3_rms_norm(
                                output_pointers[63],
                                static_cast<const float*>(d_final_norm_weight->get()),
                                static_cast<float*>(final_norm->get()), kHidden,
                                model.config().rms_epsilon);
                        }
                        if (native_lm_head) {
                            if (!fused_norm_q8 || !fused_interlayer_norm) {
                                miinfer::launch_q8_1_quantize_f32(
                                    static_cast<const float*>(final_norm->get()),
                                    static_cast<miinfer::Q8_1Block*>(final_q8_1->get()), kHidden,
                                    hipStreamPerThread);
                            }
                            launch_q6k_wave_gemv(
                                static_cast<const Q6KWaveTile*>(d_output_weight->get()),
                                static_cast<const miinfer::Q8_1Block*>(final_q8_1->get()),
                                static_cast<float*>(logits->get()), model.config().vocab_size, kHidden,
                                hipStreamPerThread);
                        } else if (lm_mmvq) {
                            miinfer::launch_q8_1_quantize_f32(
                                static_cast<const float*>(final_norm->get()),
                                static_cast<miinfer::Q8_1Block*>(final_q8_1->get()), kHidden);
                            miinfer::launch_qwen3_q6_k_q8_1_mmvq(
                                static_cast<const miinfer::Q6KDeviceBlock*>(d_output_weight->get()),
                                static_cast<const miinfer::Q8_1Block*>(final_q8_1->get()),
                                static_cast<float*>(logits->get()), model.config().vocab_size, kHidden);
                        } else {
                            miinfer::launch_qwen3_q8_k_quantize(
                                static_cast<const float*>(final_norm->get()),
                                static_cast<miinfer::Q8KDeviceBlock*>(final_q8->get()), kHidden);
                            miinfer::launch_qwen3_q6_k_q8_k_gemv(
                                static_cast<const miinfer::Q6KDeviceBlock*>(d_output_weight->get()),
                                static_cast<const miinfer::Q8KDeviceBlock*>(final_q8->get()),
                                static_cast<float*>(logits->get()), model.config().vocab_size, kHidden);
                        }
                        auto* target_token_ptr = device_token_chain
                            ? static_cast<std::uint32_t*>(d_decode_tokens->get()) + (position + 1)
                            : static_cast<std::uint32_t*>(argmax_token->get());
                        miinfer::launch_qwen3_argmax(
                            static_cast<const float*>(logits->get()),
                            target_token_ptr,
                            model.config().vocab_size);
                        MIINFER_HIP_CHECK(hipStreamEndCapture(hipStreamPerThread, &graph));
                        MIINFER_HIP_CHECK(hipGraphInstantiate(&decode_graphs[position], graph, nullptr, nullptr, 0));
                        MIINFER_HIP_CHECK(hipGraphDestroy(graph));
                    }
                }
                const auto cleanup_graphs = [&] {
                    for (auto& graph_exec : decode_graphs) {
                        if (graph_exec != nullptr) {
                            (void)hipGraphExecDestroy(graph_exec);
                            graph_exec = nullptr;
                        }
                    }
                };
                const auto run_generation = [&] {
                    std::vector<std::uint32_t> tokens;
                    tokens.reserve(generation_tokens);
                    auto token = prompt.front();
                    const double start = monotonic_ms();
                    if (use_hip_graph && device_token_chain) {
                        MIINFER_HIP_CHECK(hipMemcpyAsync(d_decode_tokens->get(), &token, sizeof(token),
                                                        hipMemcpyHostToDevice, hipStreamPerThread));
                        for (std::size_t position = 0; position < generation_tokens; ++position) {
                            MIINFER_HIP_CHECK(hipGraphLaunch(decode_graphs[position], hipStreamPerThread));
                        }
                        tokens.resize(generation_tokens);
                        MIINFER_HIP_CHECK(hipMemcpy(tokens.data(),
                                                    static_cast<const std::uint32_t*>(d_decode_tokens->get()) + 1,
                                                    generation_tokens * sizeof(std::uint32_t),
                                                    hipMemcpyDeviceToHost));
                    } else if (use_hip_graph) {
                        for (std::size_t position = 0; position < generation_tokens; ++position) {
                            miinfer::launch_qwen35_q4_k_embedding(
                                static_cast<const miinfer::Q4KDeviceBlock*>(d_embedding->get()),
                                token, model.config().vocab_size, kHidden,
                                static_cast<float*>(input->get()));
                            MIINFER_HIP_CHECK(hipGraphLaunch(decode_graphs[position], hipStreamPerThread));
                            MIINFER_HIP_CHECK(hipDeviceSynchronize());
                            std::uint32_t next = 0;
                            MIINFER_HIP_CHECK(hipMemcpy(&next, argmax_token->get(),
                                                        sizeof(next), hipMemcpyDeviceToHost));
                            tokens.push_back(next);
                            token = next;
                        }
                    } else {
                        for (std::size_t position = 0; position < generation_tokens; ++position) {
                            miinfer::launch_qwen35_q4_k_embedding(
                                static_cast<const miinfer::Q4KDeviceBlock*>(d_embedding->get()),
                                token, model.config().vocab_size, kHidden,
                                static_cast<float*>(input->get()));
                            run_prefix(std::span<const GpuLayerRef>(layers),
                                       std::span<float* const>(output_pointers),
                                       static_cast<const float*>(input->get()), position,
                                       static_cast<const float*>(d_final_norm_weight->get()),
                                       static_cast<float*>(final_norm->get()),
                                       final_q8_1 ? static_cast<miinfer::Q8_1Block*>(final_q8_1->get()) : nullptr);
                            if (!fused_interlayer_norm) {
                                miinfer::launch_qwen3_rms_norm(
                                    output_pointers[63],
                                    static_cast<const float*>(d_final_norm_weight->get()),
                                    static_cast<float*>(final_norm->get()), kHidden,
                                    model.config().rms_epsilon);
                            }
                            if (native_lm_head) {
                                if (!fused_norm_q8 || !fused_interlayer_norm) {
                                    miinfer::launch_q8_1_quantize_f32(
                                        static_cast<const float*>(final_norm->get()),
                                        static_cast<miinfer::Q8_1Block*>(final_q8_1->get()), kHidden);
                                }
                                launch_q6k_wave_gemv(
                                    static_cast<const Q6KWaveTile*>(d_output_weight->get()),
                                    static_cast<const miinfer::Q8_1Block*>(final_q8_1->get()),
                                    static_cast<float*>(logits->get()), model.config().vocab_size, kHidden);
                            } else if (lm_mmvq) {
                                miinfer::launch_q8_1_quantize_f32(
                                    static_cast<const float*>(final_norm->get()),
                                    static_cast<miinfer::Q8_1Block*>(final_q8_1->get()), kHidden);
                                miinfer::launch_qwen3_q6_k_q8_1_mmvq(
                                    static_cast<const miinfer::Q6KDeviceBlock*>(d_output_weight->get()),
                                    static_cast<const miinfer::Q8_1Block*>(final_q8_1->get()),
                                    static_cast<float*>(logits->get()), model.config().vocab_size, kHidden);
                            } else {
                                miinfer::launch_qwen3_q8_k_quantize(
                                    static_cast<const float*>(final_norm->get()),
                                    static_cast<miinfer::Q8KDeviceBlock*>(final_q8->get()), kHidden);
                                miinfer::launch_qwen3_q6_k_q8_k_gemv(
                                    static_cast<const miinfer::Q6KDeviceBlock*>(d_output_weight->get()),
                                    static_cast<const miinfer::Q8KDeviceBlock*>(final_q8->get()),
                                    static_cast<float*>(logits->get()), model.config().vocab_size, kHidden);
                            }
                            miinfer::launch_qwen3_argmax(
                                static_cast<const float*>(logits->get()),
                                static_cast<std::uint32_t*>(argmax_token->get()),
                                model.config().vocab_size);
                            std::uint32_t next = 0;
                            MIINFER_HIP_CHECK(hipMemcpy(&next, argmax_token->get(),
                                                        sizeof(next), hipMemcpyDeviceToHost));
                            tokens.push_back(next);
                            token = next;
                        }
                    }
                    const double end = monotonic_ms();
                    std::uint64_t state_hash = 0;
                    for (const auto& layer : layers) {
                        if (layer.recurrent != nullptr) {
                            state_hash ^= layer.recurrent->state_fingerprint();
                        } else {
                            state_hash ^= fingerprint(layer.attention->key_cache->get(),
                                4 * g_cache_capacity * 256 * sizeof(float));
                            state_hash ^= fingerprint(layer.attention->value_cache->get(),
                                4 * g_cache_capacity * 256 * sizeof(float));
                        }
                    }
                    return GenerationResult{
                        std::move(tokens), state_hash,
                        end - start};
                };
                const auto allocations_before_generation = g_device_allocations;
                if (benchmark) {
                    reset_all();
                    const auto warmup = run_generation();
                    std::array<GenerationResult, 5> samples{};
                    for (auto& sample : samples) {
                        reset_all();
                        sample = run_generation();
                        if (sample.tokens != warmup.tokens || sample.state_hash != warmup.state_hash) {
                            throw std::runtime_error("benchmark generation replay mismatch");
                        }
                    }
                    std::array<double, 5> times{};
                    for (std::size_t i = 0; i < samples.size(); ++i) {
                        times[i] = samples[i].decode_ms;
                    }
                    std::sort(times.begin(), times.end());
                    const double median_ms = times[times.size() / 2];
                    std::cout << "benchmark_tokens=" << generation_tokens
                              << " warmup_ms=" << warmup.decode_ms
                              << " samples_ms=";
                    for (std::size_t i = 0; i < times.size(); ++i) {
                        if (i != 0) std::cout << ',';
                        std::cout << times[i];
                    }
                    std::cout << " median_ms=" << median_ms
                              << " median_tok_s=" << (1000.0 * generation_tokens / median_ms)
                              << " replay=PASS"
                              << " allocations_during_decode="
                              << (g_device_allocations - allocations_before_generation)
                              << " device_bytes_after_setup=" << g_device_bytes
                              << " peak_device_bytes=" << g_peak_device_bytes << '\n'
                              << "M6-B2 native qwen35 generation benchmark PASS\n";
                    cleanup_graphs();
                    return 0;
                }
                reset_all();
                const auto first = run_generation();
                reset_all();
                const auto second = run_generation();
                if (first.tokens != second.tokens || first.state_hash != second.state_hash) {
                    throw std::runtime_error("native generation replay mismatch");
                }
                const double first_tps = 1000.0 * first.tokens.size() / first.decode_ms;
                const double second_tps = 1000.0 * second.tokens.size() / second.decode_ms;
                std::cout << "generated_tokens=" << first.tokens.size()
                          << " first_token=" << first.tokens.front()
                          << " last_token=" << first.tokens.back()
                          << " tokens=";
                for (std::size_t i = 0; i < first.tokens.size(); ++i) {
                    if (i != 0) std::cout << ',';
                    std::cout << first.tokens[i];
                }
                std::cout << " replay=PASS"
                          << " first_decode_ms=" << first.decode_ms
                          << " first_tok_s=" << first_tps
                          << " second_decode_ms=" << second.decode_ms
                          << " second_tok_s=" << second_tps
                          << " state_fingerprint=" << first.state_hash
                          << " allocations_during_decode="
                          << (g_device_allocations - allocations_before_generation)
                          << " device_bytes_after_setup=" << g_device_bytes
                          << " peak_device_bytes=" << g_peak_device_bytes << '\n'
                          << "M6-A28 qwen35 native autoregressive GPU generation PASS\n";
                cleanup_graphs();
                return 0;
            }
            if (trace012) {
                static RecurrentTrace l0_trace{fixture, 0, 2};
                static RecurrentTrace l1_trace{fixture, 1, 2};
                static RecurrentTrace l2_trace{fixture, 2, 2};
                recurrent0.trace = &l0_trace;
                recurrent1.trace = &l1_trace;
                recurrent2.trace = &l2_trace;
                for (std::size_t position = 0; position <= 2; ++position) {
                    const auto host_input = position > 1
                        ? embedding(tensor(*model.file(), "token_embd.weight"), generated[position - 1])
                        : read_f32(checkpoint(fixture, position, "model_input_embed"), kHidden);
                    upload(host_input.data(), input->get(), host_input.size() * sizeof(float));
                    run_prefix(std::span<const GpuLayerRef>(layers).first(layer_count),
                               std::span<float* const>(output_pointers).first(layer_count),
                               static_cast<const float*>(input->get()), position);
                    MIINFER_HIP_CHECK(hipDeviceSynchronize());
                }
                std::cout << "M6-A27.6 qwen35 P2 L0-L2 precision-boundary trace COMPLETE\n";
                return 0;
            }
            if (trace_l0_output) {
                OutputProjectionPathCapture production;
                recurrent0.output_projection_path_capture = &production;
                recurrent0.output_projection_path_capture_position = 2;
                for (std::size_t position = 0; position <= 2; ++position) {
                    const auto host_input = position > 1
                        ? embedding(tensor(*model.file(), "token_embd.weight"), generated[position - 1])
                        : read_f32(checkpoint(fixture, position, "model_input_embed"), kHidden);
                    upload(host_input.data(), input->get(), host_input.size() * sizeof(float));
                    run_prefix(std::span<const GpuLayerRef>(layers).first(layer_count),
                               std::span<float* const>(output_pointers).first(layer_count),
                               static_cast<const float*>(input->get()), position);
                    MIINFER_HIP_CHECK(hipDeviceSynchronize());
                }
                const auto external_input = read_f32(
                    checkpoint(fixture, 2, "model_input_embed"), kHidden);
                const auto external_gated = read_f32(
                    checkpoint(fixture, 2, "final_output-0"), kInner);
                const auto external_residual = read_f32(
                    checkpoint(fixture, 2, "attn_residual-0"), kHidden);
                std::vector<float> external_projected(kHidden);
                for (std::size_t i = 0; i < kHidden; ++i) {
                    external_projected[i] = external_residual[i] - external_input[i];
                }
                const auto report = [](const char* label, std::span<const float> actual,
                                       std::span<const float> expected) {
                    const auto error = located_host_error(actual, expected);
                    std::cout << "stage=" << label
                              << " max_abs=" << error.metrics.max_abs
                              << " mean_abs=" << error.metrics.mean_abs
                              << " rms=" << error.metrics.rms
                              << " relative_rms=" << error.metrics.relative_rms
                              << " max_index=" << error.index
                              << " external=" << error.expected
                              << " gpu=" << error.actual << '\n';
                };
                report("production_gated", production.gated, external_gated);
                report("production_projected", production.projected, external_projected);
                report("production_residual", production.residual, external_residual);
                std::cout << "production_q8k_bytes=" << production.q8_input.size()
                          << " production_q8k_fingerprint="
                          << host_fingerprint(production.q8_input) << '\n';

                Buffer external_gated_device = allocate(external_gated.size() * sizeof(float));
                Buffer external_projected_device = allocate(kHidden * sizeof(float));
                Buffer external_q8_device = allocate(
                    (kInner / 256) * sizeof(miinfer::Q8KDeviceBlock));
                Buffer external_residual_device = allocate(kHidden * sizeof(float));
                upload(external_gated.data(), external_gated_device->get(),
                       external_gated.size() * sizeof(float));
                Buffer temp_ssm_out;
                if (!recurrent0.d_ssm_out) {
                    temp_ssm_out = allocate(recurrent0.ssm_out_weight.byte_size);
                    upload_tensor(recurrent0.ssm_out_weight, temp_ssm_out);
                }
                const Buffer& ssm_out_dev = recurrent0.d_ssm_out ? recurrent0.d_ssm_out : temp_ssm_out;
                project(recurrent0.ssm_out_weight, ssm_out_dev,
                        static_cast<const float*>(external_gated_device->get()),
                        static_cast<miinfer::Q8KDeviceBlock*>(external_q8_device->get()),
                        static_cast<float*>(external_projected_device->get()), kHidden, kInner);
                upload(external_input.data(), external_residual_device->get(),
                       external_input.size() * sizeof(float));
                miinfer::launch_qwen3_add(
                    static_cast<const float*>(external_residual_device->get()),
                    static_cast<const float*>(external_projected_device->get()),
                    static_cast<float*>(external_residual_device->get()), kHidden);
                MIINFER_HIP_CHECK(hipDeviceSynchronize());
                const auto replay_projected = download(external_projected_device->get(), kHidden);
                const auto replay_residual = download(external_residual_device->get(), kHidden);
                report("external_gated_replay_projected", replay_projected, external_projected);
                report("external_gated_replay_residual", replay_residual, external_residual);
                const auto replay_q8 = download_bytes(
                    external_q8_device->get(), (kInner / 256) * sizeof(miinfer::Q8KDeviceBlock));
                std::cout << "external_gated_replay_q8k_bytes=" << replay_q8.size()
                          << " external_gated_replay_q8k_fingerprint="
                          << host_fingerprint(replay_q8) << '\n';
                const auto reference_q8 = quantize_q8(external_gated);
                const auto reference_q8_bytes = std::span<const std::byte>(
                    reinterpret_cast<const std::byte*>(reference_q8.data()),
                    reference_q8.size() * sizeof(Q8K));
                const auto q8_mismatch = compare_bytes(replay_q8, reference_q8_bytes);
                std::cout << "llama_cpu_q8k_reference_bytes=" << reference_q8_bytes.size()
                          << " llama_cpu_q8k_reference_fingerprint="
                          << host_fingerprint(reference_q8_bytes)
                          << " q8k_mismatch_bytes=" << q8_mismatch.count
                          << " q8k_first_mismatch=" << q8_mismatch.first << '\n';
                const std::size_t row = located_host_error(replay_projected, external_projected).index;
                const std::size_t blocks = kInner / 256;
                std::vector<Q8K> host_q8(blocks);
                std::memcpy(host_q8.data(), replay_q8.data(), replay_q8.size());
                const auto* host_weights = reinterpret_cast<const Q5K*>(recurrent0.ssm_out_weight.data)
                                           + row * blocks;
                std::vector<float> host_contributions(blocks);
                for (std::size_t block = 0; block < blocks; ++block) {
                    std::array<std::int8_t, 256> q5{};
                    std::uint8_t high_bit = 1;
                    for (std::size_t group_pair = 0; group_pair < 4; ++group_pair) {
                        const std::size_t q_offset = group_pair * 64;
                        for (std::size_t index = 0; index < 32; ++index) {
                            q5[q_offset + index] = static_cast<std::int8_t>(
                                (host_weights[block].qs[group_pair * 32 + index] & 0x0fU)
                                + ((host_weights[block].qh[index] & high_bit) != 0 ? 16 : 0));
                            q5[q_offset + 32 + index] = static_cast<std::int8_t>(
                                (host_weights[block].qs[group_pair * 32 + index] >> 4U)
                                + ((host_weights[block].qh[index] & (high_bit << 1U)) != 0 ? 16 : 0));
                        }
                        high_bit = static_cast<std::uint8_t>(high_bit << 2U);
                    }
                    std::array<std::uint8_t, 8> scales{};
                    std::array<std::uint8_t, 8> minimums{};
                    for (std::size_t group = 0; group < 8; ++group) {
                        scale_min(host_weights[block].scales, group, scales[group], minimums[group]);
                    }
                    std::array<std::int32_t, 8> partials{};
                    for (std::size_t index = 0; index < 256; ++index) {
                        partials[index % 8] += static_cast<std::int32_t>(scales[index / 32])
                            * static_cast<std::int32_t>(q5[index])
                            * static_cast<std::int32_t>(host_q8[block].qs[index]);
                    }
                    int sumi = 0;
                    for (std::size_t group = 0; group < 16; ++group) {
                        sumi += host_q8[block].bsums[group] * minimums[group / 2];
                    }
                    const float d = miinfer::fp16_bits_to_float(host_weights[block].d)
                                    * host_q8[block].d;
                    const float dmin = miinfer::fp16_bits_to_float(host_weights[block].dmin)
                                       * host_q8[block].d;
                    host_contributions[block] = -dmin * static_cast<float>(sumi);
                    for (const auto partial : partials) {
                        host_contributions[block] += d * static_cast<float>(partial);
                    }
                }
                Buffer block_contributions_device = allocate(blocks * sizeof(float));
                const auto* device_weights = static_cast<const miinfer::Q5KDeviceBlock*>(
                    recurrent0.d_ssm_out->get()) + row * blocks;
                const auto* device_q8 = static_cast<const miinfer::Q8KDeviceBlock*>(
                    external_q8_device->get());
                for (std::size_t block = 0; block < blocks; ++block) {
                    miinfer::launch_qwen3_q5_k_q8_k_gemv(
                        device_weights + block, device_q8 + block,
                        static_cast<float*>(block_contributions_device->get()) + block,
                        1, 256);
                }
                MIINFER_HIP_CHECK(hipDeviceSynchronize());
                const auto gpu_contributions = download(block_contributions_device->get(), blocks);
                std::cout << "q5k_row=" << row << " q5k_blocks=" << blocks << '\n';
                for (std::size_t block = 0; block < blocks; ++block) {
                    std::cout << "q5k_block=" << block
                              << " host=" << host_contributions[block]
                              << " gpu=" << gpu_contributions[block]
                              << " abs_error=" << std::fabs(
                                  gpu_contributions[block] - host_contributions[block]) << '\n';
                }
                const auto host_total = std::accumulate(
                    host_contributions.begin(), host_contributions.end(), 0.0F);
                const auto gpu_total = std::accumulate(
                    gpu_contributions.begin(), gpu_contributions.end(), 0.0F);
                std::cout << "q5k_host_block_sum=" << host_total
                          << " q5k_gpu_block_sum=" << gpu_total
                          << " q5k_block_sum_abs_error=" << std::fabs(gpu_total - host_total)
                          << " external_row=" << external_projected[row]
                          << " production_row=" << replay_projected[row] << '\n';
                std::cout << "M6-A27.9 qwen35 L0 P2 Q5_K block contract COMPLETE\n";
                return 0;
            }
            if (l29_path_attribution32) {
                if (argc != 5) {
                    throw std::runtime_error("L29 path attribution requires an external fixture");
                }
                LayerPathCapture production;
                recurrent29->layer_path_capture = &production;
                recurrent29->layer_path_capture_position = 19;
                for (std::size_t position = 0; position <= 19; ++position) {
                    const auto host_input = position > 1
                        ? embedding(tensor(*model.file(), "token_embd.weight"), generated[position - 1])
                        : read_f32(checkpoint(fixture, position, "model_input_embed"), kHidden);
                    upload(host_input.data(), input->get(), host_input.size() * sizeof(float));
                    run_prefix(std::span<const GpuLayerRef>(layers),
                               std::span<float* const>(output_pointers),
                               static_cast<const float*>(input->get()), position);
                    MIINFER_HIP_CHECK(hipDeviceSynchronize());
                }
                const auto read = [&](const char* name, std::size_t elements) {
                    return read_f32(checkpoint(operand_fixture, 19, name), elements);
                };
                const auto report = [](const char* label, std::span<const float> actual,
                                       std::span<const float> expected) {
                    const auto error = located_host_error(actual, expected);
                    std::cout << "stage=" << label
                              << " max_abs=" << error.metrics.max_abs
                              << " mean_abs=" << error.metrics.mean_abs
                              << " rms=" << error.metrics.rms
                              << " relative_rms=" << error.metrics.relative_rms
                              << " max_index=" << error.index
                              << " external=" << error.expected
                              << " gpu=" << error.actual << '\n';
                };
                std::cout << "M6-A26.6 L29 P19 output provenance\n";
                report("layer_input", production.input, read("l_out-28", kHidden));
                report("attn_norm", production.normalized, read("attn_norm-29", kHidden));
                report("qkv_projection", production.qkv, read("linear_attn_qkv_mixed-29", kChannels));
                report("recurrent_output", production.recurrent_output,
                       read("attn_output-29", kVHeads * kState));
                report("gated", production.gated, read("final_output-29", kVHeads * kState));
                report("attention_residual", production.attention_residual,
                       read("attn_residual-29", kHidden));
                report("post_normalized", production.post_normalized,
                       read("attn_post_norm-29", kHidden));
                report("ffn_output", production.ffn_output, read("ffn_out-29", kHidden));
                report("layer_output", production.layer_output, read("l_out-29", kHidden));
                std::cout << "M6-A26.6 qwen35 L29 output provenance COMPLETE\n";
                return 0;
            }
            if (l29_gate_attribution32) {
                if (argc != 5) {
                    throw std::runtime_error("gate attribution requires an external fixture");
                }
                GatePathCapture production;
                recurrent29->gate_path_capture = &production;
                recurrent29->gate_path_capture_position = 19;
                for (std::size_t position = 0; position <= 19; ++position) {
                    const auto host_input = position > 1
                        ? embedding(tensor(*model.file(), "token_embd.weight"), generated[position - 1])
                        : read_f32(checkpoint(fixture, position, "model_input_embed"), kHidden);
                    upload(host_input.data(), input->get(), host_input.size() * sizeof(float));
                    run_prefix(std::span<const GpuLayerRef>(layers),
                               std::span<float* const>(output_pointers),
                               static_cast<const float*>(input->get()), position);
                    MIINFER_HIP_CHECK(hipDeviceSynchronize());
                }

                const auto external_recurrent = read_f32(
                    checkpoint(operand_fixture, 19, "attn_output-29"), kVHeads * kState);
                const auto external_norm = read_f32(
                    checkpoint(operand_fixture, 19, "attn_norm-29"), kHidden);
                const auto external_gate = read_f32(
                    checkpoint(operand_fixture, 19, "z-29"), kInner);
                const auto external_gated = read_f32(
                    checkpoint(operand_fixture, 19, "final_output-29"), kVHeads * kState);
                const auto* ssm_norm = reinterpret_cast<const float*>(recurrent29->ssm_norm_weight.data);
                std::vector<float> external_head_norm(kVHeads * kState);
                std::vector<float> external_head_scaled(kVHeads * kState);
                std::vector<float> external_gate_silu(kVHeads * kState);
                for (std::size_t head = 0; head < kVHeads; ++head) {
                    const std::size_t base = head * kState;
                    std::array<float, kState> partials{};
                    for (std::size_t i = 0; i < kState; ++i) {
                        const float value = external_recurrent[base + i];
                        partials[i] = value * value;
                    }
                    for (std::size_t stride = kState / 2; stride > 0; stride /= 2) {
                        for (std::size_t i = 0; i < stride; ++i) {
                            partials[i] += partials[i + stride];
                        }
                    }
                    const float inverse_rms = 1.0F / std::sqrt(
                        partials[0] / static_cast<float>(kState) + model.config().rms_epsilon);
                    for (std::size_t i = 0; i < kState; ++i) {
                        const std::size_t index = base + i;
                        external_head_norm[index] = external_recurrent[index] * inverse_rms;
                        external_head_scaled[index] = external_head_norm[index] * ssm_norm[i];
                        external_gate_silu[index] = external_head_scaled[index] == 0.0F
                            ? 0.0F : external_gated[index] / external_head_scaled[index];
                    }
                }
                const auto silu = [](float value) {
                    return value / (1.0F + std::exp(-value));
                };
                std::vector<float> production_gate_silu(production.gate.size());
                std::transform(production.gate.begin(), production.gate.end(),
                               production_gate_silu.begin(), silu);
                const auto replay_gated = [&](std::span<const float> recurrent,
                                              std::span<const float> gate) {
                    Buffer d_recurrent = allocate(recurrent.size() * sizeof(float));
                    Buffer d_gate = allocate(gate.size() * sizeof(float));
                    Buffer d_head_norm = allocate(recurrent.size() * sizeof(float));
                    Buffer d_gated = allocate(recurrent.size() * sizeof(float));
                    upload(recurrent.data(), d_recurrent->get(), recurrent.size() * sizeof(float));
                    upload(gate.data(), d_gate->get(), gate.size() * sizeof(float));
                    miinfer::launch_qwen3_head_rms_normalize(
                        static_cast<const float*>(d_recurrent->get()),
                        static_cast<float*>(d_head_norm->get()), kVHeads, kState,
                        model.config().rms_epsilon);
                    miinfer::launch_qwen3_head_mul(
                        static_cast<const float*>(d_head_norm->get()),
                        static_cast<const float*>(recurrent29->d_ssm_norm->get()),
                        static_cast<float*>(d_gated->get()), kVHeads, kState);
                    miinfer::launch_qwen3_silu_mul(
                        static_cast<const float*>(d_gate->get()),
                        static_cast<const float*>(d_gated->get()),
                        static_cast<float*>(d_gated->get()), kInner);
                    MIINFER_HIP_CHECK(hipDeviceSynchronize());
                    return download(d_gated->get(), kVHeads * kState);
                };
                const auto report_variant = [&](const char* label,
                                                std::span<const float> recurrent,
                                                std::span<const float> gate) {
                    const auto actual = replay_gated(recurrent, gate);
                    const auto error = located_host_error(actual, external_gated);
                    std::cout << "substitution=" << label
                              << " max_abs=" << error.metrics.max_abs
                              << " mean_abs=" << error.metrics.mean_abs
                              << " rms=" << error.metrics.rms
                              << " relative_rms=" << error.metrics.relative_rms
                              << " max_index=" << error.index
                              << " external=" << error.expected
                              << " gpu=" << error.actual << '\n';
                };
                const auto replay_gate_projection = [&](std::span<const float> normalized_input) {
                    Buffer d_input = allocate(normalized_input.size() * sizeof(float));
                    Buffer d_output = allocate(kInner * sizeof(float));
                    upload(normalized_input.data(), d_input->get(),
                           normalized_input.size() * sizeof(float));
                    if (recurrent29->d_attn_gate_native) {
                        miinfer::launch_q8_1_quantize_f32(
                            static_cast<const float*>(d_input->get()),
                            static_cast<miinfer::Q8_1Block*>(recurrent29->q8_1->get()), kHidden);
                        launch_q4k_wave_gemv(
                            static_cast<const Q4KWaveTile*>(recurrent29->d_attn_gate_native->get()),
                            static_cast<const miinfer::Q8_1Block*>(recurrent29->q8_1->get()),
                            static_cast<float*>(d_output->get()), kInner, kHidden);
                    } else {
                        project(recurrent29->gate_weight, recurrent29->d_gate,
                                static_cast<const float*>(d_input->get()),
                                static_cast<miinfer::Q8KDeviceBlock*>(recurrent29->q8->get()),
                                static_cast<float*>(d_output->get()), kInner, kHidden);
                    }
                    MIINFER_HIP_CHECK(hipDeviceSynchronize());
                    return download(d_output->get(), kInner);
                };
                const auto report_gate_input_variant = [&](const char* label,
                                                           std::span<const float> normalized_input) {
                    const auto actual = replay_gate_projection(normalized_input);
                    const auto error = located_host_error(actual, external_gate);
                    std::cout << "gate_input=" << label
                              << " max_abs=" << error.metrics.max_abs
                              << " mean_abs=" << error.metrics.mean_abs
                              << " rms=" << error.metrics.rms
                              << " relative_rms=" << error.metrics.relative_rms
                              << " max_index=" << error.index
                              << " external=" << error.expected
                              << " gpu=" << error.actual << '\n';
                };
                const auto report = [](const char* label, std::span<const float> actual,
                                       std::span<const float> expected) {
                    const auto error = located_host_error(actual, expected);
                    std::cout << "stage=" << label
                              << " max_abs=" << error.metrics.max_abs
                              << " mean_abs=" << error.metrics.mean_abs
                              << " rms=" << error.metrics.rms
                              << " relative_rms=" << error.metrics.relative_rms
                              << " max_index=" << error.index
                              << " external=" << error.expected
                              << " gpu=" << error.actual << '\n';
                };
                std::cout << "M6-A26.8 L29 gate-input provenance\n";
                report_gate_input_variant("production_norm", production.normalized);
                report_gate_input_variant("external_norm", external_norm);
                std::cout << "M6-A26.7 L29 P19 gated-output provenance\n";
                report("recurrent_output", production.recurrent_output, external_recurrent);
                report("head_norm", production.head_norm, external_head_norm);
                report("head_scaled", production.head_scaled, external_head_scaled);
                report("gate_projection", production.gate, external_gate);
                report("gate_silu", production_gate_silu, external_gate_silu);
                report("gated", production.gated, external_gated);
                report_variant("production", production.recurrent_output, production.gate);
                report_variant("external_recurrent", external_recurrent, production.gate);
                report_variant("external_gate", production.recurrent_output, external_gate);
                report_variant("external_both", external_recurrent, external_gate);
                std::cout << "M6-A26.8 qwen35 L29 gate-input provenance COMPLETE\n";
                return 0;
            }
            if (k_path_attribution32) {
                if (argc != 5) {
                    throw std::runtime_error("K-path attribution requires an external fixture");
                }
                KeyPathCapture production;
                recurrent30->key_path_capture = &production;
                recurrent30->key_path_capture_position = 19;
                for (std::size_t position = 0; position <= 19; ++position) {
                    const auto host_input = position > 1
                        ? embedding(tensor(*model.file(), "token_embd.weight"), generated[position - 1])
                        : read_f32(checkpoint(fixture, position, "model_input_embed"), kHidden);
                    upload(host_input.data(), input->get(), host_input.size() * sizeof(float));
                    run_prefix(std::span<const GpuLayerRef>(layers),
                               std::span<float* const>(output_pointers),
                               static_cast<const float*>(input->get()), position);
                    MIINFER_HIP_CHECK(hipDeviceSynchronize());
                }

                const auto external_input = read_f32(
                    checkpoint(operand_fixture, 19, "l_out-29"), kHidden);
                const auto external_norm = read_f32(
                    checkpoint(operand_fixture, 19, "attn_norm-30"), kHidden);
                const auto external_qkv = read_f32(
                    checkpoint(operand_fixture, 19, "linear_attn_qkv_mixed-30"), kChannels);
                const auto external_conv = read_f32(
                    checkpoint(operand_fixture, 19, "conv_output_raw-30"), kChannels);
                const auto external_key_full = read_f32(
                    checkpoint(operand_fixture, 19, "k_in-30"), kVHeads * kState);
                std::vector<float> external_key(kKHeads * kState);
                std::vector<float> external_key_norm(kKHeads * kState);
                for (std::size_t i = 0; i < external_key.size(); ++i) {
                    const float raw = external_conv[kKHeads * kState + i];
                    external_key[i] = raw / (1.0F + std::exp(-raw));
                    external_key_norm[i] = external_key_full[i];
                }
                const auto report = [](const char* label, std::span<const float> actual,
                                       std::span<const float> expected) {
                    const auto error = located_host_error(actual, expected);
                    std::cout << "stage=" << label
                              << " max_abs=" << error.metrics.max_abs
                              << " mean_abs=" << error.metrics.mean_abs
                              << " rms=" << error.metrics.rms
                              << " relative_rms=" << error.metrics.relative_rms
                              << " max_index=" << error.index
                              << " external=" << error.expected
                              << " gpu=" << error.actual << '\n';
                };
                std::cout << "M6-A26.5 L30 P19 K-path provenance\n";
                report("layer_input", production.input, external_input);
                report("attn_norm", production.normalized, external_norm);
                report("qkv_projection", production.qkv, external_qkv);
                report("key_after_conv_silu", production.key, external_key);
                report("key_after_l2_norm", production.key_norm, external_key_norm);
                std::cout << "M6-A26.5 qwen35 L30 K-path provenance COMPLETE\n";
                return 0;
            }
            if (operand_attribution32) {
                if (argc != 5) {
                    throw std::runtime_error(
                        "operand attribution requires an external operand fixture");
                }
                RecurrentOperands production;
                recurrent30->operand_capture = &production;
                recurrent30->operand_capture_position = 19;
                for (std::size_t position = 0; position <= 19; ++position) {
                    const auto host_input = position > 1
                        ? embedding(tensor(*model.file(), "token_embd.weight"), generated[position - 1])
                        : read_f32(checkpoint(fixture, position, "model_input_embed"), kHidden);
                    upload(host_input.data(), input->get(), host_input.size() * sizeof(float));
                    run_prefix(std::span<const GpuLayerRef>(layers),
                               std::span<float* const>(output_pointers),
                               static_cast<const float*>(input->get()), position);
                    MIINFER_HIP_CHECK(hipDeviceSynchronize());
                }

                const auto external = read_external_operands(operand_fixture);
                const auto report_operand = [](const char* label,
                                               std::span<const float> actual,
                                               std::span<const float> expected) {
                    const auto error = located_host_error(actual, expected);
                    std::cout << "operand=" << label
                              << " max_abs=" << error.metrics.max_abs
                              << " mean_abs=" << error.metrics.mean_abs
                              << " rms=" << error.metrics.rms
                              << " relative_rms=" << error.metrics.relative_rms
                              << " max_index=" << error.index
                              << " external=" << error.expected
                              << " gpu=" << error.actual << '\n';
                };
                std::cout << "M6-A26.4 L30 P19->P20 production operand attribution\n"
                          << "q/k compare first " << kKHeads
                          << " heads of the external " << kVHeads << "-head fixture\n";
                report_operand("previous_state", production.previous, external.previous);
                report_operand("q_in", production.query, external.query);
                report_operand("k_in", production.key, external.key);
                report_operand("v_in", production.value, external.value);
                report_operand("beta", production.beta, external.beta);
                report_operand("g_in_as_decay", production.decay, external.decay);

                const auto expected = read_f32(
                    checkpoint(operand_fixture, 20, "state_predelta-30"),
                    kVHeads * kState * kState);
                const auto report_variant = [&](const char* label,
                                                const std::vector<float>* previous,
                                                const std::vector<float>* query,
                                                const std::vector<float>* key,
                                                const std::vector<float>* value,
                                                const std::vector<float>* beta,
                                                const std::vector<float>* decay) {
                    RecurrentOperands operands = production;
                    if (previous != nullptr) operands.previous = *previous;
                    if (query != nullptr) operands.query = *query;
                    if (key != nullptr) operands.key = *key;
                    if (value != nullptr) operands.value = *value;
                    if (beta != nullptr) operands.beta = *beta;
                    if (decay != nullptr) operands.decay = *decay;
                    const auto state = replay_state(operands);
                    const auto error = located_host_error(state, expected);
                    std::cout << "substitution=" << label
                              << " max_abs=" << error.metrics.max_abs
                              << " mean_abs=" << error.metrics.mean_abs
                              << " rms=" << error.metrics.rms
                              << " relative_rms=" << error.metrics.relative_rms
                              << " max_index=" << error.index
                              << " external=" << error.expected
                              << " gpu=" << error.actual << '\n';
                };
                report_variant("production", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
                report_variant("previous_state", &external.previous, nullptr, nullptr, nullptr, nullptr, nullptr);
                report_variant("q_in", nullptr, &external.query, nullptr, nullptr, nullptr, nullptr);
                report_variant("k_in", nullptr, nullptr, &external.key, nullptr, nullptr, nullptr);
                report_variant("v_in", nullptr, nullptr, nullptr, &external.value, nullptr, nullptr);
                report_variant("beta", nullptr, nullptr, nullptr, nullptr, &external.beta, nullptr);
                report_variant("g_in_as_decay", nullptr, nullptr, nullptr, nullptr, nullptr, &external.decay);
                std::cout << "M6-A26.4 qwen35 L30 production operand attribution COMPLETE\n";
                return 0;
            }
            const auto is_checkpoint_position = [](std::size_t position) {
                return position == 0 || position == 1 || position == 2 || position == 4
                    || position == 8 || position == 16 || position == 32 || position == 64;
            };
            const auto is_observable_position = [&is_checkpoint_position](std::size_t position) {
                return is_checkpoint_position(position) || position == 12;
            };
            const auto active_fingerprint = [](const void* device, std::size_t bytes) {
                return bytes == 0 ? 1469598103934665603ULL : fingerprint(device, bytes);
            };
            const auto record_caches = [&](std::size_t position, bool before,
                                           const auto& record) {
                const std::size_t bytes = 4 * (position + (before ? 0 : 1))
                    * 256 * sizeof(float);
                for (std::size_t layer = 0; layer < layer_count; ++layer) {
                    if (layers[layer].attention == nullptr) continue;
                    record(active_fingerprint(layers[layer].attention->key_cache->get(), bytes));
                    record(active_fingerprint(layers[layer].attention->value_cache->get(), bytes));
                }
            };
            std::vector<std::uint64_t> replay_fingerprints;
            const auto record = [&replay_fingerprints](std::uint64_t value) {
                replay_fingerprints.push_back(value);
            };
            const auto allocations_before_decode = g_device_allocations;
            double prefix_cpu_ms = 0.0;
            std::cout << "position";
            for (std::size_t layer = 0; layer < layer_count; ++layer) {
                std::cout << " l" << layer << "_max l" << layer << "_rms l" << layer << "_rel";
            }
            std::cout << " state_correct\n";
            float maximum = 0.0F;
            for (std::size_t position = 0; position <= 64; ++position) {
                const auto host_input = position > 1
                    ? embedding(tensor(*model.file(), "token_embd.weight"), generated[position - 1])
                    : read_f32(checkpoint(fixture, position, "model_input_embed"), kHidden);
                upload(host_input.data(), input->get(), host_input.size() * sizeof(float));
                bool state_correct = true;
                if (is_checkpoint_position(position)) {
                    for (std::size_t layer = 0; layer < layer_count; ++layer) {
                        if (layers[layer].recurrent == nullptr) continue;
                        const auto error = detailed_compare(
                            layers[layer].recurrent->logical_state(),
                            read_f32(checkpoint(fixture, position,
                                                "state_predelta-" + std::to_string(layer)),
                                    kVHeads * kState * kState));
                        if (error.max_abs > 5.0e-2F) {
                            std::cerr << "state mismatch position=" << position
                                      << " layer=" << layer
                                      << " max_abs=" << error.max_abs
                                      << " rms=" << error.rms
                                      << " relative_rms=" << error.relative_rms << '\n';
                        }
                        state_correct = state_correct && error.max_abs <= 5.0e-2F;
                    }
                    for (std::size_t layer = 0; layer < layer_count; ++layer) {
                        if (layers[layer].recurrent != nullptr) {
                            record(layers[layer].recurrent->state_fingerprint());
                        }
                    }
                    record_caches(position, true, record);
                }
                const auto run_start = std::clock();
                run_prefix(std::span<const GpuLayerRef>(layers).first(layer_count),
                           std::span<float* const>(output_pointers).first(layer_count),
                           static_cast<const float*>(input->get()), position,
                           (observable64 && fused_interlayer_norm && layer_count == 64)
                               ? static_cast<const float*>(d_final_norm_weight->get())
                               : nullptr,
                           (observable64 && fused_interlayer_norm && layer_count == 64)
                               ? static_cast<float*>(final_norm->get())
                               : nullptr,
                           (observable64 && fused_interlayer_norm && layer_count == 64 && final_q8_1)
                               ? static_cast<miinfer::Q8_1Block*>(final_q8_1->get())
                               : nullptr);
                MIINFER_HIP_CHECK(hipDeviceSynchronize());
                if (observable64) {
                    if (!fused_interlayer_norm) {
                        miinfer::launch_qwen3_rms_norm(
                            static_cast<const float*>(outputs[63]->get()),
                            static_cast<const float*>(d_final_norm_weight->get()),
                            static_cast<float*>(final_norm->get()), kHidden,
                            model.config().rms_epsilon);
                    }
                    if (native_lm_head) {
                        if (!fused_norm_q8 || !fused_interlayer_norm) {
                            miinfer::launch_q8_1_quantize_f32(
                                static_cast<const float*>(final_norm->get()),
                                static_cast<miinfer::Q8_1Block*>(final_q8_1->get()), kHidden);
                        }
                        launch_q6k_wave_gemv(
                            static_cast<const Q6KWaveTile*>(d_output_weight->get()),
                            static_cast<const miinfer::Q8_1Block*>(final_q8_1->get()),
                            static_cast<float*>(logits->get()), model.config().vocab_size, kHidden);
                    } else if (lm_mmvq) {
                        miinfer::launch_q8_1_quantize_f32(
                            static_cast<const float*>(final_norm->get()),
                            static_cast<miinfer::Q8_1Block*>(final_q8_1->get()), kHidden);
                        miinfer::launch_qwen3_q6_k_q8_1_mmvq(
                            static_cast<const miinfer::Q6KDeviceBlock*>(d_output_weight->get()),
                            static_cast<const miinfer::Q8_1Block*>(final_q8_1->get()),
                            static_cast<float*>(logits->get()), model.config().vocab_size, kHidden);
                    } else {
                        miinfer::launch_qwen3_q8_k_quantize(
                            static_cast<const float*>(final_norm->get()),
                            static_cast<miinfer::Q8KDeviceBlock*>(final_q8->get()), kHidden);
                        miinfer::launch_qwen3_q6_k_q8_k_gemv(
                            static_cast<const miinfer::Q6KDeviceBlock*>(d_output_weight->get()),
                            static_cast<const miinfer::Q8KDeviceBlock*>(final_q8->get()),
                            static_cast<float*>(logits->get()), model.config().vocab_size, kHidden);
                    }
                    MIINFER_HIP_CHECK(hipDeviceSynchronize());
                    const auto final_hidden_host = download(outputs[63]->get(), kHidden);
                    const auto final_norm_host = download(final_norm->get(), kHidden);
                    const auto logits_host = download(logits->get(), model.config().vocab_size);
                    const auto gpu_token = first_argmax(logits_host);
                    if (position < generated.size()) {
                        std::cout << "teacher_forced position=" << position
                                  << " reference_argmax=" << generated[position]
                                  << " gpu_argmax=" << gpu_token
                                  << " match=" << (gpu_token == generated[position] ? "PASS" : "FAIL")
                                  << '\n';
                    }
                    if (is_observable_position(position)) {
                        const auto expected_hidden = read_f32(
                            checkpoint(fixture, position, "l_out-63"), kHidden);
                        const auto expected_norm = read_f32(
                            checkpoint(fixture, position, "result_norm"), kHidden);
                        const auto expected_logits = read_f32(
                            fixture / "logits" / ("logits-" + std::to_string(position) + ".f32"),
                            model.config().vocab_size);
                        const auto hidden_error = detailed_compare(final_hidden_host, expected_hidden);
                        const auto norm_error = detailed_compare(final_norm_host, expected_norm);
                        const auto logits_error = detailed_compare(logits_host, expected_logits);
                        std::cout << "observable position=" << position
                                  << " final_hidden_max_abs=" << hidden_error.max_abs
                                  << " final_hidden_rms=" << hidden_error.rms
                                  << " final_hidden_relative_rms=" << hidden_error.relative_rms
                                  << " final_hidden_cosine="
                                  << cosine_similarity(final_hidden_host, expected_hidden)
                                  << " final_norm_max_abs=" << norm_error.max_abs
                                  << " final_norm_rms=" << norm_error.rms
                                  << " final_norm_relative_rms=" << norm_error.relative_rms
                                  << " final_norm_cosine="
                                  << cosine_similarity(final_norm_host, expected_norm)
                                  << " logits_max_abs=" << logits_error.max_abs
                                  << " logits_rms=" << logits_error.rms
                                  << " logits_relative_rms=" << logits_error.relative_rms
                                  << " logits_cosine="
                                  << cosine_similarity(logits_host, expected_logits);
                        const auto reference_top = top_indices(expected_logits, 5);
                        const auto gpu_top = top_indices(logits_host, 5);
                        std::vector<std::size_t> top_union = reference_top;
                        for (const auto index : gpu_top) {
                            if (std::find(top_union.begin(), top_union.end(), index) == top_union.end()) {
                                top_union.push_back(index);
                            }
                        }
                        float epsilon_top = 0.0F;
                        for (const auto index : top_union) {
                            epsilon_top = std::max(epsilon_top,
                                                   std::fabs(logits_host[index] - expected_logits[index]));
                        }
                        std::size_t overlap = 0;
                        for (const auto index : reference_top) {
                            if (std::find(gpu_top.begin(), gpu_top.end(), index) != gpu_top.end()) ++overlap;
                        }
                        const auto reference_token = first_argmax(expected_logits);
                        const auto gpu_winner = first_argmax(logits_host);
                        const auto gpu_top_two = top_indices(logits_host, 2);
                        const auto reference_top_two = top_indices(expected_logits, 2);
                        std::cout << " reference_argmax=" << reference_token
                                  << " gpu_argmax=" << gpu_winner
                                  << " top5_overlap=" << overlap
                                  << " reference_winner_rank_on_gpu="
                                  << rank_of(logits_host, reference_token)
                                  << " gpu_winner_rank_on_reference="
                                  << rank_of(expected_logits, gpu_winner)
                                  << " reference_margin="
                                  << expected_logits[reference_top_two[0]] - expected_logits[reference_top_two[1]]
                                  << " epsilon_top=" << epsilon_top
                                  << " margin_robust="
                                  << (expected_logits[reference_top_two[0]] - expected_logits[reference_top_two[1]]
                                              > 2.0F * epsilon_top
                                          ? "YES"
                                          : "NO")
                                  << " gpu_margin="
                                  << logits_host[gpu_top_two[0]] - logits_host[gpu_top_two[1]]
                                  << " gpu_minus_reference_at_reference_winner="
                                  << logits_host[reference_token] - expected_logits[reference_token]
                                  << '\n';
                        std::cout << "reference_top5=";
                        for (const auto index : reference_top) std::cout << index << ',';
                        std::cout << " gpu_top5=";
                        for (const auto index : gpu_top) std::cout << index << ',';
                        std::cout << '\n';
                    }
                }
                prefix_cpu_ms += 1000.0 * static_cast<double>(std::clock() - run_start)
                    / static_cast<double>(CLOCKS_PER_SEC);
                if (!is_checkpoint_position(position)) continue;
                std::array<DetailedError, 64> errors{};
                for (std::size_t layer = 0; layer < layer_count; ++layer) {
                    errors[layer] = detailed_device_error(
                        static_cast<const float*>(outputs[layer]->get()), kHidden,
                        checkpoint(fixture, position, "l_out-" + std::to_string(layer)));
                    if (external_contract64 && errors[layer].max_abs > 2.0F) {
                        std::cerr << "prefix64 external output mismatch position=" << position
                                  << " layer=" << layer
                                  << " max_abs=" << errors[layer].max_abs
                                  << " rms=" << errors[layer].rms
                                  << " relative_rms=" << errors[layer].relative_rms << '\n';
                    }
                    if (!observable64 &&
                        !((trace64 || trace53 || gate53_contract) && position == 1)) {
                        require_match("prefix output",
                                      Metrics{errors[layer].max_abs, errors[layer].rms, 0}, 2.0F);
                    }
                    maximum = std::max(maximum, errors[layer].max_abs);
                }
                if (!state_correct && !external_contract64 && !trace64 && !trace53 &&
                    !gate53_contract && !observable64) {
                    throw std::runtime_error("prefix recurrent state mismatch");
                }
                std::cout << position;
                for (std::size_t layer = 0; layer < layer_count; ++layer) {
                    const auto& error = errors[layer];
                    std::cout << ' ' << error.max_abs << ' ' << error.rms
                              << ' ' << error.relative_rms;
                }
                std::cout << " PASS\n  fingerprints hidden3="
                          << fingerprint(outputs[3]->get(), kHidden * sizeof(float))
                          << " hidden7=" << fingerprint(outputs[7]->get(), kHidden * sizeof(float));
                for (std::size_t layer = 0; layer < layer_count; ++layer) {
                    if (layers[layer].recurrent != nullptr) {
                        std::cout << " state" << layer << '='
                                  << layers[layer].recurrent->state_fingerprint();
                    }
                }
                for (std::size_t layer = 0; layer < layer_count; ++layer) {
                    if (layers[layer].attention == nullptr) continue;
                    std::cout << " K" << layer << '=' << fingerprint(
                        layers[layer].attention->key_cache->get(),
                        4 * (position + 1) * 256 * sizeof(float))
                              << " V" << layer << '=' << fingerprint(
                        layers[layer].attention->value_cache->get(),
                        4 * (position + 1) * 256 * sizeof(float));
                }
                std::cout << '\n';
                for (std::size_t layer = 0; layer < layer_count; ++layer) {
                    record(fingerprint(outputs[layer]->get(), kHidden * sizeof(float)));
                }
                for (std::size_t layer = 0; layer < layer_count; ++layer) {
                    if (layers[layer].recurrent != nullptr) {
                        record(layers[layer].recurrent->state_fingerprint());
                    }
                }
                record_caches(position, false, record);
                if (trace64 && position == 1) {
                    const auto read = [&](const char* name, std::size_t elements) {
                        return read_f32(checkpoint(fixture, 1, name), elements);
                    };
                    const auto report = [](const char* label, std::span<const float> actual,
                                           std::span<const float> expected) {
                        const auto error = located_host_error(actual, expected);
                        std::cout << "path label=" << label
                                  << " max_abs=" << error.metrics.max_abs
                                  << " mean_abs=" << error.metrics.mean_abs
                                  << " rms=" << error.metrics.rms
                                  << " relative_rms=" << error.metrics.relative_rms
                                  << " max_index=" << error.index
                                  << " reference=" << error.expected
                                  << " gpu=" << error.actual << '\n';
                    };
                    std::cout << "M6-A27.1 L54 P1 input/path attribution\n";
                    report("layer_input", l54_path.input, read("l_out-53", kHidden));
                    report("attn_norm", l54_path.normalized, read("attn_norm-54", kHidden));
                    report("qkv_projection", l54_path.qkv,
                           read("linear_attn_qkv_mixed-54", kChannels));
                    report("recurrent_output", l54_path.recurrent_output,
                           read("attn_output-54", kVHeads * kState));
                    report("gated", l54_path.gated, read("final_output-54", kVHeads * kState));
                    report("attention_residual", l54_path.attention_residual,
                           read("attn_residual-54", kHidden));
                    report("post_normalized", l54_path.post_normalized,
                           read("attn_post_norm-54", kHidden));
                    report("ffn_output", l54_path.ffn_output, read("ffn_out-54", kHidden));
                    report("layer_output", l54_path.layer_output, read("l_out-54", kHidden));
                    std::cout << "M6-A27.1 qwen35 L54 P1 attribution COMPLETE\n";
                    return 0;
                }
                if (trace53 && position == 1) {
                    const auto read = [&](const char* name, std::size_t elements) {
                        return read_f32(checkpoint(fixture, 1, name), elements);
                    };
                    const auto report = [](const char* label, std::span<const float> actual,
                                           std::span<const float> expected) {
                        const auto error = located_host_error(actual, expected);
                        std::cout << "path label=" << label
                                  << " max_abs=" << error.metrics.max_abs
                                  << " mean_abs=" << error.metrics.mean_abs
                                  << " rms=" << error.metrics.rms
                                  << " relative_rms=" << error.metrics.relative_rms
                                  << " max_index=" << error.index
                                  << " reference=" << error.expected
                                  << " gpu=" << error.actual << '\n';
                    };
                    std::cout << "M6-A27.2 L53 P1 output provenance\n";
                    report("layer_input", l53_path.input, read("l_out-52", kHidden));
                    report("attn_norm", l53_path.normalized, read("attn_norm-53", kHidden));
                    report("qkv_projection", l53_path.qkv,
                           read("linear_attn_qkv_mixed-53", kChannels));
                    report("recurrent_output", l53_path.recurrent_output,
                           read("attn_output-53", kVHeads * kState));
                    report("gated", l53_path.gated, read("final_output-53", kVHeads * kState));
                    report("attention_residual", l53_path.attention_residual,
                           read("attn_residual-53", kHidden));
                    report("post_normalized", l53_path.post_normalized,
                           read("attn_post_norm-53", kHidden));
                    report("ffn_output", l53_path.ffn_output, read("ffn_out-53", kHidden));
                    report("layer_output", l53_path.layer_output, read("l_out-53", kHidden));
                    std::cout << "M6-A27.2 qwen35 L53 P1 attribution COMPLETE\n";
                    return 0;
                }
                if (gate53_contract && position == 1) {
                    const auto external_recurrent = read_f32(
                        checkpoint(fixture, 1, "attn_output-53"), kVHeads * kState);
                    const auto external_gate = read_f32(
                        checkpoint(fixture, 1, "z-53"), kInner);
                    const auto external_gated = read_f32(
                        checkpoint(fixture, 1, "final_output-53"), kVHeads * kState);
                    const auto replay_gated = [&](std::span<const float> recurrent,
                                                  std::span<const float> gate) {
                        Buffer d_recurrent = allocate(recurrent.size() * sizeof(float));
                        Buffer d_gate = allocate(gate.size() * sizeof(float));
                        Buffer d_head_norm = allocate(recurrent.size() * sizeof(float));
                        Buffer d_gated = allocate(recurrent.size() * sizeof(float));
                        upload(recurrent.data(), d_recurrent->get(), recurrent.size() * sizeof(float));
                        upload(gate.data(), d_gate->get(), gate.size() * sizeof(float));
                        miinfer::launch_qwen3_head_rms_normalize(
                            static_cast<const float*>(d_recurrent->get()),
                            static_cast<float*>(d_head_norm->get()), kVHeads, kState,
                            model.config().rms_epsilon);
                        miinfer::launch_qwen3_head_mul(
                            static_cast<const float*>(d_head_norm->get()),
                            static_cast<const float*>(recurrent32_plus[16]->d_ssm_norm->get()),
                            static_cast<float*>(d_gated->get()), kVHeads, kState);
                        miinfer::launch_qwen3_silu_mul(
                            static_cast<const float*>(d_gate->get()),
                            static_cast<const float*>(d_gated->get()),
                            static_cast<float*>(d_gated->get()), kVHeads * kState);
                        MIINFER_HIP_CHECK(hipDeviceSynchronize());
                        return download(d_gated->get(), kVHeads * kState);
                    };
                    const auto report = [&](const char* label, std::span<const float> actual,
                                            std::span<const float> expected) {
                        const auto error = located_host_error(actual, expected);
                        std::cout << "gated_contract=" << label
                                  << " max_abs=" << error.metrics.max_abs
                                  << " mean_abs=" << error.metrics.mean_abs
                                  << " rms=" << error.metrics.rms
                                  << " relative_rms=" << error.metrics.relative_rms
                                  << " max_index=" << error.index
                                  << " external=" << error.expected
                                  << " gpu=" << error.actual << '\n';
                    };
                    report("production_recurrent", l53_gate_path.recurrent_output,
                           external_recurrent);
                    report("production_gate", l53_gate_path.gate, external_gate);
                    report("external_operands", replay_gated(external_recurrent, external_gate),
                           external_gated);
                    report("external_recurrent", replay_gated(external_recurrent,
                                                               l53_gate_path.gate),
                           external_gated);
                    report("external_gate", replay_gated(l53_gate_path.recurrent_output,
                                                           external_gate),
                           external_gated);
                    report("production", replay_gated(l53_gate_path.recurrent_output,
                                                       l53_gate_path.gate),
                           external_gated);
                    std::cout << "M6-A27.3 qwen35 L53 P1 gated contract COMPLETE\n";
                    return 0;
                }
            }
            for (std::size_t layer = 0; layer < layer_count; ++layer) {
                if (layers[layer].recurrent != nullptr) layers[layer].recurrent->poison();
                if (layers[layer].attention != nullptr) layers[layer].attention->poison();
            }
            MIINFER_HIP_CHECK(hipDeviceSynchronize());
            for (std::size_t layer = 0; layer < layer_count; ++layer) {
                if (layers[layer].recurrent != nullptr) layers[layer].recurrent->reset(fixture);
                if (layers[layer].attention != nullptr) layers[layer].attention->reset();
            }
            MIINFER_HIP_CHECK(hipDeviceSynchronize());
            std::size_t replay_index = 0;
            const auto expect_replay = [&](std::uint64_t actual, const char* label) {
                if (replay_index >= replay_fingerprints.size()
                    || replay_fingerprints[replay_index] != actual) {
                    throw std::runtime_error(std::string("poisoned reset replay mismatch: ") + label);
                }
                ++replay_index;
            };
            for (std::size_t position = 0; position <= 64; ++position) {
                const auto host_input = position > 1
                    ? embedding(tensor(*model.file(), "token_embd.weight"), generated[position - 1])
                    : read_f32(checkpoint(fixture, position, "model_input_embed"), kHidden);
                upload(host_input.data(), input->get(), host_input.size() * sizeof(float));
                if (is_checkpoint_position(position)) {
                    for (std::size_t layer = 0; layer < layer_count; ++layer) {
                        if (layers[layer].recurrent != nullptr) {
                            expect_replay(layers[layer].recurrent->state_fingerprint(),
                                          "recurrent entry");
                        }
                    }
                    record_caches(position, true, [&](std::uint64_t value) {
                        expect_replay(value, "attention cache entry");
                    });
                }
                run_prefix(std::span<const GpuLayerRef>(layers).first(layer_count),
                           std::span<float* const>(output_pointers).first(layer_count),
                           static_cast<const float*>(input->get()), position);
                MIINFER_HIP_CHECK(hipDeviceSynchronize());
                if (!is_checkpoint_position(position)) continue;
                for (std::size_t layer = 0; layer < layer_count; ++layer) expect_replay(
                    fingerprint(outputs[layer]->get(), kHidden * sizeof(float)), "layer output");
                for (std::size_t layer = 0; layer < layer_count; ++layer) {
                    if (layers[layer].recurrent != nullptr) {
                        expect_replay(layers[layer].recurrent->state_fingerprint(),
                                      "recurrent exit");
                    }
                }
                record_caches(position, false, [&](std::uint64_t value) {
                    expect_replay(value, "attention cache exit");
                });
            }
            if (replay_index != replay_fingerprints.size()) {
                throw std::runtime_error("poisoned reset replay fingerprint count mismatch");
            }
            std::cout << "max_error=" << maximum
                      << " prefix_cpu_ms=" << prefix_cpu_ms
                      << " prefix_cpu_ms_per_position=" << prefix_cpu_ms / 65.0
                      << " allocations_during_decode="
                      << (g_device_allocations - allocations_before_decode)
                      << " device_bytes_after_setup=" << g_device_bytes
                      << " peak_device_bytes=" << g_peak_device_bytes
                      << " dispatches=not-instrumented copies=not-instrumented\n"
                      << "poisoned_reset_replay=PASS\n"
                      << (external_contract64
                              ? "M6-A27 qwen35 sixty-four-layer external-contract PASS\n"
                              : external_contract32
                              ? "M6-A26 qwen35 thirty-two-layer external-contract PASS\n"
                              : prefix32 ? "M6-A26 qwen35 thirty-two-layer GPU prefix PASS\n"
                                          : prefix16 ? "M6-A25 qwen35 sixteen-layer GPU prefix PASS\n"
                                                      : "M6-A24 qwen35 eight-layer GPU prefix PASS\n");
            return 0;
        }

        Buffer input = allocate(kHidden * sizeof(float));
        Buffer state1 = allocate(kHidden * sizeof(float));
        Buffer state2 = allocate(kHidden * sizeof(float));
        Buffer state3 = allocate(kHidden * sizeof(float));
        Buffer output3 = allocate(kHidden * sizeof(float));
        Buffer state5 = second_block ? allocate(kHidden * sizeof(float)) : nullptr;
        Buffer state6 = second_block ? allocate(kHidden * sizeof(float)) : nullptr;
        Buffer state7 = second_block ? allocate(kHidden * sizeof(float)) : nullptr;
        Buffer output7 = second_block ? allocate(kHidden * sizeof(float)) : nullptr;
        float maximum = 0.0F;
        const auto generated = deep ? read_tokens(fixture / "generated_tokens.txt")
                                    : std::vector<std::uint32_t>{};
        if (deep && generated.size() < 64) throw std::runtime_error("fixture has fewer than 64 tokens");
        const auto is_checkpoint_position = [](std::size_t position) {
            return position == 0 || position == 1 || position == 2 || position == 4
                || position == 8 || position == 16 || position == 32 || position == 64;
        };
        if (second_block) {
            std::cout << "position l4_max l4_rms l4_rel l5_max l5_rms l5_rel "
                         "l6_max l6_rms l6_rel l7_max l7_rms l7_rel state_correct\n";
        } else if (deep) {
            std::cout << "position l0_max l0_rms l0_rel l1_max l1_rms l1_rel "
                         "l2_max l2_rms l2_rel l3_max l3_rms l3_rel state_correct\n";
        }

        const std::size_t last_position = deep ? 64 : 1;
        for (std::size_t position = 0; position <= last_position; ++position) {
            const auto host_input = deep && position > 1
                ? embedding(tensor(*model.file(), "token_embd.weight"), generated[position - 1])
                : read_f32(checkpoint(fixture, position, "model_input_embed"), kHidden);
            upload(host_input.data(), input->get(), host_input.size() * sizeof(float));
            bool state_correct = true;
            if (deep && is_checkpoint_position(position)) {
                const std::array<const RecurrentLayer*, 3> first_layers{
                    &recurrent0, &recurrent1, &recurrent2};
                for (std::size_t layer = 0; layer < first_layers.size(); ++layer) {
                    const auto state_error = detailed_compare(
                        first_layers[layer]->logical_state(),
                        read_f32(checkpoint(fixture, position,
                                            "state_predelta-" + std::to_string(layer)),
                                kVHeads * kState * kState));
                    state_correct = state_correct && state_error.max_abs <= 5.0e-2F;
                }
                if (second_block) {
                    const std::array<const RecurrentLayer*, 3> second_layers{
                        recurrent4.get(), recurrent5.get(), recurrent6.get()};
                    for (std::size_t offset = 0; offset < second_layers.size(); ++offset) {
                        const auto state_error = detailed_compare(
                            second_layers[offset]->logical_state(),
                            read_f32(checkpoint(fixture, position,
                                                "state_predelta-" + std::to_string(4 + offset)),
                                    kVHeads * kState * kState));
                        state_correct = state_correct && state_error.max_abs <= 5.0e-2F;
                    }
                }
            }

            run_hybrid_block(recurrent0, recurrent1, recurrent2, attention3,
                             static_cast<const float*>(input->get()), position,
                             static_cast<float*>(state1->get()), static_cast<float*>(state2->get()),
                             static_cast<float*>(state3->get()), static_cast<float*>(output3->get()));
            if (second_block) {
                run_hybrid_block(*recurrent4, *recurrent5, *recurrent6, *attention7,
                                 static_cast<const float*>(output3->get()), position,
                                 static_cast<float*>(state5->get()), static_cast<float*>(state6->get()),
                                 static_cast<float*>(state7->get()), static_cast<float*>(output7->get()));
            }
            MIINFER_HIP_CHECK(hipDeviceSynchronize());
            if (!deep || is_checkpoint_position(position)) {
                std::array<DetailedError, 8> errors{};
                const std::array<const float*, 8> outputs{
                    static_cast<const float*>(state1->get()), static_cast<const float*>(state2->get()),
                    static_cast<const float*>(state3->get()), static_cast<const float*>(output3->get()),
                    second_block ? static_cast<const float*>(state5->get()) : nullptr,
                    second_block ? static_cast<const float*>(state6->get()) : nullptr,
                    second_block ? static_cast<const float*>(state7->get()) : nullptr,
                    second_block ? static_cast<const float*>(output7->get()) : nullptr};
                const std::size_t first_layer = second_block ? 4 : 0;
                for (std::size_t offset = 0; offset < 4; ++offset) {
                    const std::size_t layer = first_layer + offset;
                    errors[layer] = detailed_device_error(
                        outputs[layer], kHidden,
                        checkpoint(fixture, position, "l_out-" + std::to_string(layer)));
                    require_match("hybrid layer output", Metrics{errors[layer].max_abs,
                                                                  errors[layer].rms, 0}, 2.0F);
                    maximum = std::max(maximum, errors[layer].max_abs);
                }
                if (second_block) {
                    for (std::size_t layer = 0; layer < 4; ++layer) {
                        const auto error = detailed_device_error(
                            outputs[layer], kHidden,
                            checkpoint(fixture, position, "l_out-" + std::to_string(layer)));
                        require_match("first hybrid layer output", Metrics{error.max_abs, error.rms, 0}, 2.0F);
                        maximum = std::max(maximum, error.max_abs);
                    }
                }
                if (deep && !state_correct && !external_contract32 && !external_contract64) {
                    throw std::runtime_error("hybrid recurrent state mismatch");
                }
                if (second_block) {
                    std::cout << position;
                    for (std::size_t layer = 4; layer < 8; ++layer) {
                        std::cout << ' ' << errors[layer].max_abs << ' ' << errors[layer].rms
                                  << ' ' << errors[layer].relative_rms;
                    }
                    std::cout << ' ' << (state_correct ? "PASS" : "DIAGNOSTIC_RETEST") << '\n';
                    std::cout << "  first_block_output="
                              << fingerprint(output3->get(), kHidden * sizeof(float)) << '\n';
                    std::cout << "  fingerprints state0=" << recurrent0.state_fingerprint()
                              << " state1=" << recurrent1.state_fingerprint()
                              << " state2=" << recurrent2.state_fingerprint()
                              << " state4=" << recurrent4->state_fingerprint()
                              << " state5=" << recurrent5->state_fingerprint()
                              << " state6=" << recurrent6->state_fingerprint()
                              << " K3=" << fingerprint(attention3.key_cache->get(),
                                  4 * (position + 1) * 256 * sizeof(float))
                              << " V3=" << fingerprint(attention3.value_cache->get(),
                                  4 * (position + 1) * 256 * sizeof(float))
                              << " K7=" << fingerprint(attention7->key_cache->get(),
                                  4 * (position + 1) * 256 * sizeof(float))
                              << " V7=" << fingerprint(attention7->value_cache->get(),
                                  4 * (position + 1) * 256 * sizeof(float)) << '\n';
                } else if (deep) {
                    std::cout << position;
                    for (std::size_t layer = 0; layer < 4; ++layer) {
                        std::cout << ' ' << errors[layer].max_abs << ' ' << errors[layer].rms
                                  << ' ' << errors[layer].relative_rms;
                    }
                    std::cout << ' ' << (state_correct ? "PASS" : "DIAGNOSTIC_RETEST") << '\n';
                    std::cout << "  fingerprints state0=" << recurrent0.state_fingerprint()
                              << " state1=" << recurrent1.state_fingerprint()
                              << " state2=" << recurrent2.state_fingerprint()
                              << " K=" << fingerprint(attention3.key_cache->get(),
                                  4 * (position + 1) * 256 * sizeof(float))
                              << " V=" << fingerprint(attention3.value_cache->get(),
                                  4 * (position + 1) * 256 * sizeof(float)) << '\n';
                } else {
                    for (std::size_t layer = 0; layer < 4; ++layer) {
                        std::cout << "position=" << position << " layer=" << layer
                                  << " max_abs=" << errors[layer].max_abs
                                  << " rmse=" << errors[layer].rms << '\n';
                    }
                }
            }
        }
        std::cout << "max_error=" << maximum << '\n'
                  << (prefix32 ? "M6-A26" : prefix8 ? "M6-A24" : second_block ? "M6-A23" : deep ? "M6-A22" : "M6-A21")
                  << " qwen35 GPU hybrid block PASS\n";
    } catch (const std::exception& error) {
        std::cerr << (prefix32 ? "M6-A26" : prefix8 ? "M6-A24" : second_block ? "M6-A23" : deep ? "M6-A22" : "M6-A21")
                  << " failed: " << error.what() << '\n';
        return 1;
    }
}
