cmake_minimum_required(VERSION 3.24)

include(cmake/MIInferOptions.cmake)

project(MIInfer VERSION 0.1.0 LANGUAGES CXX)

if(MIINFER_ENABLE_HIP)
    enable_language(HIP)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MIINFER_ENABLE_HIP)
    set(CMAKE_HIP_STANDARD 20)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    set(CMAKE_HIP_EXTENSIONS OFF)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

set(MIINFER_GIT_COMMIT "UNAVAILABLE")
set(MIINFER_GIT_DIRTY "UNAVAILABLE")
execute_process(
    COMMAND git -C "${CMAKE_SOURCE_DIR}" rev-parse --short=12 HEAD
    OUTPUT_VARIABLE MIINFER_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE MIINFER_GIT_RESULT
)
if(MIINFER_GIT_RESULT)
    set(MIINFER_GIT_COMMIT "UNAVAILABLE")
else()
    execute_process(
        COMMAND git -C "${CMAKE_SOURCE_DIR}" status --porcelain
        OUTPUT_VARIABLE MIINFER_GIT_STATUS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE MIINFER_GIT_STATUS_RESULT
    )
    if(MIINFER_GIT_STATUS_RESULT)
        set(MIINFER_GIT_DIRTY "UNAVAILABLE")
    elseif(MIINFER_GIT_STATUS)
        set(MIINFER_GIT_DIRTY "true")
    else()
        set(MIINFER_GIT_DIRTY "false")
    endif()
endif()

if(MIINFER_ENABLE_HIP)
    set(MIINFER_HIP_AVAILABLE "true")
    set(MIINFER_HIP_COMPILER "${CMAKE_HIP_COMPILER_ID} ${CMAKE_HIP_COMPILER_VERSION}")
    find_package(hipblas CONFIG REQUIRED)
else()
    set(MIINFER_HIP_AVAILABLE "false")
    set(MIINFER_HIP_COMPILER "UNAVAILABLE")
endif()

configure_file(
    cmake/miinfer/build_config.hpp.in
    "${CMAKE_CURRENT_BINARY_DIR}/generated/miinfer/build_config.hpp"
    @ONLY
)

add_library(miinfer_build_info STATIC src/build_info.cpp)
target_include_directories(miinfer_build_info
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_BINARY_DIR}/generated"
)
target_compile_features(miinfer_build_info PUBLIC cxx_std_20)
target_compile_options(miinfer_build_info PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
)

add_library(miinfer_model STATIC
    src/gguf.cpp
    src/qwen3_model.cpp
    src/qwen3_primitives.cpp
    src/qwen3_layer.cpp
    src/qwen3_tokenizer.cpp
    src/sha256.cpp
)
target_include_directories(miinfer_model
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_features(miinfer_model PUBLIC cxx_std_20)
target_compile_options(miinfer_model PRIVATE -Wall -Wextra -Wpedantic)

if(MIINFER_ENABLE_HIP)
    add_library(miinfer_hip STATIC
        src/device_validation.cpp
        src/model_plan.cpp
        src/qwen3_gpu_layer.cpp
        src/fp16_gemv_reference.cpp
        src/q4_q8_gemv_reference.cpp
        gfx906/kernels/vector_add.hip
        gfx906/kernels/fp16_gemv.hip
        gfx906/kernels/q4_q8_gemv.hip
        gfx906/kernels/q4_q8_packed_dot.hip
        gfx906/kernels/q4_q8_zero_point_dot.hip
        gfx906/kernels/qwen3_primitives.hip
        gfx906/kernels/memory_stream.hip
    )
    target_include_directories(miinfer_hip
        PUBLIC
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_BINARY_DIR}/generated"
    )
    target_compile_features(miinfer_hip PUBLIC cxx_std_20)
    target_link_libraries(miinfer_hip PUBLIC miinfer_build_info miinfer_model roc::hipblas)
    target_compile_definitions(miinfer_hip PUBLIC __HIP_PLATFORM_AMD__)
    target_compile_options(miinfer_hip PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:HIP>:-Wall;-Wextra>
    )
endif()

add_executable(miinfer-host-test tests/host_test.cpp)
target_link_libraries(miinfer-host-test PRIVATE miinfer_build_info miinfer_model)
target_compile_options(miinfer-host-test PRIVATE -Wall -Wextra -Wpedantic)

add_executable(miinfer-model-loader-test tests/model_loader_test.cpp)
target_link_libraries(miinfer-model-loader-test PRIVATE miinfer_model)
target_compile_options(miinfer-model-loader-test PRIVATE -Wall -Wextra -Wpedantic)

add_executable(miinfer-m6a2-compatibility-audit tools/m6a2_compatibility_audit.cpp)
target_link_libraries(miinfer-m6a2-compatibility-audit PRIVATE miinfer_model)
target_compile_options(miinfer-m6a2-compatibility-audit PRIVATE -Wall -Wextra -Wpedantic)

add_executable(miinfer-m6a3-qwen35-layer tools/m6a3_qwen35_layer.cpp)
target_link_libraries(miinfer-m6a3-qwen35-layer PRIVATE miinfer_model)
target_compile_options(miinfer-m6a3-qwen35-layer PRIVATE -Wall -Wextra -Wpedantic)

add_executable(miinfer-m6a4-qwen35-attention-layer tools/m6a4_qwen35_attention_layer.cpp)
target_link_libraries(miinfer-m6a4-qwen35-attention-layer PRIVATE miinfer_model)
target_compile_options(miinfer-m6a4-qwen35-attention-layer PRIVATE -Wall -Wextra -Wpedantic)

add_executable(miinfer-qwen3-primitives-host-test tests/qwen3_primitives_host_test.cpp)
target_link_libraries(miinfer-qwen3-primitives-host-test PRIVATE miinfer_model)
target_compile_options(miinfer-qwen3-primitives-host-test PRIVATE -Wall -Wextra -Wpedantic)

add_executable(miinfer-qwen3-layer0-test tests/qwen3_layer0_test.cpp)
target_link_libraries(miinfer-qwen3-layer0-test PRIVATE miinfer_model)
target_compile_options(miinfer-qwen3-layer0-test PRIVATE -Wall -Wextra -Wpedantic)

add_executable(miinfer-qwen3-kv-cache-test tests/qwen3_kv_cache_test.cpp)
target_link_libraries(miinfer-qwen3-kv-cache-test PRIVATE miinfer_model)
target_compile_options(miinfer-qwen3-kv-cache-test PRIVATE -Wall -Wextra -Wpedantic)

add_executable(miinfer-qwen3-forward-test tests/qwen3_forward_test.cpp)
target_link_libraries(miinfer-qwen3-forward-test PRIVATE miinfer_model)
target_compile_options(miinfer-qwen3-forward-test PRIVATE -Wall -Wextra -Wpedantic)

add_executable(miinfer-qwen3-trace-compare tests/qwen3_trace_compare.cpp)
target_compile_options(miinfer-qwen3-trace-compare PRIVATE -Wall -Wextra -Wpedantic)

add_executable(miinfer-qwen3-tokenizer-test tests/qwen3_tokenizer_test.cpp)
target_link_libraries(miinfer-qwen3-tokenizer-test PRIVATE miinfer_model)
target_compile_options(miinfer-qwen3-tokenizer-test PRIVATE -Wall -Wextra -Wpedantic)

if(MIINFER_ENABLE_HIP)
    add_executable(miinfer-device-info tests/device_info.cpp)
    target_link_libraries(miinfer-device-info PRIVATE miinfer_hip)
    target_compile_options(miinfer-device-info PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(miinfer-hip-smoke tests/hip_smoke.cpp)
    target_link_libraries(miinfer-hip-smoke PRIVATE miinfer_hip)
    target_compile_options(miinfer-hip-smoke PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(miinfer-bench bench/hip_smoke_bench.cpp)
    target_link_libraries(miinfer-bench PRIVATE miinfer_hip)
    target_compile_options(miinfer-bench PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(miinfer-fp16-gemv-test tests/fp16_gemv_test.cpp)
    target_link_libraries(miinfer-fp16-gemv-test PRIVATE miinfer_hip)
    target_compile_options(miinfer-fp16-gemv-test PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(miinfer-q4-q8-gemv-test tests/q4_q8_gemv_test.cpp)
    target_link_libraries(miinfer-q4-q8-gemv-test PRIVATE miinfer_hip)
    target_compile_options(miinfer-q4-q8-gemv-test PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(miinfer-q4-q8-dot4-probe tests/q4_q8_dot4_probe.cpp)
    target_link_libraries(miinfer-q4-q8-dot4-probe PRIVATE miinfer_hip)
    target_compile_options(miinfer-q4-q8-dot4-probe PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(miinfer-fp16-gemv-bench bench/fp16_gemv_bench.cpp)
    target_link_libraries(miinfer-fp16-gemv-bench PRIVATE miinfer_hip)
    target_compile_options(miinfer-fp16-gemv-bench PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(miinfer-memory-stream-bench bench/memory_stream_bench.cpp)
    target_link_libraries(miinfer-memory-stream-bench PRIVATE miinfer_hip)
    target_compile_options(miinfer-memory-stream-bench PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(miinfer-fp16-gemv-reduction-diag bench/fp16_gemv_reduction_diag.cpp)
    target_link_libraries(miinfer-fp16-gemv-reduction-diag PRIVATE miinfer_hip)
    target_compile_options(miinfer-fp16-gemv-reduction-diag PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(miinfer-fp16-gemv-k-split-bench bench/fp16_gemv_k_split_bench.cpp)
    target_link_libraries(miinfer-fp16-gemv-k-split-bench PRIVATE miinfer_hip)
    target_compile_options(miinfer-fp16-gemv-k-split-bench PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(miinfer-q4-q8-gemv-bench bench/q4_q8_gemv_bench.cpp)
    target_link_libraries(miinfer-q4-q8-gemv-bench PRIVATE miinfer_hip)
    target_compile_options(miinfer-q4-q8-gemv-bench PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(miinfer-model-info tests/model_info.cpp)
    target_link_libraries(miinfer-model-info PRIVATE miinfer_hip)
    target_compile_options(miinfer-model-info PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(miinfer-model-gpu-test tests/model_gpu_test.cpp)
    target_link_libraries(miinfer-model-gpu-test PRIVATE miinfer_hip)
    target_compile_options(miinfer-model-gpu-test PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(miinfer-qwen3-primitives-test tests/qwen3_primitives_test.cpp)
    target_link_libraries(miinfer-qwen3-primitives-test PRIVATE miinfer_hip)
    target_compile_options(miinfer-qwen3-primitives-test PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(miinfer-qwen3-layer0-gpu-test tests/qwen3_layer0_gpu_test.cpp)
    target_link_libraries(miinfer-qwen3-layer0-gpu-test PRIVATE miinfer_hip)
    target_compile_options(miinfer-qwen3-layer0-gpu-test PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(miinfer-qwen3-kv-cache-gpu-test tests/qwen3_kv_cache_gpu_test.cpp)
    target_link_libraries(miinfer-qwen3-kv-cache-gpu-test PRIVATE miinfer_hip)
    target_compile_options(miinfer-qwen3-kv-cache-gpu-test PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(miinfer-qwen3-forward-gpu-test tests/qwen3_forward_gpu_test.cpp)
    target_link_libraries(miinfer-qwen3-forward-gpu-test PRIVATE miinfer_hip)
    target_compile_options(miinfer-qwen3-forward-gpu-test PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(miinfer-qwen3-decode-gpu-test tests/qwen3_decode_gpu_test.cpp)
    target_link_libraries(miinfer-qwen3-decode-gpu-test PRIVATE miinfer_hip)
    target_compile_options(miinfer-qwen3-decode-gpu-test PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(miinfer-qwen3-decode-sequence-gpu-test
                   tests/qwen3_decode_sequence_gpu_test.cpp)
    target_link_libraries(miinfer-qwen3-decode-sequence-gpu-test PRIVATE miinfer_hip)
    target_compile_options(miinfer-qwen3-decode-sequence-gpu-test PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(miinfer-qwen3-generate tests/qwen3_generate.cpp)
    target_link_libraries(miinfer-qwen3-generate PRIVATE miinfer_hip)
    target_compile_options(miinfer-qwen3-generate PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(miinfer-qwen3-inference-bench bench/qwen3_inference_bench.cpp)
    target_link_libraries(miinfer-qwen3-inference-bench PRIVATE miinfer_hip)
    target_compile_options(miinfer-qwen3-inference-bench PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(miinfer-qwen3-decode-profile bench/qwen3_decode_profile.cpp)
    target_link_libraries(miinfer-qwen3-decode-profile PRIVATE miinfer_hip)
    target_compile_options(miinfer-qwen3-decode-profile PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(miinfer-qwen3-fast-decode-bench bench/qwen3_fast_decode_bench.cpp)
    target_link_libraries(miinfer-qwen3-fast-decode-bench PRIVATE miinfer_hip)
    target_compile_options(miinfer-qwen3-fast-decode-bench PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(miinfer-qwen3-attention-ab-bench bench/qwen3_attention_ab_bench.cpp)
    target_link_libraries(miinfer-qwen3-attention-ab-bench PRIVATE miinfer_hip)
    target_compile_options(miinfer-qwen3-attention-ab-bench PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(miinfer-qwen3-swiglu-q8-bench bench/qwen3_swiglu_q8_bench.cpp)
    target_link_libraries(miinfer-qwen3-swiglu-q8-bench PRIVATE miinfer_hip)
    target_compile_options(miinfer-qwen3-swiglu-q8-bench PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(miinfer-qwen3-position-audit bench/qwen3_position_audit.cpp)
    target_link_libraries(miinfer-qwen3-position-audit PRIVATE miinfer_hip)
    target_compile_options(miinfer-qwen3-position-audit PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(miinfer-qwen3-layer6-external-test tests/qwen3_layer6_external_test.cpp)
    target_link_libraries(miinfer-qwen3-layer6-external-test PRIVATE miinfer_hip)
    target_compile_options(miinfer-qwen3-layer6-external-test PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(miinfer-qwen3-layer35-external-test tests/qwen3_layer35_external_test.cpp)
    target_link_libraries(miinfer-qwen3-layer35-external-test PRIVATE miinfer_hip)
    target_compile_options(miinfer-qwen3-layer35-external-test PRIVATE -Wall -Wextra -Wpedantic)

endif()

include(CTest)
if(BUILD_TESTING)
    add_test(NAME host-only COMMAND miinfer-host-test)
    set_tests_properties(host-only PROPERTIES LABELS "host-only")
    add_test(NAME model-loader-host COMMAND miinfer-model-loader-test)
    set_tests_properties(model-loader-host PROPERTIES LABELS "host-only")
    add_test(NAME qwen3-primitives-host COMMAND miinfer-qwen3-primitives-host-test)
    set_tests_properties(qwen3-primitives-host PROPERTIES LABELS "host-only")
    add_test(NAME qwen3-layer0-host COMMAND miinfer-qwen3-layer0-test)
    set_tests_properties(qwen3-layer0-host PROPERTIES LABELS "host-only")
    add_test(NAME qwen3-kv-cache-host COMMAND miinfer-qwen3-kv-cache-test)
    set_tests_properties(qwen3-kv-cache-host PROPERTIES LABELS "host-only")
    add_test(NAME qwen3-forward-host COMMAND miinfer-qwen3-forward-test)
    set_tests_properties(qwen3-forward-host PROPERTIES LABELS "host-only")
    add_test(NAME qwen3-tokenizer-host COMMAND miinfer-qwen3-tokenizer-test)
    set_tests_properties(qwen3-tokenizer-host PROPERTIES LABELS "host-only")
    add_test(NAME qwen3-layer0-comparator-mutation
             COMMAND miinfer-qwen3-layer0-test --self-test-mutation)
    set_tests_properties(qwen3-layer0-comparator-mutation PROPERTIES LABELS "host-only")
    if(MIINFER_ENABLE_HIP)
        add_test(NAME hip-smoke COMMAND miinfer-hip-smoke)
        set_tests_properties(hip-smoke PROPERTIES LABELS "gpu-required")
        add_test(NAME fp16-gemv-correctness COMMAND miinfer-fp16-gemv-test)
        set_tests_properties(fp16-gemv-correctness PROPERTIES LABELS "gpu-required")
        add_test(NAME q4-q8-gemv-correctness COMMAND miinfer-q4-q8-gemv-test)
        set_tests_properties(q4-q8-gemv-correctness PROPERTIES LABELS "gpu-required")
        add_test(NAME q4-q8-dot4-probe COMMAND miinfer-q4-q8-dot4-probe)
        set_tests_properties(q4-q8-dot4-probe PROPERTIES LABELS "gpu-required")
        add_test(NAME model-gpu-arena COMMAND miinfer-model-gpu-test)
        set_tests_properties(model-gpu-arena PROPERTIES LABELS "gpu-required")
        add_test(NAME qwen3-primitives-gpu COMMAND miinfer-qwen3-primitives-test)
        set_tests_properties(qwen3-primitives-gpu PROPERTIES LABELS "gpu-required")
        add_test(NAME qwen3-layer0-gpu COMMAND miinfer-qwen3-layer0-gpu-test)
        set_tests_properties(qwen3-layer0-gpu PROPERTIES LABELS "gpu-required")
        add_test(NAME qwen3-kv-cache-gpu COMMAND miinfer-qwen3-kv-cache-gpu-test)
        set_tests_properties(qwen3-kv-cache-gpu PROPERTIES LABELS "gpu-required")
        add_test(NAME qwen3-forward-gpu COMMAND miinfer-qwen3-forward-gpu-test)
        set_tests_properties(qwen3-forward-gpu PROPERTIES LABELS "gpu-required")
        add_test(NAME qwen3-decode-gpu COMMAND miinfer-qwen3-decode-gpu-test)
        set_tests_properties(qwen3-decode-gpu PROPERTIES LABELS "gpu-required")
        add_test(NAME qwen3-decode-sequence-gpu
                 COMMAND miinfer-qwen3-decode-sequence-gpu-test)
        set_tests_properties(qwen3-decode-sequence-gpu PROPERTIES LABELS "gpu-required")
    endif()
endif()

install(TARGETS miinfer-host-test miinfer-qwen3-primitives-host-test miinfer-qwen3-layer0-test
                miinfer-qwen3-forward-test miinfer-qwen3-trace-compare
                miinfer-qwen3-tokenizer-test)
if(MIINFER_ENABLE_HIP)
    install(TARGETS miinfer-device-info miinfer-hip-smoke miinfer-bench
                    miinfer-fp16-gemv-test miinfer-fp16-gemv-bench)
    install(TARGETS miinfer-q4-q8-gemv-test)
    install(TARGETS miinfer-q4-q8-dot4-probe)
    install(TARGETS miinfer-memory-stream-bench)
    install(TARGETS miinfer-fp16-gemv-reduction-diag)
    install(TARGETS miinfer-fp16-gemv-k-split-bench)
    install(TARGETS miinfer-q4-q8-gemv-bench)
    install(TARGETS miinfer-model-info miinfer-model-gpu-test miinfer-qwen3-primitives-test)
    install(TARGETS miinfer-qwen3-decode-gpu-test)
    install(TARGETS miinfer-qwen3-decode-sequence-gpu-test)
    install(TARGETS miinfer-qwen3-generate)
    install(TARGETS miinfer-qwen3-inference-bench)
    install(TARGETS miinfer-qwen3-decode-profile)
    install(TARGETS miinfer-qwen3-fast-decode-bench)
    install(TARGETS miinfer-qwen3-attention-ab-bench)
    install(TARGETS miinfer-qwen3-position-audit)
    install(TARGETS miinfer-qwen3-layer0-gpu-test)
    install(TARGETS miinfer-qwen3-kv-cache-gpu-test)
    install(TARGETS miinfer-qwen3-forward-gpu-test)
endif()
