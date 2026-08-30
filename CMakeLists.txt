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
        src/fp16_gemv_reference.cpp
        src/q4_q8_gemv_reference.cpp
        gfx906/kernels/vector_add.hip
        gfx906/kernels/fp16_gemv.hip
        gfx906/kernels/q4_q8_gemv.hip
        gfx906/kernels/q4_q8_packed_dot.hip
        gfx906/kernels/q4_q8_zero_point_dot.hip
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

endif()

include(CTest)
if(BUILD_TESTING)
    add_test(NAME host-only COMMAND miinfer-host-test)
    set_tests_properties(host-only PROPERTIES LABELS "host-only")
    add_test(NAME model-loader-host COMMAND miinfer-model-loader-test)
    set_tests_properties(model-loader-host PROPERTIES LABELS "host-only")
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
    endif()
endif()

install(TARGETS miinfer-host-test)
if(MIINFER_ENABLE_HIP)
    install(TARGETS miinfer-device-info miinfer-hip-smoke miinfer-bench
                    miinfer-fp16-gemv-test miinfer-fp16-gemv-bench)
    install(TARGETS miinfer-q4-q8-gemv-test)
    install(TARGETS miinfer-q4-q8-dot4-probe)
    install(TARGETS miinfer-memory-stream-bench)
    install(TARGETS miinfer-fp16-gemv-reduction-diag)
    install(TARGETS miinfer-fp16-gemv-k-split-bench)
    install(TARGETS miinfer-q4-q8-gemv-bench)
    install(TARGETS miinfer-model-info miinfer-model-gpu-test)
endif()
