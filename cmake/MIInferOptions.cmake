option(MIINFER_ENABLE_HIP "Build HIP device validation, smoke test, and benchmark" ON)

set(MIINFER_TARGET_ARCH "gfx906" CACHE STRING "Only supported GPU architecture")
set(MIINFER_HIP_ARCHITECTURE "gfx906" CACHE STRING "HIP compiler architecture target")

if(NOT MIINFER_TARGET_ARCH STREQUAL "gfx906")
    message(FATAL_ERROR
        "MIInfer supports only gfx906; got MIINFER_TARGET_ARCH=${MIINFER_TARGET_ARCH}")
endif()
if(NOT MIINFER_HIP_ARCHITECTURE STREQUAL "gfx906")
    message(FATAL_ERROR
        "MIInfer supports only gfx906; got MIINFER_HIP_ARCHITECTURE=${MIINFER_HIP_ARCHITECTURE}")
endif()

if(MIINFER_ENABLE_HIP)
    set(CMAKE_HIP_ARCHITECTURES "${MIINFER_HIP_ARCHITECTURE}" CACHE STRING
        "HIP architectures compiled by MIInfer" FORCE)
endif()
