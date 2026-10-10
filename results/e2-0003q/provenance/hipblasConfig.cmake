add_library(roc::hipblas SHARED IMPORTED)
set_target_properties(roc::hipblas PROPERTIES
    IMPORTED_LOCATION "/opt/rocm-7.2.1/lib/libhipblas.so"
    INTERFACE_INCLUDE_DIRECTORIES "/opt/rocm-7.2.1/include")
