#pragma once

#include <cstddef>

namespace miinfer {

void launch_vector_add(const float* left, const float* right, float* result, std::size_t count);
bool run_vector_add(std::size_t count, float value, float tolerance);

}  // namespace miinfer
