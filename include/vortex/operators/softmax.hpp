#pragma once

#include "vortex/tensor.hpp"

namespace vortex::ops {

// Current baseline implementation:
// softmax over all elements of a 1D FP32 CUDA tensor.
void softmax(
    const Tensor& input,
    Tensor& output
);

} // namespace vortex::ops
