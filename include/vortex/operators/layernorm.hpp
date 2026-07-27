#pragma once

#include "vortex/tensor.hpp"

namespace vortex::ops {

void layernorm(
    const Tensor& input,
    const Tensor& gamma,
    const Tensor& beta,
    Tensor& output,
    float epsilon = 1e-5f
);

} // namespace vortex::ops
