#pragma once

#include "vortex/tensor.hpp"

namespace vortex::ops {

void linear_gelu_fused(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    Tensor& output
);

} // namespace vortex::ops
