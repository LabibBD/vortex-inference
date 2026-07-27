#pragma once

#include "vortex/tensor.hpp"

namespace vortex::ops {

void bias_add(
    const Tensor& input,
    const Tensor& bias,
    Tensor& output
);

void linear(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    Tensor& output
);

} // namespace vortex::ops
