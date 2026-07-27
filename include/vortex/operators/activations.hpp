#pragma once

#include "vortex/tensor.hpp"

namespace vortex::ops {

void relu(
    const Tensor& input,
    Tensor& output
);

void sigmoid(
    const Tensor& input,
    Tensor& output
);

void gelu(
    const Tensor& input,
    Tensor& output
);

} // namespace vortex::ops
