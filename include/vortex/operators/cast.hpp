#pragma once

#include "vortex/tensor.hpp"

namespace vortex::ops {

void float32_to_float16(
    const Tensor& input,
    Tensor& output
);

void float16_to_float32(
    const Tensor& input,
    Tensor& output
);

} // namespace vortex::ops
