#pragma once

#include "vortex/tensor.hpp"

namespace vortex::ops {

void add(
    const Tensor& a,
    const Tensor& b,
    Tensor& output
);

void multiply(
    const Tensor& a,
    const Tensor& b,
    Tensor& output
);

} // namespace vortex::ops
