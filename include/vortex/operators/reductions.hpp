#pragma once

#include "vortex/tensor.hpp"

namespace vortex::ops {

void reduce_sum(
    const Tensor& input,
    Tensor& output
);

void reduce_max(
    const Tensor& input,
    Tensor& output
);

} // namespace vortex::ops
