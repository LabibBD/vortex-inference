#pragma once

#include "vortex/tensor.hpp"

namespace vortex::numerics {

/*
 * Returns true only when every tensor element is finite.
 *
 * Supported:
 * - CUDA FP32
 * - CUDA FP16
 */
bool all_finite(
    const Tensor& tensor
);

} // namespace vortex::numerics
