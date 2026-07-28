#pragma once

#include "vortex/tensor.hpp"

namespace vortex::ops {

/*
 * Computes Softmax(input * scale_factor)
 * along the final tensor dimension.
 *
 * Supported shapes:
 *
 * [columns]
 * [rows, columns]
 * [batch, rows, columns]
 */
void scaled_softmax(
    const Tensor& input,
    float scale_factor,
    Tensor& output
);

} // namespace vortex::ops
