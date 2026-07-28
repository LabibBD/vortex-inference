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


/*
 * input:  [batch, tokens, input_features]
 * weight: [input_features, output_features]
 * bias:   [output_features]
 * output: [batch, tokens, output_features]
 */
void linear_3d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    Tensor& output
);

} // namespace vortex::ops
