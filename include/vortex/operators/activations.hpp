#pragma once

#include "vortex/tensor.hpp"

#include <cuda_runtime.h>

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

// Native FP16 paths
void relu_fp16(
    const Tensor& input,
    Tensor& output
);

void gelu_fp16(
    const Tensor& input,
    Tensor& output
);


void relu_async(
    const Tensor& input,
    Tensor& output,
    cudaStream_t stream
);

void gelu_async(
    const Tensor& input,
    Tensor& output,
    cudaStream_t stream
);

} // namespace vortex::ops
