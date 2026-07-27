#pragma once

#include "vortex/tensor.hpp"

namespace vortex::ops {

void matmul_naive(
    const Tensor& a,
    const Tensor& b,
    Tensor& output
);

void matmul_tiled_8(
    const Tensor& a,
    const Tensor& b,
    Tensor& output
);

void matmul_tiled(
    const Tensor& a,
    const Tensor& b,
    Tensor& output
);

void matmul_tiled_32(
    const Tensor& a,
    const Tensor& b,
    Tensor& output
);

void matmul_cublas(
    const Tensor& a,
    const Tensor& b,
    Tensor& output
);


void matmul_cublas_fp16(
    const Tensor& a,
    const Tensor& b,
    Tensor& output
);

} // namespace vortex::ops
