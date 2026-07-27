#include "vortex/operators/elementwise.hpp"
#include "vortex/cuda_check.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

namespace vortex::ops {

namespace {

constexpr int THREADS_PER_BLOCK = 256;

void validate_binary_operator(
    const Tensor& a,
    const Tensor& b,
    const Tensor& output
) {
    if (
        a.device() != Device::CUDA ||
        b.device() != Device::CUDA ||
        output.device() != Device::CUDA
    ) {
        throw std::invalid_argument(
            "VORTEX CUDA operators require CUDA tensors"
        );
    }

    if (
        a.dtype() != DataType::Float32 ||
        b.dtype() != DataType::Float32 ||
        output.dtype() != DataType::Float32
    ) {
        throw std::invalid_argument(
            "Elementwise operators currently support FP32 only"
        );
    }

    if (
        a.shape() != b.shape() ||
        a.shape() != output.shape()
    ) {
        throw std::invalid_argument(
            "Elementwise tensor shapes must match"
        );
    }
}

__global__ void add_kernel(
    const float* a,
    const float* b,
    float* output,
    std::size_t n
) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) *
        blockDim.x +
        threadIdx.x;

    if (i < n) {
        output[i] = a[i] + b[i];
    }
}

__global__ void multiply_kernel(
    const float* a,
    const float* b,
    float* output,
    std::size_t n
) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) *
        blockDim.x +
        threadIdx.x;

    if (i < n) {
        output[i] = a[i] * b[i];
    }
}

} // namespace

void add(
    const Tensor& a,
    const Tensor& b,
    Tensor& output
) {
    validate_binary_operator(a, b, output);

    const std::size_t n = a.numel();

    const int blocks =
        static_cast<int>(
            (n + THREADS_PER_BLOCK - 1) /
            THREADS_PER_BLOCK
        );

    add_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        static_cast<const float*>(a.data()),
        static_cast<const float*>(b.data()),
        static_cast<float*>(output.data()),
        n
    );

    CUDA_KERNEL_CHECK();
}

void multiply(
    const Tensor& a,
    const Tensor& b,
    Tensor& output
) {
    validate_binary_operator(a, b, output);

    const std::size_t n = a.numel();

    const int blocks =
        static_cast<int>(
            (n + THREADS_PER_BLOCK - 1) /
            THREADS_PER_BLOCK
        );

    multiply_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        static_cast<const float*>(a.data()),
        static_cast<const float*>(b.data()),
        static_cast<float*>(output.data()),
        n
    );

    CUDA_KERNEL_CHECK();
}

} // namespace vortex::ops
