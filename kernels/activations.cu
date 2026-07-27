#include "vortex/operators/activations.hpp"
#include "vortex/cuda_check.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>

namespace vortex::ops {

namespace {

constexpr int THREADS_PER_BLOCK = 256;

void validate_unary_operator(
    const Tensor& input,
    const Tensor& output
) {
    if (
        input.device() != Device::CUDA ||
        output.device() != Device::CUDA
    ) {
        throw std::invalid_argument(
            "VORTEX CUDA activations require CUDA tensors"
        );
    }

    if (
        input.dtype() != DataType::Float32 ||
        output.dtype() != DataType::Float32
    ) {
        throw std::invalid_argument(
            "Activations currently support FP32 only"
        );
    }

    if (input.shape() != output.shape()) {
        throw std::invalid_argument(
            "Input and output tensor shapes must match"
        );
    }
}

__global__ void relu_kernel(
    const float* input,
    float* output,
    std::size_t n
) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) *
        blockDim.x +
        threadIdx.x;

    if (i < n) {
        output[i] =
            input[i] > 0.0f
                ? input[i]
                : 0.0f;
    }
}

__global__ void sigmoid_kernel(
    const float* input,
    float* output,
    std::size_t n
) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) *
        blockDim.x +
        threadIdx.x;

    if (i < n) {
        output[i] =
            1.0f /
            (1.0f + expf(-input[i]));
    }
}

__global__ void gelu_kernel(
    const float* input,
    float* output,
    std::size_t n
) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) *
        blockDim.x +
        threadIdx.x;

    if (i < n) {
        const float x = input[i];

        constexpr float SQRT_2_OVER_PI =
            0.7978845608028654f;

        constexpr float GELU_COEFF =
            0.044715f;

        const float x3 =
            x * x * x;

        const float inner =
            SQRT_2_OVER_PI *
            (x + GELU_COEFF * x3);

        output[i] =
            0.5f *
            x *
            (1.0f + tanhf(inner));
    }
}

int block_count(
    std::size_t n
) {
    return static_cast<int>(
        (n + THREADS_PER_BLOCK - 1) /
        THREADS_PER_BLOCK
    );
}

} // namespace

void relu(
    const Tensor& input,
    Tensor& output
) {
    validate_unary_operator(
        input,
        output
    );

    relu_kernel<<<
        block_count(input.numel()),
        THREADS_PER_BLOCK
    >>>(
        static_cast<const float*>(input.data()),
        static_cast<float*>(output.data()),
        input.numel()
    );

    CUDA_KERNEL_CHECK();
}

void sigmoid(
    const Tensor& input,
    Tensor& output
) {
    validate_unary_operator(
        input,
        output
    );

    sigmoid_kernel<<<
        block_count(input.numel()),
        THREADS_PER_BLOCK
    >>>(
        static_cast<const float*>(input.data()),
        static_cast<float*>(output.data()),
        input.numel()
    );

    CUDA_KERNEL_CHECK();
}

void gelu(
    const Tensor& input,
    Tensor& output
) {
    validate_unary_operator(
        input,
        output
    );

    gelu_kernel<<<
        block_count(input.numel()),
        THREADS_PER_BLOCK
    >>>(
        static_cast<const float*>(input.data()),
        static_cast<float*>(output.data()),
        input.numel()
    );

    CUDA_KERNEL_CHECK();
}

} // namespace vortex::ops
