#include "vortex/operators/activations.hpp"
#include "vortex/cuda_check.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>

namespace vortex::ops {

namespace {

constexpr int THREADS_PER_BLOCK = 256;

void validate_unary_fp32(
    const Tensor& input,
    const Tensor& output
) {
    if (
        input.device() != Device::CUDA ||
        output.device() != Device::CUDA
    ) {
        throw std::invalid_argument(
            "VORTEX activations require CUDA tensors"
        );
    }

    if (
        input.dtype() != DataType::Float32 ||
        output.dtype() != DataType::Float32
    ) {
        throw std::invalid_argument(
            "FP32 activation received non-FP32 tensor"
        );
    }

    if (input.shape() != output.shape()) {
        throw std::invalid_argument(
            "Activation input/output shapes must match"
        );
    }
}

void validate_unary_fp16(
    const Tensor& input,
    const Tensor& output
) {
    if (
        input.device() != Device::CUDA ||
        output.device() != Device::CUDA
    ) {
        throw std::invalid_argument(
            "VORTEX FP16 activations require CUDA tensors"
        );
    }

    if (
        input.dtype() != DataType::Float16 ||
        output.dtype() != DataType::Float16
    ) {
        throw std::invalid_argument(
            "FP16 activation received non-FP16 tensor"
        );
    }

    if (input.shape() != output.shape()) {
        throw std::invalid_argument(
            "Activation input/output shapes must match"
        );
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

// ============================================================
// FP32 kernels
// ============================================================

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

        const float inner =
            SQRT_2_OVER_PI *
            (
                x +
                GELU_COEFF *
                x * x * x
            );

        output[i] =
            0.5f *
            x *
            (1.0f + tanhf(inner));
    }
}

// ============================================================
// FP16 kernels
// ============================================================

__global__ void relu_fp16_kernel(
    const __half* input,
    __half* output,
    std::size_t n
) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) *
        blockDim.x +
        threadIdx.x;

    if (i < n) {
        const __half zero =
            __float2half(0.0f);

        output[i] =
            __hgt(input[i], zero)
                ? input[i]
                : zero;
    }
}

__global__ void gelu_fp16_kernel(
    const __half* input,
    __half* output,
    std::size_t n
) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) *
        blockDim.x +
        threadIdx.x;

    if (i < n) {
        // Convert to FP32 for transcendental math,
        // then quantize result back to FP16.
        const float x =
            __half2float(input[i]);

        constexpr float SQRT_2_OVER_PI =
            0.7978845608028654f;

        constexpr float GELU_COEFF =
            0.044715f;

        const float inner =
            SQRT_2_OVER_PI *
            (
                x +
                GELU_COEFF *
                x * x * x
            );

        const float result =
            0.5f *
            x *
            (1.0f + tanhf(inner));

        output[i] =
            __float2half(result);
    }
}

} // namespace

// ============================================================
// FP32 API
// ============================================================

void relu(
    const Tensor& input,
    Tensor& output
) {
    validate_unary_fp32(
        input,
        output
    );

    relu_kernel<<<
        block_count(input.numel()),
        THREADS_PER_BLOCK
    >>>(
        static_cast<const float*>(
            input.data()
        ),
        static_cast<float*>(
            output.data()
        ),
        input.numel()
    );

    CUDA_KERNEL_CHECK();
}

void sigmoid(
    const Tensor& input,
    Tensor& output
) {
    validate_unary_fp32(
        input,
        output
    );

    sigmoid_kernel<<<
        block_count(input.numel()),
        THREADS_PER_BLOCK
    >>>(
        static_cast<const float*>(
            input.data()
        ),
        static_cast<float*>(
            output.data()
        ),
        input.numel()
    );

    CUDA_KERNEL_CHECK();
}

void gelu(
    const Tensor& input,
    Tensor& output
) {
    validate_unary_fp32(
        input,
        output
    );

    gelu_kernel<<<
        block_count(input.numel()),
        THREADS_PER_BLOCK
    >>>(
        static_cast<const float*>(
            input.data()
        ),
        static_cast<float*>(
            output.data()
        ),
        input.numel()
    );

    CUDA_KERNEL_CHECK();
}

// ============================================================
// FP16 API
// ============================================================

void relu_fp16(
    const Tensor& input,
    Tensor& output
) {
    validate_unary_fp16(
        input,
        output
    );

    relu_fp16_kernel<<<
        block_count(input.numel()),
        THREADS_PER_BLOCK
    >>>(
        static_cast<const __half*>(
            input.data()
        ),
        static_cast<__half*>(
            output.data()
        ),
        input.numel()
    );

    CUDA_KERNEL_CHECK();
}

void gelu_fp16(
    const Tensor& input,
    Tensor& output
) {
    validate_unary_fp16(
        input,
        output
    );

    gelu_fp16_kernel<<<
        block_count(input.numel()),
        THREADS_PER_BLOCK
    >>>(
        static_cast<const __half*>(
            input.data()
        ),
        static_cast<__half*>(
            output.data()
        ),
        input.numel()
    );

    CUDA_KERNEL_CHECK();
}


void relu_async(
    const Tensor& input,
    Tensor& output,
    cudaStream_t stream
) {
    validate_unary_fp32(
        input,
        output
    );

    relu_kernel<<<
        block_count(input.numel()),
        THREADS_PER_BLOCK,
        0,
        stream
    >>>(
        static_cast<const float*>(
            input.data()
        ),
        static_cast<float*>(
            output.data()
        ),
        input.numel()
    );

    CUDA_KERNEL_LAUNCH_CHECK();
}

void gelu_async(
    const Tensor& input,
    Tensor& output,
    cudaStream_t stream
) {
    validate_unary_fp32(
        input,
        output
    );

    gelu_kernel<<<
        block_count(input.numel()),
        THREADS_PER_BLOCK,
        0,
        stream
    >>>(
        static_cast<const float*>(
            input.data()
        ),
        static_cast<float*>(
            output.data()
        ),
        input.numel()
    );

    CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace vortex::ops
