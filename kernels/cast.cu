#include "vortex/operators/cast.hpp"
#include "vortex/cuda_check.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>

namespace vortex::ops {

namespace {

constexpr int THREADS = 256;

void validate_common(
    const Tensor& input,
    const Tensor& output
) {
    if (
        input.device() != Device::CUDA ||
        output.device() != Device::CUDA
    ) {
        throw std::invalid_argument(
            "VORTEX cast operators require CUDA tensors"
        );
    }

    if (input.shape() != output.shape()) {
        throw std::invalid_argument(
            "Cast input/output shapes must match"
        );
    }
}

__global__ void fp32_to_fp16_kernel(
    const float* input,
    __half* output,
    std::size_t n
) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) *
        blockDim.x +
        threadIdx.x;

    if (i < n) {
        output[i] =
            __float2half(input[i]);
    }
}

__global__ void fp16_to_fp32_kernel(
    const __half* input,
    float* output,
    std::size_t n
) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) *
        blockDim.x +
        threadIdx.x;

    if (i < n) {
        output[i] =
            __half2float(input[i]);
    }
}

int blocks_for(
    std::size_t n
) {
    return static_cast<int>(
        (n + THREADS - 1) /
        THREADS
    );
}

} // namespace

void float32_to_float16(
    const Tensor& input,
    Tensor& output
) {
    validate_common(
        input,
        output
    );

    if (
        input.dtype() != DataType::Float32 ||
        output.dtype() != DataType::Float16
    ) {
        throw std::invalid_argument(
            "float32_to_float16 requires FP32 -> FP16"
        );
    }

    fp32_to_fp16_kernel<<<
        blocks_for(input.numel()),
        THREADS
    >>>(
        static_cast<const float*>(
            input.data()
        ),
        static_cast<__half*>(
            output.data()
        ),
        input.numel()
    );

    CUDA_KERNEL_CHECK();
}

void float16_to_float32(
    const Tensor& input,
    Tensor& output
) {
    validate_common(
        input,
        output
    );

    if (
        input.dtype() != DataType::Float16 ||
        output.dtype() != DataType::Float32
    ) {
        throw std::invalid_argument(
            "float16_to_float32 requires FP16 -> FP32"
        );
    }

    fp16_to_fp32_kernel<<<
        blocks_for(input.numel()),
        THREADS
    >>>(
        static_cast<const __half*>(
            input.data()
        ),
        static_cast<float*>(
            output.data()
        ),
        input.numel()
    );

    CUDA_KERNEL_CHECK();
}

} // namespace vortex::ops
