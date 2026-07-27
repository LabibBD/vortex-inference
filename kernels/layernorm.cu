#include "vortex/operators/layernorm.hpp"
#include "vortex/cuda_check.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>

namespace vortex::ops {

namespace {

constexpr int THREADS = 256;

void validate_layernorm(
    const Tensor& input,
    const Tensor& gamma,
    const Tensor& beta,
    const Tensor& output
) {
    if (
        input.device() != Device::CUDA ||
        gamma.device() != Device::CUDA ||
        beta.device() != Device::CUDA ||
        output.device() != Device::CUDA
    ) {
        throw std::invalid_argument(
            "LayerNorm requires CUDA tensors"
        );
    }

    if (
        input.dtype() != DataType::Float32 ||
        gamma.dtype() != DataType::Float32 ||
        beta.dtype() != DataType::Float32 ||
        output.dtype() != DataType::Float32
    ) {
        throw std::invalid_argument(
            "LayerNorm currently supports FP32 only"
        );
    }

    if (
        input.shape().size() != 2 ||
        output.shape().size() != 2 ||
        gamma.shape().size() != 1 ||
        beta.shape().size() != 1
    ) {
        throw std::invalid_argument(
            "LayerNorm expects 2D input/output and 1D gamma/beta"
        );
    }

    if (input.shape() != output.shape()) {
        throw std::invalid_argument(
            "LayerNorm input/output shapes must match"
        );
    }

    const std::size_t features =
        input.shape()[1];

    if (
        gamma.shape()[0] != features ||
        beta.shape()[0] != features
    ) {
        throw std::invalid_argument(
            "Gamma and beta must match feature dimension"
        );
    }
}

__global__ void layernorm_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int rows,
    int features,
    float epsilon
) {
    extern __shared__ float shared[];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= rows) {
        return;
    }

    const float* row_input =
        input + row * features;

    float* row_output =
        output + row * features;

    // ---------------------------------------------
    // Mean
    // ---------------------------------------------

    float local_sum = 0.0f;

    for (
        int col = tid;
        col < features;
        col += blockDim.x
    ) {
        local_sum += row_input[col];
    }

    shared[tid] = local_sum;

    __syncthreads();

    for (
        int stride = blockDim.x / 2;
        stride > 0;
        stride >>= 1
    ) {
        if (tid < stride) {
            shared[tid] +=
                shared[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        shared[0] /=
            static_cast<float>(features);
    }

    __syncthreads();

    const float mean =
        shared[0];

    // ---------------------------------------------
    // Variance
    // ---------------------------------------------

    float local_variance = 0.0f;

    for (
        int col = tid;
        col < features;
        col += blockDim.x
    ) {
        const float difference =
            row_input[col] - mean;

        local_variance +=
            difference * difference;
    }

    shared[tid] =
        local_variance;

    __syncthreads();

    for (
        int stride = blockDim.x / 2;
        stride > 0;
        stride >>= 1
    ) {
        if (tid < stride) {
            shared[tid] +=
                shared[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        const float variance =
            shared[0] /
            static_cast<float>(features);

        shared[0] =
            rsqrtf(
                variance + epsilon
            );
    }

    __syncthreads();

    const float inverse_std =
        shared[0];

    // ---------------------------------------------
    // Normalize + affine transform
    // ---------------------------------------------

    for (
        int col = tid;
        col < features;
        col += blockDim.x
    ) {
        const float normalized =
            (row_input[col] - mean) *
            inverse_std;

        row_output[col] =
            normalized *
            gamma[col] +
            beta[col];
    }
}

} // namespace

void layernorm(
    const Tensor& input,
    const Tensor& gamma,
    const Tensor& beta,
    Tensor& output,
    float epsilon
) {
    validate_layernorm(
        input,
        gamma,
        beta,
        output
    );

    if (epsilon <= 0.0f) {
        throw std::invalid_argument(
            "LayerNorm epsilon must be positive"
        );
    }

    const int rows =
        static_cast<int>(
            input.shape()[0]
        );

    const int features =
        static_cast<int>(
            input.shape()[1]
        );

    layernorm_kernel<<<
        rows,
        THREADS,
        THREADS * sizeof(float)
    >>>(
        static_cast<const float*>(input.data()),
        static_cast<const float*>(gamma.data()),
        static_cast<const float*>(beta.data()),
        static_cast<float*>(output.data()),
        rows,
        features,
        epsilon
    );

    CUDA_KERNEL_CHECK();
}

} // namespace vortex::ops
