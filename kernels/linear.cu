#include "vortex/operators/linear.hpp"

#include "vortex/cuda_check.hpp"
#include "vortex/operators/matmul.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>

namespace vortex::ops {

namespace {

constexpr int THREADS = 256;

void validate_bias_add(
    const Tensor& input,
    const Tensor& bias,
    const Tensor& output
) {
    if (
        input.device() != Device::CUDA ||
        bias.device() != Device::CUDA ||
        output.device() != Device::CUDA
    ) {
        throw std::invalid_argument(
            "BiasAdd requires CUDA tensors"
        );
    }

    if (
        input.dtype() != DataType::Float32 ||
        bias.dtype() != DataType::Float32 ||
        output.dtype() != DataType::Float32
    ) {
        throw std::invalid_argument(
            "BiasAdd currently supports FP32 only"
        );
    }

    if (
        input.shape().size() != 2 ||
        bias.shape().size() != 1 ||
        output.shape().size() != 2
    ) {
        throw std::invalid_argument(
            "BiasAdd expects input/output 2D and bias 1D"
        );
    }

    if (input.shape() != output.shape()) {
        throw std::invalid_argument(
            "BiasAdd input/output shapes must match"
        );
    }

    if (bias.shape()[0] != input.shape()[1]) {
        throw std::invalid_argument(
            "Bias size must match final tensor dimension"
        );
    }
}

__global__ void bias_add_kernel(
    const float* input,
    const float* bias,
    float* output,
    std::size_t total,
    std::size_t columns
) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) *
        blockDim.x +
        threadIdx.x;

    if (index < total) {
        const std::size_t column =
            index % columns;

        output[index] =
            input[index] + bias[column];
    }
}

} // namespace

void bias_add(
    const Tensor& input,
    const Tensor& bias,
    Tensor& output
) {
    validate_bias_add(
        input,
        bias,
        output
    );

    const std::size_t total =
        input.numel();

    const std::size_t columns =
        input.shape()[1];

    const int blocks =
        static_cast<int>(
            (total + THREADS - 1) /
            THREADS
        );

    bias_add_kernel<<<blocks, THREADS>>>(
        static_cast<const float*>(input.data()),
        static_cast<const float*>(bias.data()),
        static_cast<float*>(output.data()),
        total,
        columns
    );

    CUDA_KERNEL_CHECK();
}

void linear(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    Tensor& output
) {
    if (
        input.shape().size() != 2 ||
        weight.shape().size() != 2 ||
        bias.shape().size() != 1 ||
        output.shape().size() != 2
    ) {
        throw std::invalid_argument(
            "Linear received invalid tensor ranks"
        );
    }

    const std::size_t batch =
        input.shape()[0];

    const std::size_t input_features =
        input.shape()[1];

    const std::size_t output_features =
        weight.shape()[1];

    if (
        weight.shape()[0] != input_features ||
        bias.shape()[0] != output_features ||
        output.shape()[0] != batch ||
        output.shape()[1] != output_features
    ) {
        throw std::invalid_argument(
            "Linear tensor dimensions are incompatible"
        );
    }

    Tensor matmul_output(
        {batch, output_features},
        DataType::Float32,
        Device::CUDA
    );

    // Our current optimized custom GEMM.
    matmul_tiled(
        input,
        weight,
        matmul_output
    );

    bias_add(
        matmul_output,
        bias,
        output
    );
}

} // namespace vortex::ops
