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


namespace {

constexpr int LINEAR_3D_TILE = 16;

__global__ void linear_3d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int rows,
    int input_features,
    int output_features
) {
    __shared__ float input_tile
        [LINEAR_3D_TILE]
        [LINEAR_3D_TILE];

    __shared__ float weight_tile
        [LINEAR_3D_TILE]
        [LINEAR_3D_TILE];

    const int local_x =
        threadIdx.x;

    const int local_y =
        threadIdx.y;

    const int row =
        blockIdx.y *
        LINEAR_3D_TILE +
        local_y;

    const int column =
        blockIdx.x *
        LINEAR_3D_TILE +
        local_x;

    float accumulator = 0.0f;

    const int tile_count =
        (
            input_features +
            LINEAR_3D_TILE -
            1
        ) /
        LINEAR_3D_TILE;

    for (
        int tile = 0;
        tile < tile_count;
        ++tile
    ) {
        const int input_column =
            tile *
            LINEAR_3D_TILE +
            local_x;

        const int weight_row =
            tile *
            LINEAR_3D_TILE +
            local_y;

        if (
            row < rows &&
            input_column <
                input_features
        ) {
            input_tile
                [local_y]
                [local_x] =
                    input[
                        row *
                        input_features +
                        input_column
                    ];
        } else {
            input_tile
                [local_y]
                [local_x] =
                    0.0f;
        }

        if (
            weight_row <
                input_features &&
            column <
                output_features
        ) {
            weight_tile
                [local_y]
                [local_x] =
                    weight[
                        weight_row *
                        output_features +
                        column
                    ];
        } else {
            weight_tile
                [local_y]
                [local_x] =
                    0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (
            int inner = 0;
            inner <
                LINEAR_3D_TILE;
            ++inner
        ) {
            accumulator +=
                input_tile
                    [local_y]
                    [inner] *
                weight_tile
                    [inner]
                    [local_x];
        }

        __syncthreads();
    }

    if (
        row < rows &&
        column <
            output_features
    ) {
        output[
            row *
            output_features +
            column
        ] =
            accumulator +
            bias[column];
    }
}

void validate_linear_3d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& output
) {
    if (
        input.device() != Device::CUDA ||
        weight.device() != Device::CUDA ||
        bias.device() != Device::CUDA ||
        output.device() != Device::CUDA
    ) {
        throw std::invalid_argument(
            "Linear3D requires CUDA tensors"
        );
    }

    if (
        input.dtype() != DataType::Float32 ||
        weight.dtype() != DataType::Float32 ||
        bias.dtype() != DataType::Float32 ||
        output.dtype() != DataType::Float32
    ) {
        throw std::invalid_argument(
            "Linear3D currently supports FP32 only"
        );
    }

    if (
        input.shape().size() != 3 ||
        weight.shape().size() != 2 ||
        bias.shape().size() != 1 ||
        output.shape().size() != 3
    ) {
        throw std::invalid_argument(
            "Linear3D received invalid tensor ranks"
        );
    }

    const std::size_t batch =
        input.shape()[0];

    const std::size_t tokens =
        input.shape()[1];

    const std::size_t input_features =
        input.shape()[2];

    const std::size_t output_features =
        weight.shape()[1];

    if (
        weight.shape()[0] != input_features ||
        bias.shape()[0] != output_features ||
        output.shape()[0] != batch ||
        output.shape()[1] != tokens ||
        output.shape()[2] != output_features
    ) {
        throw std::invalid_argument(
            "Linear3D dimensions are incompatible"
        );
    }
}

} // namespace

void linear_3d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    Tensor& output
) {
    validate_linear_3d(
        input,
        weight,
        bias,
        output
    );

    const int rows =
        static_cast<int>(
            input.shape()[0] *
            input.shape()[1]
        );

    const int input_features =
        static_cast<int>(
            input.shape()[2]
        );

    const int output_features =
        static_cast<int>(
            weight.shape()[1]
        );

    const dim3 block(
        LINEAR_3D_TILE,
        LINEAR_3D_TILE
    );

    const dim3 grid(
        (
            output_features +
            LINEAR_3D_TILE -
            1
        ) /
        LINEAR_3D_TILE,

        (
            rows +
            LINEAR_3D_TILE -
            1
        ) /
        LINEAR_3D_TILE
    );

    linear_3d_kernel<<<
        grid,
        block
    >>>(
        static_cast<const float*>(
            input.data()
        ),
        static_cast<const float*>(
            weight.data()
        ),
        static_cast<const float*>(
            bias.data()
        ),
        static_cast<float*>(
            output.data()
        ),
        rows,
        input_features,
        output_features
    );

    CUDA_KERNEL_CHECK();
}

} // namespace vortex::ops
