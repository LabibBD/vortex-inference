#include "vortex/operators/fused.hpp"
#include "vortex/cuda_check.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

namespace vortex::ops {

namespace {

constexpr int TILE = 16;

void validate(
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
            "Fused Linear+GELU requires CUDA tensors"
        );
    }

    if (
        input.dtype() != DataType::Float32 ||
        weight.dtype() != DataType::Float32 ||
        bias.dtype() != DataType::Float32 ||
        output.dtype() != DataType::Float32
    ) {
        throw std::invalid_argument(
            "Fused Linear+GELU currently supports FP32 only"
        );
    }

    if (
        input.shape().size() != 2 ||
        weight.shape().size() != 2 ||
        bias.shape().size() != 1 ||
        output.shape().size() != 2
    ) {
        throw std::invalid_argument(
            "Invalid tensor ranks for fused Linear+GELU"
        );
    }

    const std::size_t batch =
        input.shape()[0];

    const std::size_t in_features =
        input.shape()[1];

    const std::size_t out_features =
        weight.shape()[1];

    if (
        weight.shape()[0] != in_features ||
        bias.shape()[0] != out_features ||
        output.shape()[0] != batch ||
        output.shape()[1] != out_features
    ) {
        throw std::invalid_argument(
            "Invalid dimensions for fused Linear+GELU"
        );
    }
}

__device__ __forceinline__
float gelu_device(
    float x
) {
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

    return
        0.5f *
        x *
        (
            1.0f +
            tanhf(inner)
        );
}

__global__
void linear_gelu_fused_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int m,
    int n,
    int k
) {
    __shared__ float tile_input[TILE][TILE];
    __shared__ float tile_weight[TILE][TILE];

    const int tx =
        threadIdx.x;

    const int ty =
        threadIdx.y;

    const int row =
        blockIdx.y * TILE + ty;

    const int col =
        blockIdx.x * TILE + tx;

    float accumulator =
        0.0f;

    const int tile_count =
        (k + TILE - 1) / TILE;

    for (
        int tile = 0;
        tile < tile_count;
        ++tile
    ) {
        const int input_col =
            tile * TILE + tx;

        const int weight_row =
            tile * TILE + ty;

        if (
            row < m &&
            input_col < k
        ) {
            tile_input[ty][tx] =
                input[
                    row * k +
                    input_col
                ];
        } else {
            tile_input[ty][tx] =
                0.0f;
        }

        if (
            weight_row < k &&
            col < n
        ) {
            tile_weight[ty][tx] =
                weight[
                    weight_row * n +
                    col
                ];
        } else {
            tile_weight[ty][tx] =
                0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (
            int i = 0;
            i < TILE;
            ++i
        ) {
            accumulator +=
                tile_input[ty][i] *
                tile_weight[i][tx];
        }

        __syncthreads();
    }

    if (
        row < m &&
        col < n
    ) {
        const float biased =
            accumulator +
            bias[col];

        output[
            row * n +
            col
        ] =
            gelu_device(
                biased
            );
    }
}

} // namespace

void linear_gelu_fused(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    Tensor& output
) {
    validate(
        input,
        weight,
        bias,
        output
    );

    const int m =
        static_cast<int>(
            input.shape()[0]
        );

    const int k =
        static_cast<int>(
            input.shape()[1]
        );

    const int n =
        static_cast<int>(
            weight.shape()[1]
        );

    const dim3 block(
        TILE,
        TILE
    );

    const dim3 grid(
        (n + TILE - 1) / TILE,
        (m + TILE - 1) / TILE
    );

    linear_gelu_fused_kernel<<<
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
        m,
        n,
        k
    );

    CUDA_KERNEL_CHECK();
}

} // namespace vortex::ops
