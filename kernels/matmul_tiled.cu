#include "vortex/operators/matmul.hpp"
#include "vortex/cuda_check.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>

namespace vortex::ops {

namespace {

void validate_matmul(
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
            "VORTEX MatMul requires CUDA tensors"
        );
    }

    if (
        a.dtype() != DataType::Float32 ||
        b.dtype() != DataType::Float32 ||
        output.dtype() != DataType::Float32
    ) {
        throw std::invalid_argument(
            "MatMul currently supports FP32 only"
        );
    }

    if (
        a.shape().size() != 2 ||
        b.shape().size() != 2 ||
        output.shape().size() != 2
    ) {
        throw std::invalid_argument(
            "MatMul requires 2D tensors"
        );
    }

    if (a.shape()[1] != b.shape()[0]) {
        throw std::invalid_argument(
            "MatMul inner dimensions do not match"
        );
    }

    if (
        output.shape()[0] != a.shape()[0] ||
        output.shape()[1] != b.shape()[1]
    ) {
        throw std::invalid_argument(
            "MatMul output shape is incorrect"
        );
    }
}

template <int TILE>
__global__ void matmul_tiled_kernel(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k
) {
    __shared__ float tile_a[TILE][TILE];
    __shared__ float tile_b[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row =
        blockIdx.y * TILE + ty;

    const int col =
        blockIdx.x * TILE + tx;

    float sum = 0.0f;

    const int tile_count =
        (k + TILE - 1) / TILE;

    for (int tile = 0; tile < tile_count; ++tile) {

        const int a_col =
            tile * TILE + tx;

        const int b_row =
            tile * TILE + ty;

        if (row < m && a_col < k) {
            tile_a[ty][tx] =
                a[row * k + a_col];
        } else {
            tile_a[ty][tx] = 0.0f;
        }

        if (b_row < k && col < n) {
            tile_b[ty][tx] =
                b[b_row * n + col];
        } else {
            tile_b[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            sum +=
                tile_a[ty][i] *
                tile_b[i][tx];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = sum;
    }
}

template <int TILE>
void launch_tiled(
    const Tensor& a,
    const Tensor& b,
    Tensor& output
) {
    validate_matmul(
        a,
        b,
        output
    );

    const int m =
        static_cast<int>(a.shape()[0]);

    const int k =
        static_cast<int>(a.shape()[1]);

    const int n =
        static_cast<int>(b.shape()[1]);

    const dim3 block(
        TILE,
        TILE
    );

    const dim3 grid(
        (n + TILE - 1) / TILE,
        (m + TILE - 1) / TILE
    );

    matmul_tiled_kernel<TILE><<<
        grid,
        block
    >>>(
        static_cast<const float*>(a.data()),
        static_cast<const float*>(b.data()),
        static_cast<float*>(output.data()),
        m,
        n,
        k
    );

    CUDA_KERNEL_CHECK();
}

} // namespace

void matmul_tiled_8(
    const Tensor& a,
    const Tensor& b,
    Tensor& output
) {
    launch_tiled<8>(
        a,
        b,
        output
    );
}

void matmul_tiled(
    const Tensor& a,
    const Tensor& b,
    Tensor& output
) {
    launch_tiled<16>(
        a,
        b,
        output
    );
}

void matmul_tiled_32(
    const Tensor& a,
    const Tensor& b,
    Tensor& output
) {
    launch_tiled<32>(
        a,
        b,
        output
    );
}

} // namespace vortex::ops
