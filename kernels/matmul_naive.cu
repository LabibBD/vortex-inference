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
            "MatMul currently requires 2D tensors"
        );
    }

    const std::size_t m = a.shape()[0];
    const std::size_t k = a.shape()[1];

    const std::size_t b_k = b.shape()[0];
    const std::size_t n = b.shape()[1];

    if (k != b_k) {
        throw std::invalid_argument(
            "MatMul inner dimensions do not match"
        );
    }

    if (
        output.shape()[0] != m ||
        output.shape()[1] != n
    ) {
        throw std::invalid_argument(
            "MatMul output shape is incorrect"
        );
    }
}

__global__ void matmul_naive_kernel(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k
) {
    const int col =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    const int row =
        blockIdx.y * blockDim.y +
        threadIdx.y;

    if (
        row < m &&
        col < n
    ) {
        float sum = 0.0f;

        for (int i = 0; i < k; ++i) {
            sum +=
                a[row * k + i] *
                b[i * n + col];
        }

        c[row * n + col] =
            sum;
    }
}

} // namespace

void matmul_naive(
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
        static_cast<int>(
            a.shape()[0]
        );

    const int k =
        static_cast<int>(
            a.shape()[1]
        );

    const int n =
        static_cast<int>(
            b.shape()[1]
        );

    constexpr int TILE = 16;

    const dim3 block(
        TILE,
        TILE
    );

    const dim3 grid(
        (n + TILE - 1) / TILE,
        (m + TILE - 1) / TILE
    );

    matmul_naive_kernel<<<
        grid,
        block
    >>>(
        static_cast<const float*>(
            a.data()
        ),
        static_cast<const float*>(
            b.data()
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
