#include "vortex/operators/transformer.hpp"
#include "vortex/cuda_check.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace vortex::ops {

namespace {

constexpr int THREADS = 256;
constexpr int TILE = 16;

void validate_cuda_fp32(
    const Tensor& tensor,
    const char* operation
) {
    if (tensor.device() != Device::CUDA) {
        throw std::invalid_argument(
            std::string(operation) +
            " requires CUDA tensors"
        );
    }

    if (tensor.dtype() != DataType::Float32) {
        throw std::invalid_argument(
            std::string(operation) +
            " currently supports FP32 only"
        );
    }
}

int blocks_for(
    std::size_t elements
) {
    return static_cast<int>(
        (elements + THREADS - 1) /
        THREADS
    );
}

// ============================================================
// Transpose
// ============================================================

__global__ void transpose_last_two_kernel(
    const float* input,
    float* output,
    std::size_t batch,
    std::size_t rows,
    std::size_t columns
) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) *
        blockDim.x +
        threadIdx.x;

    const std::size_t total =
        batch * rows * columns;

    if (index >= total) {
        return;
    }

    const std::size_t column =
        index % columns;

    const std::size_t row =
        (index / columns) % rows;

    const std::size_t batch_index =
        index / (rows * columns);

    const std::size_t output_index =
        batch_index *
            columns *
            rows +
        column *
            rows +
        row;

    output[output_index] =
        input[index];
}

// ============================================================
// Scale
// ============================================================

__global__ void scale_kernel(
    const float* input,
    float* output,
    float factor,
    std::size_t elements
) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) *
        blockDim.x +
        threadIdx.x;

    if (index < elements) {
        output[index] =
            input[index] *
            factor;
    }
}

// ============================================================
// Batched tiled matrix multiplication
// ============================================================

__global__ void batched_matmul_kernel(
    const float* a,
    const float* b,
    float* output,
    int batch_count,
    int m,
    int n,
    int k
) {
    __shared__ float tile_a[TILE][TILE];
    __shared__ float tile_b[TILE][TILE];

    const int batch =
        blockIdx.z;

    const int row =
        blockIdx.y * TILE +
        threadIdx.y;

    const int column =
        blockIdx.x * TILE +
        threadIdx.x;

    const int local_x =
        threadIdx.x;

    const int local_y =
        threadIdx.y;

    if (batch >= batch_count) {
        return;
    }

    float accumulator =
        0.0f;

    const int tile_count =
        (k + TILE - 1) /
        TILE;

    for (
        int tile = 0;
        tile < tile_count;
        ++tile
    ) {
        const int a_column =
            tile * TILE +
            local_x;

        const int b_row =
            tile * TILE +
            local_y;

        if (
            row < m &&
            a_column < k
        ) {
            tile_a[local_y][local_x] =
                a[
                    batch * m * k +
                    row * k +
                    a_column
                ];
        } else {
            tile_a[local_y][local_x] =
                0.0f;
        }

        if (
            b_row < k &&
            column < n
        ) {
            tile_b[local_y][local_x] =
                b[
                    batch * k * n +
                    b_row * n +
                    column
                ];
        } else {
            tile_b[local_y][local_x] =
                0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (
            int inner = 0;
            inner < TILE;
            ++inner
        ) {
            accumulator +=
                tile_a[local_y][inner] *
                tile_b[inner][local_x];
        }

        __syncthreads();
    }

    if (
        row < m &&
        column < n
    ) {
        output[
            batch * m * n +
            row * n +
            column
        ] =
            accumulator;
    }
}

} // namespace

void transpose_last_two(
    const Tensor& input,
    Tensor& output
) {
    validate_cuda_fp32(
        input,
        "Transpose"
    );

    validate_cuda_fp32(
        output,
        "Transpose"
    );

    if (
        input.shape().size() != 2 &&
        input.shape().size() != 3
    ) {
        throw std::invalid_argument(
            "Transpose supports rank-2 or rank-3 tensors"
        );
    }

    if (
        output.shape().size() !=
        input.shape().size()
    ) {
        throw std::invalid_argument(
            "Transpose input/output ranks must match"
        );
    }

    std::size_t batch = 1;
    std::size_t rows = 0;
    std::size_t columns = 0;

    if (input.shape().size() == 2) {
        rows =
            input.shape()[0];

        columns =
            input.shape()[1];

        if (
            output.shape()[0] != columns ||
            output.shape()[1] != rows
        ) {
            throw std::invalid_argument(
                "Incorrect rank-2 transpose output shape"
            );
        }
    } else {
        batch =
            input.shape()[0];

        rows =
            input.shape()[1];

        columns =
            input.shape()[2];

        if (
            output.shape()[0] != batch ||
            output.shape()[1] != columns ||
            output.shape()[2] != rows
        ) {
            throw std::invalid_argument(
                "Incorrect rank-3 transpose output shape"
            );
        }
    }

    transpose_last_two_kernel<<<
        blocks_for(input.numel()),
        THREADS
    >>>(
        static_cast<const float*>(
            input.data()
        ),
        static_cast<float*>(
            output.data()
        ),
        batch,
        rows,
        columns
    );

    CUDA_KERNEL_CHECK();
}

void scale(
    const Tensor& input,
    float factor,
    Tensor& output
) {
    validate_cuda_fp32(
        input,
        "Scale"
    );

    validate_cuda_fp32(
        output,
        "Scale"
    );

    if (input.shape() != output.shape()) {
        throw std::invalid_argument(
            "Scale input/output shapes must match"
        );
    }

    scale_kernel<<<
        blocks_for(input.numel()),
        THREADS
    >>>(
        static_cast<const float*>(
            input.data()
        ),
        static_cast<float*>(
            output.data()
        ),
        factor,
        input.numel()
    );

    CUDA_KERNEL_CHECK();
}

void batched_matmul(
    const Tensor& a,
    const Tensor& b,
    Tensor& output
) {
    validate_cuda_fp32(
        a,
        "BatchedMatMul"
    );

    validate_cuda_fp32(
        b,
        "BatchedMatMul"
    );

    validate_cuda_fp32(
        output,
        "BatchedMatMul"
    );

    if (
        a.shape().size() != 3 ||
        b.shape().size() != 3 ||
        output.shape().size() != 3
    ) {
        throw std::invalid_argument(
            "BatchedMatMul requires rank-3 tensors"
        );
    }

    const std::size_t batch =
        a.shape()[0];

    const std::size_t m =
        a.shape()[1];

    const std::size_t k =
        a.shape()[2];

    const std::size_t n =
        b.shape()[2];

    if (
        b.shape()[0] != batch ||
        b.shape()[1] != k ||
        output.shape()[0] != batch ||
        output.shape()[1] != m ||
        output.shape()[2] != n
    ) {
        throw std::invalid_argument(
            "BatchedMatMul dimensions are incompatible"
        );
    }

    const dim3 block(
        TILE,
        TILE
    );

    const dim3 grid(
        static_cast<unsigned int>(
            (n + TILE - 1) /
            TILE
        ),
        static_cast<unsigned int>(
            (m + TILE - 1) /
            TILE
        ),
        static_cast<unsigned int>(
            batch
        )
    );

    batched_matmul_kernel<<<
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
        static_cast<int>(batch),
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k)
    );

    CUDA_KERNEL_CHECK();
}

} // namespace vortex::ops
