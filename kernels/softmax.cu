#include "vortex/operators/softmax.hpp"
#include "vortex/cuda_check.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <cstddef>
#include <stdexcept>

namespace vortex::ops {

namespace {

constexpr int THREADS = 256;

void validate_softmax(
    const Tensor& input,
    const Tensor& output
) {
    if (
        input.device() != Device::CUDA ||
        output.device() != Device::CUDA
    ) {
        throw std::invalid_argument(
            "Softmax requires CUDA tensors"
        );
    }

    if (
        input.dtype() != DataType::Float32 ||
        output.dtype() != DataType::Float32
    ) {
        throw std::invalid_argument(
            "Softmax currently supports FP32 only"
        );
    }

    if (input.shape() != output.shape()) {
        throw std::invalid_argument(
            "Softmax input/output shapes must match"
        );
    }

    if (
        input.shape().size() != 1 &&
        input.shape().size() != 2
    ) {
        throw std::invalid_argument(
            "Softmax currently supports 1D or 2D tensors"
        );
    }
}

__global__ void rowwise_softmax_kernel(
    const float* input,
    float* output,
    int rows,
    int columns
) {
    extern __shared__ float shared[];

    const int row =
        blockIdx.x;

    const int tid =
        threadIdx.x;

    if (row >= rows) {
        return;
    }

    const float* row_input =
        input + row * columns;

    float* row_output =
        output + row * columns;

    // ==================================================
    // Row maximum
    // ==================================================

    float local_max =
        -FLT_MAX;

    for (
        int col = tid;
        col < columns;
        col += blockDim.x
    ) {
        local_max =
            fmaxf(
                local_max,
                row_input[col]
            );
    }

    shared[tid] =
        local_max;

    __syncthreads();

    for (
        int stride = blockDim.x / 2;
        stride > 0;
        stride >>= 1
    ) {
        if (tid < stride) {
            shared[tid] =
                fmaxf(
                    shared[tid],
                    shared[tid + stride]
                );
        }

        __syncthreads();
    }

    const float row_max =
        shared[0];

    // ==================================================
    // exp(x - max) sum
    // ==================================================

    float local_sum =
        0.0f;

    for (
        int col = tid;
        col < columns;
        col += blockDim.x
    ) {
        local_sum +=
            expf(
                row_input[col] -
                row_max
            );
    }

    shared[tid] =
        local_sum;

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

    const float inverse_sum =
        1.0f / shared[0];

    // ==================================================
    // Normalize
    // ==================================================

    for (
        int col = tid;
        col < columns;
        col += blockDim.x
    ) {
        row_output[col] =
            expf(
                row_input[col] -
                row_max
            ) *
            inverse_sum;
    }
}

} // namespace

void softmax(
    const Tensor& input,
    Tensor& output
) {
    validate_softmax(
        input,
        output
    );

    int rows = 1;
    int columns = 0;

    if (input.shape().size() == 1) {
        columns =
            static_cast<int>(
                input.shape()[0]
            );
    } else {
        rows =
            static_cast<int>(
                input.shape()[0]
            );

        columns =
            static_cast<int>(
                input.shape()[1]
            );
    }

    rowwise_softmax_kernel<<<
        rows,
        THREADS,
        THREADS * sizeof(float)
    >>>(
        static_cast<const float*>(
            input.data()
        ),
        static_cast<float*>(
            output.data()
        ),
        rows,
        columns
    );

    CUDA_KERNEL_CHECK();
}

} // namespace vortex::ops
