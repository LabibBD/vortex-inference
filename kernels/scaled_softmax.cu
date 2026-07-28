#include "vortex/operators/scaled_softmax.hpp"

#include "vortex/cuda_check.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <cstddef>
#include <stdexcept>

namespace vortex::ops {

namespace {

constexpr int SOFTMAX_THREADS =
    256;

__global__ void scaled_softmax_kernel(
    const float* input,
    float* output,
    float scale_factor,
    int rows,
    int columns
) {
    extern __shared__ float shared[];

    const int row =
        static_cast<int>(
            blockIdx.x
        );

    const int thread =
        static_cast<int>(
            threadIdx.x
        );

    if (row >= rows) {
        return;
    }

    const std::size_t row_offset =
        static_cast<std::size_t>(
            row
        ) *
        static_cast<std::size_t>(
            columns
        );

    // ==================================================
    // Find maximum of the scaled row.
    // ==================================================

    float local_max =
        -FLT_MAX;

    for (
        int column = thread;
        column < columns;
        column += blockDim.x
    ) {
        const float scaled =
            input[
                row_offset +
                static_cast<std::size_t>(
                    column
                )
            ] *
            scale_factor;

        local_max =
            fmaxf(
                local_max,
                scaled
            );
    }

    shared[thread] =
        local_max;

    __syncthreads();

    for (
        int stride =
            blockDim.x / 2;
        stride > 0;
        stride >>= 1
    ) {
        if (thread < stride) {
            shared[thread] =
                fmaxf(
                    shared[thread],
                    shared[
                        thread +
                        stride
                    ]
                );
        }

        __syncthreads();
    }

    const float row_max =
        shared[0];

    // ==================================================
    // Calculate the exponential sum.
    // ==================================================

    float local_sum =
        0.0f;

    for (
        int column = thread;
        column < columns;
        column += blockDim.x
    ) {
        const float scaled =
            input[
                row_offset +
                static_cast<std::size_t>(
                    column
                )
            ] *
            scale_factor;

        local_sum +=
            expf(
                scaled -
                row_max
            );
    }

    shared[thread] =
        local_sum;

    __syncthreads();

    for (
        int stride =
            blockDim.x / 2;
        stride > 0;
        stride >>= 1
    ) {
        if (thread < stride) {
            shared[thread] +=
                shared[
                    thread +
                    stride
                ];
        }

        __syncthreads();
    }

    const float inverse_sum =
        1.0f /
        shared[0];

    // ==================================================
    // Write normalized probabilities.
    // ==================================================

    for (
        int column = thread;
        column < columns;
        column += blockDim.x
    ) {
        const float scaled =
            input[
                row_offset +
                static_cast<std::size_t>(
                    column
                )
            ] *
            scale_factor;

        output[
            row_offset +
            static_cast<std::size_t>(
                column
            )
        ] =
            expf(
                scaled -
                row_max
            ) *
            inverse_sum;
    }
}

void validate_scaled_softmax(
    const Tensor& input,
    const Tensor& output
) {
    if (
        input.device() != Device::CUDA ||
        output.device() != Device::CUDA
    ) {
        throw std::invalid_argument(
            "ScaledSoftmax requires CUDA tensors"
        );
    }

    if (
        input.dtype() != DataType::Float32 ||
        output.dtype() != DataType::Float32
    ) {
        throw std::invalid_argument(
            "ScaledSoftmax currently supports FP32 only"
        );
    }

    if (
        input.shape().empty() ||
        input.shape().size() > 3
    ) {
        throw std::invalid_argument(
            "ScaledSoftmax supports rank-1, rank-2 or rank-3 tensors"
        );
    }

    if (
        input.shape() !=
        output.shape()
    ) {
        throw std::invalid_argument(
            "ScaledSoftmax input/output shapes must match"
        );
    }

    if (
        input.shape().back() == 0
    ) {
        throw std::invalid_argument(
            "ScaledSoftmax final dimension cannot be zero"
        );
    }
}

} // namespace

void scaled_softmax(
    const Tensor& input,
    float scale_factor,
    Tensor& output
) {
    validate_scaled_softmax(
        input,
        output
    );

    const std::size_t columns =
        input.shape().back();

    const std::size_t rows =
        input.numel() /
        columns;

    scaled_softmax_kernel<<<
        static_cast<unsigned int>(
            rows
        ),
        SOFTMAX_THREADS,
        SOFTMAX_THREADS *
            sizeof(float)
    >>>(
        static_cast<const float*>(
            input.data()
        ),
        static_cast<float*>(
            output.data()
        ),
        scale_factor,
        static_cast<int>(
            rows
        ),
        static_cast<int>(
            columns
        )
    );

    CUDA_KERNEL_CHECK();
}

} // namespace vortex::ops
