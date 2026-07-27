#include "vortex/operators/reductions.hpp"

#include "vortex/cuda_check.hpp"
#include "vortex/device_buffer.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace vortex::ops {

namespace {

constexpr int THREADS_PER_BLOCK = 256;

void validate_reduction(
    const Tensor& input,
    const Tensor& output
) {
    if (
        input.device() != Device::CUDA ||
        output.device() != Device::CUDA
    ) {
        throw std::invalid_argument(
            "VORTEX reductions require CUDA tensors"
        );
    }

    if (
        input.dtype() != DataType::Float32 ||
        output.dtype() != DataType::Float32
    ) {
        throw std::invalid_argument(
            "Reductions currently support FP32 only"
        );
    }

    if (output.numel() != 1) {
        throw std::invalid_argument(
            "Reduction output tensor must contain exactly one element"
        );
    }
}

template <bool IS_MAX>
__global__ void reduction_kernel(
    const float* input,
    float* output,
    std::size_t n
) {
    extern __shared__ float shared[];

    const unsigned int tid =
        threadIdx.x;

    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) *
        blockDim.x *
        2 +
        tid;

    float value;

    if constexpr (IS_MAX) {
        value = -FLT_MAX;

        if (index < n) {
            value = input[index];
        }

        const std::size_t second =
            index + blockDim.x;

        if (second < n) {
            value = fmaxf(
                value,
                input[second]
            );
        }
    } else {
        value = 0.0f;

        if (index < n) {
            value = input[index];
        }

        const std::size_t second =
            index + blockDim.x;

        if (second < n) {
            value += input[second];
        }
    }

    shared[tid] = value;

    __syncthreads();

    // Tree reduction inside each block.
    for (
        unsigned int stride =
            blockDim.x / 2;

        stride > 0;

        stride >>= 1
    ) {
        if (tid < stride) {
            if constexpr (IS_MAX) {
                shared[tid] =
                    fmaxf(
                        shared[tid],
                        shared[tid + stride]
                    );
            } else {
                shared[tid] +=
                    shared[tid + stride];
            }
        }

        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] =
            shared[0];
    }
}

std::size_t block_count(
    std::size_t n
) {
    const std::size_t elements_per_block =
        static_cast<std::size_t>(
            THREADS_PER_BLOCK
        ) * 2;

    return (
        n +
        elements_per_block -
        1
    ) / elements_per_block;
}

template <bool IS_MAX>
void run_reduction(
    const Tensor& input,
    Tensor& output
) {
    validate_reduction(
        input,
        output
    );

    std::size_t current_n =
        input.numel();

    const std::size_t first_blocks =
        block_count(current_n);

    DeviceBuffer buffer_a(
        first_blocks * sizeof(float)
    );

    DeviceBuffer buffer_b(
        first_blocks * sizeof(float)
    );

    const float* current_input =
        static_cast<const float*>(
            input.data()
        );

    float* current_output =
        static_cast<float*>(
            buffer_a.data()
        );

    bool output_is_a = true;

    while (true) {
        const std::size_t blocks =
            block_count(current_n);

        reduction_kernel<IS_MAX><<<
            static_cast<unsigned int>(blocks),
            THREADS_PER_BLOCK,
            THREADS_PER_BLOCK * sizeof(float)
        >>>(
            current_input,
            current_output,
            current_n
        );

        CUDA_CHECK(cudaGetLastError());

        if (blocks == 1) {
            break;
        }

        current_n = blocks;

        current_input =
            current_output;

        output_is_a =
            !output_is_a;

        current_output =
            static_cast<float*>(
                output_is_a
                    ? buffer_a.data()
                    : buffer_b.data()
            );
    }

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );

    CUDA_CHECK(
        cudaMemcpy(
            output.data(),
            current_output,
            sizeof(float),
            cudaMemcpyDeviceToDevice
        )
    );
}

} // namespace

void reduce_sum(
    const Tensor& input,
    Tensor& output
) {
    run_reduction<false>(
        input,
        output
    );
}

void reduce_max(
    const Tensor& input,
    Tensor& output
) {
    run_reduction<true>(
        input,
        output
    );
}

} // namespace vortex::ops
