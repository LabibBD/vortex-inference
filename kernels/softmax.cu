#include "vortex/operators/softmax.hpp"

#include "vortex/cuda_check.hpp"
#include "vortex/operators/reductions.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>

namespace vortex::ops {

namespace {

constexpr int THREADS_PER_BLOCK = 256;

void validate_softmax(
    const Tensor& input,
    const Tensor& output
) {
    if (
        input.device() != Device::CUDA ||
        output.device() != Device::CUDA
    ) {
        throw std::invalid_argument(
            "VORTEX Softmax requires CUDA tensors"
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
            "Softmax input and output shapes must match"
        );
    }

    if (input.shape().size() != 1) {
        throw std::invalid_argument(
            "Current Softmax implementation supports 1D tensors only"
        );
    }
}

int block_count(
    std::size_t n
) {
    return static_cast<int>(
        (n + THREADS_PER_BLOCK - 1) /
        THREADS_PER_BLOCK
    );
}

__global__ void exp_shift_kernel(
    const float* input,
    const float* maximum,
    float* output,
    std::size_t n
) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) *
        blockDim.x +
        threadIdx.x;

    if (i < n) {
        output[i] =
            expf(
                input[i] - maximum[0]
            );
    }
}

__global__ void normalize_kernel(
    const float* exponentials,
    const float* sum,
    float* output,
    std::size_t n
) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) *
        blockDim.x +
        threadIdx.x;

    if (i < n) {
        output[i] =
            exponentials[i] /
            sum[0];
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

    const std::size_t n =
        input.numel();

    Tensor maximum(
        {1},
        DataType::Float32,
        Device::CUDA
    );

    Tensor exponentials(
        input.shape(),
        DataType::Float32,
        Device::CUDA
    );

    Tensor exponential_sum(
        {1},
        DataType::Float32,
        Device::CUDA
    );

    // Step 1:
    // Find max(x) for numerical stability.
    reduce_max(
        input,
        maximum
    );

    // Step 2:
    // exp(x - max(x))
    exp_shift_kernel<<<
        block_count(n),
        THREADS_PER_BLOCK
    >>>(
        static_cast<const float*>(
            input.data()
        ),
        static_cast<const float*>(
            maximum.data()
        ),
        static_cast<float*>(
            exponentials.data()
        ),
        n
    );

    CUDA_KERNEL_CHECK();

    // Step 3:
    // Sum all exponentials.
    reduce_sum(
        exponentials,
        exponential_sum
    );

    // Step 4:
    // Divide each exponential by the sum.
    normalize_kernel<<<
        block_count(n),
        THREADS_PER_BLOCK
    >>>(
        static_cast<const float*>(
            exponentials.data()
        ),
        static_cast<const float*>(
            exponential_sum.data()
        ),
        static_cast<float*>(
            output.data()
        ),
        n
    );

    CUDA_KERNEL_CHECK();
}

} // namespace vortex::ops
