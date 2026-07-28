#include "vortex/numerics.hpp"

#include "vortex/cuda_check.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>

namespace vortex::numerics {

namespace {

constexpr int THREADS =
    256;

__global__ void all_finite_fp32_kernel(
    const float* values,
    std::size_t elements,
    int* finite_flag
) {
    const std::size_t index =
        static_cast<std::size_t>(
            blockIdx.x
        ) *
        blockDim.x +
        threadIdx.x;

    if (
        index < elements &&
        !isfinite(values[index])
    ) {
        atomicExch(
            finite_flag,
            0
        );
    }
}

__global__ void all_finite_fp16_kernel(
    const __half* values,
    std::size_t elements,
    int* finite_flag
) {
    const std::size_t index =
        static_cast<std::size_t>(
            blockIdx.x
        ) *
        blockDim.x +
        threadIdx.x;

    if (index < elements) {
        const float value =
            __half2float(
                values[index]
            );

        if (!isfinite(value)) {
            atomicExch(
                finite_flag,
                0
            );
        }
    }
}

} // namespace

bool all_finite(
    const Tensor& tensor
) {
    if (
        tensor.device() !=
        Device::CUDA
    ) {
        throw std::invalid_argument(
            "all_finite currently requires a CUDA tensor"
        );
    }

    if (
        tensor.dtype() !=
            DataType::Float32 &&
        tensor.dtype() !=
            DataType::Float16
    ) {
        throw std::invalid_argument(
            "all_finite supports FP32 and FP16 only"
        );
    }

    if (tensor.numel() == 0) {
        return true;
    }

    int* device_flag =
        nullptr;

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_flag
            ),
            sizeof(int)
        )
    );

    const int initial_flag =
        1;

    CUDA_CHECK(
        cudaMemcpy(
            device_flag,
            &initial_flag,
            sizeof(int),
            cudaMemcpyHostToDevice
        )
    );

    const int blocks =
        static_cast<int>(
            (
                tensor.numel() +
                THREADS -
                1
            ) /
            THREADS
        );

    if (
        tensor.dtype() ==
        DataType::Float32
    ) {
        all_finite_fp32_kernel<<<
            blocks,
            THREADS
        >>>(
            static_cast<const float*>(
                tensor.data()
            ),
            tensor.numel(),
            device_flag
        );
    } else {
        all_finite_fp16_kernel<<<
            blocks,
            THREADS
        >>>(
            static_cast<const __half*>(
                tensor.data()
            ),
            tensor.numel(),
            device_flag
        );
    }

    CUDA_KERNEL_LAUNCH_CHECK();

    int host_flag =
        0;

    /*
     * The device-to-host copy is also the explicit
     * synchronization boundary for this inspection.
     */
    CUDA_CHECK(
        cudaMemcpy(
            &host_flag,
            device_flag,
            sizeof(int),
            cudaMemcpyDeviceToHost
        )
    );

    CUDA_CHECK(
        cudaFree(
            device_flag
        )
    );

    return host_flag != 0;
}

} // namespace vortex::numerics
