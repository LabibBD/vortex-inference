#include "vortex/operators/int8_linear.hpp"

#include "vortex/cuda_check.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace vortex::ops {

namespace {

constexpr int QUANT_THREADS =
    256;

constexpr int TILE =
    16;


__global__ void quantize_rows_kernel(
    const float* input,
    signed char* quantized,
    float* row_scales,
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

    const std::size_t offset =
        static_cast<std::size_t>(
            row
        ) *
        static_cast<std::size_t>(
            columns
        );

    float local_max =
        0.0f;

    for (
        int column = thread;
        column < columns;
        column += blockDim.x
    ) {
        local_max =
            fmaxf(
                local_max,
                fabsf(
                    input[
                        offset +
                        static_cast<std::size_t>(
                            column
                        )
                    ]
                )
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

    const float maximum =
        shared[0];

    const float scale =
        maximum > 0.0f
            ? maximum / 127.0f
            : 1.0f;

    if (thread == 0) {
        row_scales[row] =
            scale;
    }

    for (
        int column = thread;
        column < columns;
        column += blockDim.x
    ) {
        const float value =
            input[
                offset +
                static_cast<std::size_t>(
                    column
                )
            ] /
            scale;

        const int rounded =
            static_cast<int>(
                nearbyintf(value)
            );

        const int clamped =
            max(
                -127,
                min(
                    127,
                    rounded
                )
            );

        quantized[
            offset +
            static_cast<std::size_t>(
                column
            )
        ] =
            static_cast<signed char>(
                clamped
            );
    }
}


__global__ void int8_linear_kernel(
    const signed char* input,
    const signed char* weight,
    const float* row_scales,
    const float* weight_scales,
    const float* bias,
    float* output,
    int rows,
    int input_features,
    int output_features
) {
    __shared__ signed char
        input_tile[TILE][TILE];

    __shared__ signed char
        weight_tile[TILE][TILE];

    const int local_x =
        static_cast<int>(
            threadIdx.x
        );

    const int local_y =
        static_cast<int>(
            threadIdx.y
        );

    const int row =
        static_cast<int>(
            blockIdx.y
        ) *
        TILE +
        local_y;

    const int column =
        static_cast<int>(
            blockIdx.x
        ) *
        TILE +
        local_x;

    int accumulator =
        0;

    const int tile_count =
        (
            input_features +
            TILE -
            1
        ) /
        TILE;

    for (
        int tile = 0;
        tile < tile_count;
        ++tile
    ) {
        const int input_column =
            tile *
            TILE +
            local_x;

        const int weight_row =
            tile *
            TILE +
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
                    0;
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
                    0;
        }

        __syncthreads();

        #pragma unroll
        for (
            int inner = 0;
            inner < TILE;
            ++inner
        ) {
            accumulator +=
                static_cast<int>(
                    input_tile
                        [local_y]
                        [inner]
                ) *
                static_cast<int>(
                    weight_tile
                        [inner]
                        [local_x]
                );
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
            static_cast<float>(
                accumulator
            ) *
            row_scales[row] *
            weight_scales[column] +
            bias[column];
    }
}


void validate_linear(
    const Tensor& input,
    const Int8LinearWeights& weights,
    const Tensor& bias,
    const Int8LinearWorkspace& workspace,
    const Tensor& output
) {
    if (
        input.device() != Device::CUDA ||
        bias.device() != Device::CUDA ||
        output.device() != Device::CUDA
    ) {
        throw std::invalid_argument(
            "Dynamic INT8 Linear3D requires CUDA tensors"
        );
    }

    if (
        input.dtype() !=
            DataType::Float32 ||
        bias.dtype() !=
            DataType::Float32 ||
        output.dtype() !=
            DataType::Float32
    ) {
        throw std::invalid_argument(
            "Dynamic INT8 Linear3D requires FP32 input, bias and output"
        );
    }

    if (
        input.shape().size() != 3 ||
        bias.shape().size() != 1 ||
        output.shape().size() != 3
    ) {
        throw std::invalid_argument(
            "Dynamic INT8 Linear3D received invalid tensor ranks"
        );
    }

    const std::size_t rows =
        input.shape()[0] *
        input.shape()[1];

    const std::size_t input_features =
        input.shape()[2];

    const std::size_t output_features =
        weights.output_features();

    if (
        weights.input_features() !=
            input_features ||
        bias.shape()[0] !=
            output_features ||
        output.shape()[0] !=
            input.shape()[0] ||
        output.shape()[1] !=
            input.shape()[1] ||
        output.shape()[2] !=
            output_features
    ) {
        throw std::invalid_argument(
            "Dynamic INT8 Linear3D dimensions are incompatible"
        );
    }

    if (
        workspace.rows() <
            rows ||
        workspace.input_features() !=
            input_features
    ) {
        throw std::invalid_argument(
            "Dynamic INT8 Linear3D workspace is too small"
        );
    }
}

} // namespace


Int8LinearWeights::Int8LinearWeights(
    std::size_t input_features,
    std::size_t output_features
)
    : input_features_(
        input_features
      ),
      output_features_(
        output_features
      ) {
    if (
        input_features == 0 ||
        output_features == 0
    ) {
        throw std::invalid_argument(
            "INT8 weight dimensions cannot be zero"
        );
    }

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &values_
            ),
            input_features_ *
                output_features_ *
                sizeof(std::int8_t)
        )
    );

    try {
        CUDA_CHECK(
            cudaMalloc(
                reinterpret_cast<void**>(
                    &scales_
                ),
                output_features_ *
                    sizeof(float)
            )
        );
    } catch (...) {
        cudaFree(values_);
        values_ = nullptr;
        throw;
    }
}


Int8LinearWeights::~Int8LinearWeights() {
    reset();
}


Int8LinearWeights::Int8LinearWeights(
    Int8LinearWeights&& other
) noexcept
    : input_features_(
        other.input_features_
      ),
      output_features_(
        other.output_features_
      ),
      values_(other.values_),
      scales_(other.scales_) {
    other.input_features_ = 0;
    other.output_features_ = 0;
    other.values_ = nullptr;
    other.scales_ = nullptr;
}


Int8LinearWeights&
Int8LinearWeights::operator=(
    Int8LinearWeights&& other
) noexcept {
    if (this != &other) {
        reset();

        input_features_ =
            other.input_features_;

        output_features_ =
            other.output_features_;

        values_ =
            other.values_;

        scales_ =
            other.scales_;

        other.input_features_ =
            0;

        other.output_features_ =
            0;

        other.values_ =
            nullptr;

        other.scales_ =
            nullptr;
    }

    return *this;
}


void Int8LinearWeights::reset()
    noexcept {
    if (values_ != nullptr) {
        cudaFree(values_);
        values_ = nullptr;
    }

    if (scales_ != nullptr) {
        cudaFree(scales_);
        scales_ = nullptr;
    }
}


void Int8LinearWeights::copy_from_host(
    const std::int8_t* values,
    const float* scales
) {
    if (
        values == nullptr ||
        scales == nullptr
    ) {
        throw std::invalid_argument(
            "INT8 weight host data cannot be null"
        );
    }

    CUDA_CHECK(
        cudaMemcpy(
            values_,
            values,
            input_features_ *
                output_features_ *
                sizeof(std::int8_t),
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            scales_,
            scales,
            output_features_ *
                sizeof(float),
            cudaMemcpyHostToDevice
        )
    );
}


std::size_t
Int8LinearWeights::input_features()
    const noexcept {
    return input_features_;
}


std::size_t
Int8LinearWeights::output_features()
    const noexcept {
    return output_features_;
}


const std::int8_t*
Int8LinearWeights::values()
    const noexcept {
    return values_;
}


const float*
Int8LinearWeights::scales()
    const noexcept {
    return scales_;
}


Int8LinearWorkspace::Int8LinearWorkspace(
    std::size_t rows,
    std::size_t input_features
)
    : rows_(rows),
      input_features_(
        input_features
      ) {
    if (
        rows == 0 ||
        input_features == 0
    ) {
        throw std::invalid_argument(
            "INT8 workspace dimensions cannot be zero"
        );
    }

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &quantized_input_
            ),
            rows_ *
                input_features_ *
                sizeof(std::int8_t)
        )
    );

    try {
        CUDA_CHECK(
            cudaMalloc(
                reinterpret_cast<void**>(
                    &row_scales_
                ),
                rows_ *
                    sizeof(float)
            )
        );
    } catch (...) {
        cudaFree(
            quantized_input_
        );

        quantized_input_ =
            nullptr;

        throw;
    }
}


Int8LinearWorkspace::~Int8LinearWorkspace() {
    reset();
}


Int8LinearWorkspace::Int8LinearWorkspace(
    Int8LinearWorkspace&& other
) noexcept
    : rows_(other.rows_),
      input_features_(
        other.input_features_
      ),
      quantized_input_(
        other.quantized_input_
      ),
      row_scales_(
        other.row_scales_
      ) {
    other.rows_ = 0;
    other.input_features_ = 0;
    other.quantized_input_ = nullptr;
    other.row_scales_ = nullptr;
}


Int8LinearWorkspace&
Int8LinearWorkspace::operator=(
    Int8LinearWorkspace&& other
) noexcept {
    if (this != &other) {
        reset();

        rows_ =
            other.rows_;

        input_features_ =
            other.input_features_;

        quantized_input_ =
            other.quantized_input_;

        row_scales_ =
            other.row_scales_;

        other.rows_ = 0;
        other.input_features_ = 0;
        other.quantized_input_ =
            nullptr;
        other.row_scales_ =
            nullptr;
    }

    return *this;
}


void Int8LinearWorkspace::reset()
    noexcept {
    if (
        quantized_input_ !=
        nullptr
    ) {
        cudaFree(
            quantized_input_
        );

        quantized_input_ =
            nullptr;
    }

    if (row_scales_ != nullptr) {
        cudaFree(
            row_scales_
        );

        row_scales_ =
            nullptr;
    }
}


std::size_t
Int8LinearWorkspace::rows()
    const noexcept {
    return rows_;
}


std::size_t
Int8LinearWorkspace::input_features()
    const noexcept {
    return input_features_;
}


std::int8_t*
Int8LinearWorkspace::quantized_input()
    noexcept {
    return quantized_input_;
}


float*
Int8LinearWorkspace::row_scales()
    noexcept {
    return row_scales_;
}


void linear_3d_int8_dynamic(
    const Tensor& input,
    const Int8LinearWeights& weights,
    const Tensor& bias,
    Int8LinearWorkspace& workspace,
    Tensor& output
) {
    validate_linear(
        input,
        weights,
        bias,
        workspace,
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
            weights.output_features()
        );

    quantize_rows_kernel<<<
        rows,
        QUANT_THREADS,
        QUANT_THREADS *
            sizeof(float)
    >>>(
        static_cast<const float*>(
            input.data()
        ),
        reinterpret_cast<
            signed char*
        >(
            workspace.quantized_input()
        ),
        workspace.row_scales(),
        rows,
        input_features
    );

    CUDA_KERNEL_CHECK();

    const dim3 block(
        TILE,
        TILE
    );

    const dim3 grid(
        static_cast<unsigned int>(
            (
                output_features +
                TILE -
                1
            ) /
            TILE
        ),
        static_cast<unsigned int>(
            (
                rows +
                TILE -
                1
            ) /
            TILE
        )
    );

    int8_linear_kernel<<<
        grid,
        block
    >>>(
        reinterpret_cast<
            const signed char*
        >(
            workspace.quantized_input()
        ),
        reinterpret_cast<
            const signed char*
        >(
            weights.values()
        ),
        workspace.row_scales(),
        weights.scales(),
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
