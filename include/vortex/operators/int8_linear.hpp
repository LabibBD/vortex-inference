#pragma once

#include "vortex/tensor.hpp"

#include <cstddef>
#include <cstdint>

namespace vortex::ops {

/*
 * Persistent per-output-channel quantized weight matrix.
 *
 * FP32 layout before quantization:
 *     [input_features, output_features]
 *
 * INT8 layout:
 *     [input_features, output_features]
 *
 * Scale layout:
 *     [output_features]
 */
class Int8LinearWeights {
public:
    Int8LinearWeights(
        std::size_t input_features,
        std::size_t output_features
    );

    ~Int8LinearWeights();

    Int8LinearWeights(
        const Int8LinearWeights&
    ) = delete;

    Int8LinearWeights&
    operator=(
        const Int8LinearWeights&
    ) = delete;

    Int8LinearWeights(
        Int8LinearWeights&& other
    ) noexcept;

    Int8LinearWeights&
    operator=(
        Int8LinearWeights&& other
    ) noexcept;

    void copy_from_host(
        const std::int8_t* values,
        const float* scales
    );

    std::size_t input_features()
        const noexcept;

    std::size_t output_features()
        const noexcept;

    const std::int8_t* values()
        const noexcept;

    const float* scales()
        const noexcept;

private:
    void reset() noexcept;

    std::size_t input_features_{};
    std::size_t output_features_{};

    std::int8_t* values_{nullptr};
    float* scales_{nullptr};
};


/*
 * Reusable activation-quantization workspace.
 *
 * rows is normally:
 *
 *     batch * tokens
 */
class Int8LinearWorkspace {
public:
    Int8LinearWorkspace(
        std::size_t rows,
        std::size_t input_features
    );

    ~Int8LinearWorkspace();

    Int8LinearWorkspace(
        const Int8LinearWorkspace&
    ) = delete;

    Int8LinearWorkspace&
    operator=(
        const Int8LinearWorkspace&
    ) = delete;

    Int8LinearWorkspace(
        Int8LinearWorkspace&& other
    ) noexcept;

    Int8LinearWorkspace&
    operator=(
        Int8LinearWorkspace&& other
    ) noexcept;

    std::size_t rows() const noexcept;

    std::size_t input_features()
        const noexcept;

    std::int8_t* quantized_input()
        noexcept;

    float* row_scales()
        noexcept;

private:
    void reset() noexcept;

    std::size_t rows_{};
    std::size_t input_features_{};

    std::int8_t* quantized_input_{
        nullptr
    };

    float* row_scales_{nullptr};
};


/*
 * input:  [batch, tokens, input_features] FP32
 * weight: [input_features, output_features] INT8
 * scales: [output_features] FP32
 * bias:   [output_features] FP32
 * output: [batch, tokens, output_features] FP32
 */
void linear_3d_int8_dynamic(
    const Tensor& input,
    const Int8LinearWeights& weights,
    const Tensor& bias,
    Int8LinearWorkspace& workspace,
    Tensor& output
);

} // namespace vortex::ops
