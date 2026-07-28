#include "vortex/numerics.hpp"
#include "vortex/operators/int8_linear.hpp"
#include "vortex/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {

int quantize(
    float value,
    float scale
) {
    const int rounded =
        static_cast<int>(
            std::nearbyint(
                value / scale
            )
        );

    return std::max(
        -127,
        std::min(
            127,
            rounded
        )
    );
}

} // namespace


int main() {
    using namespace vortex;

    constexpr std::size_t BATCH =
        3;

    constexpr std::size_t TOKENS =
        7;

    constexpr std::size_t INPUT =
        31;

    constexpr std::size_t OUTPUT =
        29;

    constexpr std::size_t ROWS =
        BATCH * TOKENS;

    std::mt19937 generator(
        2601
    );

    std::normal_distribution<float>
        input_distribution(
            0.0f,
            1.0f
        );

    std::normal_distribution<float>
        weight_distribution(
            0.0f,
            0.25f
        );

    std::normal_distribution<float>
        bias_distribution(
            0.0f,
            0.05f
        );

    std::vector<float> input(
        ROWS * INPUT
    );

    std::vector<float> weight(
        INPUT * OUTPUT
    );

    std::vector<float> bias(
        OUTPUT
    );

    for (float& value : input) {
        value =
            input_distribution(
                generator
            );
    }

    for (float& value : weight) {
        value =
            weight_distribution(
                generator
            );
    }

    for (float& value : bias) {
        value =
            bias_distribution(
                generator
            );
    }

    // ==================================================
    // Per-output-channel weight quantization
    // ==================================================

    std::vector<std::int8_t>
        quantized_weight(
            INPUT * OUTPUT
        );

    std::vector<float>
        weight_scales(
            OUTPUT
        );

    for (
        std::size_t column = 0;
        column < OUTPUT;
        ++column
    ) {
        float maximum =
            0.0f;

        for (
            std::size_t row = 0;
            row < INPUT;
            ++row
        ) {
            maximum =
                std::max(
                    maximum,
                    std::fabs(
                        weight[
                            row *
                            OUTPUT +
                            column
                        ]
                    )
                );
        }

        const float scale =
            maximum > 0.0f
                ? maximum / 127.0f
                : 1.0f;

        weight_scales[column] =
            scale;

        for (
            std::size_t row = 0;
            row < INPUT;
            ++row
        ) {
            quantized_weight[
                row *
                OUTPUT +
                column
            ] =
                static_cast<std::int8_t>(
                    quantize(
                        weight[
                            row *
                            OUTPUT +
                            column
                        ],
                        scale
                    )
                );
        }
    }

    // ==================================================
    // CPU dynamic-INT8 reference
    // ==================================================

    std::vector<float>
        quantized_reference(
            ROWS * OUTPUT
        );

    std::vector<float>
        fp32_reference(
            ROWS * OUTPUT
        );

    for (
        std::size_t row = 0;
        row < ROWS;
        ++row
    ) {
        float input_max =
            0.0f;

        for (
            std::size_t inner = 0;
            inner < INPUT;
            ++inner
        ) {
            input_max =
                std::max(
                    input_max,
                    std::fabs(
                        input[
                            row *
                            INPUT +
                            inner
                        ]
                    )
                );
        }

        const float input_scale =
            input_max > 0.0f
                ? input_max / 127.0f
                : 1.0f;

        for (
            std::size_t column = 0;
            column < OUTPUT;
            ++column
        ) {
            int accumulator =
                0;

            double fp32_sum =
                bias[column];

            for (
                std::size_t inner = 0;
                inner < INPUT;
                ++inner
            ) {
                const int input_value =
                    quantize(
                        input[
                            row *
                            INPUT +
                            inner
                        ],
                        input_scale
                    );

                const int weight_value =
                    static_cast<int>(
                        quantized_weight[
                            inner *
                            OUTPUT +
                            column
                        ]
                    );

                accumulator +=
                    input_value *
                    weight_value;

                fp32_sum +=
                    static_cast<double>(
                        input[
                            row *
                            INPUT +
                            inner
                        ]
                    ) *
                    static_cast<double>(
                        weight[
                            inner *
                            OUTPUT +
                            column
                        ]
                    );
            }

            quantized_reference[
                row *
                OUTPUT +
                column
            ] =
                static_cast<float>(
                    accumulator
                ) *
                input_scale *
                weight_scales[column] +
                bias[column];

            fp32_reference[
                row *
                OUTPUT +
                column
            ] =
                static_cast<float>(
                    fp32_sum
                );
        }
    }

    Tensor device_input(
        {
            BATCH,
            TOKENS,
            INPUT
        },
        DataType::Float32,
        Device::CUDA
    );

    Tensor device_bias(
        {OUTPUT},
        DataType::Float32,
        Device::CUDA
    );

    Tensor output(
        {
            BATCH,
            TOKENS,
            OUTPUT
        },
        DataType::Float32,
        Device::CUDA
    );

    device_input.copy_from_host(
        input.data(),
        input.size() *
            sizeof(float)
    );

    device_bias.copy_from_host(
        bias.data(),
        bias.size() *
            sizeof(float)
    );

    ops::Int8LinearWeights
        device_weights(
            INPUT,
            OUTPUT
        );

    device_weights.copy_from_host(
        quantized_weight.data(),
        weight_scales.data()
    );

    ops::Int8LinearWorkspace
        workspace(
            ROWS,
            INPUT
        );

    ops::linear_3d_int8_dynamic(
        device_input,
        device_weights,
        device_bias,
        workspace,
        output
    );

    if (
        !numerics::all_finite(
            output
        )
    ) {
        std::cerr
            << "[VORTEX] INT8 Linear produced NaN or Inf\n";

        return 1;
    }

    std::vector<float> result(
        ROWS * OUTPUT
    );

    output.copy_to_host(
        result.data(),
        result.size() *
            sizeof(float)
    );

    float implementation_max_error =
        0.0f;

    double implementation_mean_error =
        0.0;

    float quantization_max_error =
        0.0f;

    double quantization_mean_error =
        0.0;

    for (
        std::size_t index = 0;
        index < result.size();
        ++index
    ) {
        const float implementation_error =
            std::fabs(
                result[index] -
                quantized_reference[index]
            );

        implementation_max_error =
            std::max(
                implementation_max_error,
                implementation_error
            );

        implementation_mean_error +=
            implementation_error;

        const float quantization_error =
            std::fabs(
                result[index] -
                fp32_reference[index]
            );

        quantization_max_error =
            std::max(
                quantization_max_error,
                quantization_error
            );

        quantization_mean_error +=
            quantization_error;
    }

    implementation_mean_error /=
        result.size();

    quantization_mean_error /=
        result.size();

    std::vector<float> first =
        result;

    for (int run = 0; run < 20; ++run) {
        ops::linear_3d_int8_dynamic(
            device_input,
            device_weights,
            device_bias,
            workspace,
            output
        );
    }

    output.copy_to_host(
        result.data(),
        result.size() *
            sizeof(float)
    );

    float repeat_error =
        0.0f;

    for (
        std::size_t index = 0;
        index < result.size();
        ++index
    ) {
        repeat_error =
            std::max(
                repeat_error,
                std::fabs(
                    result[index] -
                    first[index]
                )
            );
    }

    std::cout
        << "=================================\n"
        << " VORTEX Dynamic INT8 Linear3D\n"
        << "=================================\n\n";

    std::cout
        << "Shape:                 "
        << ROWS
        << " x "
        << INPUT
        << " x "
        << OUTPUT
        << '\n';

    std::cout
        << "INT8 reference max:    "
        << implementation_max_error
        << '\n';

    std::cout
        << "INT8 reference mean:   "
        << implementation_mean_error
        << '\n';

    std::cout
        << "Error vs FP32 max:     "
        << quantization_max_error
        << '\n';

    std::cout
        << "Error vs FP32 mean:    "
        << quantization_mean_error
        << '\n';

    std::cout
        << "Repeat error:          "
        << repeat_error
        << '\n';

    const bool passed =
        implementation_max_error <=
            2e-5f &&
        implementation_mean_error <=
            2e-6 &&
        quantization_max_error <=
            8e-2f &&
        quantization_mean_error <=
            1e-2 &&
        repeat_error <=
            1e-7f;

    if (!passed) {
        std::cerr
            << "\n[VORTEX] Dynamic INT8 Linear3D FAILED\n";

        return 1;
    }

    std::cout
        << "\n[VORTEX] Dynamic INT8 Linear3D PASSED\n";

    return 0;
}
