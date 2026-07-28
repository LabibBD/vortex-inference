#include "vortex/operators/scaled_softmax.hpp"
#include "vortex/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <vector>

int main() {
    using namespace vortex;

    constexpr std::size_t BATCH =
        3;

    constexpr std::size_t ROWS =
        7;

    constexpr std::size_t COLUMNS =
        19;

    constexpr float SCALE =
        0.125f;

    const std::size_t elements =
        BATCH *
        ROWS *
        COLUMNS;

    std::vector<float> input(
        elements
    );

    for (
        std::size_t index = 0;
        index < elements;
        ++index
    ) {
        input[index] =
            static_cast<float>(
                static_cast<int>(
                    index % 43
                ) -
                21
            ) *
            0.08f;
    }

    Tensor device_input(
        {
            BATCH,
            ROWS,
            COLUMNS
        },
        DataType::Float32,
        Device::CUDA
    );

    Tensor device_output(
        {
            BATCH,
            ROWS,
            COLUMNS
        },
        DataType::Float32,
        Device::CUDA
    );

    device_input.copy_from_host(
        input.data(),
        input.size() *
            sizeof(float)
    );

    ops::scaled_softmax(
        device_input,
        SCALE,
        device_output
    );

    std::vector<float> output(
        elements
    );

    device_output.copy_to_host(
        output.data(),
        output.size() *
            sizeof(float)
    );

    float max_error =
        0.0f;

    float max_row_sum_error =
        0.0f;

    bool finite =
        true;

    const std::size_t row_count =
        BATCH *
        ROWS;

    for (
        std::size_t row = 0;
        row < row_count;
        ++row
    ) {
        float maximum =
            -std::numeric_limits<float>::
                infinity();

        for (
            std::size_t column = 0;
            column < COLUMNS;
            ++column
        ) {
            maximum =
                std::max(
                    maximum,
                    input[
                        row *
                        COLUMNS +
                        column
                    ] *
                    SCALE
                );
        }

        float denominator =
            0.0f;

        for (
            std::size_t column = 0;
            column < COLUMNS;
            ++column
        ) {
            denominator +=
                std::exp(
                    input[
                        row *
                        COLUMNS +
                        column
                    ] *
                    SCALE -
                    maximum
                );
        }

        float output_sum =
            0.0f;

        for (
            std::size_t column = 0;
            column < COLUMNS;
            ++column
        ) {
            const std::size_t index =
                row *
                COLUMNS +
                column;

            const float reference =
                std::exp(
                    input[index] *
                    SCALE -
                    maximum
                ) /
                denominator;

            const float error =
                std::fabs(
                    output[index] -
                    reference
                );

            max_error =
                std::max(
                    max_error,
                    error
                );

            output_sum +=
                output[index];

            finite =
                finite &&
                std::isfinite(
                    output[index]
                );
        }

        max_row_sum_error =
            std::max(
                max_row_sum_error,
                std::fabs(
                    output_sum -
                    1.0f
                )
            );
    }

    std::cout
        << "=================================\n"
        << " VORTEX Scaled-Softmax Test\n"
        << "=================================\n\n";

    std::cout
        << "Rows:               "
        << row_count << '\n';

    std::cout
        << "Columns:            "
        << COLUMNS << '\n';

    std::cout
        << "Scale:              "
        << SCALE << '\n';

    std::cout
        << "Maximum abs error:  "
        << max_error << '\n';

    std::cout
        << "Max row-sum error:  "
        << max_row_sum_error << '\n';

    std::cout
        << "All values finite:  "
        << (
            finite
                ? "YES"
                : "NO"
        )
        << '\n';

    const bool passed =
        finite &&
        max_error <= 2e-5f &&
        max_row_sum_error <= 2e-5f;

    if (!passed) {
        std::cerr
            << "\n[VORTEX] Scaled-Softmax FAILED\n";

        return 1;
    }

    std::cout
        << "\n[VORTEX] Scaled-Softmax PASSED\n";

    return 0;
}
