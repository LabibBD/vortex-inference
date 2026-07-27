#include "vortex/operators/linear.hpp"
#include "vortex/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

int main() {
    using namespace vortex;

    constexpr std::size_t BATCH = 37;
    constexpr std::size_t IN = 83;
    constexpr std::size_t OUT = 59;

    std::vector<float> input(
        BATCH * IN
    );

    std::vector<float> weight(
        IN * OUT
    );

    std::vector<float> bias(
        OUT
    );

    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] =
            static_cast<float>(
                static_cast<int>(i % 19) - 9
            ) * 0.03f;
    }

    for (std::size_t i = 0; i < weight.size(); ++i) {
        weight[i] =
            static_cast<float>(
                static_cast<int>(i % 13) - 6
            ) * 0.02f;
    }

    for (std::size_t i = 0; i < bias.size(); ++i) {
        bias[i] =
            static_cast<float>(
                static_cast<int>(i % 7) - 3
            ) * 0.01f;
    }

    // ================================================
    // CPU reference: Y = XW + b
    // ================================================

    std::vector<float> cpu(
        BATCH * OUT,
        0.0f
    );

    for (
        std::size_t row = 0;
        row < BATCH;
        ++row
    ) {
        for (
            std::size_t col = 0;
            col < OUT;
            ++col
        ) {
            float sum = bias[col];

            for (
                std::size_t k = 0;
                k < IN;
                ++k
            ) {
                sum +=
                    input[row * IN + k] *
                    weight[k * OUT + col];
            }

            cpu[row * OUT + col] =
                sum;
        }
    }

    // ================================================
    // VORTEX
    // ================================================

    Tensor d_input(
        {BATCH, IN},
        DataType::Float32,
        Device::CUDA
    );

    Tensor d_weight(
        {IN, OUT},
        DataType::Float32,
        Device::CUDA
    );

    Tensor d_bias(
        {OUT},
        DataType::Float32,
        Device::CUDA
    );

    Tensor d_output(
        {BATCH, OUT},
        DataType::Float32,
        Device::CUDA
    );

    d_input.copy_from_host(
        input.data(),
        input.size() * sizeof(float)
    );

    d_weight.copy_from_host(
        weight.data(),
        weight.size() * sizeof(float)
    );

    d_bias.copy_from_host(
        bias.data(),
        bias.size() * sizeof(float)
    );

    ops::linear(
        d_input,
        d_weight,
        d_bias,
        d_output
    );

    std::vector<float> gpu(
        BATCH * OUT
    );

    d_output.copy_to_host(
        gpu.data(),
        gpu.size() * sizeof(float)
    );

    // ================================================
    // Validation
    // ================================================

    float max_error = 0.0f;
    double mean_error = 0.0;

    for (
        std::size_t i = 0;
        i < gpu.size();
        ++i
    ) {
        const float error =
            std::fabs(
                gpu[i] - cpu[i]
            );

        max_error =
            std::max(
                max_error,
                error
            );

        mean_error += error;
    }

    mean_error /=
        static_cast<double>(
            gpu.size()
        );

    std::cout
        << "=================================\n"
        << " VORTEX Linear Operator\n"
        << "=================================\n\n";

    std::cout
        << "Input:   "
        << BATCH << " x " << IN
        << '\n';

    std::cout
        << "Weight:  "
        << IN << " x " << OUT
        << '\n';

    std::cout
        << "Bias:    "
        << OUT
        << '\n';

    std::cout
        << "Output:  "
        << BATCH << " x " << OUT
        << "\n\n";

    std::cout
        << "Maximum abs error: "
        << max_error << '\n';

    std::cout
        << "Mean abs error:    "
        << mean_error << '\n';

    if (max_error <= 1e-4f) {
        std::cout
            << "\n[VORTEX] Linear test PASSED\n";

        return 0;
    }

    std::cerr
        << "\n[VORTEX] Linear test FAILED\n";

    return 1;
}
