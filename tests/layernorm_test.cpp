#include "vortex/operators/layernorm.hpp"
#include "vortex/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

int main() {
    using namespace vortex;

    constexpr std::size_t ROWS = 37;
    constexpr std::size_t FEATURES = 511;
    constexpr float EPSILON = 1e-5f;

    std::vector<float> input(
        ROWS * FEATURES
    );

    std::vector<float> gamma(
        FEATURES
    );

    std::vector<float> beta(
        FEATURES
    );

    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] =
            static_cast<float>(
                static_cast<int>(i % 97) - 48
            ) * 0.03f;
    }

    for (std::size_t i = 0; i < FEATURES; ++i) {
        gamma[i] =
            0.9f +
            static_cast<float>(i % 11) *
            0.01f;

        beta[i] =
            static_cast<float>(
                static_cast<int>(i % 7) - 3
            ) * 0.02f;
    }

    // ================================================
    // CPU reference
    // ================================================

    std::vector<float> cpu(
        ROWS * FEATURES
    );

    for (
        std::size_t row = 0;
        row < ROWS;
        ++row
    ) {
        double mean = 0.0;

        for (
            std::size_t col = 0;
            col < FEATURES;
            ++col
        ) {
            mean +=
                input[row * FEATURES + col];
        }

        mean /=
            static_cast<double>(FEATURES);

        double variance = 0.0;

        for (
            std::size_t col = 0;
            col < FEATURES;
            ++col
        ) {
            const double difference =
                static_cast<double>(
                    input[row * FEATURES + col]
                ) - mean;

            variance +=
                difference * difference;
        }

        variance /=
            static_cast<double>(FEATURES);

        const double inverse_std =
            1.0 /
            std::sqrt(
                variance + EPSILON
            );

        for (
            std::size_t col = 0;
            col < FEATURES;
            ++col
        ) {
            const double normalized =
                (
                    static_cast<double>(
                        input[row * FEATURES + col]
                    ) -
                    mean
                ) *
                inverse_std;

            cpu[row * FEATURES + col] =
                static_cast<float>(
                    normalized *
                    gamma[col] +
                    beta[col]
                );
        }
    }

    // ================================================
    // VORTEX CUDA
    // ================================================

    Tensor d_input(
        {ROWS, FEATURES},
        DataType::Float32,
        Device::CUDA
    );

    Tensor d_gamma(
        {FEATURES},
        DataType::Float32,
        Device::CUDA
    );

    Tensor d_beta(
        {FEATURES},
        DataType::Float32,
        Device::CUDA
    );

    Tensor d_output(
        {ROWS, FEATURES},
        DataType::Float32,
        Device::CUDA
    );

    d_input.copy_from_host(
        input.data(),
        input.size() * sizeof(float)
    );

    d_gamma.copy_from_host(
        gamma.data(),
        gamma.size() * sizeof(float)
    );

    d_beta.copy_from_host(
        beta.data(),
        beta.size() * sizeof(float)
    );

    ops::layernorm(
        d_input,
        d_gamma,
        d_beta,
        d_output,
        EPSILON
    );

    std::vector<float> gpu(
        ROWS * FEATURES
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
        << " VORTEX LayerNorm Operator\n"
        << "=================================\n\n";

    std::cout
        << "Rows:              "
        << ROWS << '\n';

    std::cout
        << "Features:          "
        << FEATURES << '\n';

    std::cout
        << "Maximum abs error: "
        << max_error << '\n';

    std::cout
        << "Mean abs error:    "
        << mean_error << '\n';

    if (max_error <= 2e-4f) {
        std::cout
            << "\n[VORTEX] LayerNorm test PASSED\n";

        return 0;
    }

    std::cerr
        << "\n[VORTEX] LayerNorm test FAILED\n";

    return 1;
}
