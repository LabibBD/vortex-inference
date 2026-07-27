#include "vortex/operators/activations.hpp"
#include "vortex/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    using namespace vortex;

    constexpr std::size_t N = 1 << 20;

    std::vector<float> input(N);

    for (std::size_t i = 0; i < N; ++i) {
        input[i] =
            static_cast<float>(
                static_cast<int>(i % 2001) - 1000
            ) * 0.01f;
    }

    Tensor d_input(
        {N},
        DataType::Float32,
        Device::CUDA
    );

    Tensor d_output(
        {N},
        DataType::Float32,
        Device::CUDA
    );

    d_input.copy_from_host(
        input.data(),
        N * sizeof(float)
    );

    std::vector<float> output(N);

    // ==================================================
    // ReLU
    // ==================================================

    ops::relu(d_input, d_output);

    d_output.copy_to_host(
        output.data(),
        N * sizeof(float)
    );

    float relu_max_error = 0.0f;

    for (std::size_t i = 0; i < N; ++i) {
        const float reference =
            std::max(0.0f, input[i]);

        relu_max_error =
            std::max(
                relu_max_error,
                std::fabs(
                    output[i] - reference
                )
            );
    }

    // ==================================================
    // Sigmoid
    // ==================================================

    ops::sigmoid(d_input, d_output);

    d_output.copy_to_host(
        output.data(),
        N * sizeof(float)
    );

    float sigmoid_max_error = 0.0f;

    for (std::size_t i = 0; i < N; ++i) {
        const float reference =
            1.0f /
            (
                1.0f +
                std::exp(-input[i])
            );

        sigmoid_max_error =
            std::max(
                sigmoid_max_error,
                std::fabs(
                    output[i] - reference
                )
            );
    }

    // ==================================================
    // GELU
    // ==================================================

    ops::gelu(d_input, d_output);

    d_output.copy_to_host(
        output.data(),
        N * sizeof(float)
    );

    float gelu_max_error = 0.0f;

    constexpr float SQRT_2_OVER_PI =
        0.7978845608028654f;

    constexpr float GELU_COEFF =
        0.044715f;

    for (std::size_t i = 0; i < N; ++i) {
        const float x =
            input[i];

        const float inner =
            SQRT_2_OVER_PI *
            (
                x +
                GELU_COEFF *
                x *
                x *
                x
            );

        const float reference =
            0.5f *
            x *
            (
                1.0f +
                std::tanh(inner)
            );

        gelu_max_error =
            std::max(
                gelu_max_error,
                std::fabs(
                    output[i] - reference
                )
            );
    }

    std::cout << "=================================\n";
    std::cout << " VORTEX Activation Operators\n";
    std::cout << "=================================\n\n";

    std::cout << "Elements:           "
              << N << '\n';

    std::cout << "ReLU max error:     "
              << relu_max_error << '\n';

    std::cout << "Sigmoid max error:  "
              << sigmoid_max_error << '\n';

    std::cout << "GELU max error:     "
              << gelu_max_error << '\n';

    const bool passed =
        relu_max_error <= 1e-6f &&
        sigmoid_max_error <= 1e-5f &&
        gelu_max_error <= 1e-5f;

    if (passed) {
        std::cout
            << "\n[VORTEX] Activation test PASSED\n";

        return 0;
    }

    std::cerr
        << "\n[VORTEX] Activation test FAILED\n";

    return 1;
}
