#include "vortex/operators/softmax.hpp"
#include "vortex/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

int main() {
    using namespace vortex;

    constexpr std::size_t N = 4096;

    std::vector<float> input(N);

    for (std::size_t i = 0; i < N; ++i) {
        input[i] =
            static_cast<float>(
                static_cast<int>(i % 101) - 50
            ) * 0.1f;
    }

    // ================================================
    // CPU reference
    // ================================================

    const float cpu_max =
        *std::max_element(
            input.begin(),
            input.end()
        );

    std::vector<double> cpu_exp(N);

    double cpu_sum = 0.0;

    for (std::size_t i = 0; i < N; ++i) {
        cpu_exp[i] =
            std::exp(
                static_cast<double>(
                    input[i] - cpu_max
                )
            );

        cpu_sum += cpu_exp[i];
    }

    std::vector<float> cpu_output(N);

    for (std::size_t i = 0; i < N; ++i) {
        cpu_output[i] =
            static_cast<float>(
                cpu_exp[i] / cpu_sum
            );
    }

    // ================================================
    // VORTEX CUDA
    // ================================================

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

    ops::softmax(
        d_input,
        d_output
    );

    std::vector<float> gpu_output(N);

    d_output.copy_to_host(
        gpu_output.data(),
        N * sizeof(float)
    );

    // ================================================
    // Validation
    // ================================================

    float max_error = 0.0f;

    double gpu_probability_sum = 0.0;

    for (std::size_t i = 0; i < N; ++i) {
        max_error =
            std::max(
                max_error,
                std::fabs(
                    gpu_output[i] -
                    cpu_output[i]
                )
            );

        gpu_probability_sum +=
            gpu_output[i];
    }

    const double sum_error =
        std::fabs(
            gpu_probability_sum - 1.0
        );

    std::cout << "=================================\n";
    std::cout << " VORTEX Softmax Operator\n";
    std::cout << "=================================\n\n";

    std::cout << "Elements:           "
              << N << '\n';

    std::cout << "CPU maximum:        "
              << cpu_max << '\n';

    std::cout << "Probability sum:    "
              << gpu_probability_sum << '\n';

    std::cout << "Sum error:          "
              << sum_error << '\n';

    std::cout << "Maximum abs error:  "
              << max_error << '\n';

    const bool passed =
        max_error <= 1e-5f &&
        sum_error <= 1e-5;

    if (passed) {
        std::cout
            << "\n[VORTEX] Softmax test PASSED\n";

        return 0;
    }

    std::cerr
        << "\n[VORTEX] Softmax test FAILED\n";

    return 1;
}
