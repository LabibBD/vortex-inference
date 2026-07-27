#include "vortex/operators/activations.hpp"
#include "vortex/operators/fused.hpp"
#include "vortex/operators/linear.hpp"
#include "vortex/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

int main() {
    using namespace vortex;

    constexpr std::size_t M = 37;
    constexpr std::size_t K = 83;
    constexpr std::size_t N = 59;

    std::vector<float> input(
        M * K
    );

    std::vector<float> weight(
        K * N
    );

    std::vector<float> bias(
        N
    );

    for (
        std::size_t i = 0;
        i < input.size();
        ++i
    ) {
        input[i] =
            static_cast<float>(
                static_cast<int>(i % 19) - 9
            ) * 0.025f;
    }

    for (
        std::size_t i = 0;
        i < weight.size();
        ++i
    ) {
        weight[i] =
            static_cast<float>(
                static_cast<int>(i % 17) - 8
            ) * 0.015f;
    }

    for (
        std::size_t i = 0;
        i < bias.size();
        ++i
    ) {
        bias[i] =
            static_cast<float>(
                static_cast<int>(i % 7) - 3
            ) * 0.01f;
    }

    Tensor d_input(
        {M, K},
        DataType::Float32,
        Device::CUDA
    );

    Tensor d_weight(
        {K, N},
        DataType::Float32,
        Device::CUDA
    );

    Tensor d_bias(
        {N},
        DataType::Float32,
        Device::CUDA
    );

    Tensor d_linear(
        {M, N},
        DataType::Float32,
        Device::CUDA
    );

    Tensor d_baseline(
        {M, N},
        DataType::Float32,
        Device::CUDA
    );

    Tensor d_fused(
        {M, N},
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

    // Baseline: three kernels internally.
    ops::linear(
        d_input,
        d_weight,
        d_bias,
        d_linear
    );

    ops::gelu(
        d_linear,
        d_baseline
    );

    // Fused path: one kernel.
    ops::linear_gelu_fused(
        d_input,
        d_weight,
        d_bias,
        d_fused
    );

    std::vector<float> baseline(
        M * N
    );

    std::vector<float> fused(
        M * N
    );

    d_baseline.copy_to_host(
        baseline.data(),
        baseline.size() * sizeof(float)
    );

    d_fused.copy_to_host(
        fused.data(),
        fused.size() * sizeof(float)
    );

    float max_error =
        0.0f;

    double mean_error =
        0.0;

    for (
        std::size_t i = 0;
        i < fused.size();
        ++i
    ) {
        const float error =
            std::fabs(
                fused[i] -
                baseline[i]
            );

        max_error =
            std::max(
                max_error,
                error
            );

        mean_error +=
            error;
    }

    mean_error /=
        static_cast<double>(
            fused.size()
        );

    std::cout
        << "=================================\n"
        << " VORTEX Kernel Fusion Test\n"
        << "=================================\n\n";

    std::cout
        << "Shape:             "
        << M << " x "
        << K << " x "
        << N << '\n';

    std::cout
        << "Baseline launches: 3\n";

    std::cout
        << "Fused launches:    1\n";

    std::cout
        << "Maximum abs error: "
        << max_error << '\n';

    std::cout
        << "Mean abs error:    "
        << mean_error << '\n';

    if (
        max_error <= 1e-5f
    ) {
        std::cout
            << "\n[VORTEX] Kernel fusion correctness PASSED\n";

        return 0;
    }

    std::cerr
        << "\n[VORTEX] Kernel fusion correctness FAILED\n";

    return 1;
}
