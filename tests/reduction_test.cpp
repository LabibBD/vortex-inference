#include "vortex/operators/reductions.hpp"
#include "vortex/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    using namespace vortex;

    // Intentionally not a power of two.
    constexpr std::size_t N =
        (1 << 20) + 123;

    std::vector<float> input(N);

    double cpu_sum = 0.0;
    float cpu_max = -1e30f;

    for (std::size_t i = 0; i < N; ++i) {
        input[i] =
            static_cast<float>(
                static_cast<int>(i % 7) - 3
            );

        cpu_sum += input[i];

        cpu_max =
            std::max(
                cpu_max,
                input[i]
            );
    }

    Tensor d_input(
        {N},
        DataType::Float32,
        Device::CUDA
    );

    Tensor d_sum(
        {1},
        DataType::Float32,
        Device::CUDA
    );

    Tensor d_max(
        {1},
        DataType::Float32,
        Device::CUDA
    );

    d_input.copy_from_host(
        input.data(),
        N * sizeof(float)
    );

    // ================================================
    // SUM
    // ================================================

    ops::reduce_sum(
        d_input,
        d_sum
    );

    float gpu_sum = 0.0f;

    d_sum.copy_to_host(
        &gpu_sum,
        sizeof(float)
    );

    // ================================================
    // MAX
    // ================================================

    ops::reduce_max(
        d_input,
        d_max
    );

    float gpu_max = 0.0f;

    d_max.copy_to_host(
        &gpu_max,
        sizeof(float)
    );

    const float sum_error =
        std::fabs(
            gpu_sum -
            static_cast<float>(cpu_sum)
        );

    const float max_error =
        std::fabs(
            gpu_max -
            cpu_max
        );

    std::cout << "=================================\n";
    std::cout << " VORTEX Reduction Operators\n";
    std::cout << "=================================\n\n";

    std::cout << "Elements:       "
              << N << '\n';

    std::cout << "CPU sum:        "
              << cpu_sum << '\n';

    std::cout << "GPU sum:        "
              << gpu_sum << '\n';

    std::cout << "Sum error:      "
              << sum_error << '\n';

    std::cout << "\nCPU max:        "
              << cpu_max << '\n';

    std::cout << "GPU max:        "
              << gpu_max << '\n';

    std::cout << "Max error:      "
              << max_error << '\n';

    const bool passed =
        sum_error <= 1e-4f &&
        max_error <= 1e-6f;

    if (passed) {
        std::cout
            << "\n[VORTEX] Reduction test PASSED\n";

        return 0;
    }

    std::cerr
        << "\n[VORTEX] Reduction test FAILED\n";

    return 1;
}
