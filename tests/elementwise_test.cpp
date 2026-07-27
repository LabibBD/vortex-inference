#include "vortex/operators/elementwise.hpp"
#include "vortex/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    using namespace vortex;

    constexpr std::size_t N = 1 << 20;

    std::vector<float> a(N);
    std::vector<float> b(N);

    for (std::size_t i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i % 1000) * 0.01f;
        b[i] = static_cast<float>(i % 500) * 0.02f;
    }

    Tensor d_a(
        {N},
        DataType::Float32,
        Device::CUDA
    );

    Tensor d_b(
        {N},
        DataType::Float32,
        Device::CUDA
    );

    Tensor d_output(
        {N},
        DataType::Float32,
        Device::CUDA
    );

    d_a.copy_from_host(
        a.data(),
        N * sizeof(float)
    );

    d_b.copy_from_host(
        b.data(),
        N * sizeof(float)
    );

    std::vector<float> gpu_output(N);

    // ================================================
    // ADD
    // ================================================

    ops::add(
        d_a,
        d_b,
        d_output
    );

    d_output.copy_to_host(
        gpu_output.data(),
        N * sizeof(float)
    );

    float add_max_error = 0.0f;

    for (std::size_t i = 0; i < N; ++i) {
        const float cpu_reference =
            a[i] + b[i];

        add_max_error = std::max(
            add_max_error,
            std::fabs(
                gpu_output[i] -
                cpu_reference
            )
        );
    }

    // ================================================
    // MULTIPLY
    // ================================================

    ops::multiply(
        d_a,
        d_b,
        d_output
    );

    d_output.copy_to_host(
        gpu_output.data(),
        N * sizeof(float)
    );

    float multiply_max_error = 0.0f;

    for (std::size_t i = 0; i < N; ++i) {
        const float cpu_reference =
            a[i] * b[i];

        multiply_max_error = std::max(
            multiply_max_error,
            std::fabs(
                gpu_output[i] -
                cpu_reference
            )
        );
    }

    std::cout << "=================================\n";
    std::cout << " VORTEX Elementwise Operators\n";
    std::cout << "=================================\n\n";

    std::cout << "Elements:             "
              << N << '\n';

    std::cout << "Add max error:        "
              << add_max_error << '\n';

    std::cout << "Multiply max error:   "
              << multiply_max_error << '\n';

    const bool passed =
        add_max_error <= 1e-5f &&
        multiply_max_error <= 1e-5f;

    if (passed) {
        std::cout
            << "\n[VORTEX] Elementwise test PASSED\n";

        return 0;
    }

    std::cerr
        << "\n[VORTEX] Elementwise test FAILED\n";

    return 1;
}
