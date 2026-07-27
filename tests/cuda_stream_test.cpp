#include "vortex/cuda_stream.hpp"
#include "vortex/operators/activations.hpp"
#include "vortex/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

int main() {
    using namespace vortex;

    constexpr std::size_t N =
        1 << 20;

    std::vector<float> input_a(N);
    std::vector<float> input_b(N);

    for (
        std::size_t i = 0;
        i < N;
        ++i
    ) {
        input_a[i] =
            static_cast<float>(
                static_cast<int>(i % 1001) - 500
            ) * 0.01f;

        input_b[i] =
            static_cast<float>(
                static_cast<int>(i % 701) - 350
            ) * 0.01f;
    }

    Tensor a(
        {N},
        DataType::Float32,
        Device::CUDA
    );

    Tensor b(
        {N},
        DataType::Float32,
        Device::CUDA
    );

    Tensor relu_output(
        {N},
        DataType::Float32,
        Device::CUDA
    );

    Tensor gelu_output(
        {N},
        DataType::Float32,
        Device::CUDA
    );

    a.copy_from_host(
        input_a.data(),
        input_a.size() * sizeof(float)
    );

    b.copy_from_host(
        input_b.data(),
        input_b.size() * sizeof(float)
    );

    CudaStream stream_a;
    CudaStream stream_b;

    // Independent work.
    ops::relu_async(
        a,
        relu_output,
        stream_a.get()
    );

    ops::gelu_async(
        b,
        gelu_output,
        stream_b.get()
    );

    // Explicit synchronization point.
    stream_a.synchronize();
    stream_b.synchronize();

    std::vector<float> relu_result(N);
    std::vector<float> gelu_result(N);

    relu_output.copy_to_host(
        relu_result.data(),
        relu_result.size() *
            sizeof(float)
    );

    gelu_output.copy_to_host(
        gelu_result.data(),
        gelu_result.size() *
            sizeof(float)
    );

    float relu_error =
        0.0f;

    float gelu_error =
        0.0f;

    constexpr float SQRT_2_OVER_PI =
        0.7978845608028654f;

    constexpr float GELU_COEFF =
        0.044715f;

    for (
        std::size_t i = 0;
        i < N;
        ++i
    ) {
        const float relu_reference =
            std::max(
                0.0f,
                input_a[i]
            );

        relu_error =
            std::max(
                relu_error,
                std::fabs(
                    relu_result[i] -
                    relu_reference
                )
            );

        const float x =
            input_b[i];

        const float gelu_reference =
            0.5f *
            x *
            (
                1.0f +
                std::tanh(
                    SQRT_2_OVER_PI *
                    (
                        x +
                        GELU_COEFF *
                        x * x * x
                    )
                )
            );

        gelu_error =
            std::max(
                gelu_error,
                std::fabs(
                    gelu_result[i] -
                    gelu_reference
                )
            );
    }

    std::cout
        << "=================================\n"
        << " VORTEX CUDA Stream Test\n"
        << "=================================\n\n";

    std::cout
        << "Independent streams: 2\n";

    std::cout
        << "Elements per branch: "
        << N << '\n';

    std::cout
        << "ReLU max error:      "
        << relu_error << '\n';

    std::cout
        << "GELU max error:      "
        << gelu_error << '\n';

    const bool passed =
        relu_error <= 1e-6f &&
        gelu_error <= 1e-5f;

    if (passed) {
        std::cout
            << "\n[VORTEX] CUDA stream execution PASSED\n";

        return 0;
    }

    std::cerr
        << "\n[VORTEX] CUDA stream execution FAILED\n";

    return 1;
}
