#include "vortex/operators/activations.hpp"
#include "vortex/operators/cast.hpp"
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

    std::vector<float> input(N);

    for (
        std::size_t i = 0;
        i < N;
        ++i
    ) {
        input[i] =
            static_cast<float>(
                static_cast<int>(
                    i % 2001
                ) - 1000
            ) * 0.005f;
    }

    Tensor d_fp32_input(
        {N},
        DataType::Float32,
        Device::CUDA
    );

    Tensor d_fp16_input(
        {N},
        DataType::Float16,
        Device::CUDA
    );

    Tensor d_fp16_output(
        {N},
        DataType::Float16,
        Device::CUDA
    );

    Tensor d_fp32_output(
        {N},
        DataType::Float32,
        Device::CUDA
    );

    d_fp32_input.copy_from_host(
        input.data(),
        N * sizeof(float)
    );

    ops::float32_to_float16(
        d_fp32_input,
        d_fp16_input
    );

    // ==================================================
    // FP16 ReLU
    // ==================================================

    ops::relu_fp16(
        d_fp16_input,
        d_fp16_output
    );

    ops::float16_to_float32(
        d_fp16_output,
        d_fp32_output
    );

    std::vector<float> relu_output(N);

    d_fp32_output.copy_to_host(
        relu_output.data(),
        N * sizeof(float)
    );

    float relu_max_error = 0.0f;

    for (
        std::size_t i = 0;
        i < N;
        ++i
    ) {
        const float reference =
            std::max(
                0.0f,
                input[i]
            );

        relu_max_error =
            std::max(
                relu_max_error,
                std::fabs(
                    relu_output[i] -
                    reference
                )
            );
    }

    // ==================================================
    // FP16 GELU
    // ==================================================

    ops::gelu_fp16(
        d_fp16_input,
        d_fp16_output
    );

    ops::float16_to_float32(
        d_fp16_output,
        d_fp32_output
    );

    std::vector<float> gelu_output(N);

    d_fp32_output.copy_to_host(
        gelu_output.data(),
        N * sizeof(float)
    );

    float gelu_max_error = 0.0f;
    double gelu_mean_error = 0.0;

    constexpr float SQRT_2_OVER_PI =
        0.7978845608028654f;

    constexpr float GELU_COEFF =
        0.044715f;

    for (
        std::size_t i = 0;
        i < N;
        ++i
    ) {
        const float x =
            input[i];

        const float reference =
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

        const float error =
            std::fabs(
                gelu_output[i] -
                reference
            );

        gelu_max_error =
            std::max(
                gelu_max_error,
                error
            );

        gelu_mean_error +=
            error;
    }

    gelu_mean_error /=
        static_cast<double>(N);

    std::cout
        << "=================================\n"
        << " VORTEX FP16 Activation Test\n"
        << "=================================\n\n";

    std::cout
        << "Elements:            "
        << N << '\n';

    std::cout
        << "FP16 ReLU max error: "
        << relu_max_error << '\n';

    std::cout
        << "FP16 GELU max error: "
        << gelu_max_error << '\n';

    std::cout
        << "FP16 GELU mean error:"
        << gelu_mean_error << '\n';

    const bool passed =
        relu_max_error <= 0.005f &&
        gelu_max_error <= 0.01f;

    if (passed) {
        std::cout
            << "\n[VORTEX] FP16 activation test PASSED\n";

        return 0;
    }

    std::cerr
        << "\n[VORTEX] FP16 activation test FAILED\n";

    return 1;
}
