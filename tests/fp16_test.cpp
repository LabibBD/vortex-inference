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
                static_cast<int>(i % 2001) - 1000
            ) * 0.001f;
    }

    Tensor fp32_input(
        {N},
        DataType::Float32,
        Device::CUDA
    );

    Tensor fp16_tensor(
        {N},
        DataType::Float16,
        Device::CUDA
    );

    Tensor fp32_output(
        {N},
        DataType::Float32,
        Device::CUDA
    );

    fp32_input.copy_from_host(
        input.data(),
        input.size() * sizeof(float)
    );

    ops::float32_to_float16(
        fp32_input,
        fp16_tensor
    );

    ops::float16_to_float32(
        fp16_tensor,
        fp32_output
    );

    std::vector<float> output(N);

    fp32_output.copy_to_host(
        output.data(),
        output.size() * sizeof(float)
    );

    float max_error = 0.0f;
    double mean_error = 0.0;

    for (
        std::size_t i = 0;
        i < N;
        ++i
    ) {
        const float error =
            std::fabs(
                output[i] - input[i]
            );

        max_error =
            std::max(
                max_error,
                error
            );

        mean_error += error;
    }

    mean_error /=
        static_cast<double>(N);

    std::cout
        << "=================================\n"
        << " VORTEX FP16 Conversion Test\n"
        << "=================================\n\n";

    std::cout
        << "Elements:          "
        << N << '\n';

    std::cout
        << "FP32 bytes:        "
        << fp32_input.bytes()
        << '\n';

    std::cout
        << "FP16 bytes:        "
        << fp16_tensor.bytes()
        << '\n';

    std::cout
        << "Memory reduction:  "
        << static_cast<double>(
               fp32_input.bytes()
           ) /
           static_cast<double>(
               fp16_tensor.bytes()
           )
        << "x\n";

    std::cout
        << "Maximum abs error: "
        << max_error << '\n';

    std::cout
        << "Mean abs error:    "
        << mean_error << '\n';

    const bool passed =
        fp16_tensor.bytes() * 2 ==
            fp32_input.bytes() &&
        max_error <= 1e-3f;

    if (passed) {
        std::cout
            << "\n[VORTEX] FP16 conversion test PASSED\n";

        return 0;
    }

    std::cerr
        << "\n[VORTEX] FP16 conversion test FAILED\n";

    return 1;
}
