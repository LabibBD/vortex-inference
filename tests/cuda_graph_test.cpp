#include "vortex/cuda_graph.hpp"
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

    std::vector<float> input(N);

    for (
        std::size_t i = 0;
        i < N;
        ++i
    ) {
        input[i] =
            static_cast<float>(
                static_cast<int>(
                    i % 1001
                ) - 500
            ) * 0.01f;
    }

    Tensor d_input(
        {N},
        DataType::Float32,
        Device::CUDA
    );

    Tensor temporary(
        {N},
        DataType::Float32,
        Device::CUDA
    );

    Tensor output(
        {N},
        DataType::Float32,
        Device::CUDA
    );

    d_input.copy_from_host(
        input.data(),
        input.size() *
            sizeof(float)
    );

    CudaStream stream;
    CudaGraph graph;

    graph.capture(
        stream,
        [&](cudaStream_t cuda_stream) {
            ops::relu_async(
                d_input,
                temporary,
                cuda_stream
            );

            ops::gelu_async(
                temporary,
                output,
                cuda_stream
            );
        }
    );

    if (!graph.valid()) {
        std::cerr
            << "[VORTEX] Graph capture failed\n";

        return 1;
    }

    // Replay several times.
    for (int i = 0; i < 3; ++i) {
        graph.launch(
            stream.get()
        );
    }

    stream.synchronize();

    std::vector<float> result(N);

    output.copy_to_host(
        result.data(),
        result.size() *
            sizeof(float)
    );

    constexpr float SQRT_2_OVER_PI =
        0.7978845608028654f;

    constexpr float GELU_COEFF =
        0.044715f;

    float max_error =
        0.0f;

    for (
        std::size_t i = 0;
        i < N;
        ++i
    ) {
        const float relu =
            std::max(
                0.0f,
                input[i]
            );

        const float reference =
            0.5f *
            relu *
            (
                1.0f +
                std::tanh(
                    SQRT_2_OVER_PI *
                    (
                        relu +
                        GELU_COEFF *
                        relu *
                        relu *
                        relu
                    )
                )
            );

        max_error =
            std::max(
                max_error,
                std::fabs(
                    result[i] -
                    reference
                )
            );
    }

    std::cout
        << "=================================\n"
        << " VORTEX CUDA Graph Test\n"
        << "=================================\n\n";

    std::cout
        << "Captured kernels: 2\n";

    std::cout
        << "Graph replays:    3\n";

    std::cout
        << "Elements:         "
        << N << '\n';

    std::cout
        << "Maximum error:    "
        << max_error << '\n';

    if (max_error <= 1e-5f) {
        std::cout
            << "\n[VORTEX] CUDA Graph execution PASSED\n";

        return 0;
    }

    std::cerr
        << "\n[VORTEX] CUDA Graph execution FAILED\n";

    return 1;
}
